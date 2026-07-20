"""
evaluate_early_deficit_calibration.py — the final, decisive check in the early-match
volatility investigation. Every mechanistic hypothesis (blend-weight sensitivity, n0
sizing, feature staleness, determinism, double-counting, orientation) has been
individually traced and individually ruled out on real data. The remaining question is no
longer "is there a bug" — it's "is the smoothed engine's stated probability empirically
justified in exactly the score states where it looked most extreme (an early break
deficit), or merely unsurprising given clean, honestly-fed inputs."

Buckets real points by a concrete, operational definition of "early-set break deficit"
(still in set 1, tracked player has 0 games while the opponent has 2+), then compares
each engine's stated win probability in that bucket against the REAL, eventual match
outcome — a reliability check specifically for the situation the investigation's example
(row 31, 0-4 in games) came from, not a whole-match average that would hide this.

Reuses calibration_table (the same reliability-table logic used in every other evaluation
this project has run) rather than building new binning logic, and reuses
ml_informed_markov_predict directly (not a reimplementation) for the smoothed engine's
predictions, exactly matching real production behavior.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_live_engines_v2 import (
    tracked_player_is_winner, markov_p_winner, HOLDOUT_YEAR, N_MATCHES, RANDOM_STATE,
    POINT_FILES, PROCESSED,
)
from tennis_intel.live.match_state_conversion import row_to_match_state
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.ml_informed_markov import (
    ml_informed_markov_predict, ml_informed_point_probabilities, ServeReturnPosterior,
    build_pretrained_prior,
)
from tennis_intel.live.live_win_probability import prob_a_wins_match_from_state
from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error, calibration_table,
)
from generate_publication_trajectory import compute_composite_prematch_probability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def is_early_set_break_deficit(row: dict, track_winner: bool) -> bool:
    """Operational definition of the bucket this whole investigation started from: still
    in set 1 (a_sets == b_sets == 0), and the TRACKED player ('A' — the real winner if
    track_winner, else the real loser) has 0 games in the current set while the opponent
    has at least 2 — a genuine, meaningful early deficit, not a narrow single score line,
    so the bucket has enough real matches in it to be statistically meaningful."""
    if row.get("Set1", 0) != 0 or row.get("Set2", 0) != 0:
        return False
    p1_is_winner = bool(row["player1_is_winner"])
    # "A" tracks the winner throughout this project's convention; map games accordingly.
    a_is_p1 = p1_is_winner if track_winner else (not p1_is_winner)
    a_games = row["Gm1"] if a_is_p1 else row["Gm2"]
    b_games = row["Gm2"] if a_is_p1 else row["Gm1"]
    return a_games == 0 and b_games >= 2


def main() -> None:
    logger.info("Loading trained classifier...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

    logger.info("Building point dataset (same selection as evaluate_live_engines_v2.py)...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["match_year"] = points["match_id"].str[:4].astype(int)
    test_points = points[points["match_year"] >= HOLDOUT_YEAR].copy()
    test_points["player1_is_winner"] = (test_points["Svr"] == 1) == test_points["server_is_winner"]

    match_ids = np.sort(test_points["match_id"].unique())
    n_use = min(N_MATCHES, len(match_ids))
    selected = np.random.RandomState(RANDOM_STATE).choice(match_ids, size=n_use, replace=False)
    eval_df = test_points[test_points["match_id"].isin(selected)].copy()
    eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
    logger.info("Evaluating %d matches, %d points total", n_use, len(eval_df))

    markov_preds, smoothed_preds, unsmoothed_preds, targets, in_bucket_flags = [], [], [], [], []
    current_match_id, posterior = None, None

    for row in eval_df.to_dict("records"):
        if row["match_id"] != current_match_id:
            current_match_id = row["match_id"]
            p0_a_wins = compute_composite_prematch_probability(row)
            # BUG FIX (external review, 2026-07): see return_seed.py's module docstring.
            p_a_return_seed = compute_p_a_return_seed(row, track_winner=True)
            elo_a = row.get("elo_matches_played_pre_winner")
            elo_b = row.get("elo_matches_played_pre_loser")
            h2h = None
            if pd.notna(row.get("winner_h2h_wins_pre_match")) and pd.notna(row.get("loser_h2h_wins_pre_match")):
                h2h = float(row["winner_h2h_wins_pre_match"]) + float(row["loser_h2h_wins_pre_match"])
            best_of_val = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3

            p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
                p0_a_wins, p_a_return_seed, best_of_val,
                elo_matches_played_a=elo_a, elo_matches_played_b=elo_b, h2h_meetings=h2h,
            )
            posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)

        track_winner = tracked_player_is_winner(row["match_id"])
        state = row_to_match_state(row)

        p_markov_winner = markov_p_winner(row)
        p_smoothed_winner, posterior = ml_informed_markov_predict(state, row, model, feature_cols, posterior)

        p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(row, model, feature_cols)
        p_a_serve_raw_c = float(np.clip(p_a_serve_raw, 0.01, 0.99))
        p_a_return_raw_c = float(np.clip(p_a_return_raw, 0.01, 0.99))
        p_unsmoothed_winner = prob_a_wins_match_from_state(state, p_a_serve_raw_c, p_a_return_raw_c)

        in_bucket = is_early_set_break_deficit(row, track_winner)

        if track_winner:
            markov_preds.append(p_markov_winner)
            smoothed_preds.append(p_smoothed_winner)
            unsmoothed_preds.append(p_unsmoothed_winner)
            targets.append(1.0)
        else:
            markov_preds.append(1.0 - p_markov_winner)
            smoothed_preds.append(1.0 - p_smoothed_winner)
            unsmoothed_preds.append(1.0 - p_unsmoothed_winner)
            targets.append(0.0)
        in_bucket_flags.append(in_bucket)

    y = np.array(targets)
    markov_arr = np.array(markov_preds)
    smoothed_arr = np.array(smoothed_preds)
    unsmoothed_arr = np.array(unsmoothed_preds)
    in_bucket = np.array(in_bucket_flags)

    n_bucket = in_bucket.sum()
    logger.info("Points in the early-set break-deficit bucket: %d / %d (%.1f%%)",
               n_bucket, len(y), 100 * n_bucket / len(y))
    if n_bucket < 30:
        logger.warning("Bucket sample size is small (%d) — treat results as indicative, "
                       "not conclusive.", n_bucket)

    print(f"\n=== Whole-match metrics (for reference) ===")
    print(f"{'Engine':<20} {'LogLoss':>10} {'Brier':>10} {'ECE':>10}")
    for name, arr in [("Pure Markov", markov_arr), ("ML-Informed (smoothed)", smoothed_arr),
                       ("ML-Informed (unsmoothed)", unsmoothed_arr)]:
        print(f"{name:<20} {compute_log_loss(y, arr):>10.4f} {compute_brier_score(y, arr):>10.4f} "
              f"{expected_calibration_error(y, arr):>10.4f}")

    print(f"\n=== EARLY-SET BREAK-DEFICIT BUCKET ONLY (n={n_bucket}) — the actual question ===")
    print("(A tracked player down 0 games to 2+ in set 1 — the exact situation row 31's")
    print(" example came from. This is what settles whether the smoothed engine's")
    print(" extremity there was justified or not.)\n")
    print(f"{'Engine':<20} {'LogLoss':>10} {'Brier':>10} {'ECE':>10} {'MeanPred':>10}")
    for name, arr in [("Pure Markov", markov_arr), ("ML-Informed (smoothed)", smoothed_arr),
                       ("ML-Informed (unsmoothed)", unsmoothed_arr)]:
        y_b, p_b = y[in_bucket], arr[in_bucket]
        if len(y_b) == 0:
            print(f"{name:<20} {'(no data in bucket)':>44}")
            continue
        print(f"{name:<20} {compute_log_loss(y_b, p_b):>10.4f} {compute_brier_score(y_b, p_b):>10.4f} "
              f"{expected_calibration_error(y_b, p_b, n_bins=5):>10.4f} {p_b.mean():>10.4f}")

    print(f"\n=== Reliability table, smoothed engine, bucket only ===")
    y_b, p_b = y[in_bucket], smoothed_arr[in_bucket]
    if len(y_b) > 0:
        print(calibration_table(y_b, p_b, n_bins=5).to_string(index=False))
    print(f"\nActual bucket win rate for the tracked (deficit) player: {y_b.mean():.4f} "
          f"({int(y_b.sum())}/{len(y_b)})")
    print("If the smoothed engine's mean prediction in this bucket is CLOSE to this real")
    print("rate, its low-probability calls here were empirically justified, not overreacting.")
    print("If its mean prediction is notably LOWER than the real rate, it IS overconfident")
    print("in this specific situation, even though no data-feeding bug was found.")


if __name__ == "__main__":
    main()