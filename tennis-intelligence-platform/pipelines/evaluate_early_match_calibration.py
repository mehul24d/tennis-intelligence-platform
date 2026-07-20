"""
evaluate_early_match_calibration.py — the mandatory first step before touching n0 or
building any sensitivity-gated shrinkage, per the reviewer's framing: pure Markov's low
early-match variance is a structural artifact of using career-rate constants with
effectively infinite n0, not evidence of being better-calibrated. Matching that flatness
isn't the goal. The real question is whether the smoothed engine's finite-n0,
sensitivity-amplified early-match volatility is EARNED by better calibration, or just
noise for no benefit — and that can only be answered by measuring it directly, not by
eyeballing a chart.

Buckets purely by points_played_so_far_in_match (0-10, 10-25, 25-50) — NOT points
remaining, NOT game-score phase — reusing the already-computed
points_played_so_far_in_match column directly (added earlier this session specifically
for graded fatigue-proxy purposes, reused here for its more literal meaning: how far into
the match this point is). Compares LogLoss/Brier/ECE for all three engines within just
these early buckets across the full holdout set.
"""

from __future__ import annotations

import argparse
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
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.ml_informed_markov import (
    ml_informed_markov_predict, ml_informed_point_probabilities, ServeReturnPosterior,
    build_pretrained_prior,
)
from tennis_intel.live.live_win_probability import prob_a_wins_match_from_state
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error,
)
from generate_publication_trajectory import compute_composite_prematch_probability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EARLY_MATCH_BINS = [0, 10, 25, 50]
EARLY_MATCH_LABELS = ["0-10", "10-25", "25-50"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run on the full holdout set instead of the 150-match "
                             "smoke-test sample.")
    args = parser.parse_args()

    logger.info("Loading trained classifier...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

    logger.info("Building point dataset...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["match_year"] = points["match_id"].str[:4].astype(int)
    test_points = points[points["match_year"] >= HOLDOUT_YEAR].copy()
    test_points["player1_is_winner"] = (test_points["Svr"] == 1) == test_points["server_is_winner"]

    match_ids = np.sort(test_points["match_id"].unique())
    if args.full:
        selected = match_ids
        logger.info("Running FULL holdout set: %d matches.", len(selected))
    else:
        n_use = min(N_MATCHES, len(match_ids))
        selected = np.random.RandomState(RANDOM_STATE).choice(match_ids, size=n_use, replace=False)
        logger.info("Running smoke-test sample: %d matches.", n_use)

    eval_df = test_points[test_points["match_id"].isin(selected)].copy()
    eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)

    # points_played_so_far_in_match already exists (added earlier this session for
    # fatigue-proxy purposes) — reused here directly rather than recomputed.
    eval_df["early_bin"] = pd.cut(
        eval_df["points_played_so_far_in_match"], bins=EARLY_MATCH_BINS,
        labels=EARLY_MATCH_LABELS, right=False, include_lowest=True,
    )
    # Restrict entirely to points within the first 50 -- the whole point of this backtest
    # is the EARLY-match window specifically, not the whole match with an early bucket
    # tacked on.
    eval_df = eval_df[eval_df["points_played_so_far_in_match"] < 50].copy()
    logger.info("Restricted to points_played_so_far_in_match < 50: %d points remain "
               "(from %d matches)", len(eval_df), len(selected))

    records = []
    current_match_id, posterior = None, None

    for idx, row in enumerate(eval_df.to_dict("records")):
        if idx % 10000 == 0 and idx > 0:
            logger.info("  %d / %d points (%.1f%%)", idx, len(eval_df), 100 * idx / len(eval_df))

        if row["match_id"] != current_match_id:
            current_match_id = row["match_id"]
            p0_a_wins = compute_composite_prematch_probability(row)
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

        target = 1.0 if track_winner else 0.0
        markov_p = p_markov_winner if track_winner else (1.0 - p_markov_winner)
        smoothed_p = p_smoothed_winner if track_winner else (1.0 - p_smoothed_winner)
        unsmoothed_p = p_unsmoothed_winner if track_winner else (1.0 - p_unsmoothed_winner)

        records.append({
            "early_bin": row["early_bin"], "target": target,
            "markov_p": markov_p, "smoothed_p": smoothed_p, "unsmoothed_p": unsmoothed_p,
        })

    df = pd.DataFrame(records)

    print(f"\n=== Early-match (points_played_so_far < 50) calibration, all three engines ===\n")
    print(f"{'bin':<8} {'engine':<26} {'n':>8} {'log_loss':>10} {'brier':>10} {'ece':>8}")
    for label in EARLY_MATCH_LABELS:
        sub_all = df[df["early_bin"] == label]
        if len(sub_all) < 30:
            print(f"{label:<8} (too few points, n={len(sub_all)})")
            continue
        for col, name in [("markov_p", "Pure Markov"), ("smoothed_p", "ML-Informed (smoothed)"),
                           ("unsmoothed_p", "ML-Informed (unsmoothed)")]:
            y, p = sub_all["target"].values, sub_all[col].values
            ll = compute_log_loss(y, p)
            br = compute_brier_score(y, p)
            ece = expected_calibration_error(y, p, n_bins=5)
            print(f"{label:<8} {name:<26} {len(sub_all):>8} {ll:>10.4f} {br:>10.4f} {ece:>8.4f}")
        print()

    print("Interpretation:")
    print("- If smoothed's LogLoss/ECE is BETTER OR COMPARABLE to Markov's within these")
    print("  early bins despite looking visually louder, the volatility is buying real")
    print("  accuracy -- any fix belongs at the DISPLAY layer (e.g. an EMA purely on what")
    print("  gets charted), not the underlying mechanism.")
    print("- If smoothed's LogLoss/ECE is MEASURABLY WORSE than Markov's specifically in")
    print("  these early bins, that's the green light to tune n0 or add sensitivity-gated")
    print("  shrinkage -- validated against THIS backtest, not against how calm the chart")
    print("  looks.")


if __name__ == "__main__":
    main()