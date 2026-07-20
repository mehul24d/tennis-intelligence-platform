"""
diagnose_points_remaining.py — resolves the confound flagged after
diagnose_deciding_set_gap.py's point-level checks came back flat: "deciding set" and
"points remaining until the match ends" are correlated but not identical, and the
match_point bucket's unusually low log loss (0.27-0.34) in the earlier phase table is
itself evidence that points-remaining, not phase label, may be the real driver of
match-level log loss variation.

Makes points_remaining the PRIMARY grouping variable from the start (not a secondary
confirmation): bins ALL points by how many points remain until their match ends,
completely ignoring deciding-set status for the first pass, across the full dataset, for
all three engines. Then, as the confirmatory step, holds points_remaining fixed within a
single bin and checks whether deciding vs. non-deciding still differ — if the gap
vanishes once points_remaining is controlled for, deciding-set status was a proxy, not a
cause; if a real residual gap persists, that's a genuinely new, structural finding.

points_remaining is a simple reverse-count within each match (already point-sorted) —
not a new feature pipeline, just a different grouping key on data already in hand.
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
from tennis_intel.evaluation.metrics import compute_log_loss, compute_brier_score
from generate_publication_trajectory import compute_composite_prematch_probability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POINTS_REMAINING_BINS = [0, 10, 25, 50, 100, 200, np.inf]
POINTS_REMAINING_LABELS = ["0-10", "10-25", "25-50", "50-100", "100-200", "200+"]


def is_deciding_set(row: dict) -> bool:
    best_of = int(row.get("best_of", 3)) if pd.notna(row.get("best_of")) else 3
    sets_needed = (best_of // 2) + 1
    return row.get("Set1", 0) == row.get("Set2", 0) == sets_needed - 1


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

    eval_df["points_remaining"] = eval_df.groupby("match_id").cumcount(ascending=False)
    eval_df["points_remaining_bin"] = pd.cut(
        eval_df["points_remaining"], bins=POINTS_REMAINING_BINS,
        labels=POINTS_REMAINING_LABELS, right=False, include_lowest=True,
    )

    logger.info("Evaluating %d matches, %d points total", len(selected), len(eval_df))

    records = []
    current_match_id, posterior = None, None

    for idx, row in enumerate(eval_df.to_dict("records")):
        if idx % 10000 == 0 and idx > 0:
            logger.info("  %d / %d points (%.1f%%)", idx, len(eval_df), 100 * idx / len(eval_df))

        if row["match_id"] != current_match_id:
            current_match_id = row["match_id"]
            p0_a_wins = compute_composite_prematch_probability(row)
            loser_serve_surface = row.get("loser_first_serve_win_pct_surface_career")
            loser_serve_career = row.get("loser_first_serve_win_pct_career")
            if loser_serve_surface is not None and pd.notna(loser_serve_surface):
                opponent_serve = float(loser_serve_surface)
            elif loser_serve_career is not None and pd.notna(loser_serve_career):
                opponent_serve = float(loser_serve_career)
            else:
                opponent_serve = 0.65
            p_a_return_seed = 1.0 - opponent_serve
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
            "points_remaining_bin": row["points_remaining_bin"],
            "is_deciding": is_deciding_set(row), "target": target,
            "markov_p": markov_p, "smoothed_p": smoothed_p, "unsmoothed_p": unsmoothed_p,
        })

    df = pd.DataFrame(records)

    print(f"\n=== PRIMARY PASS: log loss binned PURELY by points_remaining "
          f"(deciding-set status ignored) ===")
    for col, name in [("markov_p", "Pure Markov"), ("smoothed_p", "ML-Informed (smoothed)"),
                       ("unsmoothed_p", "ML-Informed (unsmoothed)")]:
        print(f"\n{name}:")
        print(f"{'bin':<10} {'n':>8} {'log_loss':>10} {'brier':>10}")
        for label in POINTS_REMAINING_LABELS:
            sub = df[df["points_remaining_bin"] == label]
            if len(sub) < 30:
                print(f"{label:<10} {len(sub):>8}  (too few points)")
                continue
            y, p = sub["target"].values, sub[col].values
            print(f"{label:<10} {len(sub):>8} {compute_log_loss(y, p):>10.4f} "
                  f"{compute_brier_score(y, p):>10.4f}")

    print(f"\n=== CONFIRMATORY PASS: within each points_remaining bin, "
          f"deciding vs non-deciding ===")
    for col, name in [("markov_p", "Pure Markov"), ("smoothed_p", "ML-Informed (smoothed)"),
                       ("unsmoothed_p", "ML-Informed (unsmoothed)")]:
        print(f"\n{name}:")
        print(f"{'bin':<10} {'segment':<14} {'n':>8} {'log_loss':>10}")
        for label in POINTS_REMAINING_LABELS:
            bin_df = df[df["points_remaining_bin"] == label]
            for deciding, seg_label in [(False, "non-deciding"), (True, "deciding")]:
                sub = bin_df[bin_df["is_deciding"] == deciding]
                if len(sub) < 30:
                    continue
                y, p = sub["target"].values, sub[col].values
                print(f"{label:<10} {seg_label:<14} {len(sub):>8} {compute_log_loss(y, p):>10.4f}")

    print("\nInterpretation:")
    print("- PRIMARY PASS: if log loss rises smoothly and monotonically as")
    print("  points_remaining increases, largely independent of engine, that confirms")
    print("  this is a generic long-horizon-prediction property (more future uncertainty")
    print("  to integrate over), not a deciding-set-specific effect.")
    print("- CONFIRMATORY PASS: within a single points_remaining bin, if deciding and")
    print("  non-deciding log loss are close, deciding-set status was a PROXY for")
    print("  points_remaining, not an independent cause -- fully closes the investigation.")
    print("  If a real gap persists even after conditioning on points_remaining, that IS")
    print("  a genuinely new, structural, deciding-set-specific finding (e.g. pressure")
    print("  effects not captured by any current feature) worth chasing further.")


if __name__ == "__main__":
    main()