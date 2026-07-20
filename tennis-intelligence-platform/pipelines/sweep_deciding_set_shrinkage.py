"""
sweep_deciding_set_shrinkage.py — measures the deciding-set shrinkage mechanism
(sensitivity_aware_blend's is_deciding_set/deciding_set_shrinkage_factor, the fourth and
final attempt against the confirmed, structural deciding-set log-loss gap) at several
candidate shrinkage factors, using the SAME points-remaining-controlled confirmatory
methodology already validated in diagnose_points_remaining.py — because a plausible
mechanism is not evidence it works, per this project's own established discipline, and
every prior attempt (a binary flag, two fatigue proxies, a break-point-return feature, an
ordinal pressure index) looked reasonable going in and needed exactly this kind of direct
measurement to confirm or refute.

Reports, for each candidate factor (including 1.0 = no shrinkage, the current baseline),
the SAME within-points_remaining-bin deciding-vs-non-deciding log loss comparison used
throughout this whole investigation — the one metric that actually distinguishes "moved
the gap" from "looked plausible."
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
    tracked_player_is_winner, HOLDOUT_YEAR, N_MATCHES, RANDOM_STATE, POINT_FILES, PROCESSED,
)
from tennis_intel.live.match_state_conversion import row_to_match_state
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.ml_informed_markov import (
    ml_informed_markov_predict, ServeReturnPosterior, build_pretrained_prior,
)
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.evaluation.metrics import compute_log_loss
from generate_publication_trajectory import compute_composite_prematch_probability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POINTS_REMAINING_BINS = [0, 10, 25, 50, 100, 200, np.inf]
POINTS_REMAINING_LABELS = ["0-10", "10-25", "25-50", "50-100", "100-200", "200+"]

# Candidate shrinkage factors to test in one run. 1.0 = current baseline (no shrinkage,
# byte-identical to every existing measurement). Values below 1.0 progressively trust
# the raw classifier more, and the prior/posterior less, specifically in deciding sets.
SHRINKAGE_FACTORS = [1.0, 0.8, 0.6, 0.4, 0.2]


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
    logger.info("Evaluating %d matches, %d points, %d shrinkage factors",
               len(selected), len(eval_df), len(SHRINKAGE_FACTORS))
    logger.info("NOTE: this re-runs the full per-point loop ONCE PER shrinkage factor "
               "(%d total passes) — expect roughly %dx the runtime of "
               "diagnose_points_remaining.py's single-pass run on the same match count.",
               len(SHRINKAGE_FACTORS), len(SHRINKAGE_FACTORS))

    # Run the full per-point loop ONCE PER SHRINKAGE FACTOR — the posterior update
    # itself is unaffected by shrinkage (only the BLENDED value changes, not what the
    # posterior learns from real outcomes), but re-running per factor is the simplest,
    # least error-prone way to guarantee no state leaks between factors.
    results_by_factor = {}
    for factor in SHRINKAGE_FACTORS:
        logger.info("Running with deciding_set_shrinkage_factor=%.2f...", factor)
        records = []
        current_match_id, posterior = None, None

        for row in eval_df.to_dict("records"):
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

            p_smoothed_winner, posterior = ml_informed_markov_predict(
                state, row, model, feature_cols, posterior,
                deciding_set_shrinkage_factor=factor,
            )

            target = 1.0 if track_winner else 0.0
            smoothed_p = p_smoothed_winner if track_winner else (1.0 - p_smoothed_winner)

            records.append({
                "points_remaining_bin": row["points_remaining_bin"],
                "is_deciding": bool(row.get("deciding_set", False)),
                "target": target, "smoothed_p": smoothed_p,
            })

        results_by_factor[factor] = pd.DataFrame(records)

    print(f"\n=== Deciding-set log loss at matched points_remaining, by shrinkage factor ===\n")
    print(f"{'bin':<10} " + "".join(f"factor={f:<10.1f}" for f in SHRINKAGE_FACTORS))
    for label in POINTS_REMAINING_LABELS:
        row_str = f"{label:<10} "
        for factor in SHRINKAGE_FACTORS:
            df = results_by_factor[factor]
            sub = df[(df["points_remaining_bin"] == label) & (df["is_deciding"])]
            if len(sub) < 30:
                row_str += f"{'(n<30)':<17}"
                continue
            ll = compute_log_loss(sub["target"].values, sub["smoothed_p"].values)
            row_str += f"{ll:<17.4f}"
        print(row_str)

    print(f"\n=== Non-deciding log loss at matched points_remaining, by shrinkage factor "
          f"(should stay ~constant across factors — shrinkage only applies to deciding "
          f"sets) ===\n")
    print(f"{'bin':<10} " + "".join(f"factor={f:<10.1f}" for f in SHRINKAGE_FACTORS))
    for label in POINTS_REMAINING_LABELS:
        row_str = f"{label:<10} "
        for factor in SHRINKAGE_FACTORS:
            df = results_by_factor[factor]
            sub = df[(df["points_remaining_bin"] == label) & (~df["is_deciding"])]
            if len(sub) < 30:
                row_str += f"{'(n<30)':<17}"
                continue
            ll = compute_log_loss(sub["target"].values, sub["smoothed_p"].values)
            row_str += f"{ll:<17.4f}"
        print(row_str)

    print("\nInterpretation:")
    print("- factor=1.0 is the current baseline (identical to every prior measurement).")
    print("- If a lower factor meaningfully REDUCES deciding-set log loss at matched")
    print("  points_remaining, while non-deciding log loss stays flat (confirming the")
    print("  mechanism is correctly isolated to deciding sets only), that's the first")
    print("  confirmed improvement across FOUR attempts against this gap.")
    print("- If log loss is flat or WORSE across all factors, that's strong, final")
    print("  evidence the gap is not fixable by adjusting how much the recursion trusts")
    print("  its own prior/posterior either — consistent with the same-day")
    print("  quality-gap explanation, which no adjustment to EITHER the classifier's")
    print("  inputs OR the blend's trust weighting can address, since it's a property")
    print("  of the specific match that no historical data could have anticipated.")


if __name__ == "__main__":
    main()