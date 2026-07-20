"""
diagnose_deciding_set_gap.py — the deciding-set-specific diagnostic proposed after
evaluate_full_match_calibration.py's full-scale run found the smoothed engine's LogLoss
advantage over pure Markov nearly disappears in deciding sets (0.596 vs 0.614, only +3%
relative) despite a dramatic advantage everywhere else (0.480 vs 0.557, +14%).

Separates two genuinely different hypotheses that a match-level metric alone cannot tell
apart:
  (a) The raw point-level classifier itself gets worse at predicting individual POINT
      outcomes in deciding sets (a genuine modeling gap — missing fatigue/pressure
      features, matching the "heightened pressure-sensitivity" hypothesis).
  (b) The classifier is fine at the point level, but the posterior/blend mechanism
      specifically loses value relative to what the raw classifier alone already offers
      by the time a match reaches a deciding set (matching the "posteriors have
      accumulated so much evidence they've converged toward something less
      discriminating" hypothesis).

Computes POINT-LEVEL (not match-level) log loss/Brier for three things, deciding vs
non-deciding set: the raw classifier's own prediction, the posterior mean alone (no
blend), and the actual blended value fed into the recursion — all evaluated against
whether the point's actual server won THAT point, not the eventual match outcome.
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
    ml_informed_point_probabilities, ServeReturnPosterior, build_pretrained_prior,
    recursion_sensitivity, sensitivity_aware_blend,
)
from tennis_intel.evaluation.metrics import compute_log_loss, compute_brier_score
from generate_publication_trajectory import compute_composite_prematch_probability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
        logger.info("Running smoke-test sample: %d matches. Deciding-set points are a "
                   "minority of all points, so this sample may still be too small for a "
                   "confident conclusion here — treat as a first look, prefer --full "
                   "before acting on the result.", n_use)

    eval_df = test_points[test_points["match_id"].isin(selected)].copy()
    eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
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

        state = row_to_match_state(row)
        p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(row, model, feature_cols)

        sens_serve = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "serve")
        sens_return = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "return")
        pts_obs_serve = posterior.points_observed_serve()
        pts_obs_return = posterior.points_observed_return()

        blended_serve = sensitivity_aware_blend(p_a_serve_raw, posterior.mean_serve(), sens_serve,
                                                points_observed=pts_obs_serve)
        blended_return = sensitivity_aware_blend(p_a_return_raw, posterior.mean_return(), sens_return,
                                                 points_observed=pts_obs_return)

        deciding = is_deciding_set(row)

        # POINT-LEVEL target: did A win THIS point? (not the eventual match outcome)
        pt_winner = row.get("PtWinner")
        if pd.notna(pt_winner):
            p1_is_winner = bool(row.get("player1_is_winner", True))
            a_won_point = (int(pt_winner) == 1) if p1_is_winner else (int(pt_winner) == 2)

            if state.server_is_a:
                # A is serving: raw/posterior/blend serve-side predictions apply
                records.append({
                    "is_deciding": deciding, "role": "serve", "a_won_point": float(a_won_point),
                    "raw_pred": p_a_serve_raw, "posterior_pred": posterior.mean_serve(),
                    "blend_pred": blended_serve,
                })
                posterior = posterior.update_serve(a_won_point)
            else:
                # A is returning: raw/posterior/blend return-side predictions apply
                records.append({
                    "is_deciding": deciding, "role": "return", "a_won_point": float(a_won_point),
                    "raw_pred": p_a_return_raw, "posterior_pred": posterior.mean_return(),
                    "blend_pred": blended_return,
                })
                posterior = posterior.update_return(a_won_point)

    df = pd.DataFrame(records)
    df["raw_pred_c"] = df["raw_pred"].clip(0.001, 0.999)
    df["posterior_pred_c"] = df["posterior_pred"].clip(0.001, 0.999)
    df["blend_pred_c"] = df["blend_pred"].clip(0.001, 0.999)

    print(f"\n=== POINT-LEVEL accuracy (predicting THIS point's winner, not the match) ===")
    print(f"{'Segment':<16} {'Source':<12} {'n':>8} {'log_loss':>10} {'brier':>10}")
    for deciding, seg_label in [(False, "non-deciding"), (True, "deciding")]:
        sub = df[df["is_deciding"] == deciding]
        if len(sub) < 30:
            print(f"{seg_label:<16} {'(too few points, n=' + str(len(sub)) + ')':<12}")
            continue
        y = sub["a_won_point"].values
        for col, label in [("raw_pred_c", "raw classifier"), ("posterior_pred_c", "posterior only"),
                            ("blend_pred_c", "blend (actual)")]:
            p = sub[col].values
            print(f"{seg_label:<16} {label:<12} {len(sub):>8} "
                  f"{compute_log_loss(y, p):>10.4f} {compute_brier_score(y, p):>10.4f}")

    print(f"\n=== Relative degradation, deciding vs non-deciding (higher = worse in deciding) ===")
    for col, label in [("raw_pred_c", "raw classifier"), ("posterior_pred_c", "posterior only"),
                        ("blend_pred_c", "blend (actual)")]:
        non_dec = df[~df["is_deciding"]]
        dec = df[df["is_deciding"]]
        if len(dec) < 30:
            continue
        ll_non_dec = compute_log_loss(non_dec["a_won_point"].values, non_dec[col].values)
        ll_dec = compute_log_loss(dec["a_won_point"].values, dec[col].values)
        pct_change = 100 * (ll_dec - ll_non_dec) / ll_non_dec
        print(f"{label:<16} non-deciding LogLoss={ll_non_dec:.4f}  deciding LogLoss={ll_dec:.4f}  "
              f"relative change={pct_change:+.1f}%")

    print("\nInterpretation:")
    print("- If 'raw classifier' degrades sharply in deciding sets (large +% change), the")
    print("  point-level model itself is missing signal there -- a genuine modeling gap")
    print("  (e.g. fatigue/pressure features), independent of the smoothing mechanism.")
    print("- If 'raw classifier' stays roughly stable but 'posterior only' or")
    print("  'blend (actual)' degrades much more, the smoothing/blend mechanism itself is")
    print("  specifically losing value in deciding sets relative to what the classifier")
    print("  alone already knew -- pointing at the posterior's accumulated evidence (by")
    print("  a deciding set, often 150+ points deep) being LESS discriminating than a")
    print("  fresh per-point read, not a data or feature problem.")


if __name__ == "__main__":
    main()