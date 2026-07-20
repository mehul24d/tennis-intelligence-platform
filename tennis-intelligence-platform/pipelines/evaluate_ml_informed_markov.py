"""
evaluate_ml_informed_markov.py — evaluates the ML-informed Markov engine (Day 9 point
classifier's context-aware predictions fed into the validated Markov recursion, see
src/tennis_intel/live/ml_informed_markov.py) against pure Markov and pure ML+MC, on the
SAME 150 matches / same RANDOM_STATE / same match-selection logic as
evaluate_live_engines_v2.py, for direct comparability with the existing Day 11 numbers.

Reuses tracked_player_is_winner, _row_to_match_state, and markov_p_winner directly from
evaluate_live_engines_v2.py rather than duplicating them — this new engine differs from
that file ONLY in what point-probability feeds the (unmodified) Markov recursion.

RUNS SEQUENTIALLY, NOT IN PARALLEL: unlike ML+MC (hundreds of Monte Carlo simulations per
point), this engine costs a single classifier call pair plus a closed-form recursion per
point — cheap enough that the parallel worker-pool machinery in evaluate_live_engines_v2.py
is unnecessary complexity here, not a missing optimization.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_live_engines_v2 import (
    tracked_player_is_winner, _row_to_match_state, markov_p_winner,
    HOLDOUT_YEAR, N_MATCHES, RANDOM_STATE, POINT_FILES, PROCESSED,
)
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.ml_informed_markov import (
    ml_informed_markov_predict, ServeReturnPosterior, build_pretrained_prior,
)
from generate_publication_trajectory import compute_composite_prematch_probability
from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error, paired_bootstrap_diff,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Loading trained classifier...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model = payload["gradient_boosting"]
    feature_cols = payload["feature_cols"]

    logger.info("Building point dataset (same as evaluate_live_engines_v2.py)...")
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
    logger.info("Evaluating %d matches, %d points (SAME selection as Day 11)", n_use, len(eval_df))

    markov_preds, ml_informed_preds, targets = [], [], []
    t0 = time.perf_counter()
    posterior = None
    current_match_id = None

    for i, row in enumerate(eval_df.to_dict("records")):
        # Reset the Beta-Binomial posterior at the start of EACH match — it represents
        # accumulated in-match evidence for one specific pair of players, and must NOT
        # carry over from the previous (unrelated) match in this multi-match evaluation
        # loop. Uses the CORRECTED construction (build_pretrained_prior): the feature-rich
        # pre-match win probability, inverted through the Markov recursion, with a
        # confidence-derived n0 — not a raw career rate with one fixed n0 for every match.
        if row["match_id"] != current_match_id:
            current_match_id = row["match_id"]
            # BUG FIX (external review, 2026-07): this was the ONE remaining call site
            # still using compute_ml_pre_match_probability (a 200-trial Monte Carlo
            # rollout, confirmed systematically overconfident — ~0.87 vs a real-world
            # ~55-60% expectation, diagnosed directly on the 2025 Roland Garros final)
            # as its ACTUAL seeding input, not just a diagnostic comparison printed
            # alongside the real one — every other evaluation script in this project
            # (replay_match.py, generate_publication_trajectory.py,
            # evaluate_full_match_calibration.py, diagnose_points_remaining.py, etc.)
            # already used compute_composite_prematch_probability as the real seed. This
            # script's results were, until now, computed from a systematically
            # overconfident pre-match prior inconsistent with every other measurement in
            # the project.
            p0_a_wins = compute_composite_prematch_probability(row)

            # BUG FIX (external review, 2026-07): see return_seed.py's module docstring
            # for the full derivation. The old inline construction (1 - opponent's
            # first_serve_win_pct_career) systematically understated the returner's true
            # seed by ignoring second-serve points entirely.
            p_a_return_seed = compute_p_a_return_seed(row, track_winner=True)

            elo_matches_played_a = row.get("elo_matches_played_pre_winner")
            elo_matches_played_b = row.get("elo_matches_played_pre_loser")
            best_of_val = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3

            # Composite n0 upgrade (external audit, 2026-07, Architecture Review finding
            # C): matchup-specific H2H depth, not just career match count.
            h2h_meetings = None
            winner_h2h = row.get("winner_h2h_wins_pre_match")
            loser_h2h = row.get("loser_h2h_wins_pre_match")
            if pd.notna(winner_h2h) and pd.notna(loser_h2h):
                h2h_meetings = float(winner_h2h) + float(loser_h2h)

            tourney_h2h_meetings = None
            winner_tourney_h2h = row.get("winner_tourney_h2h_wins_pre_match")
            loser_tourney_h2h = row.get("loser_tourney_h2h_wins_pre_match")
            if pd.notna(winner_tourney_h2h) and pd.notna(loser_tourney_h2h):
                tourney_h2h_meetings = float(winner_tourney_h2h) + float(loser_tourney_h2h)

            p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
                p0_a_wins, p_a_return_seed, best_of_val,
                elo_matches_played_a=elo_matches_played_a,
                elo_matches_played_b=elo_matches_played_b,
                h2h_meetings=h2h_meetings, tourney_h2h_meetings=tourney_h2h_meetings,
            )
            posterior = ServeReturnPosterior.from_pretrained_prior(
                p_serve0, n0_serve, p_return0, n0_return
            )

        track_winner = tracked_player_is_winner(row["match_id"])
        state = _row_to_match_state(row)

        p_markov_winner = markov_p_winner(row)
        p_ml_informed_winner, posterior = ml_informed_markov_predict(
            state, row, model, feature_cols, posterior
        )

        if track_winner:
            markov_preds.append(p_markov_winner)
            ml_informed_preds.append(p_ml_informed_winner)
            targets.append(1.0)
        else:
            markov_preds.append(1.0 - p_markov_winner)
            ml_informed_preds.append(1.0 - p_ml_informed_winner)
            targets.append(0.0)

        if (i + 1) % 2000 == 0:
            elapsed = time.perf_counter() - t0
            logger.info("  %d / %d points (%.1fs elapsed)", i + 1, len(eval_df), elapsed)

    elapsed = time.perf_counter() - t0
    logger.info("Done in %.1fs (%.1f points/sec)", elapsed, len(eval_df) / elapsed)

    y = np.array(targets)
    markov_arr = np.array(markov_preds)
    ml_informed_arr = np.array(ml_informed_preds)

    print(f"\n=== ML-Informed Markov vs. Pure Markov ===")
    print(f"Matches: {n_use}, points: {len(eval_df):,}, "
          f"target balance: {y.mean():.3f}\n")
    print(f"{'Engine':<20} {'LogLoss':>10} {'Brier':>10} {'ECE':>10}")
    for name, preds in [("Pure Markov", markov_arr), ("ML-Informed Markov", ml_informed_arr)]:
        ll = compute_log_loss(y, preds)
        brier = compute_brier_score(y, preds)
        ece = expected_calibration_error(y, preds)
        print(f"{name:<20} {ll:>10.4f} {brier:>10.4f} {ece:>10.4f}")

    result = paired_bootstrap_diff(y, ml_informed_arr, markov_arr,
                                   metric_fn=compute_log_loss, metric_name="log_loss")
    direction = "ML-informed better" if result.point_estimate_diff < 0 else "ML-informed worse"
    sig = "SIGNIFICANT" if not result.zero_in_ci else "not significant"
    print(f"\nPaired bootstrap (ML-informed - Pure Markov): diff={result.point_estimate_diff:+.4f} "
          f"95% CI=[{result.ci_lower:+.4f}, {result.ci_upper:+.4f}] -> {sig} ({direction})")

    print(f"\n(For reference, ML+MC's POST-LEAKAGE-FIX numbers on this SAME match "
          f"selection were log_loss=0.5390, brier=0.1793, ece=0.0744 — measured after "
          f"server_is_winner was removed and the classifier retrained, see the audit "
          f"remediation's Step 2 impact analysis. The earlier 0.2652/0.0781/0.0468 figures "
          f"quoted in this comment for months were from the PRE-fix, leaky classifier and "
          f"must not be used as a comparison baseline — they overstate ML+MC's real "
          f"performance by a wide margin, exactly like every other pre-fix number in this "
          f"project. Compare ML-Informed Markov above against THESE corrected figures, not "
          f"the old ones.)")

    out_df = pd.DataFrame({
        "match_id": eval_df["match_id"], "Pt": eval_df["Pt"],
        "markov_pred": markov_arr, "ml_informed_pred": ml_informed_arr, "target": y,
    })
    out_path = PROCESSED / "ml_informed_markov_predictions.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"\nSaved per-point predictions to {out_path}")


if __name__ == "__main__":
    main()