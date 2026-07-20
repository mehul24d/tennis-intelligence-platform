"""
analyze_n0_fix_impact.py — quantifies the actual impact of the elo_matches_played_b fix
(n0_return_a previously used elo_matches_played_a instead of elo_matches_played_b) on the
real 150-match evaluation sample, per the prescribed roadmap: measure before assuming.

Runs BOTH the pre-fix and post-fix versions of build_pretrained_prior side by side on the
SAME real match selection, in ONE pass, so the comparison is exact and not subject to
match-selection drift between two separate runs. The pre-fix logic is reproduced here in
an isolated, clearly-labeled function (_old_build_pretrained_prior) purely for this
comparison — the shipped production code (ml_informed_markov.py) remains correctly fixed
and is never modified or reverted.

Reports:
  - Distribution of per-match maximum probability differences (old vs new)
  - Count of matches affected by >1%, >5%, >10% maximum probability shift
  - Whether the impact concentrates in matches with large player-experience asymmetry
    (elo_matches_played_a vs elo_matches_played_b) — the specific, targeted hypothesis the
    bug's own mechanism implies, not just an aggregate before/after number
  - Full log loss / Brier / ECE for both versions on the same sample, for direct comparison
    against the two separate runs already reported in chat
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
    tracked_player_is_winner, _row_to_match_state, HOLDOUT_YEAR, N_MATCHES, RANDOM_STATE,
    POINT_FILES, PROCESSED,
)
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.ml_informed_markov import (
    ml_informed_markov_predict, ServeReturnPosterior, invert_prematch_probability,
    build_pretrained_prior,
)
from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error,
)
from generate_publication_trajectory import compute_composite_prematch_probability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _old_build_pretrained_prior(
    p0_a_wins: float, p_a_return_seed: float, best_of: int,
    elo_matches_played_a: float | None = None,
    base_n0: float = 20.0, min_n0: float = 5.0, max_n0: float = 60.0,
    reference_matches: float = 150.0,
) -> tuple[float, float, float, float]:
    """PRE-FIX behavior, reproduced in isolation ONLY for this comparison: n0_return_a
    incorrectly used elo_matches_played_a (A's own match count) instead of
    elo_matches_played_b (the opponent's) — see ml_informed_markov.py's
    build_pretrained_prior docstring for the full explanation of why this was wrong."""
    p_serve0_a = invert_prematch_probability(p0_a_wins, p_a_return_seed, best_of)

    def _confidence_n0(matches_played):
        if matches_played is None or (isinstance(matches_played, float) and np.isnan(matches_played)):
            return base_n0
        scale = min(matches_played / reference_matches, 1.0)
        return float(np.clip(base_n0 + scale * (max_n0 - base_n0), min_n0, max_n0))

    n0_serve_a = _confidence_n0(elo_matches_played_a)
    n0_return_a = _confidence_n0(elo_matches_played_a)  # BUG: should be elo_matches_played_b
    return p_serve0_a, n0_serve_a, p_a_return_seed, n0_return_a


def main() -> None:
    logger.info("Loading trained classifier...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

    logger.info("Building point dataset (same as evaluate_ml_informed_markov.py)...")
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
    logger.info("Evaluating %d matches, %d points, OLD and NEW versions side by side",
               n_use, len(eval_df))

    old_preds, new_preds, targets = [], [], []
    per_match_max_diff = {}
    per_match_experience_gap = {}
    current_match_id, old_posterior, new_posterior = None, None, None

    for row in eval_df.to_dict("records"):
        if row["match_id"] != current_match_id:
            current_match_id = row["match_id"]
            p0_a_wins = compute_composite_prematch_probability(row)

            # BUG FIX (external review, 2026-07): see return_seed.py's module docstring.
            p_a_return_seed = compute_p_a_return_seed(row, track_winner=True)

            elo_a = row.get("elo_matches_played_pre_winner")
            elo_b = row.get("elo_matches_played_pre_loser")
            best_of_val = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3

            old_p_serve0, old_n0s, old_p_return0, old_n0r = _old_build_pretrained_prior(
                p0_a_wins, p_a_return_seed, best_of_val, elo_matches_played_a=elo_a,
            )
            new_p_serve0, new_n0s, new_p_return0, new_n0r = build_pretrained_prior(
                p0_a_wins, p_a_return_seed, best_of_val,
                elo_matches_played_a=elo_a, elo_matches_played_b=elo_b,
            )
            old_posterior = ServeReturnPosterior.from_pretrained_prior(
                old_p_serve0, old_n0s, old_p_return0, old_n0r
            )
            new_posterior = ServeReturnPosterior.from_pretrained_prior(
                new_p_serve0, new_n0s, new_p_return0, new_n0r
            )
            per_match_max_diff[current_match_id] = 0.0
            if elo_a is not None and elo_b is not None and pd.notna(elo_a) and pd.notna(elo_b):
                per_match_experience_gap[current_match_id] = abs(float(elo_a) - float(elo_b))
            else:
                per_match_experience_gap[current_match_id] = np.nan

        track_winner = tracked_player_is_winner(row["match_id"])
        state = _row_to_match_state(row)

        p_old_winner, old_posterior = ml_informed_markov_predict(
            state, row, model, feature_cols, old_posterior
        )
        p_new_winner, new_posterior = ml_informed_markov_predict(
            state, row, model, feature_cols, new_posterior
        )

        diff = abs(p_old_winner - p_new_winner)
        per_match_max_diff[current_match_id] = max(per_match_max_diff[current_match_id], diff)

        if track_winner:
            old_preds.append(p_old_winner)
            new_preds.append(p_new_winner)
            targets.append(1.0)
        else:
            old_preds.append(1.0 - p_old_winner)
            new_preds.append(1.0 - p_new_winner)
            targets.append(0.0)

    y = np.array(targets)
    old_arr = np.array(old_preds)
    new_arr = np.array(new_preds)

    print(f"\n=== Impact of the elo_matches_played_b fix, same {n_use}-match sample ===\n")
    print(f"{'Version':<10} {'LogLoss':>10} {'Brier':>10} {'ECE':>10}")
    for name, arr in [("OLD (buggy)", old_arr), ("NEW (fixed)", new_arr)]:
        print(f"{name:<10} {compute_log_loss(y, arr):>10.4f} "
              f"{compute_brier_score(y, arr):>10.4f} {expected_calibration_error(y, arr):>10.4f}")

    diffs = np.abs(old_arr - new_arr)
    print(f"\nPer-POINT probability difference (old vs new): "
          f"mean={diffs.mean():.4f}, max={diffs.max():.4f}, "
          f"95th percentile={np.percentile(diffs, 95):.4f}")

    max_diffs = pd.Series(per_match_max_diff)
    print(f"\nPer-MATCH maximum probability difference:")
    print(f"  mean={max_diffs.mean():.4f}, median={max_diffs.median():.4f}, "
          f"max={max_diffs.max():.4f}")
    print(f"  matches affected by >1% max shift: {(max_diffs > 0.01).sum()} / {len(max_diffs)}")
    print(f"  matches affected by >5% max shift: {(max_diffs > 0.05).sum()} / {len(max_diffs)}")
    print(f"  matches affected by >10% max shift: {(max_diffs > 0.10).sum()} / {len(max_diffs)}")

    experience_gap = pd.Series(per_match_experience_gap)
    combined = pd.DataFrame({"max_diff": max_diffs, "experience_gap": experience_gap}).dropna()
    if len(combined) > 5:
        corr = combined["max_diff"].corr(combined["experience_gap"])
        print(f"\nCorrelation between per-match max probability shift and player-experience "
              f"gap (|elo_matches_played_a - elo_matches_played_b|): {corr:.4f}")
        print("(A positive correlation here would directly confirm the bug's impact "
              "concentrates specifically in matches with mismatched player experience — "
              "exactly the mechanism the fix targets — rather than being spread evenly "
              "and randomly across all matches.)")


if __name__ == "__main__":
    main()