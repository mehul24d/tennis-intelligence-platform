"""
evaluate_all_engines_unified.py — merges the two independently-run evaluation outputs
(evaluate_live_engines_v2.py's Markov + ML+MC, and evaluate_ml_informed_markov.py's
ML-Informed Markov with the corrected Elo/H2H-inverted prior) into one comparison, with
formal paired bootstrap significance tests between EVERY pair of engines — not just each
new engine against pure Markov individually, which is all that existed before this script.

Reuses both already-computed prediction files directly — no recomputation, no re-running
the expensive Day 11 rollout or even the cheap ML-Informed Markov pass. Both source
scripts use the IDENTICAL 150-match selection (same RANDOM_STATE, same
tracked_player_is_winner hash) and the SAME underlying markov_p_winner computation, so a
built-in cross-check (both files' independently-computed markov_pred columns must agree
almost exactly for the same match_id/Pt) is verified before trusting anything else — if
that check fails, the two runs are not actually comparable and the rest of this script's
output should not be trusted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error, paired_bootstrap_diff,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

DAY11_FILE = PROCESSED / "day11_head_to_head_v2_predictions.parquet"
ML_INFORMED_FILE = PROCESSED / "ml_informed_markov_predictions.parquet"

ENGINES = {
    "Markov": "markov_pred",
    "ML+MC": "ml_pred",
    "ML-Informed Markov": "ml_informed_pred",
}


def main() -> None:
    if not DAY11_FILE.exists():
        raise SystemExit(f"'{DAY11_FILE}' not found. Run evaluate_live_engines_v2.py first.")
    if not ML_INFORMED_FILE.exists():
        raise SystemExit(f"'{ML_INFORMED_FILE}' not found. Run evaluate_ml_informed_markov.py first.")

    day11 = pd.read_parquet(DAY11_FILE)
    ml_informed = pd.read_parquet(ML_INFORMED_FILE)

    logger.info("Day 11 file: %d points, %d matches", len(day11), day11["match_id"].nunique())
    logger.info("ML-Informed file: %d points, %d matches", len(ml_informed), ml_informed["match_id"].nunique())

    n_before = len(day11)
    merged = day11.merge(
        ml_informed[["match_id", "Pt", "markov_pred", "ml_informed_pred", "target"]],
        on=["match_id", "Pt"], how="inner", suffixes=("", "_ml_informed_file"),
    )
    logger.info("Merged: %d points (%d matches) — %d points from Day 11 did not have a "
               "matching row in the ML-Informed file and were dropped from this comparison",
               len(merged), merged["match_id"].nunique(), n_before - len(merged))

    if len(merged) == 0:
        raise SystemExit("Zero overlapping points between the two files — they were run "
                         "on completely different match selections. Cannot compare.")

    # CROSS-CHECK, must pass before trusting anything else: both files independently
    # computed markov_pred for the same points using the same function — they must agree
    # almost exactly (small floating-point differences are fine; a large or systematic
    # disagreement means the two runs are not actually comparable).
    markov_diff = (merged["markov_pred"] - merged["markov_pred_ml_informed_file"]).abs()
    logger.info("Cross-check: max |markov_pred difference| between the two independently-"
               "computed columns = %.2e (should be ~0)", markov_diff.max())
    if markov_diff.max() > 1e-6:
        raise AssertionError(
            f"The two files' independently-computed markov_pred columns disagree by up to "
            f"{markov_diff.max():.4f} — this means the two evaluation runs are NOT directly "
            f"comparable (different match selection, different code version, or a real bug). "
            f"Do not trust the comparison below until this is understood."
        )
    logger.info("Cross-check PASSED — the two runs are directly comparable.\n")

    # Target cross-check too: both files use the SAME tracked_player_is_winner hash, so
    # targets must match exactly for every point.
    target_mismatches = (merged["target"] != merged["target_ml_informed_file"]).sum()
    if target_mismatches > 0:
        raise AssertionError(
            f"{target_mismatches} point(s) have a DIFFERENT target between the two files — "
            f"the tracked-player convention is inconsistent between the two runs."
        )

    y = merged["target"].values
    print(f"\n=== Unified Three-Engine Comparison ===")
    print(f"Matches: {merged['match_id'].nunique()}, points: {len(merged):,}, "
          f"target balance: {y.mean():.3f}\n")

    print(f"{'Engine':<22} {'LogLoss':>10} {'Brier':>10} {'ECE':>10}")
    metrics_by_engine = {}
    for name, col in ENGINES.items():
        p = merged[col].values
        ll = compute_log_loss(y, p)
        brier = compute_brier_score(y, p)
        ece = expected_calibration_error(y, p)
        metrics_by_engine[name] = (ll, brier, ece)
        print(f"{name:<22} {ll:>10.4f} {brier:>10.4f} {ece:>10.4f}")

    print(f"\n=== Paired Bootstrap: every pair of engines, log loss ===")
    engine_names = list(ENGINES.keys())
    for i in range(len(engine_names)):
        for j in range(i + 1, len(engine_names)):
            name_a, name_b = engine_names[i], engine_names[j]
            col_a, col_b = ENGINES[name_a], ENGINES[name_b]
            result = paired_bootstrap_diff(
                y, merged[col_a].values, merged[col_b].values,
                metric_fn=compute_log_loss, metric_name="log_loss",
            )
            sig = "SIGNIFICANT" if not result.zero_in_ci else "not significant"
            direction = f"{name_a} better" if result.point_estimate_diff < 0 else f"{name_b} better"
            print(f"{name_a} vs {name_b}: diff={result.point_estimate_diff:+.4f} "
                  f"95% CI=[{result.ci_lower:+.4f}, {result.ci_upper:+.4f}] -> "
                  f"{sig} ({direction})")

    print(f"\n=== Paired Bootstrap: every pair of engines, Brier ===")
    for i in range(len(engine_names)):
        for j in range(i + 1, len(engine_names)):
            name_a, name_b = engine_names[i], engine_names[j]
            col_a, col_b = ENGINES[name_a], ENGINES[name_b]
            result = paired_bootstrap_diff(
                y, merged[col_a].values, merged[col_b].values,
                metric_fn=compute_brier_score, metric_name="brier",
            )
            sig = "SIGNIFICANT" if not result.zero_in_ci else "not significant"
            direction = f"{name_a} better" if result.point_estimate_diff < 0 else f"{name_b} better"
            print(f"{name_a} vs {name_b}: diff={result.point_estimate_diff:+.4f} "
                  f"95% CI=[{result.ci_lower:+.4f}, {result.ci_upper:+.4f}] -> "
                  f"{sig} ({direction})")


if __name__ == "__main__":
    main()