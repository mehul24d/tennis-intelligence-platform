"""
train_day6_comparison.py — the actual empirical contribution: does adding MCP point-level
serve/return data improve win-probability prediction beyond Elo + rolling match-level form?

Runs the IDENTICAL evaluation harness from Milestone 5 (temporal CV, same models, same
metrics) TWICE on the EXACT SAME subset of matches (those with real serve/return data, so
the comparison is apples-to-apples and not confounded by which matches happen to have
richer data):
  (a) baseline features only (Elo + rolling form, from Milestone 5)
  (b) baseline + serve/return diffs

Usage (from project root, with .venv activated):
    python pipelines/train_day6_comparison.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tennis_intel.modeling.build_symmetric_dataset import build_symmetric_dataset, FEATURE_PAIRS
from tennis_intel.modeling.train_and_evaluate import run_model_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DAY6_PATH = PROCESSED_DIR / "matches_with_day6_features.parquet"

SERVE_RETURN_PAIRS = [
    (f"winner_{rate}_career", f"loser_{rate}_career", f"{rate}_career")
    for rate in ["first_serve_in_pct", "first_serve_win_pct", "second_serve_win_pct",
                 "ace_rate", "df_rate", "bp_saved_pct", "return_pts_won_pct", "bp_converted_pct"]
]

# Broad enough range to capture most of the charted-match era (MCP coverage starts ~2011)
TEST_YEARS = list(range(2015, 2026))


def main() -> None:
    if not DAY6_PATH.exists():
        raise FileNotFoundError(f"{DAY6_PATH} not found — run pipelines/build_day6_features.py first.")

    matches = pd.read_parquet(DAY6_PATH)
    logger.info("Loaded %d matches", len(matches))

    # Build BOTH feature sets on the full dataset, then restrict both to the SAME subset
    # where serve/return data actually exists — this is the apples-to-apples requirement.
    baseline_only = build_symmetric_dataset(matches, feature_pairs=FEATURE_PAIRS)
    with_serve_return = build_symmetric_dataset(matches, feature_pairs=FEATURE_PAIRS + SERVE_RETURN_PAIRS)

    # Restrict both feature sets to the SAME subset where serve/return data actually
    # exists — checked directly on the enhanced frame's own diff column.
    key_col = f"{SERVE_RETURN_PAIRS[0][2]}_diff"
    subset_mask = with_serve_return[key_col].notna()
    n_subset = subset_mask.sum()
    logger.info("Restricting comparison to %d matches (%.1f%%) with real serve/return data",
                n_subset, 100 * n_subset / len(matches))

    if n_subset < 500:
        logger.warning("Very small comparison subset (%d matches) — results below will have "
                        "wide confidence intervals. Interpret accordingly.", n_subset)

    baseline_subset = baseline_only[subset_mask].copy()
    enhanced_subset = with_serve_return[subset_mask].copy()

    print("\n" + "=" * 70)
    print("BASELINE (Elo + rolling form only) — evaluated on serve/return-available subset")
    print("=" * 70)
    baseline_result = run_model_comparison(baseline_subset, test_years=TEST_YEARS)
    baseline_agg = baseline_result.aggregate_by_model()
    print(baseline_agg.to_string(index=False))

    print("\n" + "=" * 70)
    print("ENHANCED (Elo + rolling form + serve/return) — same subset, same folds")
    print("=" * 70)
    enhanced_result = run_model_comparison(enhanced_subset, test_years=TEST_YEARS)
    enhanced_agg = enhanced_result.aggregate_by_model()
    print(enhanced_agg.to_string(index=False))

    print("\n" + "=" * 70)
    print("INCREMENTAL VALUE (baseline log loss - enhanced log loss; positive = improvement)")
    print("=" * 70)
    comparison = baseline_agg[["model", "mean_log_loss"]].merge(
        enhanced_agg[["model", "mean_log_loss"]], on="model", suffixes=("_baseline", "_enhanced")
    )
    comparison["log_loss_improvement"] = comparison["mean_log_loss_baseline"] - comparison["mean_log_loss_enhanced"]
    comparison["relative_improvement_pct"] = 100 * comparison["log_loss_improvement"] / comparison["mean_log_loss_baseline"]
    print(comparison.to_string(index=False))

    baseline_agg.to_csv(PROCESSED_DIR / "day6_baseline_on_subset.csv", index=False)
    enhanced_agg.to_csv(PROCESSED_DIR / "day6_enhanced_aggregate.csv", index=False)
    comparison.to_csv(PROCESSED_DIR / "day6_incremental_value.csv", index=False)
    print(f"\nWrote comparison tables to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()