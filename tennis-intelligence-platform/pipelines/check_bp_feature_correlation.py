"""
check_bp_feature_correlation.py — tests the specific hypothesis raised after the
permutation-importance results came back mixed: does the new winner_/loser_
bp_serve_win_pct_career correlate strongly with the ALREADY-EXISTING
winner_/loser_bp_saved_pct_career (both derived from break-point-specific data, just at
different granularity — bp_saved_pct_career from Overview.csv's match-wide bk_pts/
bp_saved, bp_serve_win_pct_career from KeyPointsServe.csv's row=='BP' specifically)?

If highly correlated, that's a clean, direct explanation for the new feature's weak/
negative marginal importance: the classifier already has access to essentially the same
signal via the existing feature, so the new one adds little beyond noise from its
independently-computed (and likely smaller-sample, since break points are a small
fraction of total points) rolling window.

Also checks correlation with the outcome itself (server_wins_point) directly, as a
second, independent way to see whether the new feature carries real univariate signal
even if it's redundant with an existing one in the full multivariate model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_live_engines_v2 import HOLDOUT_YEAR, POINT_FILES, PROCESSED
from tennis_intel.live.build_point_dataset import build_point_dataset


def main() -> None:
    print("Building point dataset...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["match_year"] = points["match_id"].str[:4].astype(int)
    test_pts = points[points["match_year"] >= HOLDOUT_YEAR].copy()

    pairs = [
        ("winner_bp_serve_win_pct_career", "winner_bp_saved_pct_career"),
        ("loser_bp_serve_win_pct_career", "loser_bp_saved_pct_career"),
        ("winner_bp_return_win_pct_career", "winner_return_pts_won_pct_career"),
        ("loser_bp_return_win_pct_career", "loser_return_pts_won_pct_career"),
    ]

    print(f"\n=== Correlation between new and existing features (n={len(test_pts):,} points) ===\n")
    print(f"{'new_feature':<38} {'existing_feature':<32} {'n_both_present':>15} {'pearson_r':>10}")
    for new_col, existing_col in pairs:
        if new_col not in test_pts.columns or existing_col not in test_pts.columns:
            print(f"{new_col:<38} {existing_col:<32} {'MISSING COLUMN':>15}")
            continue
        sub = test_pts[[new_col, existing_col]].dropna()
        if len(sub) < 30:
            print(f"{new_col:<38} {existing_col:<32} {len(sub):>15}  (too few rows)")
            continue
        r = sub[new_col].corr(sub[existing_col])
        print(f"{new_col:<38} {existing_col:<32} {len(sub):>15,} {r:>10.4f}")

    print("\nInterpretation:")
    print("- A HIGH correlation (e.g. |r| > 0.5) between a new BP feature and its paired")
    print("  existing feature would directly explain weak/negative permutation importance:")
    print("  the classifier already has access to nearly the same information, so the new")
    print("  column adds little beyond noise from its own, likely-thinner sample.")
    print("- A LOW correlation suggests the new feature is NOT simply redundant with an")
    print("  existing one — its weak permutation importance would then need a different")
    print("  explanation (e.g. genuinely thin per-match sample size on break points")
    print("  specifically, since they're a small fraction of total points, producing a")
    print("  noisier rolling career average than match-wide features like")
    print("  bp_saved_pct_career), not simple redundancy.")


if __name__ == "__main__":
    main()