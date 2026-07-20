"""
check_second_serve_correlation.py — tests the specific hypothesis raised after this
turn's permutation-importance results came back genuinely inconsistent across the four
second-serve variants (winner/loser x career/surface): winner and loser versions of the
SAME stat landing on opposite sides of zero, and surface vs. non-surface flipping which
one is positive, is not a coherent pattern on its own. The leading hypothesis: these
features are highly correlated with the ALREADY-EXISTING first_serve_win_pct_career
(plausible, since players who serve well on their first serve tend to also serve well
on their second — server quality is a real, shared underlying trait), causing
permutation importance to split a shared signal unstably across correlated columns
(permuting one alone barely hurts performance if the model can lean on its correlated
partner instead).

Checks BOTH first/second-serve correlation for the SAME player, AND career/surface
correlation for the SAME serve type, since either could independently explain
instability, and it's worth knowing which (or both) actually applies here.
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

    print(f"\n=== First-serve vs. second-serve correlation, SAME player (n={len(test_pts):,} points) ===\n")
    print(f"{'first_serve_col':<40} {'second_serve_col':<40} {'n_both':>10} {'pearson_r':>10}")
    first_second_pairs = [
        ("winner_first_serve_win_pct_career", "winner_second_serve_win_pct_career"),
        ("loser_first_serve_win_pct_career", "loser_second_serve_win_pct_career"),
        ("winner_first_serve_win_pct_surface_career", "winner_second_serve_win_pct_surface_career"),
        ("loser_first_serve_win_pct_surface_career", "loser_second_serve_win_pct_surface_career"),
    ]
    for first_col, second_col in first_second_pairs:
        sub = test_pts[[first_col, second_col]].dropna()
        if len(sub) < 30:
            print(f"{first_col:<40} {second_col:<40} {len(sub):>10}  (too few rows)")
            continue
        r = sub[first_col].corr(sub[second_col])
        print(f"{first_col:<40} {second_col:<40} {len(sub):>10,} {r:>10.4f}")

    print(f"\n=== Career vs. surface correlation, SAME serve type + player (n={len(test_pts):,} points) ===\n")
    print(f"{'career_col':<40} {'surface_col':<40} {'n_both':>10} {'pearson_r':>10}")
    career_surface_pairs = [
        ("winner_second_serve_win_pct_career", "winner_second_serve_win_pct_surface_career"),
        ("loser_second_serve_win_pct_career", "loser_second_serve_win_pct_surface_career"),
    ]
    for career_col, surface_col in career_surface_pairs:
        sub = test_pts[[career_col, surface_col]].dropna()
        if len(sub) < 30:
            print(f"{career_col:<40} {surface_col:<40} {len(sub):>10}  (too few rows)")
            continue
        r = sub[career_col].corr(sub[surface_col])
        print(f"{career_col:<40} {surface_col:<40} {len(sub):>10,} {r:>10.4f}")

    print("\nInterpretation:")
    print("- A HIGH first/second correlation (e.g. |r| > 0.5) directly explains today's")
    print("  unstable, sign-flipping permutation importance: the classifier already has")
    print("  access to nearly the same information via first_serve_win_pct_career, so")
    print("  splitting the shared signal across two correlated columns makes each")
    print("  individual importance estimate noisy and order-dependent.")
    print("- If confirmed, the fix is NOT 'drop second serve' -- it's using the already-")
    print("  built combined_serve_win_pct_career (first+second properly weighted into")
    print("  ONE feature, from the return-seed-fix pipeline) as the classifier input")
    print("  INSTEAD OF the two separate, correlated first/second columns -- consolidate,")
    print("  don't remove real signal.")
    print("- A HIGH career/surface correlation would separately suggest the surface-")
    print("  conditioned second-serve version isn't adding much beyond the career")
    print("  version specifically for second serve (distinct from the first/second")
    print("  question above).")


if __name__ == "__main__":
    main()