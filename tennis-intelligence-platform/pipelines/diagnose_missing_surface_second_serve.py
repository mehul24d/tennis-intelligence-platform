"""
diagnose_missing_surface_second_serve.py — localizes exactly where
second_serve_win_pct_surface_career disappears in the Day 6 pipeline. The career
version (second_serve_win_pct_career) works correctly, confirmed by its presence in
the trained classifier's feature list — but the surface-conditioned version never
reaches the final Day 6 parquet file, and code inspection of attach_surface and
compute_rolling_surface_serve_return_features didn't reveal an obvious cause. This
prints the actual column presence at each stage, on the real data, to settle it
directly rather than continue guessing from code alone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from tennis_intel.features.serve_return_features import (
    load_and_prepare_stats, attach_player_ids_and_chronology,
)
from tennis_intel.features.surface_serve_return_features import (
    attach_surface, compute_rolling_surface_serve_return_features, SURFACE_RATE_COLS,
)

PROCESSED_DIR = Path("data/processed")
RAW_MCP_DIR = Path("data/raw/tennis_MatchChartingProject")
STATS_PATH = RAW_MCP_DIR / "charting-m-stats-Overview.csv"
FROZEN_JOIN_PATH = PROCESSED_DIR / "joined_matches_m.parquet"


def main() -> None:
    frozen_join = pd.read_parquet(FROZEN_JOIN_PATH)

    print("=== Step 1: load_and_prepare_stats ===")
    stats = load_and_prepare_stats(STATS_PATH)
    print(f"'second_serve_win_pct' in stats.columns: {'second_serve_win_pct' in stats.columns}")
    if "second_serve_win_pct" in stats.columns:
        print(f"  non-null count: {stats['second_serve_win_pct'].notna().sum()} / {len(stats)}")

    print("\n=== Step 2: attach_player_ids_and_chronology ===")
    stats_with_ids = attach_player_ids_and_chronology(stats, frozen_join)
    print(f"'second_serve_win_pct' in stats_with_ids.columns: "
          f"{'second_serve_win_pct' in stats_with_ids.columns}")
    if "second_serve_win_pct" in stats_with_ids.columns:
        print(f"  non-null count: {stats_with_ids['second_serve_win_pct'].notna().sum()} / {len(stats_with_ids)}")
    print(f"  Row count: {len(stats_with_ids)} (started at {len(stats)})")

    print("\n=== Step 3: attach_surface ===")
    stats_with_surface = attach_surface(stats_with_ids, frozen_join)
    print(f"'second_serve_win_pct' in stats_with_surface.columns: "
          f"{'second_serve_win_pct' in stats_with_surface.columns}")
    if "second_serve_win_pct" in stats_with_surface.columns:
        print(f"  non-null count: {stats_with_surface['second_serve_win_pct'].notna().sum()} / {len(stats_with_surface)}")
    print(f"  Row count: {len(stats_with_surface)} (started at {len(stats_with_ids)})")
    print(f"  Full column list: {sorted(stats_with_surface.columns.tolist())}")

    print(f"\n=== Step 4: is 'second_serve_win_pct' in SURFACE_RATE_COLS? ===")
    print(f"SURFACE_RATE_COLS: {SURFACE_RATE_COLS}")
    print(f"'second_serve_win_pct' in SURFACE_RATE_COLS: {'second_serve_win_pct' in SURFACE_RATE_COLS}")

    print("\n=== Step 5: compute_rolling_surface_serve_return_features ===")
    surface_rolling = compute_rolling_surface_serve_return_features(stats_with_surface)
    surface_cols = [c for c in surface_rolling.columns if "_surface_" in c]
    print(f"All '_surface_' columns produced: {sorted(surface_cols)}")
    print(f"'second_serve_win_pct_surface_career' present: "
          f"{'second_serve_win_pct_surface_career' in surface_rolling.columns}")


if __name__ == "__main__":
    main()