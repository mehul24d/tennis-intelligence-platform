"""
check_loser_second_serve_sparsity.py — tests the specific hypothesis proposed for why
loser_second_serve_win_pct_career has landed negative across four separate retrains
while winner_second_serve_win_pct_career stays stably positive: is the loser-side
feature disproportionately populated by thinner-history players (since "loser" is
assigned per-match, a career underdog appears as "loser" far more often than "winner"),
making its estimate noisier and net-harmful for the model to lean on?

Checks (1) missingness rate for winner- vs. loser-side second-serve, and (2) the mean
elo_matches_played_pre_{winner,loser} conditioned on the feature being present, which
directly tests whether loser-side rows have systematically thinner history behind them.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

    # Reduce to one row per match, since these are all match-level (pre-match) features
    # -- no need to look at every point, just every distinct match in the holdout set.
    matches = test_pts.drop_duplicates(subset="match_id")
    print(f"n matches in holdout: {len(matches):,}\n")

    print("=== Missingness rate ===")
    for col in ["winner_second_serve_win_pct_career", "loser_second_serve_win_pct_career"]:
        n_missing = matches[col].isna().sum()
        pct_missing = 100 * n_missing / len(matches)
        print(f"{col:<40} missing: {n_missing:>6,} / {len(matches):,} ({pct_missing:.1f}%)")

    print("\n=== Mean elo_matches_played_pre_{winner,loser}, conditioned on second-serve feature being present ===")
    for side, feat_col, elo_col in [
        ("winner", "winner_second_serve_win_pct_career", "elo_matches_played_pre_winner"),
        ("loser", "loser_second_serve_win_pct_career", "elo_matches_played_pre_loser"),
    ]:
        present = matches[matches[feat_col].notna()]
        mean_elo_matches = present[elo_col].mean()
        median_elo_matches = present[elo_col].median()
        print(f"{side:<8} (n={len(present):,}): mean elo_matches_played={mean_elo_matches:.1f}, "
              f"median={median_elo_matches:.1f}")

    print("\n=== Direct comparison: mean elo_matches_played_pre_winner vs. _pre_loser, "
          "ACROSS ALL MATCHES regardless of second-serve presence ===")
    mean_winner_all = matches["elo_matches_played_pre_winner"].mean()
    mean_loser_all = matches["elo_matches_played_pre_loser"].mean()
    print(f"winner: {mean_winner_all:.1f}, loser: {mean_loser_all:.1f}, "
          f"diff: {mean_winner_all - mean_loser_all:+.1f}")

    print("\nInterpretation:")
    print("- A meaningfully LOWER mean/median elo_matches_played_pre_loser (vs. winner)")
    print("  directly confirms the proposed hypothesis: loser-side rows systematically")
    print("  have thinner career history behind them, making the feature noisier and")
    print("  explaining the consistent negative importance -- the fix is dropping this")
    print("  specific feature (keeping winner-side), not better imputation, since the")
    print("  underlying data genuinely is thinner, not just poorly handled.")
    print("- If the two are comparable, the hypothesis is NOT confirmed, and the")
    print("  negative importance needs a different explanation before acting on it.")


if __name__ == "__main__":
    main()