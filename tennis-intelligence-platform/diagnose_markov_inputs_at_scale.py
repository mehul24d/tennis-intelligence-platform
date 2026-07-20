"""Diagnostic: checks whether missing/fallback career stats or degenerate predictions are
driving Markov's near-chance-level performance in the Day 11 evaluation."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "pipelines")

import pandas as pd
import numpy as np
from tennis_intel.live.build_point_dataset import build_point_dataset

RAW_MCP = "data/raw/tennis_MatchChartingProject"
POINT_FILES = [f"{RAW_MCP}/charting-m-points-to-2009.csv",
               f"{RAW_MCP}/charting-m-points-2010s.csv",
               f"{RAW_MCP}/charting-m-points-2020s.csv"]

frozen_join = pd.read_parquet("data/processed/joined_matches_m.parquet")
day6 = pd.read_parquet("data/processed/matches_with_day6_features.parquet")
points = build_point_dataset(POINT_FILES, frozen_join, day6)
points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]
points["match_year"] = points["match_id"].str[:4].astype(int)
test_points = points[points["match_year"] >= 2022]

# Check missingness of the two key columns feeding Markov's p_serve/p_return
for col in ["winner_first_serve_win_pct_career", "loser_first_serve_win_pct_career"]:
    n_missing = test_points[col].isna().sum()
    pct = 100 * n_missing / len(test_points)
    print(f"{col}: {n_missing} missing / {len(test_points)} ({pct:.1f}%)")

# Distribution of the actual values used (not just missingness)
print("\nDistribution of winner_first_serve_win_pct_career (test set):")
print(test_points["winner_first_serve_win_pct_career"].describe())
print("\nDistribution of loser_first_serve_win_pct_career (test set):")
print(test_points["loser_first_serve_win_pct_career"].describe())

# Distribution of the IMPLIED SKILL GAP (winner_serve - loser_serve) -- if this is often
# near zero or even NEGATIVE (winner's own serve stat lower than loser's), that would
# directly explain near-chance-level Markov performance
gap = test_points["winner_first_serve_win_pct_career"] - test_points["loser_first_serve_win_pct_career"]
print(f"\nImplied serve-rate gap (winner - loser), per point (n={len(gap)}):")
print(gap.describe())
print(f"\nFraction of points where winner's OWN serve stat is actually LOWER than loser's: "
      f"{100*(gap < 0).mean():.1f}%")
print("(If this is high, it means the person who WON the match often had a WORSE career")
print(" serve stat than the person who lost -- meaning career serve rate alone is a weak")
print(" predictor of who wins, which would directly explain near-chance Markov performance)")