"""Pulls the actual Set1/Set2/Gm1/Gm2 values around wherever the 2008 Wimbledon final
replay crashed, to check whether the game counts are genuinely anomalous (a data quality
issue) rather than a logic bug in the recursion's termination conditions."""
import sys
sys.path.insert(0, "src")
import pandas as pd
from tennis_intel.live.build_point_dataset import build_point_dataset

RAW_MCP = "data/raw/tennis_MatchChartingProject"
POINT_FILES = [f"{RAW_MCP}/charting-m-points-to-2009.csv",
               f"{RAW_MCP}/charting-m-points-2010s.csv",
               f"{RAW_MCP}/charting-m-points-2020s.csv"]

frozen_join = pd.read_parquet("data/processed/joined_matches_m.parquet")
day6 = pd.read_parquet("data/processed/matches_with_day6_features.parquet")
points = build_point_dataset(POINT_FILES, frozen_join, day6)

candidates = points[points["match_id"].str.contains("2008", na=False) &
                    points["match_id"].str.contains("Wimbledon", na=False, case=False) &
                    points["match_id"].str.contains("Federer", na=False) &
                    points["match_id"].str.contains("Nadal", na=False)]

if candidates.empty:
    print("Match not found via this exact filter — check the real match_id string.")
else:
    match_id = candidates["match_id"].iloc[0]
    match = candidates.sort_values("Pt")
    print(f"Match: {match_id}, {len(match)} points\n")

    print("Max Gm1/Gm2 values across the whole match (checking for anomalies):")
    print(f"  Gm1: min={match['Gm1'].min()}, max={match['Gm1'].max()}")
    print(f"  Gm2: min={match['Gm2'].min()}, max={match['Gm2'].max()}")
    print(f"  Set1: min={match['Set1'].min()}, max={match['Set1'].max()}")
    print(f"  Set2: min={match['Set2'].min()}, max={match['Set2'].max()}")

    print("\nTail of the match (final set, where the crash likely occurs):")
    cols = ["Pt", "Set1", "Set2", "Gm1", "Gm2", "Svr", "is_tiebreak_game", "best_of"]
    print(match[cols].tail(30).to_string(index=False))

    print("\nAny row where Gm1 or Gm2 exceeds a realistic bound (say, >25 games in one set)?")
    anomalous = match[(match["Gm1"] > 25) | (match["Gm2"] > 25)]
    print(f"  {len(anomalous)} anomalous row(s) found" if len(anomalous) else "  None found")
    if len(anomalous):
        print(anomalous[cols].to_string(index=False))