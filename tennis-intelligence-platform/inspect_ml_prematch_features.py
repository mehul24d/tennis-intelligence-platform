"""Pulls the EXACT real feature values feeding compute_ml_pre_match_probability for the
2025 Roland Garros final, to find precisely what's driving the 0.86 ML pre-match number —
same disciplined "pull real numbers, don't guess" approach as the earlier Markov 0.995
investigation."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "pipelines")
import pandas as pd
from tennis_intel.live.build_point_dataset import build_point_dataset

RAW_MCP = "data/raw/tennis_MatchChartingProject"
POINT_FILES = [f"{RAW_MCP}/charting-m-points-to-2009.csv",
               f"{RAW_MCP}/charting-m-points-2010s.csv",
               f"{RAW_MCP}/charting-m-points-2020s.csv"]
MATCH_ID = "20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz"

frozen_join = pd.read_parquet("data/processed/joined_matches_m.parquet")
day6 = pd.read_parquet("data/processed/matches_with_day6_features.parquet")
points = build_point_dataset(POINT_FILES, frozen_join, day6)
points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]

row = points[points["match_id"] == MATCH_ID].iloc[0]
p1_is_winner = bool(row["player1_is_winner"])
print(f"player1_is_winner: {p1_is_winner}  (winner_* columns = whichever of Sinner/Alcaraz "
      f"actually won this real match, i.e. Sinner LOST so 'loser_*' = Sinner's real stats)")
print("=" * 70)

cols_to_check = [
    "elo_pre_match_winner", "elo_pre_match_loser",
    "elo_surface_pre_match_winner", "elo_surface_pre_match_loser",
    "elo_matches_played_pre_winner", "elo_matches_played_pre_loser",
    "winner_win_pct_last10", "loser_win_pct_last10",
    "winner_surface_win_pct_last10", "loser_surface_win_pct_last10",
    "winner_first_serve_win_pct_career", "loser_first_serve_win_pct_career",
    "winner_first_serve_win_pct_surface_career", "loser_first_serve_win_pct_surface_career",
    "winner_h2h_wins_pre_match", "loser_h2h_wins_pre_match",
    "winner_tourney_h2h_wins_pre_match", "loser_tourney_h2h_wins_pre_match",
    "winner_tourney_win_pct_last10", "loser_tourney_win_pct_last10",
]
for c in cols_to_check:
    val = row.get(c, "MISSING COLUMN")
    print(f"  {c:50s} = {val}")

print("\n" + "=" * 70)
print("REMEMBER: 'winner_*' = Sinner's real stats (he won), 'loser_*' = Alcaraz's real stats")
print("(since the raw TML labels are based on the actual outcome of this specific match)")