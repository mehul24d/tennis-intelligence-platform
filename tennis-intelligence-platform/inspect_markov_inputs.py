"""Diagnostic: print the exact career serve/return values feeding the Markov pre-match
calculation for the 2025 Roland Garros final, to check whether 0.995 reflects real,
reasonable career numbers or a data-quality issue."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "pipelines")

import pandas as pd
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.markov_baseline import prob_win_match

RAW_MCP = "data/raw/tennis_MatchChartingProject"
POINT_FILES = [
    f"{RAW_MCP}/charting-m-points-to-2009.csv",
    f"{RAW_MCP}/charting-m-points-2010s.csv",
    f"{RAW_MCP}/charting-m-points-2020s.csv",
]
MATCH_ID = "20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz"

frozen_join = pd.read_parquet("data/processed/joined_matches_m.parquet")
day6 = pd.read_parquet("data/processed/matches_with_day6_features.parquet")
points = build_point_dataset(POINT_FILES, frozen_join, day6)
points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]

row = points[points["match_id"] == MATCH_ID].iloc[0]

print("=" * 70)
print(f"Match: {MATCH_ID}")
print(f"player1_is_winner: {row['player1_is_winner']}  (True = Sinner is Player 1 AND won)")
print("=" * 70)
print("\nRaw career feature values used by the Markov pre-match calculation:\n")
for col in ["winner_first_serve_win_pct_career", "loser_first_serve_win_pct_career",
            "winner_first_serve_in_pct_career", "loser_first_serve_in_pct_career",
            "winner_bp_saved_pct_career", "loser_bp_saved_pct_career",
            "winner_return_pts_won_pct_career", "loser_return_pts_won_pct_career",
            "elo_pre_match_winner", "elo_pre_match_loser",
            "winner_win_pct_last10", "loser_win_pct_last10"]:
    val = row.get(col, "MISSING COLUMN")
    print(f"  {col:45s} = {val}")

print("\n--- Reconstructing the exact Markov pre-match call ---")
p1_is_winner = bool(row["player1_is_winner"])
ps_key = "winner_first_serve_win_pct_career" if p1_is_winner else "loser_first_serve_win_pct_career"
pr_key = "winner_return_pts_won_pct_career" if p1_is_winner else "loser_return_pts_won_pct_career"
ps = row.get(ps_key, 0.65)
pr = row.get(pr_key, 0.38)
print(f"p_serve (Player 1's serve-win rate used): {ps}")
print(f"p_return (Player 1's return-win rate used): {pr}")
best_of = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3
p = prob_win_match(float(ps) if pd.notna(ps) else 0.65,
                   float(pr) if pd.notna(pr) else 0.38, best_of=best_of)
print(f"\nResulting Markov pre-match P(Player 1 wins): {p:.4f}")