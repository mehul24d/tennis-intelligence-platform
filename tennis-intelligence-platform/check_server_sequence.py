"""
check_server_sequence.py — settles the (a) vs (b) question directly: does Sinner or
Alcaraz serve game 1 of this match? This determines whether Alcaraz's return posterior
had ~10-16 real observations by point 16 (if Sinner serves first) or exactly zero (if
Alcaraz serves first) -- a precise, falsifiable distinction the earlier "alternating
serve/return" explanation got wrong by assuming point-to-point alternation, which is not
how tennis scoring works (the server is fixed for an entire game).
"""
import sys
sys.path.insert(0, "src")
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
match = points[points["match_id"] == MATCH_ID].sort_values("Pt").reset_index(drop=True)

p1_is_winner = bool(match.iloc[0]["player1_is_winner"])
print(f"player1_is_winner (is Sinner, Player1, the real winner?): {p1_is_winner}")
print(f"(Should be False -- Alcaraz actually won this match)\n")

print(f"{'Pt':>4} {'Svr':>4} {'server':>12} {'gm1':>4} {'gm2':>4} {'A_is_serving':>13}")
for i in range(30):
    row = match.iloc[i]
    svr = int(row["Svr"])
    server_name = "Sinner(P1)" if svr == 1 else "Alcaraz(P2)"
    # "A" = the tracked winner = Alcaraz (since p1_is_winner should be False)
    a_is_serving = (svr == 2) if not p1_is_winner else (svr == 1)
    print(f"{i+1:>4} {svr:>4} {server_name:>12} {int(row['Gm1']):>4} {int(row['Gm2']):>4} "
          f"{str(a_is_serving):>13}")

print("\nDirect answer:")
first_server = "Sinner" if int(match.iloc[0]['Svr']) == 1 else "Alcaraz"
print(f"Game 1 is served by: {first_server}")
if first_server == "Sinner":
    print("-> Case (a): Alcaraz RETURNS in game 1. His return posterior should have")
    print("   ~10-16 real observations by point 16, not near-zero. The 'insufficient")
    print("   data' explanation for the flat early trajectory does NOT hold -- the flat")
    print("   trajectory instead reflects a specific sequence of return outcomes that")
    print("   happened to net out close to the seeded prior, not a lack of evidence.")
else:
    print("-> Case (b): Alcaraz SERVES in game 1. His return posterior genuinely has")
    print("   ZERO real observations through point 16 (his own SERVE posterior is the")
    print("   one accumulating evidence instead). Check whether the smoothed engine's")
    print("   output starts diverging from the pre-fix trajectory specifically once the")
    print("   server switches to Sinner (watch for the first point where Svr==1).")