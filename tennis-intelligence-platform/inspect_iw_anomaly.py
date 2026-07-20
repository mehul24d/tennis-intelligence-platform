"""Pulls the FULL raw row data for the two anomalous points (same match, same
reconstructed state, but different real predictions) to find exactly what my
reconstruction script is missing."""
import sys
sys.path.insert(0, "src")
import pandas as pd

df = pd.read_parquet("data/processed/day11_head_to_head_v2_predictions.parquet")
match_id = df[df["markov_pred"].notna()]
# Find the exact match_id from the earlier output (truncated in that log)
candidates = df[df["match_id"].str.contains("Indian_Wells_Masters-R16-Grig", na=False)]
mid = candidates["match_id"].iloc[0]
print(f"Full match_id: {mid}\n")

rows = df[(df["match_id"] == mid) & (df["Pt"].isin([122, 123, 124, 125, 126, 127]))]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
cols = ["Pt", "Set1", "Set2", "Gm1", "Gm2", "p1_points", "p2_points", "tb_p1_points",
        "tb_p2_points", "Svr", "is_tiebreak_game", "is_match_point", "player1_is_winner",
        "best_of", "markov_pred", "target"]
print(rows[cols].to_string(index=False))