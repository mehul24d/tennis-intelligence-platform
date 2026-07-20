"""Traces exactly why ml_informed_markov_p1 (smoothed) diverges sharply from
markov_p1, ml_mc_p1, AND its own unsmoothed sibling at the SAME game states (points
16-30, gm1=1-2/gm2=0-1, all still within set 1, which Player1/Sinner actually won 6-4).
First confirms who player1_is_winner actually is for this match, then reconstructs the
EXACT same blend_serve/blend_return/posterior values my earlier diagnostic computed,
converted correctly to Player1's perspective, to see whether they match this CSV's
reported ml_informed_markov_p1 -- if they don't match, the discrepancy is between two
DIFFERENT computations, not a misreading."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "pipelines")
import joblib
import pandas as pd
import numpy as np
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.live_win_probability import prob_a_wins_match_from_state
from tennis_intel.live.ml_informed_markov import (
    build_pretrained_prior, ServeReturnPosterior, ml_informed_point_probabilities,
    recursion_sensitivity, sensitivity_aware_blend, ml_informed_markov_predict,
)
from replay_match import _row_to_match_state
from generate_publication_trajectory import compute_composite_prematch_probability

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

first_row = match.iloc[0].to_dict()
print(f"player1_is_winner for this match: {first_row['player1_is_winner']}")
print(f"(If True: Player1=Sinner IS the tracked real winner -- but Alcaraz actually won")
print(f" this real match, so this should be False. If it prints True, that ITSELF would")
print(f" be a serious, separate data-orientation bug worth stopping on immediately.)\n")

payload = joblib.load("data/processed/day9_point_classifiers.joblib")
model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

p0_a_wins = compute_composite_prematch_probability(first_row)
loser_serve_surface = first_row.get("loser_first_serve_win_pct_surface_career")
loser_serve_career = first_row.get("loser_first_serve_win_pct_career")
opponent_serve = float(loser_serve_surface) if pd.notna(loser_serve_surface) else float(loser_serve_career)
p_a_return_seed = 1.0 - opponent_serve
elo_a = first_row.get("elo_matches_played_pre_winner")
elo_b = first_row.get("elo_matches_played_pre_loser")
h2h = None
if pd.notna(first_row.get("winner_h2h_wins_pre_match")) and pd.notna(first_row.get("loser_h2h_wins_pre_match")):
    h2h = float(first_row["winner_h2h_wins_pre_match"]) + float(first_row["loser_h2h_wins_pre_match"])
best_of_val = int(first_row["best_of"]) if pd.notna(first_row.get("best_of")) else 5

p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
    p0_a_wins, p_a_return_seed, best_of_val,
    elo_matches_played_a=elo_a, elo_matches_played_b=elo_b, h2h_meetings=h2h,
)
print(f"p0_a_wins (XGBoost, P(A=tracked winner wins)) = {p0_a_wins:.4f}")
print(f"p_serve0 (A's inverted point-level serve prior) = {p_serve0:.4f}\n")

posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)

print(f"{'Pt':>4} {'gm1':>4} {'gm2':>4} {'p1_won':>7} {'A_post_serve':>12} {'p_smoothed_A':>13} "
      f"{'p_smoothed_P1(reconstructed)':>29} {'CSV_ml_informed_p1':>19}")

csv = pd.read_csv("docs/trajectories/replay_20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz.csv")

p1_is_winner = bool(first_row["player1_is_winner"])

for i in range(30):
    row = match.iloc[i].to_dict()
    state = _row_to_match_state(row)

    p_smoothed_A, posterior_after = ml_informed_markov_predict(state, row, model, feature_cols, posterior)
    p_smoothed_p1_reconstructed = p_smoothed_A if p1_is_winner else (1.0 - p_smoothed_A)

    pt_winner = row.get("PtWinner")
    p1_won_str = "-"
    if pd.notna(pt_winner):
        p1_won_str = "Y" if int(pt_winner) == 1 else "N"

    csv_val = csv.iloc[i]["ml_informed_markov_p1"] if i < len(csv) else float("nan")

    print(f"{i+1:>4} {int(row['Gm1']):>4} {int(row['Gm2']):>4} {p1_won_str:>7} "
          f"{posterior.mean_serve():>12.4f} {p_smoothed_A:>13.4f} "
          f"{p_smoothed_p1_reconstructed:>29.4f} {csv_val:>19.4f}")

    posterior = posterior_after