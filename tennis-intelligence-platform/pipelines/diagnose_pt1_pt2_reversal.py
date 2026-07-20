"""Traces exactly why ml_informed_markov_p1 (smoothed) drops sharply at point 1->2 while
its own unsmoothed sibling barely moves at all -- a genuine, unexplained-so-far
inconsistency flagged directly from the chart, not assumed to be sensitivity/n0 behavior
without checking."""
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
    recursion_sensitivity, sensitivity_aware_blend,
)
from replay_match import _row_to_match_state
from generate_publication_trajectory import compute_composite_prematch_probability
from tennis_intel.live.return_seed import compute_p_a_return_seed

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

payload = joblib.load("data/processed/day9_point_classifiers.joblib")
model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

first_row = match.iloc[0].to_dict()
p0_a_wins = compute_composite_prematch_probability(first_row)
p_a_return_seed = compute_p_a_return_seed(first_row, track_winner=True)
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
posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)
print(f"Seeded: p_serve0={p_serve0:.4f}, p_return0={p_return0:.4f}\n")

print(f"{'Pt':>4} {'PtWinner':>9} {'p1_won':>7} {'raw_serve':>10} {'raw_return':>11} "
      f"{'post_serve':>11} {'post_return':>12} {'blend_serve':>12} {'blend_return':>13} "
      f"{'unsmoothed_out':>14} {'smoothed_out':>13}")

for i in range(6):
    row = match.iloc[i].to_dict()
    state = _row_to_match_state(row)
    p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(row, model, feature_cols)

    sens_serve = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "serve")
    sens_return = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "return")
    pts_obs_serve = posterior.points_observed_serve()
    pts_obs_return = posterior.points_observed_return()

    blended_serve = sensitivity_aware_blend(p_a_serve_raw, posterior.mean_serve(), sens_serve,
                                            points_observed=pts_obs_serve)
    blended_return = sensitivity_aware_blend(p_a_return_raw, posterior.mean_return(), sens_return,
                                             points_observed=pts_obs_return)
    blended_serve_c = float(np.clip(blended_serve, 0.01, 0.99))
    blended_return_c = float(np.clip(blended_return, 0.01, 0.99))

    p_a_serve_raw_c = float(np.clip(p_a_serve_raw, 0.01, 0.99))
    p_a_return_raw_c = float(np.clip(p_a_return_raw, 0.01, 0.99))
    unsmoothed_out = prob_a_wins_match_from_state(state, p_a_serve_raw_c, p_a_return_raw_c)
    smoothed_out = prob_a_wins_match_from_state(state, blended_serve_c, blended_return_c)

    pt_winner = row.get("PtWinner")
    p1_is_winner = bool(row.get("player1_is_winner", True))
    p1_won = "-"
    if pd.notna(pt_winner):
        p1_won = "Y" if int(pt_winner) == 1 else "N"

    print(f"{i+1:>4} {int(pt_winner) if pd.notna(pt_winner) else -1:>9} {p1_won:>7} "
          f"{p_a_serve_raw:>10.4f} {p_a_return_raw:>11.4f} "
          f"{posterior.mean_serve():>11.4f} {posterior.mean_return():>12.4f} "
          f"{blended_serve:>12.4f} {blended_return:>13.4f} "
          f"{unsmoothed_out:>14.4f} {smoothed_out:>13.4f}")

    if pd.notna(pt_winner):
        a_won = (int(pt_winner) == 1) if p1_is_winner else (int(pt_winner) == 2)
        if state.server_is_a:
            posterior = posterior.update_serve(a_won)
        else:
            posterior = posterior.update_return(a_won)

print("\nCheck: does blend_serve/blend_return move sharply between pt1 and pt2 while")
print("raw_serve/raw_return (what unsmoothed uses) barely change? If so, the divergence")
print("is coming from the BLEND/POSTERIOR side specifically, not the raw classifier --")
print("worth explaining precisely why, not just naming 'sensitivity' generically.")