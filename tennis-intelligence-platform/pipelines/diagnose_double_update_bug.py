"""Checks the actual mechanism behind the 0.31 (unsmoothed) vs 0.065 (smoothed) gap at
row 31 of the Djokovic-Alcaraz trace. Corrects the hypothesis to match the real data
model: ServeReturnPosterior tracks only A's OWN serve rate and A's OWN return rate --
there is no separate "B's serve posterior" object. "B's serve rate" only ever exists
implicitly as (1 - A's return rate). Prints both of A's posteriors AND the actual blended
p_a_serve/p_a_return fed into the recursion, to find which one has become unreasonably
extreme and why."""
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

RAW_MCP = "data/raw/tennis_MatchChartingProject"
POINT_FILES = [f"{RAW_MCP}/charting-m-points-to-2009.csv",
               f"{RAW_MCP}/charting-m-points-2010s.csv",
               f"{RAW_MCP}/charting-m-points-2020s.csv"]
MATCH_ID = "20230716-M-Wimbledon-F-Novak_Djokovic-Carlos_Alcaraz"

frozen_join = pd.read_parquet("data/processed/joined_matches_m.parquet")
day6 = pd.read_parquet("data/processed/matches_with_day6_features.parquet")
points = build_point_dataset(POINT_FILES, frozen_join, day6)
points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]
match = points[points["match_id"] == MATCH_ID].sort_values("Pt").reset_index(drop=True)

payload = joblib.load("data/processed/day9_point_classifiers.joblib")
model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

first_row = match.iloc[0].to_dict()
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

p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
    p0_a_wins, p_a_return_seed, int(first_row["best_of"]),
    elo_matches_played_a=elo_a, elo_matches_played_b=elo_b, h2h_meetings=h2h,
)
print(f"Seeded: p_serve0={p_serve0:.4f}, p_return0={p_return0:.4f}, "
      f"n0_serve={n0_serve:.2f}, n0_return={n0_return:.2f}\n")

posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)

print(f"{'Pt':>4} {'gm':>5} {'srv':>4} {'post_serve':>10} {'post_return':>11} "
      f"{'blend_serve':>11} {'blend_return':>12} {'sens_srv':>8} {'sens_ret':>8} "
      f"{'unsmoothed':>10} {'smoothed':>9}")

for i in range(35):
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

    srv_label = "A" if state.server_is_a else "B"
    gm_score = f"{int(row['Gm1'])}-{int(row['Gm2'])}"
    print(f"{i+1:>4} {gm_score:>5} {srv_label:>4} {posterior.mean_serve():>10.4f} "
          f"{posterior.mean_return():>11.4f} {blended_serve:>11.4f} {blended_return:>12.4f} "
          f"{sens_serve:>8.2f} {sens_return:>8.2f} {unsmoothed_out:>10.4f} {smoothed_out:>9.4f}")

    pt_winner = row.get("PtWinner")
    if pd.notna(pt_winner):
        p1_is_winner = bool(row.get("player1_is_winner", True))
        a_won = (int(pt_winner) == 1) if p1_is_winner else (int(pt_winner) == 2)
        if state.server_is_a:
            posterior = posterior.update_serve(a_won)
        else:
            posterior = posterior.update_return(a_won)

print("\nWhat to check at row 31 specifically:")
print("- Is 'post_return' (A's return posterior mean) unreasonably LOW, dragging")
print("  'blend_return' down with it, and is THAT the main driver of smoothed_out's")
print("  extra pessimism versus unsmoothed_out?")
print("- Compare 'blend_serve' and 'blend_return' against the SAME point's raw classifier")
print("  values (not shown here, already in v3) -- are BOTH blended values MORE extreme")
print("  than what either the posterior alone OR the raw classifier alone would suggest,")
print("  which would indicate the blend formula itself pushes both in the same direction")
print("  simultaneously rather than one offsetting the other?")