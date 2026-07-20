"""Directly distinguishes whether early-match volatility comes from the Beta-Binomial
posterior itself moving too fast (n0 too small) or from the blend mechanics overriding a
well-behaved posterior (sensitivity_aware_blend's weighting) -- rather than guessing
between the two, per the reviewer's own falsifiable-check framing."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "pipelines")
import joblib
import pandas as pd
import numpy as np
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.ml_informed_markov import (
    build_pretrained_prior, ServeReturnPosterior, ml_informed_point_probabilities,
    recursion_sensitivity, sensitivity_aware_blend,
)
from pipelines.replay_match import _row_to_match_state
from pipelines.generate_publication_trajectory import compute_composite_prematch_probability

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
posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)
print(f"n0_serve={n0_serve:.2f}, n0_return={n0_return:.2f}, p_serve0={p_serve0:.4f}\n")

print(f"{'Pt':>4} {'raw_clf':>8} {'posterior_mean':>15} {'blended':>9} {'sensitivity':>11} "
      f"{'weight_evid':>11} {'pts_obs':>8}")
for i in range(30):
    row = match.iloc[i].to_dict()
    state = _row_to_match_state(row)
    p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(row, model, feature_cols)

    posterior_mean_before = posterior.mean_serve()
    sens = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "serve")
    pts_obs = posterior.points_observed_serve()
    weight_evid = 200.0 / (200.0 + pts_obs)
    blended = sensitivity_aware_blend(p_a_serve_raw, posterior_mean_before, sens,
                                      points_observed=pts_obs)

    print(f"{i+1:>4} {p_a_serve_raw:>8.4f} {posterior_mean_before:>15.4f} {blended:>9.4f} "
          f"{sens:>11.4f} {weight_evid:>11.4f} {pts_obs:>8.1f}")

    pt_winner = row.get("PtWinner")
    if pd.notna(pt_winner):
        p1_is_winner = bool(row["player1_is_winner"])
        a_won = (int(pt_winner) == 1) if p1_is_winner else (int(pt_winner) == 2)
        if state.server_is_a:
            posterior = posterior.update_serve(a_won)
        else:
            posterior = posterior.update_return(a_won)

print("\nDIAGNOSIS:")
print("- If 'posterior_mean' column moves SLOWLY (small changes point to point) but")
print("  'blended' swings MUCH more -- the posterior is well-behaved, and the BLEND")
print("  (sensitivity_aware_blend, specifically the sensitivity-based weight, since")
print("  weight_evid should already be near 1.0 this early) is the actual source of")
print("  volatility, NOT n0.")
print("- If 'posterior_mean' ITSELF swings by more than a few % per point, n0 is")
print("  genuinely too small and the reviewer's top-ranked fix (raise n0_base) applies.")