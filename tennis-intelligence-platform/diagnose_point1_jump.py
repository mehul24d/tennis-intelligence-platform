"""Traces the EXACT computation at point 0 (true pre-match) vs point 1 (first real
charted point) for the Sinner-Alcaraz match, to find precisely where the visible jump
in the chart originates -- rather than reasoning from synthetic states."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "pipelines")
import pandas as pd
import numpy as np
import joblib
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.ml_informed_markov import (
    build_pretrained_prior, ServeReturnPosterior, ml_informed_markov_predict,
    recursion_sensitivity, ml_informed_point_probabilities, sensitivity_aware_blend,
)
from tennis_intel.live.live_win_probability import MatchState
from tennis_intel.live.markov_baseline import prob_win_match
from pipelines.generate_publication_trajectory import compute_composite_prematch_probability

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
print(f"p0_a_wins (composite, Alcaraz) = {p0_a_wins:.4f}")

loser_serve_surface = first_row.get("loser_first_serve_win_pct_surface_career")
p_a_return_seed = 1.0 - float(loser_serve_surface)
elo_matches_played_a = first_row.get("elo_matches_played_pre_winner")
best_of_val = int(first_row["best_of"])

p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
    p0_a_wins, p_a_return_seed, best_of_val, elo_matches_played_a=elo_matches_played_a,
)
print(f"p_serve0={p_serve0:.4f}, n0_serve={n0_serve:.1f}, p_return0={p_return0:.4f}, n0_return={n0_return:.1f}")

# TRUE pre-match point (point 0): run the prior straight through the recursion, matching
# what generate_publication_trajectory.py actually plots as the pre-match dot
true_pre_match_p_a_wins = prob_win_match(p_serve0, p_return0, best_of=best_of_val)
print(f"\nTRUE pre-match dot: P(Alcaraz wins) = {true_pre_match_p_a_wins:.4f}")

# Point 1: the ACTUAL first row's real state, using the REAL ml_informed_markov_predict
posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)
from pipelines.replay_match import _row_to_match_state
state1 = _row_to_match_state(first_row)
print(f"\nPoint 1 real state: a_sets={state1.a_sets}, a_games={state1.a_games}, "
      f"a_points={state1.a_points}, b_points={state1.b_points}, server_is_a={state1.server_is_a}")

p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(first_row, model, feature_cols)
print(f"\nRaw classifier prediction at point 1: p_a_serve_raw={p_a_serve_raw:.4f}, "
      f"p_a_return_raw={p_a_return_raw:.4f}")
print(f"Posterior mean BEFORE point 1 update: mean_serve={posterior.mean_serve():.4f}, "
      f"mean_return={posterior.mean_return():.4f}")

sens_serve = recursion_sensitivity(state1, p_a_serve_raw, p_a_return_raw, "serve")
sens_return = recursion_sensitivity(state1, p_a_serve_raw, p_a_return_raw, "return")
print(f"\nSensitivity at point 1's REAL state: sens_serve={sens_serve:.4f}, sens_return={sens_return:.4f}")

weight_serve = 0.2 + 0.6 * min(sens_serve / 3.0, 1.0)
weight_return = 0.2 + 0.6 * min(sens_return / 3.0, 1.0)
print(f"weight_on_posterior: serve={weight_serve:.4f}, return={weight_return:.4f}")

p_a_serve_blended = sensitivity_aware_blend(p_a_serve_raw, posterior.mean_serve(), sens_serve)
p_a_return_blended = sensitivity_aware_blend(p_a_return_raw, posterior.mean_return(), sens_return)
print(f"\nBlended: p_a_serve={p_a_serve_blended:.4f}, p_a_return={p_a_return_blended:.4f}")

p_match_point1, _ = ml_informed_markov_predict(state1, first_row, model, feature_cols, posterior)
print(f"\nFull match prediction at point 1: P(Alcaraz wins) = {p_match_point1:.4f}")
print(f"\n*** JUMP: {true_pre_match_p_a_wins:.4f} (pre-match) -> {p_match_point1:.4f} (point 1) ***")