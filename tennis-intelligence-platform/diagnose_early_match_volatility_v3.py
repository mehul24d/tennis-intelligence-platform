"""v3: adds real within-game point score (e.g., "15-0", "30-15", "AD-40") to directly
confirm whether the observed out_p swings map onto genuine point-by-point progress within
the opening game, per the reviewer's concrete next step."""
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

PT_NAMES = {0: "0", 1: "15", 2: "30", 3: "40"}
def point_score_str(p1, p2, is_tb):
    if is_tb:
        return f"{int(p1)}-{int(p2)} (TB)"
    p1, p2 = int(p1), int(p2)
    if p1 >= 3 and p2 >= 3:
        if p1 == p2:
            return "Deuce"
        return "AD-40" if p1 > p2 else "40-AD"
    p1n = PT_NAMES.get(p1, str(p1))
    p2n = PT_NAMES.get(p2, str(p2))
    return f"{p1n}-{p2n}"

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

print(f"{'Pt':>4} {'gm_score':>8} {'pt_score':>10} {'server':>7} {'raw_clf':>8} {'sens':>6} "
      f"{'out_p':>8} {'out_delta':>10}")

prev_out_p = None
for i in range(30):
    row = match.iloc[i].to_dict()
    state = _row_to_match_state(row)
    p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(row, model, feature_cols)

    is_tb = bool(row["is_tiebreak_game"])
    if is_tb:
        p1_pts_raw, p2_pts_raw = row.get("tb_p1_points"), row.get("tb_p2_points")
    else:
        p1_pts_raw, p2_pts_raw = row.get("p1_points"), row.get("p2_points")
    p1_pts = p1_pts_raw if pd.notna(p1_pts_raw) else 0
    p2_pts = p2_pts_raw if pd.notna(p2_pts_raw) else 0
    pt_score = point_score_str(p1_pts, p2_pts, is_tb)
    server_label = "P1" if row["Svr"] == 1 else "P2"

    sens_serve = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "serve")
    pts_obs_serve = posterior.points_observed_serve()
    pts_obs_return = posterior.points_observed_return()
    blended_serve = sensitivity_aware_blend(p_a_serve_raw, posterior.mean_serve(), sens_serve,
                                            points_observed=pts_obs_serve)
    sens_return = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "return")
    blended_return = sensitivity_aware_blend(p_a_return_raw, posterior.mean_return(), sens_return,
                                             points_observed=pts_obs_return)
    blended_serve_clip = float(np.clip(blended_serve, 0.01, 0.99))
    blended_return_clip = float(np.clip(blended_return, 0.01, 0.99))

    out_p = prob_a_wins_match_from_state(state, blended_serve_clip, blended_return_clip)
    out_delta = (out_p - prev_out_p) if prev_out_p is not None else 0.0
    prev_out_p = out_p

    gm_score = f"{int(row['Gm1'])}-{int(row['Gm2'])}"
    print(f"{i+1:>4} {gm_score:>8} {pt_score:>10} {server_label:>7} {p_a_serve_raw:>8.4f} "
          f"{sens_serve:>6.2f} {out_p:>8.4f} {out_delta:>+10.4f}")

    pt_winner = row.get("PtWinner")
    if pd.notna(pt_winner):
        p1_is_winner = bool(row.get("player1_is_winner", True))
        a_won = (int(pt_winner) == 1) if p1_is_winner else (int(pt_winner) == 2)
        if state.server_is_a:
            posterior = posterior.update_serve(a_won)
        else:
            posterior = posterior.update_return(a_won)

print("\nCheck: do out_p swings correspond to REAL, meaningful point-score progress")
print("(e.g., reaching 40-15 or a break point) or do they occur between two point-score")
print("states that a human wouldn't consider very different in leverage?")