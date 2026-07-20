"""Extended diagnostic per reviewer's four concrete checks:
1. Score state + point type (routine/BP/deuce/SP) alongside sensitivity
2. Actual output win-probability delta per point, to check whether it matches
   sensitivity's implied magnitude
3. (Analysis, not code) whether a variance-aware, sensitivity-scaled shrinkage is needed
4. Whether raw_clf repeating identical values across consecutive points reflects genuinely
   identical feature vectors or a stale-feature-set bug
"""
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

print(f"{'Pt':>4} {'score':>8} {'pt_type':>8} {'raw_clf':>8} {'sens':>7} {'blend_serve':>11} "
      f"{'out_p':>8} {'out_delta':>10} {'feat_hash':>10}")

prev_out_p = None
prev_feat_hash = None
for i in range(30):
    row = match.iloc[i].to_dict()
    state = _row_to_match_state(row)
    p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(row, model, feature_cols)

    # Point type classification
    if row.get("is_break_point"):
        pt_type = "BP"
    elif row.get("is_set_point"):
        pt_type = "SP"
    elif row.get("is_match_point"):
        pt_type = "MP"
    elif row.get("is_tiebreak_game"):
        pt_type = "TB"
    else:
        # rough deuce check from raw point score if available
        pt_type = "routine"

    sens_serve = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "serve")
    sens_return = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "return")

    pts_obs_serve = posterior.points_observed_serve()
    pts_obs_return = posterior.points_observed_return()
    blended_serve = sensitivity_aware_blend(p_a_serve_raw, posterior.mean_serve(), sens_serve,
                                            points_observed=pts_obs_serve)
    blended_return = sensitivity_aware_blend(p_a_return_raw, posterior.mean_return(), sens_return,
                                             points_observed=pts_obs_return)
    blended_serve_clip = float(np.clip(blended_serve, 0.01, 0.99))
    blended_return_clip = float(np.clip(blended_return, 0.01, 0.99))

    out_p = prob_a_wins_match_from_state(state, blended_serve_clip, blended_return_clip)
    out_delta = (out_p - prev_out_p) if prev_out_p is not None else 0.0
    prev_out_p = out_p

    # Check 4: are consecutive identical raw_clf values from identical feature vectors,
    # or a stale/unchanging feature-set artifact?
    row_a_serves = dict(row)
    p1_is_winner = bool(row.get("player1_is_winner", True))
    row_a_serves["server_is_player1"] = p1_is_winner
    feat_vector = tuple(row_a_serves.get(c) for c in feature_cols if c in row_a_serves)
    feat_hash = hash(feat_vector) % 10000
    same_as_prev_feat = "SAME" if feat_hash == prev_feat_hash else ""
    prev_feat_hash = feat_hash

    score_str = f"{int(row['Gm1'])}-{int(row['Gm2'])}"
    print(f"{i+1:>4} {score_str:>8} {pt_type:>8} {p_a_serve_raw:>8.4f} {sens_serve:>7.3f} "
          f"{blended_serve:>11.4f} {out_p:>8.4f} {out_delta:>+10.4f} {feat_hash:>6} {same_as_prev_feat}")

    pt_winner = row.get("PtWinner")
    if pd.notna(pt_winner):
        a_won = (int(pt_winner) == 1) if p1_is_winner else (int(pt_winner) == 2)
        if state.server_is_a:
            posterior = posterior.update_serve(a_won)
        else:
            posterior = posterior.update_return(a_won)

print("\nWhat to check:")
print("1. Do large |out_delta| values line up with high 'sens' values? If yes, confirms")
print("   sensitivity is genuinely amplifying small blend changes into large output swings.")
print("2. Is |out_delta| much bigger than the corresponding change in 'blend_serve' from")
print("   the previous row, by roughly a factor matching 'sens'? That's the direct")
print("   confirmation the reviewer asked for.")
print("3. Do any 'SAME' flags appear for DIFFERENT raw_clf values, or do identical raw_clf")
print("   values always carry a 'SAME' flag? If raw_clf repeats WITHOUT 'SAME', the")
print("   classifier is coincidentally producing the same output for different feature")
print("   vectors (fine). If feat_hash is SAME across many rows even as game state")
print("   visibly changes, that would indicate a real feature-staleness bug.")