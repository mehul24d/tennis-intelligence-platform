"""Extends instrumentation to points 20-50 of the real Djokovic-Alcaraz match, where the
chart shows the smoothed (cyan) line diverging much further from the unsmoothed (green)
line than any single point's swing could explain. Checks: does the posterior's trajectory
match the REAL, ground-truth serve win/loss record over this window (legitimate
accumulation), or does it diverge from that ground truth (a sign of double-counting or
some other compounding bug)?"""
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
print(f"Seeded: p_serve0={p_serve0:.4f}, n0_serve={n0_serve:.2f}\n")

posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)

# Track GROUND TRUTH independently: real serve points won/lost by A, computed directly
# from PtWinner, with ZERO dependence on the posterior or classifier -- this is the
# independent check the posterior's own trajectory must be validated against.
real_a_serve_wins = 0
real_a_serve_total = 0

print(f"{'Pt':>4} {'gm':>5} {'server':>7} {'real_won':>8} {'real_a_rate':>11} "
      f"{'posterior_mean':>14} {'unsmoothed_out':>14} {'smoothed_out':>13}")

for i in range(50):
    row = match.iloc[i].to_dict()
    state = _row_to_match_state(row)

    p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(row, model, feature_cols)
    p_a_serve_raw_c = float(np.clip(p_a_serve_raw, 0.01, 0.99))
    p_a_return_raw_c = float(np.clip(p_a_return_raw, 0.01, 0.99))
    unsmoothed_out = prob_a_wins_match_from_state(state, p_a_serve_raw_c, p_a_return_raw_c)

    smoothed_out, posterior_after = ml_informed_markov_predict(state, row, model, feature_cols, posterior)

    # Ground truth tracking (A's serve points only, independent of posterior internals)
    if state.server_is_a:
        pt_winner = row.get("PtWinner")
        if pd.notna(pt_winner):
            p1_is_winner = bool(row.get("player1_is_winner", True))
            a_won = (int(pt_winner) == 1) if p1_is_winner else (int(pt_winner) == 2)
            real_a_serve_total += 1
            if a_won:
                real_a_serve_wins += 1

    real_rate = (real_a_serve_wins / real_a_serve_total) if real_a_serve_total > 0 else float("nan")
    server_label = "A" if state.server_is_a else "B"
    a_won_str = "-"
    pt_winner = row.get("PtWinner")
    if pd.notna(pt_winner) and state.server_is_a:
        p1_is_winner = bool(row.get("player1_is_winner", True))
        a_won_str = str(bool((int(pt_winner) == 1) if p1_is_winner else (int(pt_winner) == 2)))

    gm_score = f"{int(row['Gm1'])}-{int(row['Gm2'])}"
    print(f"{i+1:>4} {gm_score:>5} {server_label:>7} {a_won_str:>8} {real_rate:>11.4f} "
          f"{posterior.mean_serve():>14.4f} {unsmoothed_out:>14.4f} {smoothed_out:>13.4f}")

    posterior = posterior_after

print(f"\nFinal real A-serve win rate over these {real_a_serve_total} real serve points: "
      f"{real_a_serve_wins}/{real_a_serve_total} = {real_a_serve_wins/real_a_serve_total:.4f}")
print(f"Final posterior.mean_serve(): {posterior.mean_serve():.4f}")
print(f"Pre-match prior p_serve0 was: {p_serve0:.4f}")
print("\nCheck: does posterior.mean_serve() end up CLOSE to the real observed rate above")
print("(legitimate accumulation of real evidence), or does it overshoot BEYOND what the")
print("real win/loss record justifies (a sign of double-counting or compounding)?")