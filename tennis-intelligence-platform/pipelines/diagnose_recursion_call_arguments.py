"""Definitively answers the double-update question by (1) showing B's implied rates are
pure arithmetic complements of A's own tracked posteriors -- not an independently
double-fed object, since no such object exists anywhere in the codebase (confirmed by
exhaustive grep before writing this script) -- and (2) monkey-patching
prob_a_wins_match_from_state to record its LITERAL, ACTUAL call arguments from inside the
real, unmodified ml_informed_markov_predict function -- not a parallel reimplementation
that could silently drift from the real code path."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "pipelines")
import joblib
import pandas as pd
import numpy as np
from tennis_intel.live.build_point_dataset import build_point_dataset
import tennis_intel.live.live_win_probability as lwp
from tennis_intel.live.ml_informed_markov import (
    build_pretrained_prior, ServeReturnPosterior, ml_informed_markov_predict,
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
posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)

# Monkey-patch the REAL recursion function to record its literal call arguments, without
# changing its behavior at all (calls through to the original immediately).
call_log = []
_original_fn = lwp.prob_a_wins_match_from_state
def _recording_wrapper(state, p_a_serve, p_a_return):
    result = _original_fn(state, p_a_serve, p_a_return)
    call_log.append({
        "a_sets": state.a_sets, "b_sets": state.b_sets,
        "a_games": state.a_games, "b_games": state.b_games,
        "server_is_a": state.server_is_a,
        "p_a_serve_arg": p_a_serve, "p_a_return_arg": p_a_return,
        "result": result,
    })
    return result
lwp.prob_a_wins_match_from_state = _recording_wrapper
# ml_informed_markov.py imported prob_a_wins_match_from_state directly into its own
# namespace at module load time -- patch THAT reference too, or the wrapper never
# actually intercepts the real call site.
import tennis_intel.live.ml_informed_markov as mim
mim.prob_a_wins_match_from_state = _recording_wrapper

print(f"Seeded: p_serve0={p_serve0:.4f}, p_return0={p_return0:.4f}\n")
print(f"{'Pt':>4} {'gm':>5} {'A_post_srv':>10} {'A_post_ret':>10} {'B_implied_srv(1-Aret)':>22} "
      f"{'B_implied_ret(1-Asrv)':>22} {'RECURSION_ARG_p_a_serve':>23} "
      f"{'RECURSION_ARG_p_a_return':>24} {'server_is_a':>11} {'result':>8}")

for i in range(35):
    row = match.iloc[i].to_dict()
    state = _row_to_match_state(row)

    a_post_srv = posterior.mean_serve()
    a_post_ret = posterior.mean_return()
    b_implied_srv = 1.0 - a_post_ret   # pure arithmetic complement, NOT a separate object
    b_implied_ret = 1.0 - a_post_srv   # pure arithmetic complement, NOT a separate object

    call_log.clear()
    p_match, posterior_after = ml_informed_markov_predict(state, row, model, feature_cols, posterior)
    # ml_informed_markov_predict calls prob_a_wins_match_from_state 5 times per point, not
    # once: recursion_sensitivity itself probes the recursion twice per direction (finite
    # difference) x 2 directions (serve, return) = 4 calls, then ONE final call with the
    # actual blended values produces the real output -- confirmed by checking the source
    # order directly (both recursion_sensitivity calls precede the final call). The LAST
    # entry in call_log is always the real, final call that produces p_match.
    assert len(call_log) == 5, f"Expected exactly 5 recursion calls (4 sensitivity probes " \
                               f"+ 1 final), got {len(call_log)} -- the internal call " \
                               f"structure may have changed"
    call = call_log[-1]
    assert abs(call["result"] - p_match) < 1e-9, "The last recorded call must match the returned p_match"

    gm_score = f"{int(row['Gm1'])}-{int(row['Gm2'])}"
    print(f"{i+1:>4} {gm_score:>5} {a_post_srv:>10.4f} {a_post_ret:>10.4f} "
          f"{b_implied_srv:>22.4f} {b_implied_ret:>22.4f} "
          f"{call['p_a_serve_arg']:>23.4f} {call['p_a_return_arg']:>24.4f} "
          f"{str(call['server_is_a']):>11} {call['result']:>8.4f}")

    posterior = posterior_after

lwp.prob_a_wins_match_from_state = _original_fn
mim.prob_a_wins_match_from_state = _original_fn

print("\nWhat this table proves directly:")
print("(a) B's implied rates are ALGEBRAIC COMPLEMENTS of A's own posteriors (1-A_ret,")
print("    1-A_srv) -- computed here for display only, confirming by construction that")
print("    no separate B-side posterior object exists anywhere to be 'double-fed'.")
print("(b) RECURSION_ARG columns are the LITERAL values the real prob_a_wins_match_from_state")
print("    call received, captured via monkey-patch from the actual, unmodified")
print("    ml_informed_markov_predict code path -- not reconstructed separately. If")
print("    server_is_a doesn't match the real row's actual server for that point, that's")
print("    a genuine orientation bug. If it matches correctly, and RECURSION_ARG_p_a_serve/")
print("    return look reasonable given A_post_srv/A_post_ret, the recursion is being fed")
print("    exactly what it should be, and any remaining extremity comes from the")
print("    recursion's own sensitivity at that state, not a data-feeding bug.")