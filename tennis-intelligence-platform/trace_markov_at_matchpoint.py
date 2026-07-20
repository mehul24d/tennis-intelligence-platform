"""Corrected version: replicates the FULL Day 11 pipeline logic exactly, including the
target-adjustment (track random player, flip if tracking the loser) step that the first
version of this script incorrectly omitted."""
import sys
sys.path.insert(0, "src")
import hashlib
import pandas as pd
from tennis_intel.live.live_win_probability import MatchState, prob_a_wins_match_from_state

def tracked_player_is_winner(match_id: str) -> bool:
    digest = hashlib.md5(match_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 2 == 0

df = pd.read_parquet("data/processed/day11_head_to_head_v2_predictions.parquet")
mp_rows = df[df["is_match_point"] == True].copy()

print("--- Correctly reconstructing Markov's prediction, INCLUDING target-adjustment ---")
n_match = 0
n_mismatch = 0
for i, row in mp_rows.head(15).iterrows():
    p1_is_winner = bool(row["player1_is_winner"])
    if p1_is_winner:
        a_sets, b_sets, a_games, b_games = row["Set1"], row["Set2"], row["Gm1"], row["Gm2"]
        a_points, b_points = row["p1_points"], row["p2_points"]
        server_is_a = (row["Svr"] == 1)
    else:
        a_sets, b_sets, a_games, b_games = row["Set2"], row["Set1"], row["Gm2"], row["Gm1"]
        a_points, b_points = row["p2_points"], row["p1_points"]
        server_is_a = (row["Svr"] == 2)

    state = MatchState(
        a_sets=int(a_sets), b_sets=int(b_sets), a_games=int(a_games), b_games=int(b_games),
        a_points=int(a_points) if pd.notna(a_points) else 0,
        b_points=int(b_points) if pd.notna(b_points) else 0,
        server_is_a=bool(server_is_a), is_tiebreak=bool(row["is_tiebreak_game"]),
        best_of=int(row["best_of"]) if pd.notna(row["best_of"]) else 3,
    )
    ps_key = "winner_first_serve_win_pct_career" if p1_is_winner else "loser_first_serve_win_pct_career"
    opp_key = "loser_first_serve_win_pct_career" if p1_is_winner else "winner_first_serve_win_pct_career"
    ps = row.get(ps_key)
    opp_serve = row.get(opp_key)
    ps = 0.65 if pd.isna(ps) else float(ps)
    opp_serve = 0.65 if pd.isna(opp_serve) else float(opp_serve)
    pr = 1.0 - opp_serve

    p_winner_wins = prob_a_wins_match_from_state(state, ps, pr)

    # Apply the SAME target-adjustment as evaluate_live_engines_v2.py
    track_winner = tracked_player_is_winner(row["match_id"])
    if track_winner:
        adjusted_pred = p_winner_wins
        expected_target = 1.0
    else:
        adjusted_pred = 1.0 - p_winner_wins
        expected_target = 0.0

    matches = abs(adjusted_pred - row["markov_pred"]) < 0.01
    target_matches = abs(expected_target - row["target"]) < 0.01
    n_match += 1
    if not matches:
        n_mismatch += 1

    print(f"\nmatch_id={row['match_id'][:40]}, Pt={row['Pt']}")
    print(f"  P(winner wins) [raw recursion] = {p_winner_wins:.4f}")
    print(f"  track_winner (hash-based) = {track_winner}")
    print(f"  adjusted_pred (should match recorded) = {adjusted_pred:.4f}, "
          f"RECORDED markov_pred = {row['markov_pred']:.4f}, MATCH={matches}")
    print(f"  expected_target = {expected_target}, RECORDED target = {row['target']}, "
          f"MATCH={target_matches}")

print(f"\n\n=== SUMMARY: {n_match - n_mismatch}/{n_match} rows correctly reproduced "
      f"once target-adjustment is properly included ===")