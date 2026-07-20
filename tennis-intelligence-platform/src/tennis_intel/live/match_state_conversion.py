"""
match_state_conversion.py — the single, canonical implementation of the point-row-to-
MatchState conversion, extracted per the external audit's Architecture Review finding E /
Code Review finding #5: three near-identical implementations existed independently across
evaluate_live_engines_v2.py, replay_match.py, and evaluate_live_engines.py, which the audit
correctly flagged as "a known recurrence vector" — orientation bugs have already happened
twice in this exact conversion logic (the tb_points bug and the "A=Player1 vs A=winner"
mismatch bug, both documented below), and duplicated implementations mean a fix applied to
one copy can silently fail to reach the others (exactly what happened: evaluate_live_engines.py,
the pre-v2 original, never received the tb_points fix that v2 and replay_match.py both got).

This module is the ONLY place this conversion should be implemented going forward. All
three pipeline scripts import this function rather than defining their own copy.
"""

from __future__ import annotations

import pandas as pd

from tennis_intel.live.live_win_probability import MatchState


def row_to_match_state(row: dict) -> MatchState:
    """
    Converts a point-dataset row into the MatchState the live engines expect.

    CONVENTION: "A" always means the TRACKED WINNER (the real, eventual winner of this
    historical match), not "Player 1" — this matches batch_simulate_dynamic's own
    hard-coded internal assumption (see that function's docstring: "player1_is_winner:
    bool, # maps sim's A/B (A = tracked winner) to MCP Player1/2"). Callers that want a
    Player-1-oriented probability must invert the result when player1_is_winner is False
    — see markov_p_player1/ml_p_player1-style wrappers in the calling scripts for the
    exact invert-if-needed pattern.

    BUG FIX #1 (found via a single-match replay, 2026-07): p1_points/p2_points are only
    populated for REGULAR games — point_level_features.py deliberately leaves them NaN
    during a tiebreak, storing the real count in tb_p1_points/tb_p2_points instead. An
    earlier version of this function (still present, unfixed, in the pre-v2
    evaluate_live_engines.py before this centralization) read only p1_points/p2_points
    unconditionally, silently feeding every tiebreak point to both engines as "the
    tiebreak just started, 0-0" regardless of the real score.

    BUG FIX #2 (found via a real chart discontinuity between the pre-match point and the
    first replayed point, 2026-07): an earlier version of replay_match.py's own copy of
    this function constructed "A" to always mean Player 1, regardless of who actually won
    — mismatched against batch_simulate_dynamic's own "A=winner" assumption, causing an
    inverted server_is_winner-equivalent feature on every point whenever Player 1 lost.
    """
    is_tb = bool(row["is_tiebreak_game"])
    if is_tb:
        a_pts_raw, b_pts_raw = row.get("tb_p1_points"), row.get("tb_p2_points")
    else:
        a_pts_raw, b_pts_raw = row.get("p1_points"), row.get("p2_points")

    p1_is_winner = bool(row["player1_is_winner"])
    if p1_is_winner:
        a_sets, b_sets, a_games, b_games = row["Set1"], row["Set2"], row["Gm1"], row["Gm2"]
        a_points, b_points = a_pts_raw, b_pts_raw
        server_is_a = (row["Svr"] == 1)
    else:
        a_sets, b_sets, a_games, b_games = row["Set2"], row["Set1"], row["Gm2"], row["Gm1"]
        a_points, b_points = b_pts_raw, a_pts_raw
        server_is_a = (row["Svr"] == 2)

    return MatchState(
        a_sets=int(a_sets), b_sets=int(b_sets),
        a_games=int(a_games), b_games=int(b_games),
        a_points=int(a_points) if pd.notna(a_points) else 0,
        b_points=int(b_points) if pd.notna(b_points) else 0,
        server_is_a=bool(server_is_a),
        is_tiebreak=is_tb,
        best_of=int(row["best_of"]) if pd.notna(row.get("best_of")) else 3,
    )