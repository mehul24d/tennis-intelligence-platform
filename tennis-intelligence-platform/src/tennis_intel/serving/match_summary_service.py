"""
match_summary_service.py — the service layer behind the Match Summary cards
(Largest Comeback, Largest Probability Swing, Longest Winning Streak, Longest Service
Hold, Break Points Created/Converted, Total Winners).

DESIGNATED PROBABILITY ENGINE: "Largest Comeback" and "Largest Probability Swing" are
inherently engine-dependent — five different engines would give five different
answers. This module uses ML-Informed Markov (smoothed) throughout, since it is this
project's own primary, most-validated engine (confirmed via
evaluate_full_match_calibration.py / evaluate_ml_informed_markov.py across this
project's history as the best-calibrated of the five). This choice is made explicit
here rather than silently picked, so a future reader isn't left guessing why one
engine's numbers appear in a "Match Summary" card that doesn't otherwise name an
engine.

REUSES tennis_intel.serving.replay_service.compute_five_engine_trajectory — an
earlier version of this file had its OWN separate copy of the seeding + per-point
loop logic, duplicating replay_match_by_id's own construction. Refactored to share
one implementation (also used by model_agreement_service.py and
point_timeline_service.py) rather than risk four independent copies silently
drifting apart over time.

PLACEHOLDERS, precise about WHY: "Total Winners" and "Total Unforced Errors" data
genuinely EXISTS in charting-m-stats-Overview.csv (confirmed earlier this session —
that file has real "winners"/"unforced" columns, already used elsewhere in this
project's feature engineering) — but that file is NOT currently loaded into
ReplayContext (only frozen_join/day6/points are), so it is not wired into THIS
service yet. This is a "not yet connected" gap, not a "data doesn't exist" gap.
"""

from __future__ import annotations

from tennis_intel.serving.replay_service import ReplayContext, compute_five_engine_trajectory


def _signed_streak_lengths(won_flags: list[bool]) -> list[int]:
    """Signed consecutive-run lengths — positive N means N consecutive wins ending
    at that index, negative N means N consecutive losses. Same idea as this
    project's own points_streak feature (point_level_features.py), reimplemented
    minimally here since this is display logic, not a model feature."""
    lengths = []
    current = 0
    for won in won_flags:
        if won:
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
        lengths.append(current)
    return lengths


def get_match_summary(ctx: ReplayContext, match_id: str) -> dict:
    """Computes the Match Summary card stats for one match. Raises ValueError if
    match_id isn't in the frozen-join corpus (via compute_five_engine_trajectory)."""
    computed = compute_five_engine_trajectory(ctx, match_id)
    records = computed["records"]
    final_winner_is_p1 = computed["final_winner_is_p1"]
    smoothed_p1 = computed["ml_informed_p1"]
    n_points = len(records)

    # BUG FIX (2026-07, see docs/known_issue_ml_informed_markov_pre_point_state.md and
    # point_timeline_service.py's matching fix): smoothed_p1[i] is "the probability just
    # before point i is played" (row_to_match_state's pre-point convention); point i's
    # own outcome is only reflected in smoothed_p1[i+1] (or, for the final point, the
    # actual terminal outcome). The previous `all_p1 = [prematch_p1] + smoothed_p1` /
    # `swings[i] = |all_p1[i] - all_p1[i-1]|` construction paired point i with the
    # PREVIOUS point's before/after values, mislabeling which point caused each swing.
    swings = [
        abs(
            (smoothed_p1[i + 1] if i + 1 < n_points else (1.0 if final_winner_is_p1 else 0.0))
            - smoothed_p1[i]
        )
        for i in range(n_points)
    ]
    max_swing_idx = max(range(len(swings)), key=lambda i: swings[i])
    swing_after = (
        smoothed_p1[max_swing_idx + 1] if max_swing_idx + 1 < n_points
        else (1.0 if final_winner_is_p1 else 0.0)
    )
    largest_swing = {
        "point_index": int(records[max_swing_idx]["Pt"]),
        "probability_before": round(smoothed_p1[max_swing_idx], 6),
        "probability_after": round(swing_after, 6),
        "swing": round(swings[max_swing_idx], 6),
    }

    winner_p = smoothed_p1 if final_winner_is_p1 else [1.0 - p for p in smoothed_p1]
    min_winner_p = min(winner_p)
    largest_comeback = {
        "lowest_win_probability": round(min_winner_p, 6),
        "comeback_margin": round(max(0.0, 0.5 - min_winner_p), 6),
        "point_index_of_low": int(records[winner_p.index(min_winner_p)]["Pt"]),
    }

    # CONVENTION, SETTLED 2026-07 (see ml_informed_markov.py's ml_informed_markov_predict
    # docstring and docs/ptwinner_convention_correction.md for the full investigation):
    # PtWinner is LITERAL, fixed-player-relative — PtWinner==1 means player 1 won the
    # point, PERIOD, independent of who served. A same-day "fix" here previously claimed
    # the opposite (server-relative), citing check_ptwinner_disagreement_at_scale.py's
    # "0.00% disagreement" — that script only ever checks internal self-consistency
    # between PtWinner and fixed-player Pts on INTERIOR (non-game-boundary) points, which
    # cannot distinguish "PtWinner is server-relative" from "PtWinner is literal" (the two
    # are mirror images of each other and only diverge at Svr==2, which the script never
    # examines). Checked directly against Gm1/Gm2 at real game boundaries instead: literal
    # PtWinner matches at 99.91% corpus-wide, symmetric across Svr==1/2; server-relative
    # matches only ~51% (chance) at boundaries.
    def _point_winner_is_p1(row: dict) -> bool:
        return row["PtWinner"] == 1

    p1_won_flags = [_point_winner_is_p1(row) for row in records]
    p1_streaks = _signed_streak_lengths(p1_won_flags)
    p1_longest_streak = max(p1_streaks) if p1_streaks else 0
    p2_longest_streak = -min(p1_streaks) if p1_streaks else 0

    server_win_run = 0
    longest_server_run = 0
    prev_svr = None
    for row in records:
        # "Did the SERVER win this point" now genuinely requires cross-referencing
        # PtWinner (literal, player-relative) against who's serving — PtWinner==1
        # alone tells us player 1 won, not that the server won. A same-day "fix" here
        # removed this cross-reference on the (now-reverted) assumption that PtWinner
        # was already server-relative; restored, since literal PtWinner requires it.
        server_is_p1 = row["Svr"] == 1
        server_won = (row["PtWinner"] == 1) == server_is_p1
        if row["Svr"] != prev_svr:
            server_win_run = 0
        if server_won:
            server_win_run += 1
            longest_server_run = max(longest_server_run, server_win_run)
        else:
            server_win_run = 0
        prev_svr = row["Svr"]

    bp_rows = [row for row in records if row.get("is_break_point")]
    # p1_bp_created: unaffected by the PtWinner convention (only uses Svr) — P1 created
    # a break point whenever P2 (Svr==2) is serving at a break point, regardless of who
    # ultimately won it.
    p1_bp_created = sum(1 for row in bp_rows if row["Svr"] == 2)
    # p1_bp_converted: P2 serving (Svr==2), AND P1 (the receiver) won the point — under
    # literal PtWinner, P1 winning is directly PtWinner==1 (not !=1, which a same-day
    # "fix" here previously used under the now-reverted server-relative assumption).
    p1_bp_converted = sum(1 for row in bp_rows if row["Svr"] == 2 and row["PtWinner"] == 1)
    p2_bp_created = sum(1 for row in bp_rows if row["Svr"] == 1)
    # p2_bp_converted: P1 serving (Svr==1), AND P2 (the receiver) won — under literal
    # PtWinner, P2 winning is PtWinner==2, i.e. PtWinner != 1 (PtWinner only takes
    # values 1/2, so this form happens to be correct under BOTH conventions here —
    # unlike p1_bp_converted above, which is not symmetric).
    p2_bp_converted = sum(1 for row in bp_rows if row["Svr"] == 1 and row["PtWinner"] != 1)

    return {
        "match_id": match_id,
        "largest_probability_swing": largest_swing,
        "largest_comeback": largest_comeback,
        "longest_winning_streak_points": {
            "player1": p1_longest_streak, "player2": p2_longest_streak,
        },
        "longest_service_hold_points": longest_server_run,
        "break_points": {
            "player1_created": p1_bp_created, "player1_converted": p1_bp_converted,
            "player2_created": p2_bp_created, "player2_converted": p2_bp_converted,
        },
        "total_winners": None,
        "total_unforced_errors": None,
        "serve_percentage": None,
    }