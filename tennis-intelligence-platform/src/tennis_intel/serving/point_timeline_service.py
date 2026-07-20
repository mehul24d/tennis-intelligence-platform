"""
point_timeline_service.py — the service layer behind the interactive Point Timeline
table (every point: server, receiver, winner, score before/after, probability
before/after, swing, filters for break/set/match/tiebreak points and swing size).

DESIGNATED PROBABILITY ENGINE: same choice as match_summary_service.py — ML-Informed
Markov (smoothed), this project's primary, most-validated engine.

REUSES tennis_intel.serving.replay_service.compute_five_engine_trajectory — same
shared computation as every other service in this file family.
"""

from __future__ import annotations

from tennis_intel.serving.replay_service import ReplayContext, compute_five_engine_trajectory

_POINT_ORDINAL_DISPLAY = {0: "0", 1: "15", 2: "30", 3: "40", 4: "AD"}


def _server_perspective_score(row: dict) -> str | None:
    """
    Builds a "server's score - receiver's score" display string — e.g. a genuine
    break point (receiver at 40, server at 0) displays as "0-40", not the raw,
    fixed-player1/player2 "Pts" string re-used verbatim.

    BUG FIX (2026-07, found via a screenshot showing every 'Break points' filtered
    row as a service-game score like server=Nuno Borges, score=40-0, won by=Nuno
    Borges — i.e. looking exactly like the SERVER winning a routine service point,
    not the receiver winning a break point): the raw `Pts` column is FIXED
    player1/player2 notation for regular games (confirmed via
    point_level_features.py's own docstring — independently validated against
    PtWinner at 97.7%+ accuracy) — it does NOT reorder based on who's serving. The
    frontend's "Server" column, by contrast, IS correctly computed relative to
    Svr. Displaying the raw Pts string next to the correctly-computed Server name
    silently mixed two different perspectives: when player 2 served, "40-0" read
    as if player 1 (the receiver) had 40 and the server had 0 — a genuine break
    point — but with no way to tell that from the digits alone, since the SAME
    string "40-0" would also occur, meaning the opposite, whenever player 1 served.
    This function reorders using the already-parsed, unambiguous p1_points/
    p2_points ordinals (0-3, 4=Advantage) so the displayed score always matches
    the server named alongside it, exactly as a human reading a live scoreboard
    would expect.

    Tiebreak points use tb_p1_points/tb_p2_points directly, WITHOUT reordering —
    those columns are already server-first by construction (see
    point_level_features.py's own "TIEBREAK NOTATION FIX" docstring), unlike the
    misleadingly-named-the-same-way regular-game p1_points/p2_points.
    """
    is_tiebreak = bool(row.get("is_tiebreak_game", False))
    server_is_p1 = row["Svr"] == 1

    if is_tiebreak:
        # NOTE: despite the "tb_p1/tb_p2" naming, these are ALREADY server-first —
        # see the docstring above. The parser's own "tb_p1" slot is filled with
        # whichever player is CURRENTLY SERVING in a tiebreak, not literally player 1.
        server_pts = row.get("tb_p1_points")
        receiver_pts = row.get("tb_p2_points")
        if server_pts is None or receiver_pts is None:
            return None
        return f"{int(server_pts)}-{int(receiver_pts)}"

    p1_pts, p2_pts = row.get("p1_points"), row.get("p2_points")
    if p1_pts is None or p2_pts is None:
        return None
    server_pts = p1_pts if server_is_p1 else p2_pts
    receiver_pts = p2_pts if server_is_p1 else p1_pts
    if server_pts not in _POINT_ORDINAL_DISPLAY or receiver_pts not in _POINT_ORDINAL_DISPLAY:
        return None
    return f"{_POINT_ORDINAL_DISPLAY[int(server_pts)]}-{_POINT_ORDINAL_DISPLAY[int(receiver_pts)]}"


def get_point_timeline(
    ctx: ReplayContext, match_id: str,
    break_points_only: bool = False, set_points_only: bool = False,
    match_points_only: bool = False, tiebreak_only: bool = False,
    min_swing: float | None = None,
) -> dict:
    """
    Computes the full point-by-point timeline for one match, with optional filters.
    Raises ValueError if match_id isn't in the frozen-join corpus.

    min_swing: if set, only returns points whose |probability swing| is at least this
    value (e.g. 0.05 for "swings greater than 5%", 0.10 for "greater than 10%").
    Filters compose — passing multiple filters returns only points satisfying ALL of
    them, not any.
    """
    computed = compute_five_engine_trajectory(ctx, match_id)
    records = computed["records"]
    smoothed_p1 = computed["ml_informed_p1"]
    final_winner_is_p1 = computed["final_winner_is_p1"]
    n_points = len(records)

    largest_swing_so_far = 0.0

    # First pass: compute every point's row (unfiltered), tracking the single largest
    # swing in the match so "is_largest_swing" reflects the match's TRUE largest
    # swing, not just the largest within whatever filtered subset gets returned.
    all_rows = []
    for i, row in enumerate(records):
        # BUG FIX (2026-07, see docs/known_issue_ml_informed_markov_pre_point_state.md):
        # smoothed_p1[i] is computed at row i's PRE-POINT state (row_to_match_state's
        # documented convention) -- i.e. smoothed_p1[i] IS "the probability just before
        # point i is played". Point i's own outcome is only reflected in
        # smoothed_p1[i+1] (row i+1's pre-point state is, by construction of consecutive
        # point-dataset rows, exactly the post-point-i state) -- or, for the match's
        # final point, in the actual terminal outcome. The previous version built
        # `all_p1 = [prematch_p1] + smoothed_p1` and paired point i with
        # (all_p1[i], all_p1[i+1]) = (smoothed_p1[i-1], smoothed_p1[i]) for i>0 -- the
        # PREVIOUS point's before/after pair, mislabeled as point i's. Verified via a
        # hand-computed synthetic case (tests/unit/test_point_probability_indexing.py)
        # and confirmed as the dominant cause of near-chance probability-direction
        # correctness on a real match, even after the separate PtWinner fix in
        # ml_informed_markov.py.
        prob_before = smoothed_p1[i]
        prob_after = smoothed_p1[i + 1] if i + 1 < n_points else (
            1.0 if final_winner_is_p1 else 0.0
        )
        swing = abs(prob_after - prob_before)
        largest_swing_so_far = max(largest_swing_so_far, swing)

        server_is_p1 = row["Svr"] == 1
        # CONVENTION, SETTLED 2026-07 (see ml_informed_markov.py's
        # ml_informed_markov_predict docstring and docs/ptwinner_convention_correction.md
        # for the full investigation): PtWinner is LITERAL, fixed-player-relative —
        # PtWinner==1 means player 1 won the point, PERIOD, independent of who served.
        # A same-day "fix" here previously claimed the opposite (PtWinner is
        # server-relative), citing check_ptwinner_disagreement_at_scale.py's "0.00%
        # disagreement" — that script's ground truth is fixed-player Pts and it
        # explicitly skips every game-boundary row, so it can only ever confirm
        # INTERNAL self-consistency between PtWinner and Pts, not which of the two
        # self-consistent conventions (server-relative+fixed-Pts vs.
        # literal+server-first-Pts) actually matches the independently-recorded Gm1/Gm2
        # columns. Checked directly against Gm1/Gm2 at real game boundaries: literal
        # PtWinner matches at 99.91% (corpus-wide, symmetric across Svr==1/2); the
        # server-relative reading this comment previously described matches only ~51%
        # (chance level) at boundaries — invisible to the old script because it never
        # examined boundaries. This is simply `row["PtWinner"] == 1` — no Svr
        # adjustment needed, since "p1" here already means the named Player 1
        # (computed["p1_name"]), exactly what PtWinner==1 directly denotes.
        point_winner_is_p1 = row["PtWinner"] == 1

        all_rows.append({
            "point_index": int(row["Pt"]),
            "server": computed["p1_name"] if server_is_p1 else computed["p2_name"],
            "receiver": computed["p2_name"] if server_is_p1 else computed["p1_name"],
            "winner": computed["p1_name"] if point_winner_is_p1 else computed["p2_name"],
            "score_before": _server_perspective_score(row),
            "set1": int(row["Set1"]), "set2": int(row["Set2"]),
            "gm1": int(row["Gm1"]), "gm2": int(row["Gm2"]),
            "probability_before_p1": round(prob_before, 6),
            "probability_after_p1": round(prob_after, 6),
            "probability_swing": round(swing, 6),
            "is_break_point": bool(row.get("is_break_point", False)),
            "is_set_point": bool(row.get("is_set_point", False)),
            "is_match_point": bool(row.get("is_match_point", False)),
            "is_tiebreak_point": bool(row.get("is_tiebreak_game", False)),
        })

    for entry in all_rows:
        entry["is_largest_swing"] = abs(entry["probability_swing"] - largest_swing_so_far) < 1e-9

    filtered = all_rows
    if break_points_only:
        filtered = [r for r in filtered if r["is_break_point"]]
    if set_points_only:
        filtered = [r for r in filtered if r["is_set_point"]]
    if match_points_only:
        filtered = [r for r in filtered if r["is_match_point"]]
    if tiebreak_only:
        filtered = [r for r in filtered if r["is_tiebreak_point"]]
    if min_swing is not None:
        filtered = [r for r in filtered if r["probability_swing"] >= min_swing]

    return {
        "match_id": match_id,
        "n_points_total": len(all_rows),
        "n_points_returned": len(filtered),
        "points": filtered,
    }