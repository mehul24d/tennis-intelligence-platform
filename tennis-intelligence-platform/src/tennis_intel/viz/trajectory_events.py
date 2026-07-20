"""
trajectory_events.py — extracts structural events (set boundaries, breaks, tiebreaks,
match/championship points) from a point-by-point match dataframe. Pure data extraction,
no plotting — kept separate so future styling changes never require touching this logic,
and this logic can be unit-tested independently of matplotlib (per requirement 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SetBoundary:
    point_index: int      # cumulative point number where the set ended
    set_number: int
    score_str: str         # e.g. "6-4"
    winner_is_p1: bool


@dataclass
class MatchEvent:
    point_index: int
    kind: str               # "break", "tiebreak_start", "tiebreak_end", "match_point", "championship_point"
    label: str               # short text for the annotation
    winner_is_p1: bool | None = None


def detect_set_boundaries(df: pd.DataFrame) -> list[SetBoundary]:
    """A set boundary is any point where the cumulative sets-won total increases versus
    the previous point. Returns one SetBoundary per completed set, in point order."""
    boundaries = []
    sets_total_prev = None
    set_number = 0
    game_start_idx = 0

    for i, row in df.reset_index(drop=True).iterrows():
        sets_total = int(row["Set1"]) + int(row["Set2"])
        if sets_total_prev is not None and sets_total > sets_total_prev:
            set_number += 1
            prev_row = df.reset_index(drop=True).iloc[i - 1]
            p1_gained = int(row["Set1"]) > int(prev_row["Set1"])
            # BUG FIX (external review, 2026-07): games score of the set just completed.
            # prev_row's Gm1/Gm2 is the score ENTERING the set-deciding point (this
            # project's established convention: a row's own Set/Gm columns describe the
            # state before that row's own point is played) — i.e. one game short of the
            # set's real final tally, since prev_row's point hadn't been played yet when
            # that row was recorded. The winner of the set clinched it by winning exactly
            # one more game (a regular game or a tiebreak — tiebreak games don't increment
            # Gm1/Gm2 until the tiebreak itself concludes, so this +1 is correct in both
            # cases), so the true final score is prev_row's count with +1 added to
            # whichever player actually won the set.
            gm1, gm2 = int(prev_row["Gm1"]), int(prev_row["Gm2"])
            if p1_gained:
                gm1 += 1
            else:
                gm2 += 1
            score_str = f"{gm1}-{gm2}"
            boundaries.append(SetBoundary(
                point_index=int(row["point_index"]) if "point_index" in row else i + 1,
                set_number=set_number, score_str=score_str, winner_is_p1=p1_gained,
            ))
        sets_total_prev = sets_total

    return boundaries


def detect_events(df: pd.DataFrame) -> list[MatchEvent]:
    """
    Detects: service breaks (game won by the receiver), tiebreak start/end, match points
    reached, and championship points (match point in the final set of a deciding format).
    Uses the same is_break_point / is_tiebreak_game / is_match_point flags already computed
    by the frozen Day 7 point-level feature pipeline — no new score logic is invented here.
    """
    events = []
    df = df.reset_index(drop=True)
    prev_games_total = None
    prev_is_tb = False
    total_sets_at_row = None

    for i, row in df.iterrows():
        pidx = int(row["point_index"]) if "point_index" in row else i + 1
        games_total = int(row["Gm1"]) + int(row["Gm2"])
        is_tb = bool(row.get("is_tiebreak_game", False))

        # Tiebreak start: transition into a tiebreak game
        if is_tb and not prev_is_tb:
            events.append(MatchEvent(pidx, "tiebreak_start", "Tiebreak begins"))
        # Tiebreak end: transition out of a tiebreak game (games_total changed while prev was tb)
        if prev_is_tb and not is_tb and prev_games_total is not None and games_total != prev_games_total:
            events.append(MatchEvent(pidx - 1, "tiebreak_end", "Tiebreak ends"))

        # Break of serve: the game just concluded (games_total incremented) and the player
        # who was serving that game (Svr on the last point of the game, i.e. the PRECEDING
        # row) did NOT win it — determined directly from which of Gm1/Gm2 incremented vs.
        # who was serving, no inference needed.
        if prev_games_total is not None and games_total > prev_games_total and not is_tb:
            prev_row = df.iloc[i - 1]
            server_was_p1 = (prev_row["Svr"] == 1)
            p1_won_game = int(row["Gm1"]) > int(prev_row["Gm1"])
            is_break = (server_was_p1 and not p1_won_game) or (not server_was_p1 and p1_won_game)
            if is_break:
                winner_is_p1 = p1_won_game
                events.append(MatchEvent(
                    pidx - 1, "break",
                    f"Break — {'Player 1' if winner_is_p1 else 'Player 2'} breaks serve",
                    winner_is_p1=winner_is_p1,
                ))

        if bool(row.get("is_match_point", False)):
            sets_needed = (int(row.get("best_of", 3)) // 2) + 1
            p1_sets, p2_sets = int(row["Set1"]), int(row["Set2"])
            is_champ = (p1_sets == sets_needed - 1) or (p2_sets == sets_needed - 1)
            kind = "championship_point" if is_champ else "match_point"
            label = "Championship point" if is_champ else "Match point"
            if not events or events[-1].kind != kind or events[-1].point_index != pidx:
                events.append(MatchEvent(pidx, kind, label))

        prev_games_total = games_total
        prev_is_tb = is_tb

    return events