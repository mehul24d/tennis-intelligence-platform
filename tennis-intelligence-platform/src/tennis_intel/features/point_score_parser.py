"""
point_score_parser.py — parses MCP's point-by-point score notation into structured state.

Grounded in direct inspection of real data (2026-07-03), not assumed:
  - `Pt` is a clean, gapless 1..N counter per match — safe to sort on directly.
  - `Pts` is standard tennis point notation: "0-15", "40-40", "AD-40", or (in a tiebreak)
    plain digit pairs like "3-2".
  - `TbSet` does NOT mean "this point is in a tiebreak" — confirmed True for an entire
    match including ordinary non-tiebreak games late in the match. Do not use it as a
    per-point tiebreak flag. Tiebreak status is derived instead from Gm1==6 and Gm2==6
    (entering a game at 6-games-all in the current set) — unambiguous, doesn't depend on
    a column that's demonstrably not doing what its name suggests.
  - `Set1`/`Set2` = completed sets won by player 1/2 so far (not the in-progress set).
  - `Gm1`/`Gm2` = games won by player 1/2 within the CURRENT set.
  - `Svr` = 1 or 2, matching the same player-slot convention as `Player 1`/`Player 2` in
    charting-m-matches.csv (Sackmann's standard convention across his datasets).
  - `1st`/`2nd` are dense shot-charting notation (serve direction, shot type, error type) —
    deliberately NOT parsed in v1 (see live_win_probability_extension_analysis.md's scoping
    note). The only signal extracted from them is whether the point reached a second serve
    (`2nd` non-null), which requires no notation decoding at all.
  - `PtWinner` (1 or 2) is already a clean target — no derivation needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REGULAR_POINT_VALUES = {"0", "15", "30", "40", "AD"}


@dataclass
class ParsedPointScore:
    p1_points: int | None = None   # ordinal point count in a REGULAR game (0,1,2,3=40,4=AD)
    p2_points: int | None = None
    is_tiebreak_score: bool = False
    tb_p1_points: int | None = None  # raw tiebreak point count, if applicable
    tb_p2_points: int | None = None
    parse_ok: bool = False


_REGULAR_ORDINAL = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def parse_pts(raw_pts: str, is_tiebreak_game: bool) -> ParsedPointScore:
    """
    Parses the `Pts` column. `is_tiebreak_game` must be determined by the caller from
    Gm1==6 and Gm2==6 (see module docstring) — the score string alone is ambiguous between
    a very early regular game and a tiebreak in some edge cases, so tiebreak context is
    passed in rather than guessed from the string.
    """
    if raw_pts is None or not isinstance(raw_pts, str):
        return ParsedPointScore(parse_ok=False)

    parts = raw_pts.strip().split("-")
    if len(parts) != 2:
        return ParsedPointScore(parse_ok=False)

    p1_raw, p2_raw = parts

    if is_tiebreak_game:
        try:
            return ParsedPointScore(
                is_tiebreak_score=True,
                tb_p1_points=int(p1_raw), tb_p2_points=int(p2_raw),
                parse_ok=True,
            )
        except ValueError:
            return ParsedPointScore(parse_ok=False)

    if p1_raw in REGULAR_POINT_VALUES and p2_raw in REGULAR_POINT_VALUES:
        return ParsedPointScore(
            p1_points=_REGULAR_ORDINAL[p1_raw], p2_points=_REGULAR_ORDINAL[p2_raw],
            parse_ok=True,
        )

    return ParsedPointScore(parse_ok=False)


def is_break_point(server_is_p1: bool, p1_points: int, p2_points: int,
                    is_tiebreak: bool, tb_p1: int | None = None, tb_p2: int | None = None) -> bool:
    """A break point exists when the RETURNER is one point away from winning the game."""
    if is_tiebreak:
        # In a tiebreak, "break point" isn't the standard concept (serve alternates every
        # point/every-2-points) — treated as False here; tiebreak-specific pressure points
        # are a distinct, deferred concept (see module docstring scoping).
        return False
    returner_points = p2_points if server_is_p1 else p1_points
    server_points = p1_points if server_is_p1 else p2_points
    if returner_points == 4:  # returner at Advantage
        return True
    if returner_points == 3 and server_points < 3:  # returner at 40, server below 40
        return True
    return False


def would_win_game_next_point(player_is_p1: bool, p1_points: int, p2_points: int,
                               is_tiebreak: bool, tb_p1: int | None = None,
                               tb_p2: int | None = None, tb_target: int = 7) -> bool:
    """True if the given player would win the CURRENT GAME by winning the next point."""
    if is_tiebreak:
        player_pts = tb_p1 if player_is_p1 else tb_p2
        opp_pts = tb_p2 if player_is_p1 else tb_p1
        next_player_pts = player_pts + 1
        return next_player_pts >= tb_target and (next_player_pts - opp_pts) >= 2

    player_pts = p1_points if player_is_p1 else p2_points
    opp_pts = p2_points if player_is_p1 else p1_points
    if player_pts == 4:  # already at Advantage
        return True
    if player_pts == 3 and opp_pts < 3:  # at 40, opponent below 40
        return True
    return False


def would_win_set_by_winning_this_game(
    player_is_p1: bool, p1_games: int, p2_games: int,
) -> bool:
    """
    True if the given player winning the CURRENT GAME would also win the current set,
    given their games-won count BEFORE this game. Standard ad-scoring set rules: first to
    6 games with a 2-game lead, or winning a tiebreak at 6-6 (which brings the game count
    to 7, decisively). Does not attempt to handle non-standard formats (no-ad sets, match
    tiebreaks in lieu of a final set, short sets) — those are out of scope for v1, and this
    function should not be trusted for tournaments known to use them without verification.
    """
    player_games_after = (p1_games if player_is_p1 else p2_games) + 1
    opp_games = p2_games if player_is_p1 else p1_games

    if player_games_after == 7 and opp_games == 6:
        return True  # won the 6-6 tiebreak game, set goes 7-6
    if player_games_after >= 6 and (player_games_after - opp_games) >= 2:
        return True
    return False


def is_set_point(server_is_p1: bool, p1_points: int, p2_points: int,
                  is_tiebreak: bool, p1_games: int, p2_games: int,
                  tb_p1: int | None = None, tb_p2: int | None = None) -> bool:
    """True if EITHER player winning the current point would win both the game and the set."""
    for player_is_p1 in (True, False):
        if would_win_game_next_point(player_is_p1, p1_points, p2_points, is_tiebreak, tb_p1, tb_p2) \
           and would_win_set_by_winning_this_game(player_is_p1, p1_games, p2_games):
            return True
    return False


def is_match_point(server_is_p1: bool, p1_points: int, p2_points: int,
                    is_tiebreak: bool, p1_games: int, p2_games: int,
                    p1_sets: int, p2_sets: int, best_of: int,
                    tb_p1: int | None = None, tb_p2: int | None = None) -> bool:
    """True if EITHER player winning the current point would win the game, the set, AND
    the match (i.e. reach the required number of sets for the given best_of format)."""
    sets_needed = (best_of // 2) + 1
    for player_is_p1 in (True, False):
        wins_game = would_win_game_next_point(player_is_p1, p1_points, p2_points, is_tiebreak, tb_p1, tb_p2)
        wins_set = would_win_set_by_winning_this_game(player_is_p1, p1_games, p2_games)
        if wins_game and wins_set:
            player_sets_after = (p1_sets if player_is_p1 else p2_sets) + 1
            if player_sets_after >= sets_needed:
                return True
    return False