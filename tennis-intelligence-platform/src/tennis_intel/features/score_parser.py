"""
score_parser.py — parses TML-Database's `score` field into structured game/set statistics.

Score format (Sackmann/TML convention): space-separated sets, each "winnerGames-loserGames",
optionally with a tiebreak score in parentheses, e.g. "6-4 7-6(4)". Retirements/defaults are
suffixed ("6-2 3-1 RET"), walkovers have no real score ("W/O").

This module is defensive by design: malformed or unusual score strings (there WILL be some,
across 58 years of historical data) do not crash the pipeline — they're logged and returned
with parse_ok=False and NaN numeric fields, so downstream code can filter them explicitly
rather than silently getting corrupted numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WALKOVER_TOKENS = {"W/O", "WO", "WALKOVER", ""}
RETIREMENT_SUFFIXES = {"RET", "RETIRED", "DEF", "DEFAULT", "ABD", "ABANDONED"}


@dataclass
class ParsedScore:
    sets_won: int | None = None       # from the match WINNER's perspective
    sets_lost: int | None = None
    games_won: int | None = None
    games_lost: int | None = None
    n_sets_played: int | None = None
    straight_sets: bool | None = None  # winner lost zero sets
    retired: bool = False
    walkover: bool = False
    parse_ok: bool = False


def _parse_set_token(token: str) -> tuple[int, int] | None:
    """Parses a single set token like '6-4' or '7-6(4)' into (winner_games, loser_games).
    Returns None if the token doesn't match the expected pattern (defensive, not a crash)."""
    token = token.strip()
    # Strip tiebreak parenthetical, e.g. "7-6(4)" -> "7-6"
    token = re.sub(r"\(\d+\)", "", token)
    match = re.match(r"^(\d+)-(\d+)$", token)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _is_complete_set(winner_games: int, loser_games: int) -> bool:
    """A set is complete if it reaches a valid finished-set score: 6+ games with a 2-game
    margin, or a 7-6/7-5 tiebreak-adjacent finish. An incomplete/abandoned set (e.g. "3-1"
    when a player retires mid-set) will NOT match any of these patterns."""
    hi, lo = max(winner_games, loser_games), min(winner_games, loser_games)
    if hi >= 6 and (hi - lo) >= 2:
        return True
    if hi == 7 and lo in (5, 6):
        return True
    return False


def parse_score(raw_score: str | float | None) -> ParsedScore:
    """Parses a raw TML score string into a ParsedScore. Never raises — malformed input
    produces a ParsedScore with parse_ok=False rather than an exception, since a single bad
    row must not crash processing of 198k matches."""
    if raw_score is None or (isinstance(raw_score, float)):  # NaN
        return ParsedScore(walkover=True, parse_ok=False)

    raw = str(raw_score).strip()
    if raw.upper() in WALKOVER_TOKENS:
        return ParsedScore(walkover=True, parse_ok=False)

    tokens = raw.split()
    retired = False
    set_tokens = []
    for t in tokens:
        cleaned = t.strip().upper().rstrip(".")
        if cleaned in RETIREMENT_SUFFIXES:
            retired = True
            continue
        set_tokens.append(t)

    if not set_tokens:
        return ParsedScore(retired=retired, walkover=(not retired), parse_ok=False)

    parsed_sets = [_parse_set_token(t) for t in set_tokens]
    if any(p is None for p in parsed_sets):
        # At least one set token didn't parse — don't guess at partial data, flag it
        return ParsedScore(retired=retired, parse_ok=False)

    if retired and parsed_sets:
        # Two real conventions exist in this data: some retirement scores include the
        # abandoned set's partial score as the last token (e.g. "6-2 3-1 RET" — "3-1" was
        # never finished); others just list completed sets with no partial trailing score
        # (e.g. "6-2 6-3 RET" — retirement happened cleanly at the set boundary). We can't
        # tell which convention applies from position alone, so check validity directly:
        # drop the last token ONLY if it does not represent a valid completed-set score.
        last_w, last_l = parsed_sets[-1]
        if not _is_complete_set(last_w, last_l):
            parsed_sets = parsed_sets[:-1]

    if not parsed_sets:
        # Retired during the very first set — no completed sets exist at all
        return ParsedScore(
            sets_won=0, sets_lost=0, games_won=0, games_lost=0, n_sets_played=0,
            straight_sets=None, retired=retired, walkover=False, parse_ok=True,
        )

    games_won = sum(p[0] for p in parsed_sets)
    games_lost = sum(p[1] for p in parsed_sets)
    sets_won = sum(1 for p in parsed_sets if p[0] > p[1])
    sets_lost = sum(1 for p in parsed_sets if p[0] < p[1])

    return ParsedScore(
        sets_won=sets_won,
        sets_lost=sets_lost,
        games_won=games_won,
        games_lost=games_lost,
        n_sets_played=len(parsed_sets),
        straight_sets=(sets_lost == 0),
        retired=retired,
        walkover=False,
        parse_ok=True,
    )