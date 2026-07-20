"""
Canonicalization utilities for joining TML-Database (match-level) against
tennis_MatchChartingProject / MCP (point-level) data.

These two datasets are maintained independently and have no shared match ID, so joining
them requires normalizing free-text fields (player names, tournament names, round labels)
to a common form before matching.

Design principle: every normalization function is pure (no I/O, no globals mutated) and
unit-testable in isolation. See tests/unit/test_canonical_player_names.py.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_player_name(raw_name: str | float | None) -> str:
    """
    Normalize a player name for matching across datasets.

    Handles: accented characters (Federer vs Fédérer-style variants), case, punctuation,
    extra whitespace, and common suffix noise ("Jr.", "III").

    Does NOT handle: nickname variants (e.g. "Alex" vs "Alexander") or full transpositions
    (e.g. maiden-name changes) — those require the alias table in KNOWN_ALIASES below.
    """
    if raw_name is None or (isinstance(raw_name, float)):  # NaN
        return ""

    name = str(raw_name).strip()

    # Strip accents: "Félix Auger-Aliassime" -> "Felix Auger-Aliassime"
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Lowercase, collapse whitespace
    name = name.lower()
    name = re.sub(r"\s+", " ", name).strip()

    # Remove periods (e.g. "J.J. Wolf" -> "jj wolf"), keep hyphens/apostrophes (real name chars)
    name = name.replace(".", "")

    # Strip common suffixes that appear inconsistently across sources
    name = re.sub(r"\b(jr|sr|iii|ii|iv)\b\.?$", "", name).strip()

    return name


# Manually curated aliases for players whose name representation differs structurally between
# TML-Database and MCP (nicknames, alternate romanizations, etc.) rather than just accents/case.
# Populate this incrementally as unmatched-pair review (Stage 4) surfaces real cases — do not
# pre-guess entries without evidence from the actual unmatched list.
KNOWN_ALIASES: dict[str, str] = {
    # normalized_variant: normalized_canonical_form
    # e.g. "stan wawrinka": "stanislas wawrinka"  — add only entries verified against real
    # unmatched-pair review in Stage 4, never speculative guesses.
}


def apply_alias(normalized_name: str) -> str:
    """Map a normalized name through the alias table if a mapping exists, else return unchanged."""
    return KNOWN_ALIASES.get(normalized_name, normalized_name)


def normalize_tournament_name(raw_name: str | float | None) -> str:
    """
    Normalize a tournament name for matching. Handles sponsor-name churn (tournaments are
    frequently renamed for sponsorship, e.g. "Rogers Cup" vs "Canadian Open") only for the
    common cases below — extend TOURNAMENT_ALIASES as real mismatches are found in Stage 4.
    """
    if raw_name is None or isinstance(raw_name, float):
        return ""

    name = str(raw_name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"\s+", " ", name).strip()

    # Strip common noise tokens
    for token in ["masters 1000", "atp", "wta", "presented by", "sponsored by"]:
        name = name.replace(token, "")
    name = re.sub(r"\s+", " ", name).strip()

    return TOURNAMENT_ALIASES.get(name, name)


TOURNAMENT_ALIASES: dict[str, str] = {
    # normalized_variant: normalized_canonical_form
    # populate from real Stage 4 unmatched review, e.g.:
    # "rogers cup": "canadian open",
}


ROUND_MAP: dict[str, str] = {
    # Canonical round codes. TML uses Sackmann-style short codes; MCP tends to use similar
    # short codes but VERIFY against real data in Stage 2 rather than trusting this blindly.
    "f": "F",
    "final": "F",
    "sf": "SF",
    "semifinal": "SF",
    "qf": "QF",
    "quarterfinal": "QF",
    "r16": "R16",
    "round of 16": "R16",
    "r32": "R32",
    "round of 32": "R32",
    "r64": "R64",
    "round of 64": "R64",
    "r128": "R128",
    "round of 128": "R128",
    "rr": "RR",
    "round robin": "RR",
}


def normalize_round(raw_round: str | float | None) -> str:
    if raw_round is None or isinstance(raw_round, float):
        return ""
    key = str(raw_round).strip().lower()
    return ROUND_MAP.get(key, key.upper())