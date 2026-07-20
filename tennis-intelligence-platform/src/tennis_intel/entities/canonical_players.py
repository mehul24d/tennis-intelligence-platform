"""
canonical_players.py — name normalization and matching-key generation for player identity
resolution.

Extends tennis_intel.data.canonical_player_names (which handles accents/case/punctuation)
with two additional capabilities needed for player identity resolution specifically:
  1. Comma-format reordering: "Nadal, Rafael" -> "Rafael Nadal"
  2. Initials-key generation: "R. Nadal" and "Rafael Nadal" both produce "r nadal", so a
     name seen only as an abbreviation can still be matched against a full-name registry
     entry when the (first-initial, last-name) combination is unique in that registry.

Initials matching is inherently lossy — "R. Nadal" could theoretically be "Rafael Nadal" or
"Roger Nadal" if both existed. This module surfaces ambiguity explicitly (returns None /
multiple candidates) rather than guessing; see player_aliases.py for how ambiguous initials
matches are handled (logged as unresolved, never silently picked).
"""

from __future__ import annotations

import re

from tennis_intel.data.canonical_player_names import normalize_player_name


def reorder_comma_name(raw_name: str | float | None) -> str:
    """
    Detects "Last, First" format and reorders to "First Last". Returns the input unchanged
    (as a string) if no comma is present. Does NOT normalize otherwise — call
    normalize_player_name on the result separately.
    """
    if raw_name is None or isinstance(raw_name, float):
        return ""
    raw = str(raw_name).strip()
    if "," not in raw:
        return raw
    parts = [p.strip() for p in raw.split(",", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return raw
    last, first = parts
    return f"{first} {last}"


def full_name_key(raw_name: str | float | None) -> str:
    """The primary matching key: comma-reordered, then fully normalized. This is what most
    names will match on directly."""
    return normalize_player_name(reorder_comma_name(raw_name))


def initials_key(normalized_full_name: str) -> str | None:
    """
    Given an already-normalized full name (e.g. "rafael nadal"), returns a
    (first-initial, last-name) key (e.g. "r nadal") for matching against abbreviated forms
    like "R. Nadal" or "Nadal R.". Returns None if the name doesn't have at least two tokens
    (can't safely derive an initial + surname).
    """
    tokens = normalized_full_name.split()
    if len(tokens) < 2:
        return None
    first_initial = tokens[0][0]
    surname = tokens[-1]
    return f"{first_initial} {surname}"


def loose_key(normalized_full_name: str) -> str:
    """
    A further-relaxed key that bridges hyphen/apostrophe variants:
      - hyphens are replaced with a SPACE ("auger-aliassime" -> "auger aliassime"), since a
        hyphenated compound surname is semantically two space-separated tokens in variant
        spellings ("Felix Auger Aliassime")
      - apostrophes are REMOVED entirely, not spaced ("o'connell" -> "oconnell"), since that
        is how the no-apostrophe variant actually appears in practice ("Oconnell", not
        "O Connell")
    Applied as a fallback tier AFTER exact full_name_key match fails — it's slightly lossier
    (could in principle merge two genuinely different compound surnames that only differ by
    a hyphen/apostrophe), so it's not the primary key.
    """
    result = normalized_full_name.replace("-", " ").replace("'", "")
    return re.sub(r"\s+", " ", result).strip()


def query_keys(raw_name: str | float | None) -> dict[str, str | None]:
    """
    Produces every candidate matching key for a raw input name, used both when building the
    registry (indexing) and when resolving an unknown name against it (lookup). Returns a
    dict so callers can try keys in priority order (full name first, then loose/punctuation-
    stripped, then initials as last resort) and log which strategy succeeded.
    """
    full = full_name_key(raw_name)
    return {
        "full_name": full,
        "loose": loose_key(full) if full else None,
        "initials": initials_key(full) if full else None,
    }