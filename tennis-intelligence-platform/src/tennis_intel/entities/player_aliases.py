"""
player_aliases.py — resolves arbitrary name strings (primarily from MCP, which has no
native player ID) to canonical player_ids from the TML-derived registry.

Two resolution paths, in priority order:

  1. Join-derived aliases (free, ground-truth): for every MCP row already matched to a TML
     row by the frozen join pipeline (join_pipeline_v1), we know the corresponding
     winner_id/loser_id directly — no fuzzy matching needed. This resolves the large
     majority of MCP names for free, since 79.1% of MCP matches already joined.

  2. Registry lookup for everything else: full-name key match first, then initials-key
     match ONLY if the initials key maps to a single unambiguous registry entry. Multiple
     registry entries sharing an initials key (e.g. two different "R. Nadal"-shaped players)
     are left unresolved rather than guessed — see resolve_name().

Every resolution and non-resolution is logged via AliasResolutionLog for the validation
report in validate_players.py — nothing is silently dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tennis_intel.entities.canonical_players import query_keys

logger = logging.getLogger(__name__)


@dataclass
class AliasResolution:
    raw_name: str
    player_id: str | None
    strategy: str  # "join_derived", "full_name_match", "initials_match", "unresolved"
    detail: str = ""


@dataclass
class AliasResolutionResult:
    resolutions: list[AliasResolution] = field(default_factory=list)
    name_to_id: dict[str, str] = field(default_factory=dict)  # raw_name -> player_id, resolved only

    def unresolved_names(self) -> list[str]:
        return [r.raw_name for r in self.resolutions if r.player_id is None]


def build_join_derived_aliases(joined_matches: pd.DataFrame) -> dict[str, str]:
    """
    Extracts free, ground-truth name->id mappings from the already-joined TML<->MCP dataset.
    For each joined row, MCP's Player 1/Player 2 raw strings are matched (by normalized name)
    against TML's winner_name/loser_name to determine which TML id each corresponds to.
    """
    aliases: dict[str, str] = {}

    required_cols = {
        "mcp_Player 1", "mcp_Player 2", "mcp_player1_norm", "mcp_player2_norm",
        "tml_winner_name_norm", "tml_loser_name_norm", "tml_winner_id", "tml_loser_id",
        "mcp_Player 1", "mcp_Player 2",
    }
    missing = required_cols - set(joined_matches.columns)
    if missing:
        raise ValueError(f"joined_matches is missing expected columns: {missing}")

    for _, row in joined_matches.iterrows():
        pairs = [
            (row["mcp_Player 1"], row["mcp_player1_norm"]),
            (row["mcp_Player 2"], row["mcp_player2_norm"]),
        ]
        for raw, norm in pairs:
            if norm == row["tml_winner_name_norm"]:
                aliases[raw] = row["tml_winner_id"]
            elif norm == row["tml_loser_name_norm"]:
                aliases[raw] = row["tml_loser_id"]
            # if neither matches (shouldn't happen given how the join was built), leave
            # unresolved here — it'll fall through to registry lookup in resolve_names()

    return aliases


def resolve_name(raw_name: str, registry: pd.DataFrame, full_name_index: dict[str, str],
                  loose_index: dict[str, list[str]],
                  initials_index: dict[str, list[str]]) -> AliasResolution:
    """Resolves a single raw name against the registry, trying three tiers in order:
    exact full-name match, then loose (hyphen/apostrophe-stripped) match, then initials
    match. The last two are only accepted if unambiguous (single candidate) — never guessed."""
    keys = query_keys(raw_name)

    full_id = full_name_index.get(keys["full_name"])
    if full_id is not None:
        return AliasResolution(raw_name, full_id, "full_name_match")

    if keys["loose"] is not None:
        candidates = loose_index.get(keys["loose"], [])
        if len(candidates) == 1:
            return AliasResolution(
                raw_name, candidates[0], "loose_match",
                detail=f"unambiguous loose key '{keys['loose']}' (hyphen/apostrophe variant)",
            )
        if len(candidates) > 1:
            return AliasResolution(
                raw_name, None, "unresolved",
                detail=f"loose key '{keys['loose']}' matches {len(candidates)} players, ambiguous",
            )

    if keys["initials"] is not None:
        candidates = initials_index.get(keys["initials"], [])
        if len(candidates) == 1:
            return AliasResolution(
                raw_name, candidates[0], "initials_match",
                detail=f"unambiguous initials key '{keys['initials']}'",
            )
        if len(candidates) > 1:
            return AliasResolution(
                raw_name, None, "unresolved",
                detail=f"initials key '{keys['initials']}' matches {len(candidates)} players, ambiguous",
            )

    return AliasResolution(raw_name, None, "unresolved", detail="no full-name, loose, or initials match")


def resolve_names(
    raw_names: list[str],
    registry: pd.DataFrame,
    join_derived_aliases: dict[str, str] | None = None,
) -> AliasResolutionResult:
    """Resolves a list of raw name strings against the registry, preferring join-derived
    aliases (free, ground-truth) before falling back to fuzzy registry lookup."""
    join_derived_aliases = join_derived_aliases or {}

    full_name_index: dict[str, str] = dict(
        zip(registry["canonical_name_key"], registry["player_id"])
    )
    loose_index: dict[str, list[str]] = {}
    initials_index: dict[str, list[str]] = {}
    for _, row in registry.iterrows():
        keys = query_keys(row["canonical_name"])
        if keys["loose"] is not None:
            loose_index.setdefault(keys["loose"], []).append(row["player_id"])
        if keys["initials"] is not None:
            initials_index.setdefault(keys["initials"], []).append(row["player_id"])

    result = AliasResolutionResult()
    for raw_name in raw_names:
        if raw_name in join_derived_aliases:
            res = AliasResolution(raw_name, join_derived_aliases[raw_name], "join_derived")
        else:
            res = resolve_name(raw_name, registry, full_name_index, loose_index, initials_index)

        result.resolutions.append(res)
        if res.player_id is not None:
            result.name_to_id[raw_name] = res.player_id

    return result