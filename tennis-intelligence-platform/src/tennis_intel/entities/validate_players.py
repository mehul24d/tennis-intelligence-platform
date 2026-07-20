"""
validate_players.py — diagnostic reporting for the player identity layer.

Prints the summary format the project owner specified:
    Unique player strings: ...
    Canonical players: ...
    Merged aliases: ...
    Unresolved: ...

Plus a strategy breakdown (join_derived / full_name_match / initials_match / unresolved)
so any spike in fuzzy-matched (as opposed to ground-truth join-derived) resolutions is
visible before it silently propagates into Elo/rolling-stats features.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from tennis_intel.entities.player_aliases import AliasResolutionResult


@dataclass
class PlayerValidationReport:
    unique_player_strings: int
    canonical_players: int
    merged_aliases: int
    unresolved: int
    strategy_breakdown: dict[str, int]
    unresolved_sample: list[str]

    def render(self) -> str:
        lines = [
            "=== Player Identity Resolution Report ===",
            f"Unique player strings:  {self.unique_player_strings:,}",
            f"Canonical players:      {self.canonical_players:,}",
            f"Merged aliases:         {self.merged_aliases:,}",
            f"Unresolved:             {self.unresolved:,}",
            "",
            "Breakdown by resolution strategy:",
        ]
        for strategy, count in sorted(self.strategy_breakdown.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {strategy:20s} {count:,}")
        if self.unresolved_sample:
            lines.append("")
            lines.append(f"Sample of unresolved names (up to 20 of {self.unresolved}):")
            for name in self.unresolved_sample:
                lines.append(f"  {name}")
        return "\n".join(lines)


def build_player_validation_report(
    registry: pd.DataFrame, resolution: AliasResolutionResult
) -> PlayerValidationReport:
    strategy_counts = Counter(r.strategy for r in resolution.resolutions)

    # "merged aliases" = names that resolved to a player_id but are NOT that player's own
    # canonical_name (i.e. an actual alias merge happened, not a trivial self-match)
    canonical_names = set(registry["canonical_name"])
    merged = sum(
        1 for r in resolution.resolutions
        if r.player_id is not None and r.raw_name not in canonical_names
    )

    unresolved = resolution.unresolved_names()

    return PlayerValidationReport(
        unique_player_strings=len(resolution.resolutions),
        canonical_players=len(registry),
        merged_aliases=merged,
        unresolved=len(unresolved),
        strategy_breakdown=dict(strategy_counts),
        unresolved_sample=unresolved[:20],
    )


def check_registry_integrity(registry: pd.DataFrame) -> list[str]:
    """Basic sanity checks on the registry itself — run before trusting it downstream."""
    problems = []

    if registry["player_id"].duplicated().any():
        n_dup = registry["player_id"].duplicated().sum()
        problems.append(f"{n_dup} duplicate player_id(s) in registry — should be impossible, investigate")

    if registry["canonical_name"].isna().any():
        problems.append(f"{registry['canonical_name'].isna().sum()} players with no canonical_name")

    empty_keys = (registry["canonical_name_key"] == "").sum()
    if empty_keys:
        problems.append(f"{empty_keys} players with an empty canonical_name_key (normalization failure?)")

    return problems