"""
join_validation.py — Stage 5 of the TML-Database <-> MCP join pipeline.

Produces the automated validation report: coverage, duplicate joins, ambiguous joins,
unmatched counts, and a breakdown by matching strategy. Deliberately separate from
join_tml_mcp.py so the report can be regenerated from a saved JoinResult / log without
re-running the (more expensive) join itself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from tennis_intel.data.join_tml_mcp import JoinLogEntry, JoinResult


@dataclass
class ValidationReport:
    tml_total: int
    mcp_total: int
    joined_total: int
    coverage_pct: float
    duplicate_joins: int
    ambiguous_unresolved: int
    unmatched_total: int
    strategy_breakdown: dict[str, int]

    def render(self) -> str:
        lines = [
            "=== TML <-> MCP Join Validation Report ===",
            f"TML matches (total pool):       {self.tml_total:,}",
            f"MCP matches (total pool):       {self.mcp_total:,}",
            f"Successfully joined:            {self.joined_total:,}",
            f"Coverage (joined / MCP total):  {self.coverage_pct:.1f}%",
            f"Duplicate joins (same TML row matched >1x): {self.duplicate_joins:,}",
            f"Ambiguous, unresolved:          {self.ambiguous_unresolved:,}",
            f"Unmatched:                      {self.unmatched_total:,}",
            "",
            "Breakdown by matching strategy:",
        ]
        for strategy, count in sorted(self.strategy_breakdown.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {strategy:35s} {count:,}")
        return "\n".join(lines)


def build_validation_report(
    tml_total: int, mcp_total: int, result: JoinResult
) -> ValidationReport:
    strategy_counts = Counter(entry.strategy for entry in result.log)

    joined_total = len(result.joined)
    unmatched_total = len(result.unmatched_mcp)
    ambiguous = strategy_counts.get("ambiguous_unresolved", 0)

    duplicate_joins = _count_duplicate_tml_matches(result)

    coverage_pct = (joined_total / mcp_total * 100) if mcp_total else 0.0

    return ValidationReport(
        tml_total=tml_total,
        mcp_total=mcp_total,
        joined_total=joined_total,
        coverage_pct=coverage_pct,
        duplicate_joins=duplicate_joins,
        ambiguous_unresolved=ambiguous,
        unmatched_total=unmatched_total,
        strategy_breakdown=dict(strategy_counts),
    )


def _count_duplicate_tml_matches(result: JoinResult) -> int:
    """Counts how many TML rows got matched to more than one MCP row — a red flag indicating
    either a genuine data duplicate or a join-key collision that needs investigation, not a
    result to silently accept."""
    if result.joined.empty or "tml_tourney_id" not in result.joined.columns:
        return 0
    # tourney_id + match_num should uniquely identify a TML match
    key_cols = [c for c in ["tml_tourney_id", "tml_match_num"] if c in result.joined.columns]
    if not key_cols:
        return 0
    counts = result.joined.groupby(key_cols).size()
    return int((counts > 1).sum())


def sample_unmatched(result: JoinResult, n: int = 20) -> pd.DataFrame:
    """Returns a sample of unmatched MCP rows for manual review — this is the primary input
    for populating KNOWN_ALIASES / TOURNAMENT_ALIASES in canonical_player_names.py. Review
    this manually; do not auto-generate alias entries without eyeballing the actual mismatch."""
    cols = [c for c in ["match_id", "Player 1", "Player 2", "Tournament", "Round", "Date"]
            if c in result.unmatched_mcp.columns]
    return result.unmatched_mcp[cols].head(n)