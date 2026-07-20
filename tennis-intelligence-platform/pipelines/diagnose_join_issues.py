"""
diagnose_join_issues.py — investigates the two flagged issues from the join validation report:
  1. Duplicate joins: TML rows matched to more than one MCP row
  2. Ambiguous, unresolved: MCP rows with multiple TML candidates, unresolved even by date band

Run this AFTER build_joined_dataset.py. It re-runs the join (cheap enough at this data size)
and prints full detail on both categories so real patterns can be reviewed before deciding
whether they need a code fix or are an acceptable, documented margin.

Usage:
    python pipelines/diagnose_join_issues.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tennis_intel.data.join_tml_mcp import (
    load_tml_matches,
    load_mcp_matches,
    normalize_tml,
    normalize_mcp,
    deterministic_join,
    fallback_join,
    _nearest_by_date,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TML_DIR = PROJECT_ROOT / "data" / "raw" / "TML-Database"
MCP_DIR = PROJECT_ROOT / "data" / "raw" / "tennis_MatchChartingProject"


def diagnose_duplicates(joined: pd.DataFrame) -> None:
    print("=" * 70)
    print("DUPLICATE JOINS — TML rows matched to more than one MCP row")
    print("=" * 70)

    key_cols = [c for c in ["tml_tourney_id", "tml_match_num"] if c in joined.columns]
    if not key_cols:
        print("Could not identify TML key columns — skipping.")
        return

    counts = joined.groupby(key_cols).size()
    dup_keys = counts[counts > 1].index

    if len(dup_keys) == 0:
        print("None found.")
        return

    print(f"{len(dup_keys)} distinct TML matches were matched more than once.\n")

    display_cols = [
        c for c in [
            "tml_tourney_id", "tml_tourney_name", "tml_round", "tml_winner_name", "tml_loser_name",
            "mcp_match_id", "mcp_Player 1", "mcp_Player 2", "mcp_Round", "mcp_Date",
        ]
        if c in joined.columns
    ]

    for key in dup_keys[:15]:  # cap output for readability
        if len(key_cols) == 1:
            mask = joined[key_cols[0]] == key
        else:
            mask = (joined[key_cols] == pd.Series(key, index=key_cols)).all(axis=1)
        subset = joined.loc[mask, display_cols]
        print(f"--- TML key {key} ({mask.sum()} MCP matches) ---")
        print(subset.to_string(index=False))
        print()


def diagnose_ambiguous(tml_norm: pd.DataFrame, mcp_norm: pd.DataFrame) -> None:
    print("=" * 70)
    print("AMBIGUOUS, UNRESOLVED — MCP rows with multiple TML candidates, unresolved")
    print("even by nearest-date disambiguation (exact-tie distance)")
    print("=" * 70)

    tml_index: dict[tuple, list[int]] = {}
    for idx, row in tml_norm.iterrows():
        key = (row["tourney_name_norm"], row["round_norm"], row["player_pair"])
        tml_index.setdefault(key, []).append(idx)

    shown = 0
    for _, mrow in mcp_norm.iterrows():
        key = (mrow["tournament_norm"], mrow["round_norm"], mrow["player_pair"])
        candidates = tml_index.get(key, [])
        if len(candidates) <= 1:
            continue
        chosen, reason = _nearest_by_date(candidates, tml_norm, mrow["Date"])
        if chosen is not None:
            continue  # this one WAS resolved, not ambiguous

        shown += 1
        print(f"\n--- MCP: {mrow['match_id']} (unresolved: {reason}) ---")
        print(f"  {mrow['Player 1']} vs {mrow['Player 2']}, {mrow['Tournament']}, "
              f"{mrow['Round']}, played {mrow['Date'].date() if pd.notna(mrow['Date']) else 'NaT'}")
        print(f"  {len(candidates)} TML candidates (tourney_id, date, winner, loser):")
        for c in candidates:
            r = tml_norm.loc[c]
            print(f"    {r['tourney_id']}, {r['tourney_date'].date() if pd.notna(r['tourney_date']) else 'NaT'}, "
                  f"{r['winner_name']} beat {r['loser_name']}")
        if shown >= 15:
            print("\n... (capped at 15 for readability)")
            break

    if shown == 0:
        print("None found (all resolved by nearest-date disambiguation).")


def main() -> None:
    print("Loading and normalizing data (this re-runs the join, ~10s)...\n")
    tml_raw = load_tml_matches(TML_DIR)
    mcp_raw = load_mcp_matches(MCP_DIR, gender="m")
    tml_norm = normalize_tml(tml_raw)
    mcp_norm = normalize_mcp(mcp_raw)

    stage3 = deterministic_join(tml_norm, mcp_norm)
    consumed = {e.tml_row_index for e in stage3.log if e.tml_row_index is not None}
    stage4 = fallback_join(tml_norm, stage3.unmatched_mcp, stage3.log, consumed_tml=consumed)
    joined = pd.concat([stage3.joined, stage4.joined], ignore_index=True)

    diagnose_duplicates(joined)
    print()
    diagnose_ambiguous(tml_norm, mcp_norm)


if __name__ == "__main__":
    main()