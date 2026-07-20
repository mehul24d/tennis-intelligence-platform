"""
build_player_registry.py — pipeline entrypoint for the player identity layer (Week 2, Day 3).

Runs:
    1. Build the canonical registry from TML-Database (players.parquet)
    2. Resolve every unique name appearing in the frozen TML<->MCP join against that
       registry (join-derived aliases first, then fuzzy fallback)
    3. Print the validation report
    4. Write matches_with_player_ids.parquet (TML matches, which already carry winner_id/
       loser_id natively — this step mainly re-exports them alongside the registry for
       downstream convenience, and verifies every id referenced actually exists in the
       registry)

Usage (from project root, with .venv activated):
    python pipelines/build_player_registry.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tennis_intel.data.join_tml_mcp import load_tml_matches
from tennis_intel.entities.player_registry import build_registry, write_registry
from tennis_intel.entities.player_aliases import build_join_derived_aliases, resolve_names
from tennis_intel.entities.validate_players import build_player_validation_report, check_registry_integrity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TML_DIR = PROJECT_ROOT / "data" / "raw" / "TML-Database"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
JOINED_MATCHES_PATH = PROCESSED_DIR / "joined_matches_m.parquet"


def main() -> None:
    logger.info("=== Step 1: Building canonical player registry from TML-Database ===")
    registry = build_registry(TML_DIR)

    problems = check_registry_integrity(registry)
    if problems:
        print("\n⚠️  Registry integrity issues found:")
        for p in problems:
            print(f"  - {p}")
        print()
    else:
        print("\n✅ Registry integrity checks passed (no duplicate IDs, no missing names/keys)\n")

    write_registry(registry, PROCESSED_DIR / "players.parquet")

    logger.info("=== Step 2: Resolving MCP names against the registry ===")
    if not JOINED_MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"{JOINED_MATCHES_PATH} not found — run pipelines/build_joined_dataset.py first "
            "(the join pipeline is frozen as of docs/join_pipeline_v1_freeze.md, but its "
            "output is still a required input here)."
        )
    joined = pd.read_parquet(JOINED_MATCHES_PATH)

    join_derived = build_join_derived_aliases(joined)
    logger.info("Extracted %d name->id aliases directly from the frozen join", len(join_derived))

    # Every unique MCP player name string across all charted matches, not just the joined
    # subset — this is the real test of the fallback resolver, since names outside the
    # joined 79.1% have no ground-truth alias available.
    from tennis_intel.data.join_tml_mcp import load_mcp_matches
    mcp_all = load_mcp_matches(PROJECT_ROOT / "data" / "raw" / "tennis_MatchChartingProject", gender="m")
    unique_names = pd.unique(pd.concat([mcp_all["Player 1"], mcp_all["Player 2"]]).dropna())

    resolution = resolve_names(list(unique_names), registry, join_derived_aliases=join_derived)

    report = build_player_validation_report(registry, resolution)
    print(report.render())

    # Step 3: write matches_with_player_ids.parquet — TML matches already carry native IDs;
    # this step verifies referential integrity against the registry and re-exports for
    # downstream convenience (feature pipelines should read this file, not raw TML CSVs).
    logger.info("=== Step 3: Verifying and writing matches_with_player_ids.parquet ===")
    tml_matches = load_tml_matches(TML_DIR)
    registry_ids = set(registry["player_id"])

    winner_has_id = tml_matches["winner_id"].notna()
    loser_has_id = tml_matches["loser_id"].notna()
    orphan_winners = winner_has_id & ~tml_matches["winner_id"].isin(registry_ids)
    orphan_losers = loser_has_id & ~tml_matches["loser_id"].isin(registry_ids)
    n_missing_ids = (~winner_has_id).sum() + (~loser_has_id).sum()
    n_true_orphans = (orphan_winners | orphan_losers).sum()

    if n_missing_ids:
        print(f"\nℹ️  {n_missing_ids} winner_id/loser_id value(s) are NaN in the source data "
              "(not a registry problem — these rows never had an id to begin with).")

    if n_true_orphans:
        print(f"\n⚠️  {n_true_orphans} match(es) reference a NON-NULL player_id not found in "
              "the registry — this WOULD be a real integrity bug. Offending row(s):")
        cols = [c for c in ["tourney_id", "tourney_name", "round", "winner_id", "winner_name",
                             "loser_id", "loser_name"] if c in tml_matches.columns]
        print(tml_matches.loc[orphan_winners | orphan_losers, cols].to_string(index=False))
    else:
        print(f"\n✅ Referential integrity confirmed: every NON-NULL winner_id/loser_id in "
              f"{len(tml_matches):,} TML matches exists in the registry.")

    output_path = PROCESSED_DIR / "matches_with_player_ids.parquet"
    tml_matches.to_parquet(output_path, index=False)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()