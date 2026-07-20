"""
player_registry.py — builds the canonical player registry from TML-Database.

TML already ships alphanumeric player IDs (winner_id/loser_id in match files, id in
ATP_Database.csv) — we do not invent a new ID scheme, we adopt TML's directly and extend it.
This means every match already resolvable via TML gets a free, ground-truth player_id with
zero fuzzy matching required; fuzzy matching (see player_aliases.py) is only needed for
names appearing in MCP that don't come from an already-TML-joined match.

Registry schema (players.parquet):
    player_id       str   TML's native alphanumeric ID, adopted as-is
    canonical_name  str   most frequently-observed spelling of the player's name in TML
    birth_date      date  from ATP_Database.csv, if available
    hand            str   from ATP_Database.csv, if available
    country         str   from ATP_Database.csv (ioc), if available
    n_matches       int   how many TML matches this player appears in (winner or loser) —
                          useful for sanity-checking rare/likely-noise IDs later
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tennis_intel.data.join_tml_mcp import load_tml_matches
from tennis_intel.entities.canonical_players import full_name_key

logger = logging.getLogger(__name__)


def _read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def build_registry(tml_dir: Path) -> pd.DataFrame:
    """
    Builds the player registry by:
      1. Loading all TML matches, collecting (id, name) pairs from both winner and loser
         columns, and picking the most frequent spelling per ID as the canonical_name
         (handles the rare case where a name's exact spelling varies slightly across years
         for the same underlying player ID).
      2. Left-joining ATP_Database.csv for biographical detail (birth_date, hand, country)
         where available — not every player has a complete biographical row.
    """
    matches = load_tml_matches(tml_dir)

    winner_pairs = matches[["winner_id", "winner_name"]].rename(
        columns={"winner_id": "player_id", "winner_name": "name"}
    )
    loser_pairs = matches[["loser_id", "loser_name"]].rename(
        columns={"loser_id": "player_id", "loser_name": "name"}
    )
    all_pairs = pd.concat([winner_pairs, loser_pairs], ignore_index=True).dropna(subset=["player_id"])

    # Most frequent spelling per ID becomes canonical_name; also count total appearances.
    name_counts = all_pairs.groupby(["player_id", "name"]).size().reset_index(name="count")
    canonical = (
        name_counts.sort_values("count", ascending=False)
        .drop_duplicates(subset="player_id", keep="first")
        [["player_id", "name"]]
        .rename(columns={"name": "canonical_name"})
    )
    n_matches = all_pairs.groupby("player_id").size().reset_index(name="n_matches")
    registry = canonical.merge(n_matches, on="player_id", how="left")

    bio_path = tml_dir / "ATP_Database.csv"
    if bio_path.exists():
        bio = _read_csv_robust(bio_path)
        bio_slim = bio[["id", "birthdate", "hand", "ioc"]].rename(
            columns={"id": "player_id", "birthdate": "birth_date", "ioc": "country"}
        )
        registry = registry.merge(bio_slim, on="player_id", how="left")
        logger.info("Merged biographical data for %d / %d players",
                    registry["birth_date"].notna().sum(), len(registry))
    else:
        logger.warning("ATP_Database.csv not found at %s — registry will lack biographical data", bio_path)
        registry["birth_date"] = pd.NA
        registry["hand"] = pd.NA
        registry["country"] = pd.NA

    registry["canonical_name_key"] = registry["canonical_name"].apply(full_name_key)

    logger.info("Built registry: %d unique players from %d TML matches", len(registry), len(matches))
    return registry


def write_registry(registry: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(output_path, index=False)
    logger.info("Wrote %d players to %s", len(registry), output_path)