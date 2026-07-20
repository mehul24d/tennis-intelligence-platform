"""player_documents.py — turns v1's career_stats_service player profiles into
retrievable RagDocuments.

REUSES, DOES NOT REIMPLEMENT: every stat here (current_elo, peak_elo, career record,
surface breakdown, Grand Slam record) comes straight from
tennis_intel.serving.career_stats_service.get_player_profile — the same, already-
validated function backing the v1 Player Profile page. This module only formats that
output into embeddable text, exactly matching the discipline replay_service.py and
match_documents.py already follow (reuse the service layer, don't re-read raw parquet
columns for stats that already have a canonical computation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from rag_engine import _v1_path  # noqa: F401 — sets up sys.path for the import below
from rag_engine.ingest.types import RagDocument

from tennis_intel.serving.career_stats_service import (
    CareerStatsContext,
    get_player_profile,
    load_career_stats_context,
)

DEFAULT_DAY6_PATH = (
    Path(__file__).resolve().parents[4]
    / "tennis-intelligence-platform" / "data" / "processed" / "matches_with_day6_features.parquet"
)

# A player needs at least this many career matches to get a profile document — avoids
# flooding the index with one-match qualifiers/wildcards that have no meaningful
# career signal to retrieve.
MIN_CAREER_MATCHES = 10


def _surface_clause(surface_stats: dict) -> str:
    parts = []
    for surface, stats in sorted(
        surface_stats.items(), key=lambda kv: kv[1]["win_pct"], reverse=True
    ):
        if stats["matches"] == 0:
            continue
        parts.append(
            f"{surface} {stats['win_pct']:.1%} ({stats['wins']}/{stats['matches']})"
        )
    if not parts:
        return ""
    best = parts[0]
    return f" Surface breakdown: {', '.join(parts)}, best surface {best.split(' ')[0]}."


def _player_text(profile: dict) -> str:
    name = profile["player_name"]
    sentence = (
        f"{name} — career profile. {profile['career_matches']} career matches, "
        f"{profile['career_wins']} wins, {profile['career_losses']} losses "
        f"({profile['career_win_pct']:.1%} win rate). Current Elo: "
        f"{profile['current_elo']:.1f} (peak {profile['peak_elo']:.1f})."
    )
    sentence += _surface_clause(profile["surface_stats"])

    gs = profile.get("grand_slam_stats")
    if gs and gs["matches"] > 0:
        gs_losses = gs["matches"] - gs["wins"]
        sentence += (
            f" Grand Slam record: {gs['wins']}-{gs_losses} ({gs['win_pct']:.1%} win rate)."
        )
    return sentence


def _player_metadata(profile: dict) -> dict:
    surface_stats = profile["surface_stats"]
    best_surface = ""
    if surface_stats:
        best_surface = max(
            (s for s, v in surface_stats.items() if v["matches"] > 0),
            key=lambda s: surface_stats[s]["win_pct"],
            default="",
        )
    return {
        "doc_type": "player_profile",
        "player_id": str(profile["player_id"]),
        "player": str(profile["player_name"]),
        "career_matches": int(profile["career_matches"]),
        "career_win_pct": float(profile["career_win_pct"]),
        "current_elo": float(profile["current_elo"]),
        "best_surface": best_surface,
    }


def build_player_documents(
    day6_path: Path = DEFAULT_DAY6_PATH, limit: int | None = None
) -> Iterator[RagDocument]:
    """Yields one RagDocument per player with >= MIN_CAREER_MATCHES career matches.
    `limit` caps the number of PLAYERS processed (ordered by career match count,
    descending — most-documented players first), for fast iteration while developing.
    """
    day6 = pd.read_parquet(day6_path)
    ctx: CareerStatsContext = load_career_stats_context(day6)

    winner_counts = ctx.day6["winner_id"].value_counts()
    loser_counts = ctx.day6["loser_id"].value_counts()
    match_counts = winner_counts.add(loser_counts, fill_value=0).sort_values(ascending=False)
    player_ids = match_counts[match_counts >= MIN_CAREER_MATCHES].index.tolist()
    if limit is not None:
        player_ids = player_ids[:limit]

    for player_id in player_ids:
        profile = get_player_profile(ctx, player_id)
        yield RagDocument(
            doc_id=f"player:{player_id}",
            text=_player_text(profile),
            metadata=_player_metadata(profile),
        )
