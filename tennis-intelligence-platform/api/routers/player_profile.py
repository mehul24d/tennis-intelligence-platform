"""
routers/player_profile.py — the Player Profile page endpoints (search, profile,
head-to-head).

Uses match_list.get_career_stats_context (the FULL day6 corpus, ~198,062 matches),
NOT get_match_list_context (frozen-join-limited to ~6,000 matches).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.schemas.player_profile import (
    PlayerProfileResponse, HeadToHeadResponse, PlayerSearchResult,
)
from api.routers.match_list import get_career_stats_context
from tennis_intel.serving.career_stats_service import (
    get_player_profile, search_players_by_name, get_head_to_head,
)

router = APIRouter(prefix="/api/players", tags=["player-profile"])


@router.get("/search", response_model=list[PlayerSearchResult])
def search_players(request: Request, q: str, limit: int = 20) -> list[PlayerSearchResult]:
    """Name-based player search/autocomplete, across the FULL TML corpus."""
    ctx = get_career_stats_context(request)
    results = search_players_by_name(ctx, q, limit=limit)
    return [PlayerSearchResult(**r) for r in results]


@router.get("/{player_id}", response_model=PlayerProfileResponse)
def get_player_profile_endpoint(player_id: str, request: Request) -> PlayerProfileResponse:
    """Full Player Profile payload, computed across EVERY TML match for this
    player, not just their Match Charting Project-covered subset."""
    ctx = get_career_stats_context(request)
    try:
        result = get_player_profile(ctx, player_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return PlayerProfileResponse(**result)


@router.get("/{player_id_a}/head-to-head/{player_id_b}", response_model=HeadToHeadResponse)
def get_head_to_head_endpoint(
    player_id_a: str, player_id_b: str, request: Request,
) -> HeadToHeadResponse:
    """Head-to-head record between two players, across the FULL TML corpus."""
    ctx = get_career_stats_context(request)
    result = get_head_to_head(ctx, player_id_a, player_id_b)
    return HeadToHeadResponse(**result)