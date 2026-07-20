"""
routers/rankings.py — the Rankings dashboard endpoints (current Elo, peak Elo,
surface Elo, biggest upsets).

Uses match_list.get_career_stats_context (the FULL day6 corpus, ~198,062 matches),
NOT get_match_list_context (frozen-join-limited to ~6,000 matches).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas.rankings import (
    CurrentEloEntry, PeakEloEntry, SurfaceEloEntry, PeakSurfaceEloEntry, UpsetEntry,
)
from api.routers.match_list import get_career_stats_context
from tennis_intel.serving.career_stats_service import (
    get_current_elo_rankings, get_peak_elo_rankings, get_surface_elo_rankings,
    get_peak_surface_elo_rankings, get_biggest_upsets,
)

router = APIRouter(prefix="/api/rankings", tags=["rankings"])


@router.get("/current-elo", response_model=list[CurrentEloEntry])
def current_elo_rankings(request: Request, limit: int = 100) -> list[CurrentEloEntry]:
    ctx = get_career_stats_context(request)
    return [CurrentEloEntry(**r) for r in get_current_elo_rankings(ctx, limit=limit)]


@router.get("/peak-elo", response_model=list[PeakEloEntry])
def peak_elo_rankings(request: Request, limit: int = 100) -> list[PeakEloEntry]:
    ctx = get_career_stats_context(request)
    return [PeakEloEntry(**r) for r in get_peak_elo_rankings(ctx, limit=limit)]


@router.get("/surface-elo", response_model=list[SurfaceEloEntry])
def surface_elo_rankings(request: Request, surface: str, limit: int = 100) -> list[SurfaceEloEntry]:
    """surface: one of 'Clay', 'Hard', 'Grass' (must match day6's real, exact
    values — case-sensitive). Returns each player's CURRENT (most recent match on
    this surface) surface Elo."""
    ctx = get_career_stats_context(request)
    return [SurfaceEloEntry(**r) for r in get_surface_elo_rankings(ctx, surface, limit=limit)]


@router.get("/peak-surface-elo", response_model=list[PeakSurfaceEloEntry])
def peak_surface_elo_rankings(request: Request, surface: str, limit: int = 100) -> list[PeakSurfaceEloEntry]:
    """surface: one of 'Clay', 'Hard', 'Grass' (case-sensitive). Returns each
    player's CAREER-HIGH surface Elo ON THAT SURFACE, at any point in their history —
    distinct from /surface-elo, which returns their most recent (possibly declined)
    rating. A player who peaked on Clay years ago but has since dropped off will
    still show their true historical best here, plus the date it was achieved."""
    ctx = get_career_stats_context(request)
    return [PeakSurfaceEloEntry(**r) for r in get_peak_surface_elo_rankings(ctx, surface, limit=limit)]


@router.get("/biggest-upsets", response_model=list[UpsetEntry])
def biggest_upsets(request: Request, limit: int = 100) -> list[UpsetEntry]:
    ctx = get_career_stats_context(request)
    return [UpsetEntry(**r) for r in get_biggest_upsets(ctx, limit=limit)]