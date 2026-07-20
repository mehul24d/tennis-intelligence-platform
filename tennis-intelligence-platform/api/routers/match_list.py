"""
routers/match_list.py — the Match Explorer table endpoint (search/filter/paginate).

DISTINCT FROM routers/matches.py's GET /api/matches: that endpoint returns bare
match_id strings (for the replay/search-by-substring use case, e.g. picking a specific
match to feed into the replay endpoint). THIS endpoint (GET /api/matches/search)
returns the full, enriched Match Explorer table — tournament, surface, round, final
score, Elo, etc. — for browsing/filtering. Two genuinely different purposes behind
similar-sounding names; kept as separate routers/functions rather than overloading one
endpoint with two different response shapes depending on query params.

Builds its own MatchListContext lazily from ReplayContext's already-loaded
frozen_join/day6 (cached on first call, via app.state) rather than reloading either
parquet file a second time.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas.match_list import MatchListResponse
from api.schemas.full_match_list import FullMatchListResponse
from tennis_intel.serving.match_list_service import load_match_list_context, get_match_list
from tennis_intel.serving.career_stats_service import load_career_stats_context, get_full_match_list

router = APIRouter(prefix="/api/matches", tags=["match-list"])


def get_match_list_context(request: Request):
    """Lazily builds and caches the MatchListContext on first call — the merge itself
    is cheap (a left-merge on ~6,000 rows), but no reason to redo it on every request
    when the underlying data doesn't change within a server's lifetime. PUBLIC (not
    module-private) since player_profile.py and rankings.py routers both need this
    exact same context and shouldn't each build their own separate copy."""
    if not hasattr(request.app.state, "match_list_context"):
        replay_ctx = request.app.state.replay_context
        request.app.state.match_list_context = load_match_list_context(
            replay_ctx.frozen_join, replay_ctx.day6
        )
    return request.app.state.match_list_context


def get_career_stats_context(request: Request):
    """Lazily builds and caches the CareerStatsContext (the FULL day6 table, ~198,062
    matches, independent of Match Charting Project coverage) on first call — used by
    player_profile.py and rankings.py, which need the complete TML corpus, not just
    the frozen-join-limited ~6,000-match subset that get_match_list_context provides.
    See tennis_intel.serving.career_stats_service's own docstring for the full
    reasoning on why these two features specifically need the full corpus."""
    if not hasattr(request.app.state, "career_stats_context"):
        replay_ctx = request.app.state.replay_context
        request.app.state.career_stats_context = load_career_stats_context(replay_ctx.day6)
    return request.app.state.career_stats_context


@router.get("/search", response_model=MatchListResponse)
def search_matches(
    request: Request,
    player: str | None = None,
    tournament: str | None = None,
    year: int | None = None,
    surface: str | None = None,
    round: str | None = None,
    tourney_level: str | None = None,
    best_of: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> MatchListResponse:
    """
    Filterable, paginated match list for the Match Explorer table. See
    tennis_intel.serving.match_list_service.get_match_list's own docstring for the
    exact semantics of each filter, including the documented gap for ATP500/ATP250
    (not separable from tourney_level='A' in the current data).
    """
    ctx = get_match_list_context(request)
    result = get_match_list(
        ctx, player=player, tournament=tournament, year=year, surface=surface,
        round_=round, tourney_level=tourney_level, best_of=best_of,
        limit=limit, offset=offset,
    )
    return MatchListResponse(**result)


@router.get("/search-full", response_model=FullMatchListResponse)
def search_matches_full(
    request: Request,
    player: str | None = None,
    surface: str | None = None,
    year: int | None = None,
    tourney_level: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> FullMatchListResponse:
    """
    Match Explorer, searching the FULL TML corpus (~198,062 matches) rather than
    only the ~6,000 with Match Charting Project point-by-point coverage. Every real
    TML match is browsable here; each result's has_replay_data flag tells the
    frontend whether "Open analysis" (full point-by-point replay) is actually
    available for that specific match, or whether only the brief score/tournament
    summary can be shown. See
    tennis_intel.serving.career_stats_service.get_full_match_list's own docstring
    for the full reasoning and the exact cross-referencing logic.
    """
    ctx = get_career_stats_context(request)
    replay_ctx = request.app.state.replay_context
    result = get_full_match_list(
        ctx, replay_ctx.frozen_join, player=player, surface=surface,
        year=year, tourney_level=tourney_level, limit=limit, offset=offset,
    )
    return FullMatchListResponse(**result)