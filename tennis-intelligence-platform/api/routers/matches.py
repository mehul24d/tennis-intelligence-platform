"""
routers/matches.py — match-replay and match-search endpoints.

Every response body here matches tennis_intel.serving.replay_service's own return
shapes directly (see schemas/match.py's own docstring on this) — this router is a
thin HTTP wrapper, not a place for any new business logic.

DISTINCT FROM routers/match_list.py's GET /api/matches/search: THIS file's
list_matches (GET /api/matches, no path suffix) returns bare match_id strings, meant
for the replay/lookup use case. match_list.py's endpoint returns the full, enriched
Match Explorer table (tournament, surface, Elo, etc.) for browsing/filtering — a
genuinely different response shape and purpose, kept in a separate router/file.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.schemas.match import MatchReplayResponse, MatchSearchResponse
from api.schemas.match_summary import MatchSummaryResponse
from api.schemas.model_agreement import ModelAgreementResponse
from api.schemas.point_timeline import PointTimelineResponse
from tennis_intel.serving.replay_service import (
    replay_match_by_id, search_match_ids, list_available_match_ids,
)
from tennis_intel.serving.match_summary_service import get_match_summary
from tennis_intel.serving.model_agreement_service import get_model_agreement
from tennis_intel.serving.point_timeline_service import get_point_timeline

router = APIRouter(prefix="/api/matches", tags=["matches"])


@router.get("", response_model=MatchSearchResponse)
def list_matches(request: Request, search: list[str] | None = None) -> MatchSearchResponse:
    """
    Lists match_ids available for replay. Pass one or more `search` query params
    (e.g. ?search=Sinner&search=Alcaraz&search=Roland_Garros) to filter by substring
    match against the match_id, same semantics as replay_match.py's own --search flag.
    With no search terms, returns every available match_id (2,000+ matches — the
    frontend's Match Explorer should paginate/virtualize this list, not render it
    all at once).
    """
    ctx = request.app.state.replay_context
    if search:
        return MatchSearchResponse(match_ids=search_match_ids(ctx, search))
    return MatchSearchResponse(match_ids=list_available_match_ids(ctx))


@router.get("/{match_id}/replay", response_model=MatchReplayResponse)
def get_match_replay(match_id: str, request: Request) -> MatchReplayResponse:
    """
    Full point-by-point replay for one match, across all five prediction engines —
    the data source for the Match Analysis page's centerpiece probability chart.
    """
    ctx = request.app.state.replay_context
    try:
        result = replay_match_by_id(ctx, match_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return MatchReplayResponse(**result)


@router.get("/{match_id}/summary", response_model=MatchSummaryResponse)
def get_match_summary_endpoint(match_id: str, request: Request) -> MatchSummaryResponse:
    """
    Match Summary card stats — largest comeback, largest probability swing (both
    computed using ML-Informed Markov (smoothed), this project's primary engine — see
    match_summary_service.py's own docstring for why that specific engine was chosen),
    longest winning streak, longest service hold, and break points created/converted
    for both players. total_winners/total_unforced_errors/serve_percentage are
    currently null placeholders — see that same docstring for exactly why (the
    underlying data exists in Overview.csv but isn't yet wired into this endpoint).
    """
    ctx = request.app.state.replay_context
    try:
        result = get_match_summary(ctx, match_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return MatchSummaryResponse(**result)


@router.get("/{match_id}/model-agreement", response_model=ModelAgreementResponse)
def get_model_agreement_endpoint(match_id: str, request: Request) -> ModelAgreementResponse:
    """
    Model Agreement Panel data — per-point highest/lowest/average/std-dev probability
    across all five engines, max disagreement, most/least confident engine, and which
    engine is changing fastest, plus a match-wide summary of how often engines
    disagree by more than 5%/10%/20%.
    """
    ctx = request.app.state.replay_context
    try:
        result = get_model_agreement(ctx, match_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ModelAgreementResponse(**result)


@router.get("/{match_id}/timeline", response_model=PointTimelineResponse)
def get_point_timeline_endpoint(
    match_id: str, request: Request,
    break_points_only: bool = False, set_points_only: bool = False,
    match_points_only: bool = False, tiebreak_only: bool = False,
    min_swing: float | None = None,
) -> PointTimelineResponse:
    """
    Interactive Point Timeline table — every point's server, receiver, winner, score
    before, probability before/after (ML-Informed Markov smoothed), swing, and
    break/set/match/tiebreak flags. Supports the spec's filters directly as query
    params (e.g. ?break_points_only=true, ?min_swing=0.05 for "swings greater than
    5%"). Filters compose — all provided filters must be satisfied, not any.
    """
    ctx = request.app.state.replay_context
    try:
        result = get_point_timeline(
            ctx, match_id,
            break_points_only=break_points_only, set_points_only=set_points_only,
            match_points_only=match_points_only, tiebreak_only=tiebreak_only,
            min_swing=min_swing,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return PointTimelineResponse(**result)