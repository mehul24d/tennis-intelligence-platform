"""routers/query.py — POST /query: fuses live CV features (if a complete job_id is
given) with rag_engine's retrieval and llm_agent's grounded generation."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from v2_serving.job_store import job_store
from v2_serving.models import QueryRequest, QueryResponse, SourcesUsedResponse
from v2_serving.query_pipeline import build_live_feature_snapshot, run_query

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    live_features = None
    live_features_used = False
    live_features_note = None

    if request.job_id is not None:
        job = job_store.get(request.job_id)
        if job is None:
            live_features_note = f"job_id '{request.job_id}' not found -- proceeding on historical context alone."
        elif job.status != "complete":
            live_features_note = f"job '{request.job_id}' is '{job.status}', not complete -- proceeding on historical context alone."
        else:
            clip_name = Path(job.video_path).stem
            live_features = build_live_feature_snapshot(job.result, clip_name)
            live_features_used = True

    question = request.question
    if request.player or request.opponent:
        matchup = " vs ".join(p for p in [request.player, request.opponent] if p)
        question = f"[Players: {matchup}] {question}"

    try:
        response = run_query(question, live_features)
    except Exception as exc:  # noqa: BLE001 -- surface as a clear 502, never a silent empty answer
        raise HTTPException(status_code=502, detail=f"query failed: {type(exc).__name__}: {exc}")

    return QueryResponse(
        answer=response.answer,
        sources_used=SourcesUsedResponse(
            live_features=response.sources_used.live_features,
            retrieved_docs=response.sources_used.retrieved_docs,
        ),
        sources_offered=response.sources_offered,
        live_features_used=live_features_used,
        live_features_note=live_features_note,
    )
