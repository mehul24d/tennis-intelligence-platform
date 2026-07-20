"""query_pipeline.py — builds a LiveFeatureSnapshot from a completed cv_pipeline
job's live-estimate result (see video_pipeline.py) and hands it, alongside the
caller's question, to llm_agent's TennisAnalystAgent, which internally runs
rag_engine's retriever for historical context. This is pure orchestration -- no
changes to rag_engine/ or llm_agent/'s own grounding discipline.

The VectorStore and GeminiClient/model are expensive to construct (loads the
sentence-transformer + opens the persisted Chroma index) -- cached as module-level
singletons rather than rebuilt per request, same rationale as job_store.py's
singleton (single-process dev server).
"""

from __future__ import annotations

import sys
from pathlib import Path

RAG_ENGINE_SRC = Path(__file__).resolve().parents[3] / "rag_engine" / "src"
LLM_AGENT_SRC = Path(__file__).resolve().parents[3] / "llm_agent" / "src"
for _p in (RAG_ENGINE_SRC, LLM_AGENT_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from llm_agent.live_features import LiveFeature, LiveFeatureSnapshot

_vector_store = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        from rag_engine.index.vector_store import VectorStore
        _vector_store = VectorStore()
    return _vector_store


def build_live_feature_snapshot(job_result: dict, clip_name: str) -> LiveFeatureSnapshot:
    """Turns a completed job's live-estimate CV result into a LiveFeatureSnapshot.
    Every feature here is a live-inference ESTIMATE (is_estimate=True) -- none of
    this is ground-truth-validated (see video_pipeline.py's "_live_estimate" field
    naming), and that distinction is preserved through to the agent's prompt via
    the same [L#]-tag "(model estimate)" annotation llm_agent already applies to
    win-probability features."""
    features: list[LiveFeature] = []

    near = job_result.get("near_player_detection_live_estimate", {})
    if near.get("status") == "measured":
        features.append(LiveFeature(
            "Near player detected (live CV estimate, this clip segment)",
            f"{near['rate']:.0%} of {near['n']} processed frames", is_estimate=True,
        ))
    far = job_result.get("far_player_detection_live_estimate", {})
    if far.get("status") == "measured":
        features.append(LiveFeature(
            "Far player detected (live CV estimate, this clip segment -- see note: "
            "not directly comparable to Phase 3's ground-truth-validated figures)",
            f"{far['rate']:.0%} of {far['n']} processed frames", is_estimate=True,
        ))
    ball = job_result.get("ball_detection_live_estimate", {})
    if ball.get("status") == "measured":
        features.append(LiveFeature(
            "Ball detected (live CV estimate, this clip segment)",
            f"{ball['rate']:.0%} of {ball['n']} processed frames", is_estimate=True,
        ))
    tracking = job_result.get("tracking", {})
    n_near_ids = len(tracking.get("near_player_distinct_track_ids", []))
    n_far_ids = len(tracking.get("far_player_distinct_track_ids", []))
    if n_near_ids or n_far_ids:
        features.append(LiveFeature(
            "Tracking ID stability (live CV estimate)",
            f"{n_near_ids} distinct ID(s) for near player, {n_far_ids} for far player "
            f"across the processed segment (more than 1 suggests a track loss/swap)",
            is_estimate=True,
        ))
    homography = job_result.get("homography", {})
    features.append(LiveFeature(
        "Real-world-distance CV metrics availability for this clip",
        f"{homography.get('status', 'unknown')} -- {homography.get('note', '')}",
        is_estimate=False,  # this is a stated fact about data quality, not a model prediction
    ))

    return LiveFeatureSnapshot(
        match_id=clip_name, p1_name="near player", p2_name="far player", features=features,
    )


def run_query(question: str, live_features: LiveFeatureSnapshot | None):
    from llm_agent.agent import TennisAnalystAgent

    agent = TennisAnalystAgent(vector_store=_get_vector_store())
    return agent.ask(question, live_features=live_features)
