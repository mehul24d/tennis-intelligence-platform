"""test_query.py — POST /query. Mocks run_query (no live LLM/Gemini calls, no
retrieval index load) so the suite stays fast -- the real end-to-end grounding
behavior (live CV features + RAG context fused, citation audit intact) was
manually verified against a real Gemini call during Phase 4 development (see
PROGRESS.md); this suite instead locks in the API's CONTRACT: that whatever
sources_used/sources_offered llm_agent returns passes through unflattened, and
that a missing/incomplete job_id degrades gracefully rather than blocking the query.
"""

from __future__ import annotations

from unittest.mock import patch

import v2_serving.routers.jobs as jobs_router
import v2_serving.routers.query as query_router
from llm_agent.agent import AgentResponse, SourcesUsed

FIXTURE_ANSWER = AgentResponse(
    answer="Norrie won 74% of first-serve points [D2]; the live clip shows the near "
           "player detected in 100% of processed frames [L1].",
    sources_used=SourcesUsed(
        live_features=["Near player detected (live CV estimate, this clip segment): 100% of 120 processed frames"],
        retrieved_docs=["match_summary: Cameron Norrie defeated ... in the R32 of the Wimbledon ..."],
    ),
    sources_offered={
        "L1": "Near player detected (live CV estimate, this clip segment): 100% of 120 processed frames",
        "L2": "Far player detected (live CV estimate, this clip segment): 100% of 120 processed frames",
        "D1": "match_summary: some other, uncited match",
        "D2": "match_summary: Cameron Norrie defeated ... in the R32 of the Wimbledon ...",
    },
)

JOB_FIXTURE_RESULT = {"clip": "video1.mp4", "near_player_detection_live_estimate": {"status": "measured", "rate": 1.0, "n": 120}}


def test_query_without_job_id_runs_on_historical_context_alone(client):
    with patch.object(query_router, "run_query", return_value=FIXTURE_ANSWER) as mock_run:
        response = client.post("/query", json={"question": "How does Norrie perform on grass?"})
    assert response.status_code == 200
    body = response.json()
    assert body["live_features_used"] is False
    assert body["live_features_note"] is None
    # live_features arg passed to run_query should be None when no job_id given
    assert mock_run.call_args.args[1] is None


def test_query_sources_used_and_sources_offered_pass_through_unflattened(client):
    with patch.object(query_router, "run_query", return_value=FIXTURE_ANSWER):
        response = client.post("/query", json={"question": "How does Norrie perform on grass?"})
    body = response.json()
    assert body["answer"] == FIXTURE_ANSWER.answer
    # sources_used: only what was actually cited
    assert body["sources_used"]["live_features"] == FIXTURE_ANSWER.sources_used.live_features
    assert body["sources_used"]["retrieved_docs"] == FIXTURE_ANSWER.sources_used.retrieved_docs
    # sources_offered: everything given to the model, cited or not -- L2 and D1
    # were offered but NOT cited, and must still appear here (this is the whole
    # point of the citation-audit transparency, not just echoing sources_used twice).
    assert body["sources_offered"] == FIXTURE_ANSWER.sources_offered
    assert "L2" in body["sources_offered"]
    assert "L2" not in body["sources_used"]["live_features"] and not any(
        "Far player" in s for s in body["sources_used"]["live_features"]
    )


def test_query_with_complete_job_id_fuses_live_features(client):
    with patch.object(jobs_router, "run_video_analysis", return_value=JOB_FIXTURE_RESULT):
        submit = client.post("/analyze-video", json={"video_path": "/fake/video1.mp4", "frame_limit": 120})
        job_id = submit.json()["job_id"]

    with patch.object(query_router, "run_query", return_value=FIXTURE_ANSWER) as mock_run:
        response = client.post("/query", json={"job_id": job_id, "question": "How is the near player doing?"})

    assert response.status_code == 200
    body = response.json()
    assert body["live_features_used"] is True
    assert body["live_features_note"] is None
    # a real (non-None) LiveFeatureSnapshot must have been built and passed to run_query
    live_features_arg = mock_run.call_args.args[1]
    assert live_features_arg is not None
    assert live_features_arg.match_id == "video1"


def test_query_with_incomplete_job_id_degrades_gracefully_not_blocked(client):
    with patch.object(jobs_router, "run_video_analysis", return_value=JOB_FIXTURE_RESULT):
        # don't let the background task run by not calling GET -- but since
        # TestClient runs BackgroundTasks synchronously (verified), simulate an
        # in-progress job directly via the job store instead.
        pass
    from v2_serving.job_store import job_store
    job = job_store.create(video_path="/fake/video1.mp4", frame_limit=120)
    job_store.set_status(job.job_id, "processing")

    with patch.object(query_router, "run_query", return_value=FIXTURE_ANSWER) as mock_run:
        response = client.post("/query", json={"job_id": job.job_id, "question": "How is the near player doing?"})

    assert response.status_code == 200
    body = response.json()
    assert body["live_features_used"] is False
    assert "processing" in body["live_features_note"]
    assert mock_run.call_args.args[1] is None


def test_query_with_unknown_job_id_degrades_gracefully(client):
    with patch.object(query_router, "run_query", return_value=FIXTURE_ANSWER):
        response = client.post("/query", json={"job_id": "does-not-exist", "question": "Anything?"})
    assert response.status_code == 200
    body = response.json()
    assert body["live_features_used"] is False
    assert "not found" in body["live_features_note"]


def test_query_player_and_opponent_are_folded_into_the_question(client):
    with patch.object(query_router, "run_query", return_value=FIXTURE_ANSWER) as mock_run:
        client.post("/query", json={"question": "How do they match up?", "player": "Norrie", "opponent": "Draper"})
    question_arg = mock_run.call_args.args[0]
    assert "Norrie" in question_arg and "Draper" in question_arg
