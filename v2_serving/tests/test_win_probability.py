"""test_win_probability.py — GET /win-probability/{job_id}.

The two match_id cases below make REAL calls into v1's engine (not mocked) --
deliberately, as regression guards: these exact values (0.7818, 0.9093) were
verified bit-for-bit against the original, slower compute_five_engine_trajectory()
path before the fast path was adopted (see PROGRESS.md's Phase 4 "performance
finding" entry). If the fast path in win_probability_pipeline.py ever drifts from
what v1's full engine would produce -- the exact class of subtle bug this project
has caught before (e.g. the PtWinner convention investigation) -- these tests
catch it immediately rather than silently. First run pays v1's one-time
load_replay_context() cost (~15-20s); acceptable for a regression guard, not run
per-commit-critical-path.
"""

from __future__ import annotations

from unittest.mock import patch

import v2_serving.routers.jobs as jobs_router

DJOKOVIC_GOFFIN = "20190710-M-Wimbledon-QF-Novak_Djokovic-David_Goffin"
DJOKOVIC_KOHLSCHREIBER = "20190701-M-Wimbledon-R128-Novak_Djokovic-Philipp_Kohlschreiber"


def _make_complete_job(client) -> str:
    with patch.object(jobs_router, "run_video_analysis", return_value={"clip": "video1.mp4"}):
        submit = client.post("/analyze-video", json={"video_path": "/fake/video1.mp4", "frame_limit": 120})
    return submit.json()["job_id"]


def test_win_probability_unknown_job_returns_404(client):
    response = client.get("/win-probability/does-not-exist")
    assert response.status_code == 404


def test_win_probability_without_match_id_reports_not_available_plainly(client):
    job_id = _make_complete_job(client)
    response = client.get(f"/win-probability/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["prematch_baseline"]["status"] == "not_available"
    assert "match_id" in body["prematch_baseline"]["reason"]
    assert body["live_adjustment"]["status"] == "not_available"


def test_win_probability_live_adjustment_always_not_available_today(client):
    """cv_pipeline has no point-level score/serve extraction -- this must stay
    'not_available' with a plain reason regardless of match_id, not silently
    start returning a fabricated number."""
    job_id = _make_complete_job(client)
    response = client.get(f"/win-probability/{job_id}", params={"match_id": DJOKOVIC_GOFFIN})
    body = response.json()
    assert body["live_adjustment"]["status"] == "not_available"
    assert "point-by-point" in body["live_adjustment"]["reason"]


def test_win_probability_djokovic_goffin_exact_regression_value(client):
    job_id = _make_complete_job(client)
    response = client.get(f"/win-probability/{job_id}", params={"match_id": DJOKOVIC_GOFFIN})
    assert response.status_code == 200
    baseline = response.json()["prematch_baseline"]
    assert baseline["status"] == "available"
    assert baseline["p1_name"] == "Novak Djokovic"
    assert baseline["p2_name"] == "David Goffin"
    # Exact regression value -- rounded to 4dp by the API, matching the verified
    # bit-for-bit comparison (0.7818396461367739) done before adopting the fast path.
    assert baseline["p1_win_probability_prematch"] == 0.7818


def test_win_probability_djokovic_kohlschreiber_exact_regression_value(client):
    job_id = _make_complete_job(client)
    response = client.get(f"/win-probability/{job_id}", params={"match_id": DJOKOVIC_KOHLSCHREIBER})
    assert response.status_code == 200
    baseline = response.json()["prematch_baseline"]
    assert baseline["status"] == "available"
    assert baseline["p1_name"] == "Novak Djokovic"
    assert baseline["p2_name"] == "Philipp Kohlschreiber"
    assert baseline["p1_win_probability_prematch"] == 0.9093


def test_win_probability_unknown_match_id_reports_not_available_not_a_crash(client):
    job_id = _make_complete_job(client)
    response = client.get(f"/win-probability/{job_id}", params={"match_id": "not-a-real-match-id"})
    assert response.status_code == 200
    assert response.json()["prematch_baseline"]["status"] == "not_available"
