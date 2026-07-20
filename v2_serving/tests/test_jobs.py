"""test_jobs.py — POST /analyze-video + GET /jobs/{job_id}.

Uses a mocked run_video_analysis (a pre-computed fixture result, same shape as a
real one) for the success path -- real cv_pipeline inference (YOLO+ByteTrack+
MediaPipe across a whole segment) is slow and belongs in Phase 3's own validation
scripts, not this fast unit suite. The FAILURE path below is NOT mocked -- it
exercises run_video_analysis's real file-existence check, which is fast (fails
before any model loading) and is exactly the real behavior observed manually
during Phase 4 development (a relative path resolved against the server's cwd
surfaced as a clean 'failed' job, not a crash) -- a regression test for that
actual finding, not a hypothetical.
"""

from __future__ import annotations

from unittest.mock import patch

import v2_serving.routers.jobs as jobs_router

# Same shape as a real video_pipeline.run_video_analysis() result (see
# PROGRESS.md's Phase 4 entries) -- Status enum values included deliberately so
# passthrough-fidelity tests have real values to check against.
FIXTURE_RESULT = {
    "ground_truth": "NONE -- this is live inference output on an unannotated video, not a validated accuracy figure.",
    "clip": "video1.mp4",
    "n_frames_processed": 120,
    "source_fps": 60.08721279576819,
    "processing_time_s": 20.6,
    "player_selection_method": ["court_position_plausibility"],
    "homography": {
        "status": "measured",
        "note": "Independently validated in Phase 3 against the baseline center hash mark (~13px / ~8cm real-world error).",
    },
    "near_player_detection_live_estimate": {"status": "measured", "rate": 1.0, "n": 120},
    "far_player_detection_live_estimate": {"status": "measured", "rate": 1.0, "n": 120, "note": "..."},
    "ball_detection_live_estimate": {"status": "measured", "rate": 0.0167, "n": 120},
    "near_player_pose_live_estimate": {"status": "measured", "success_rate": 1.0, "n_attempted": 120},
    "far_player_pose_live_estimate": {"status": "not_detected", "note": "pose attempted on 120 frame(s), landmarks found on 0"},
    "tracking": {
        "near_player_distinct_track_ids": [1],
        "far_player_distinct_track_ids": [2],
        "note": "...",
    },
}


def test_analyze_video_returns_job_id_immediately(client):
    with patch.object(jobs_router, "run_video_analysis", return_value=FIXTURE_RESULT):
        response = client.post("/analyze-video", json={"video_path": "/fake/video1.mp4", "frame_limit": 120})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert "job_id" in body


def test_job_completes_and_result_is_passed_through_unflattened(client):
    with patch.object(jobs_router, "run_video_analysis", return_value=FIXTURE_RESULT):
        submit = client.post("/analyze-video", json={"video_path": "/fake/video1.mp4", "frame_limit": 120})
        job_id = submit.json()["job_id"]
        status = client.get(f"/jobs/{job_id}")

    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "complete"
    assert body["result"] == FIXTURE_RESULT


def test_status_enum_values_survive_the_api_boundary_unflattened(client):
    """The specific thing Phase 4 was told not to do: collapse cv_pipeline's
    Status enum vocabulary into a simpler-looking but less truthful shape."""
    with patch.object(jobs_router, "run_video_analysis", return_value=FIXTURE_RESULT):
        submit = client.post("/analyze-video", json={"video_path": "/fake/video1.mp4", "frame_limit": 120})
        job_id = submit.json()["job_id"]
        result = client.get(f"/jobs/{job_id}").json()["result"]

    assert result["homography"]["status"] == "measured"
    assert result["near_player_detection_live_estimate"]["status"] == "measured"
    # not_detected is a distinct, meaningful status (attempted, found nothing) --
    # must not collapse into False/None/a generic "failed".
    assert result["far_player_pose_live_estimate"]["status"] == "not_detected"


def test_unknown_job_id_returns_404(client):
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_relative_or_missing_video_path_fails_cleanly_not_a_crash(client):
    """Regression test for a real finding during manual Phase 4 testing: a
    relative video_path resolved against the server's cwd (not the repo root)
    surfaced as a clean 'failed' job with a clear FileNotFoundError message, not
    a crash or a silently-stuck 'processing' job. NOT mocked -- exercises the
    real file-existence check in video_pipeline.run_video_analysis."""
    submit = client.post("/analyze-video", json={"video_path": "data/does/not/exist.mp4", "frame_limit": 10})
    job_id = submit.json()["job_id"]

    status = client.get(f"/jobs/{job_id}")
    body = status.json()
    assert body["status"] == "failed"
    assert body["result"] is None
    assert "FileNotFoundError" in body["error"]
    assert "does/not/exist.mp4" in body["error"]
