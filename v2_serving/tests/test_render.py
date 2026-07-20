"""test_render.py — POST /render-video, GET /render-jobs/{id}, GET
/rendered-video/{filename}.

Same testing philosophy as test_jobs.py: a mocked render_annotated_video for
the success path (real cv2 video writing is slow and belongs in a manual
smoke test against a real clip, not this fast unit suite -- see PROGRESS.md's
'Step 3: Full Output Video Render' entry for that manual verification). The
error paths below are NOT mocked -- they exercise the real
job_store/render_job_store lookups.
"""

from __future__ import annotations

from unittest.mock import patch

import v2_serving.routers.render as render_router
from v2_serving.job_store import job_store

FIXTURE_RESULT = {
    "clip": "video1.mp4",
    "source_fps": 59.94,
    "video_width": 1920,
    "video_height": 1080,
    "frames": [],
    "homography": {"status": "measured", "court_corners": {"BL": [1, 2], "BR": [3, 4], "TR": [5, 6], "TL": [7, 8]}},
}


def _complete_source_job() -> str:
    job = job_store.create(video_path="/fake/video1.mp4", frame_limit=120)
    job_store.set_result(job.job_id, FIXTURE_RESULT)
    return job.job_id


def test_render_video_rejects_unknown_source_job(client):
    response = client.post("/render-video", json={"job_id": "does-not-exist"})
    assert response.status_code == 404


def test_render_video_rejects_incomplete_source_job(client):
    job = job_store.create(video_path="/fake/video1.mp4", frame_limit=120)  # still "pending"
    response = client.post("/render-video", json={"job_id": job.job_id})
    assert response.status_code == 400
    assert "not complete" in response.json()["detail"]


def test_render_video_returns_render_job_id_immediately(client):
    source_job_id = _complete_source_job()
    fake_output = {"output_path": "/fake/out.mp4", "n_frames_total": 120, "n_frames_annotated": 120, "elapsed_s": 1.0}
    with patch.object(render_router, "render_annotated_video", return_value=fake_output):
        response = client.post("/render-video", json={"job_id": source_job_id})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert "render_job_id" in body


def test_render_job_completes_and_exposes_output_filename(client):
    # output_filename is deterministically {render_job_id}.mp4 -- the router
    # builds this path itself (see routers/render.py's _process_render_job)
    # and passes it to render_annotated_video as the WRITE target, so the
    # mock's return value doesn't drive it; only that render_annotated_video
    # doesn't raise matters for this test.
    source_job_id = _complete_source_job()
    fake_output = {"output_path": "irrelevant", "n_frames_total": 120, "n_frames_annotated": 120, "elapsed_s": 1.0}
    with patch.object(render_router, "render_annotated_video", return_value=fake_output):
        submit = client.post("/render-video", json={"job_id": source_job_id})
        render_job_id = submit.json()["render_job_id"]
        status = client.get(f"/render-jobs/{render_job_id}")

    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "complete"
    assert body["source_job_id"] == source_job_id
    assert body["output_filename"] == f"{render_job_id}.mp4"


def test_render_job_reports_failure_not_a_crash(client):
    source_job_id = _complete_source_job()
    with patch.object(render_router, "render_annotated_video", side_effect=RuntimeError("boom")):
        submit = client.post("/render-video", json={"job_id": source_job_id})
        render_job_id = submit.json()["render_job_id"]
        status = client.get(f"/render-jobs/{render_job_id}")

    body = status.json()
    assert body["status"] == "failed"
    assert "boom" in body["error"]


def test_unknown_render_job_id_returns_404(client):
    response = client.get("/render-jobs/does-not-exist")
    assert response.status_code == 404


def test_unknown_rendered_video_filename_returns_404(client):
    response = client.get("/rendered-video/does-not-exist.mp4")
    assert response.status_code == 404


def test_rendered_video_filename_rejects_path_traversal(client):
    response = client.get("/rendered-video/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in (400, 404)  # never a raw filesystem read outside RENDER_OUTPUT_DIR
