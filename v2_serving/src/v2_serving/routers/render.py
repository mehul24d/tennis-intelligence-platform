"""routers/render.py — POST /render-video (burns an already-complete
/analyze-video job's overlay data into a real downloadable .mp4, via
video_render.render_annotated_video), GET /render-jobs/{render_job_id}
(poll), and GET /rendered-video/{filename} (fetch the finished file). The
server-side counterpart to the dashboard's live VideoOverlay.jsx canvas
overlay -- see video_render.py's module docstring for what gets drawn and
why. Async/job-based for consistency with /analyze-video (routers/jobs.py),
even though a render itself is comparatively cheap (no model inference, just
video I/O + cv2 drawing) -- a full-length clip's frame-by-frame write can
still take real wall-clock time, and the event loop shouldn't block on it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from v2_serving.job_store import job_store
from v2_serving.models import RenderJobStatusResponse, RenderVideoRequest, RenderVideoResponse
from v2_serving.render_job_store import render_job_store
from v2_serving.video_render import render_annotated_video

router = APIRouter(tags=["render"])

REPO_ROOT = Path(__file__).resolve().parents[4]
RENDER_OUTPUT_DIR = REPO_ROOT / "data" / "rendered"


def _process_render_job(render_job_id: str, video_path: str, result: dict) -> None:
    job = render_job_store.get(render_job_id)
    if job is None:
        return
    render_job_store.set_status(render_job_id, "processing")
    try:
        output_path = RENDER_OUTPUT_DIR / f"{render_job_id}.mp4"
        render_annotated_video(video_path, result, str(output_path))
        render_job_store.set_result(render_job_id, str(output_path))
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, same reasoning as
        # routers/jobs.py's _process_job: a failed render must be reported via job
        # status, never crash the background-task thread silently.
        render_job_store.set_failed(render_job_id, f"{type(exc).__name__}: {exc}")


@router.post("/render-video", response_model=RenderVideoResponse)
def render_video(request: RenderVideoRequest, background_tasks: BackgroundTasks) -> RenderVideoResponse:
    source_job = job_store.get(request.job_id)
    if source_job is None:
        raise HTTPException(status_code=404, detail=f"no job with id {request.job_id}")
    if source_job.status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"job {request.job_id} is not complete (status: {source_job.status}) -- "
                   "render draws from an already-computed analysis result, nothing to render yet",
        )

    render_job = render_job_store.create(source_job_id=request.job_id)
    background_tasks.add_task(
        _process_render_job, render_job.render_job_id, source_job.video_path, source_job.result,
    )
    return RenderVideoResponse(render_job_id=render_job.render_job_id, status=render_job.status)


@router.get("/render-jobs/{render_job_id}", response_model=RenderJobStatusResponse)
def get_render_job(render_job_id: str) -> RenderJobStatusResponse:
    job = render_job_store.get(render_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no render job with id {render_job_id}")
    return RenderJobStatusResponse(
        render_job_id=job.render_job_id,
        status=job.status,
        source_job_id=job.source_job_id,
        error=job.error,
        output_filename=Path(job.output_path).name if job.output_path else None,
    )


@router.get("/rendered-video/{filename}")
def get_rendered_video(filename: str):
    # basename only -- rejects path-separator tricks before touching the
    # filesystem, same pattern as routers/media.py's _resolve_known_clip.
    if filename != Path(filename).name:
        raise HTTPException(status_code=400, detail="filename must not contain a path")
    path = RENDER_OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no rendered video named '{filename}'")
    return FileResponse(path, media_type="video/mp4")
