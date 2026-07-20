"""routers/jobs.py — POST /analyze-video (kicks off async cv_pipeline inference)
and GET /jobs/{job_id} (poll status / fetch result)."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from v2_serving.job_store import job_store
from v2_serving.models import AnalyzeVideoRequest, AnalyzeVideoResponse, JobStatusResponse
from v2_serving.video_pipeline import run_video_analysis

router = APIRouter(tags=["jobs"])


def _process_job(job_id: str) -> None:
    job = job_store.get(job_id)
    if job is None:
        return
    job_store.set_status(job_id, "processing")
    try:
        result = run_video_analysis(job.video_path, job.frame_limit)
        job_store.set_result(job_id, result)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: a failed job should
        # be reported via job status, never crash the background-task thread silently.
        job_store.set_failed(job_id, f"{type(exc).__name__}: {exc}")


@router.post("/analyze-video", response_model=AnalyzeVideoResponse)
def analyze_video(request: AnalyzeVideoRequest, background_tasks: BackgroundTasks) -> AnalyzeVideoResponse:
    job = job_store.create(video_path=request.video_path, frame_limit=request.frame_limit)
    background_tasks.add_task(_process_job, job.job_id)
    return AnalyzeVideoResponse(job_id=job.job_id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job with id {job_id}")
    return JobStatusResponse(
        job_id=job.job_id, status=job.status, video_path=job.video_path,
        frame_limit=job.frame_limit, error=job.error, result=job.result,
    )
