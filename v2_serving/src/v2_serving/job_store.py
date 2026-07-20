"""job_store.py — the in-memory job store for async video-analysis jobs.

WHY IN-PROCESS / IN-MEMORY, NOT CELERY/RAY: this is a single-developer-machine
(M2, no GPU) serving layer with one worker process. FastAPI's own `BackgroundTasks`
already solves the actual problem here -- "don't block the HTTP response while
cv_pipeline grinds through frames" -- by running the task in a threadpool via
anyio, so the event loop stays free to answer `GET /jobs/{job_id}` polls while a
video is processing. A real task queue (Celery/Ray) earns its complexity when you
need: multiple worker processes/machines, task retries/persistence across process
restarts, or a job backlog that outlives the API process. None of those apply here
-- introducing one now would be solving a scaling problem this deployment doesn't
have yet. If v2_serving ever needs multi-worker horizontal scaling, this is the
first thing that would need to change (the in-memory dict doesn't survive a
process restart or work across workers) -- noted here explicitly rather than
silently assumed to scale.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from v2_serving.models import JobStatusLiteral


@dataclass
class Job:
    job_id: str
    video_path: str
    frame_limit: int
    status: JobStatusLiteral = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, video_path: str, frame_limit: int) -> Job:
        job = Job(job_id=str(uuid.uuid4()), video_path=video_path, frame_limit=frame_limit)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: JobStatusLiteral) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = status

    def set_result(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].result = result
                self._jobs[job_id].status = "complete"

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].error = error
                self._jobs[job_id].status = "failed"


# Module-level singleton -- fine for a single-process dev server (see module
# docstring on why this isn't a real queue/DB).
job_store = JobStore()
