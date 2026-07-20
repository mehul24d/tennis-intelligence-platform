"""render_job_store.py — in-memory job store for async video-render jobs,
same minimal single-process pattern as job_store.py (see that module's own
docstring for why an in-memory dict is the right call here, not a real
queue). Kept as its own store rather than folded into job_store.Job: a render
job's result is an output FILE PATH, not an analysis JSON, and a render job
additionally references a source analyze-video job -- different enough
shapes that reusing one dataclass for both would need type-branching
throughout, for no real benefit over two small parallel stores.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from v2_serving.models import JobStatusLiteral


@dataclass
class RenderJob:
    render_job_id: str
    source_job_id: str
    status: JobStatusLiteral = "pending"
    output_path: str | None = None
    error: str | None = None


class RenderJobStore:
    def __init__(self):
        self._jobs: dict[str, RenderJob] = {}
        self._lock = threading.Lock()

    def create(self, source_job_id: str) -> RenderJob:
        job = RenderJob(render_job_id=str(uuid.uuid4()), source_job_id=source_job_id)
        with self._lock:
            self._jobs[job.render_job_id] = job
        return job

    def get(self, render_job_id: str) -> RenderJob | None:
        with self._lock:
            return self._jobs.get(render_job_id)

    def set_status(self, render_job_id: str, status: JobStatusLiteral) -> None:
        with self._lock:
            if render_job_id in self._jobs:
                self._jobs[render_job_id].status = status

    def set_result(self, render_job_id: str, output_path: str) -> None:
        with self._lock:
            if render_job_id in self._jobs:
                self._jobs[render_job_id].output_path = output_path
                self._jobs[render_job_id].status = "complete"

    def set_failed(self, render_job_id: str, error: str) -> None:
        with self._lock:
            if render_job_id in self._jobs:
                self._jobs[render_job_id].error = error
                self._jobs[render_job_id].status = "failed"


# Module-level singleton -- same reasoning as job_store's (see its docstring).
render_job_store = RenderJobStore()
