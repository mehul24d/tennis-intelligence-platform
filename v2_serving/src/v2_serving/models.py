"""models.py — Pydantic request/response models for the Phase 4 API. Kept in one
place so the shapes returned to callers are explicit and reviewable, rather than
inferred from ad hoc dicts scattered across router functions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatusLiteral = Literal["pending", "processing", "complete", "failed"]


class AnalyzeVideoRequest(BaseModel):
    """LOCAL/DEV MODE: accepts a filesystem path, not an uploaded file -- see
    video_pipeline.py's module docstring for why file-upload isn't implemented in
    this pass (a real, documented scope decision, not an oversight)."""

    video_path: str = Field(..., description="Filesystem path to a video file, local/dev use.")
    frame_limit: int = Field(
        150, ge=1, le=5000,
        description="Number of frames to process from the start of the video. Kept "
                    "modest by default so a job completes in a reasonable time on a "
                    "single M2 CPU worker -- see PROGRESS.md for measured throughput.",
    )


class AnalyzeVideoResponse(BaseModel):
    job_id: str
    status: JobStatusLiteral


class QueryRequest(BaseModel):
    job_id: str | None = Field(
        None, description="If given and the job is complete, live CV features from "
                           "that clip are fused into the agent's context. If omitted, "
                           "not found, or not yet complete, the query still runs on "
                           "RAG-retrieved historical context alone -- never blocks.",
    )
    question: str
    player: str | None = None
    opponent: str | None = None


class SourcesUsedResponse(BaseModel):
    live_features: list[str]
    retrieved_docs: list[str]


class QueryResponse(BaseModel):
    answer: str
    sources_used: SourcesUsedResponse = Field(
        ..., description="Citation audit: exactly which [L#]/[D#] tags the model "
                          "actually cited in its answer -- passed through from "
                          "llm_agent unflattened, per Phase 4's transparency requirement.",
    )
    sources_offered: dict[str, str] = Field(
        ..., description="Every source (live feature or retrieved doc) the model "
                          "was given, cited or not -- lets a caller see what was "
                          "available versus what was actually used.",
    )
    live_features_used: bool = Field(..., description="Whether a job_id resolved to usable, complete CV features.")
    live_features_note: str | None = None


class WinProbabilityResponse(BaseModel):
    job_id: str
    prematch_baseline: dict = Field(
        ..., description="v1's pre-match-only Monte Carlo/Markov baseline. Always "
                          "present in the response (even when unavailable, with a "
                          "plain reason) -- never silently omitted.",
    )
    live_adjustment: dict = Field(
        ..., description="Any in-match adjustment derived from this job's CV "
                          "features. Reports 'not_available' with a plain reason "
                          "rather than fabricating a plausible-looking number when "
                          "the job's features don't support one.",
    )


class RenderVideoRequest(BaseModel):
    job_id: str = Field(
        ..., description="An existing, COMPLETE /analyze-video job. Render draws from "
                          "its already-computed result (court lines, boxes, ball, shot "
                          "events) -- it does not re-run any detection/pose/ball model.",
    )


class RenderVideoResponse(BaseModel):
    render_job_id: str
    status: JobStatusLiteral


class RenderJobStatusResponse(BaseModel):
    render_job_id: str
    status: JobStatusLiteral
    source_job_id: str
    error: str | None = None
    output_filename: str | None = Field(
        None, description="Present only when status == 'complete'. Fetch the file itself "
                           "from GET /rendered-video/{output_filename}.",
    )


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatusLiteral
    video_path: str
    frame_limit: int
    error: str | None = None
    result: dict[str, Any] | None = Field(
        None, description="Present only when status == 'complete'. Structured "
                           "cv_pipeline-style output, Status enum values passed "
                           "through faithfully (measured/not_detected/"
                           "excluded_known_issue/not_attempted/unvalidated/etc.), "
                           "never collapsed into a flattened boolean or omitted.",
    )
