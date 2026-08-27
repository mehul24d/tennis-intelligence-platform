"""main_public.py — the LIVE, PUBLIC deployment entrypoint. A separate app from
main.py, not a config flag on it: main.py wires up all five routers (jobs, query,
win_probability, media, render), three of which (jobs, win_probability, media)
exist specifically for the video-upload/CV-analysis feature and transitively
import cv_pipeline -> ultralytics/torch/mediapipe. Measured directly (see v1's
own Render deployment history) that stack is far too heavy for a free-tier host —
frame-by-frame YOLO+pose inference would OOM or timeout on any real request, not
just an edge case. Rather than deploy that and have it fail, or gate it behind a
runtime flag that still pays the import cost, this app only ever imports the
`query` router — RAG+LLM tactical Q&A, which query_pipeline.py's own docstring
confirms has no cv_pipeline import anywhere in its call chain. Video analysis
stays a real, working, LOCAL-only feature (run `uvicorn v2_serving.main:app`) —
just not part of the public site.

CORS: reads ALLOWED_ORIGINS (comma-separated) the same way v1's api/main.py
does, for the same reason — the deployed frontend's origin isn't known at
write-time, only at deploy-time.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from v2_serving.routers import query

app = FastAPI(
    title="Tennis Intelligence Platform v2 API (public)",
    description="Public serving layer: RAG+LLM tactical Q&A only. Video analysis "
                "is a local-only feature not exposed here — see this module's "
                "docstring for why.",
    version="0.1.0",
)

_default_origins = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
