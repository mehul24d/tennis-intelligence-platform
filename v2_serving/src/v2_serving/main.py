"""main.py — Phase 4 FastAPI app: orchestrates rag_engine, llm_agent, cv_pipeline,
and v1's win-probability engine behind one API. This module only wires things
together -- no analysis logic lives here, and nothing in the three existing
components (rag_engine/, llm_agent/, cv_pipeline/) or v1's backend is modified by
this phase.

CORS: added in Phase 5 (v2_dashboard) -- the dashboard is a Vite dev server on
localhost:5173, a different origin than this API (127.0.0.1:8734), so browser
fetches would otherwise be blocked regardless of the API itself working (curl
never hits this restriction, only real browser requests do -- worth noting
explicitly since it's exactly the kind of thing that looks like a backend bug
from the frontend side but isn't one). Scoped to localhost dev origins only.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from v2_serving.routers import jobs, media, query, render, win_probability

app = FastAPI(
    title="Tennis Intelligence Platform v2 API",
    description="Phase 4 serving layer: video analysis, RAG+LLM tactical Q&A, and win-probability, unified.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(query.router)
app.include_router(win_probability.router)
app.include_router(media.router)
app.include_router(render.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
