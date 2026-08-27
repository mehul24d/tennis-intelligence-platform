"""
api/main.py — the FastAPI application entrypoint.

Run with:
    uvicorn api.main:app --reload --port 8000

The replay context (trained classifier + full point-level dataset) is loaded ONCE at
startup via the lifespan handler below — NOT per-request — since building the full
point-level dataset takes real time (see replay_match.py's own "one-time cost" comment)
and must not be repeated on every API call.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tennis_intel.serving.replay_service import load_replay_context
from api.routers import matches, match_list, model_comparison, research_dashboard, player_profile, rankings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading replay context (classifier + full point-level dataset)...")
    app.state.replay_context = load_replay_context()
    print("Replay context loaded. API ready.")
    yield


app = FastAPI(
    title="Tennis Intelligence Platform API",
    description="Serves match replay, model comparison, and calibration data for the "
                "Tennis Intelligence Platform frontend.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allowed origins default to local frontend dev; set ALLOWED_ORIGINS (comma-separated)
# in the deployment environment to add the deployed frontend's origin(s).
_default_origins = ["http://localhost:3000"]
_extra_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(matches.router)
app.include_router(match_list.router)
app.include_router(model_comparison.router)
app.include_router(research_dashboard.router)
app.include_router(player_profile.router)
app.include_router(rankings.router)


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}