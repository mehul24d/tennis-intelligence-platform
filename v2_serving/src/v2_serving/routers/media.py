"""routers/media.py — GET /video-file/{filename}: serves a known demo clip's raw
bytes over HTTP so the dashboard's <video> element can actually play it.

REAL GAP FOUND while building Phase 5's video player (not assumed away):
cv_pipeline's demo clips live on the local filesystem
(data/cv_annotated/videos/videoN.mp4, data/tennis_clip.mp4) -- a browser cannot
load a local filesystem path via <video src="/Users/.../video1.mp4">, only a
served HTTP URL. This endpoint closes that gap for the known demo clips only
(never an arbitrary path -- filename is matched against a fixed allow-list
built from the same two directories video_pipeline.py already knows about, so
this can't become a path-traversal read-any-file endpoint).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["media"])

REPO_ROOT = Path(__file__).resolve().parents[4]
# data/tennis/ added 2026-07-19: 3.mp4/4.mp4/5.mp4 (the "Run Validated Reference
# Pipeline End-to-End on Three New Clips" clips) only exist under this
# subdirectory, not directly under data/ like 1.mp4/2.mp4 -- their dashboard
# jobs' video-file requests were 404ing (browser <video> tag would silently
# fail to load) before this was added, even though the analysis JSON itself
# was otherwise correct. Real gap found while finishing 5.mp4's Step 5, not
# assumed away -- see PROGRESS.md.
VIDEO_DIRS = [REPO_ROOT / "data" / "cv_annotated" / "videos", REPO_ROOT / "data" / "tennis", REPO_ROOT / "data"]


def _resolve_known_clip(filename: str) -> Path:
    # basename only -- rejects any path-separator tricks outright before even
    # touching the filesystem.
    if filename != Path(filename).name:
        raise HTTPException(status_code=400, detail="filename must not contain a path")
    for directory in VIDEO_DIRS:
        candidate = directory / filename
        if candidate.is_file() and candidate.suffix == ".mp4":
            return candidate
    raise HTTPException(status_code=404, detail=f"no known demo clip named '{filename}'")


@router.get("/video-file/{filename}")
def get_video_file(filename: str):
    path = _resolve_known_clip(filename)
    return FileResponse(path, media_type="video/mp4")
