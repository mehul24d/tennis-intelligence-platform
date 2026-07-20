"""ball_detection.py — ball detection via YOLOv8's built-in COCO "sports ball" class
(id=32). Tried first because it's zero-extra-cost (same pretrained model already used
for players, no custom training/download) -- per the plan's own framing ("propose an
approach given that generic YOLO often misses small fast-moving balls"), this is the
cheapest option to try before reaching for a color/motion detector or a specialized
model. Its accuracy (or lack thereof) is exactly what this step's report measures --
not assumed good or bad ahead of time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_BALL_MATCH_DISTANCE_PX = 100.0  # tighter than the player threshold -- the ball
# ground truth is a genuine point (not a foot-position proxy), and a "close" false
# match matters more for a small fast object.
SPORTS_BALL_CLASS_ID = 32  # COCO class id


@dataclass(frozen=True)
class BallDetectionResult:
    frame_index: int
    n_candidates: int
    error_px: float | None  # None if no ground truth this frame, or no match within threshold


def box_center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def run_ball_frame_detection(model, frame, frame_index: int, ann) -> BallDetectionResult:
    """ann: a cv_pipeline.annotations.FrameAnnotation. ann.ball is already None for
    sentinel/missing-ground-truth frames (see annotations.py), so this function simply
    reports error_px=None for those without needing its own sentinel logic."""
    results = model.predict(frame, classes=[SPORTS_BALL_CLASS_ID], verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
    centers = [box_center(b) for b in boxes]

    if ann.ball is None or not centers:
        return BallDetectionResult(frame_index=frame_index, n_candidates=len(boxes), error_px=None)

    dists = [np.hypot(ann.ball[0] - c[0], ann.ball[1] - c[1]) for c in centers]
    best = min(dists)
    error_px = best if best <= MAX_BALL_MATCH_DISTANCE_PX else None
    return BallDetectionResult(frame_index=frame_index, n_candidates=len(boxes), error_px=error_px)
