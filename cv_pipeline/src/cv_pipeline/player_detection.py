"""player_detection.py — runs pre-trained YOLOv8 (person class, inference-only, CPU)
on video frames and compares detected bounding-box centers against the ground-truth
player_r/player_l pixel points.

MATCHING: ground truth is 2 labeled points per frame (player_r, player_l), YOLO can
return 0, 1, 2, or more person detections per frame (false positives -- ball kids,
umpires, spectators near the fence are all "person" too). Greedy nearest-neighbor
matching: for each ground-truth point, find the closest unclaimed YOLO detection
point; a match only counts if within MAX_MATCH_DISTANCE_PX (else the ground-truth
point is scored as "not detected" rather than forced onto a distant, wrong box).

DETECTION POINT CONVENTION: ground truth is a foot/ground-contact point, not a body
center -- confirmed directly (not assumed) by comparing a real YOLO box against
video1 frame_400's ground truth: box BOTTOM-CENTER landed 28px from ground truth,
while box CENTER (mid-body) landed 179px away, a spurious "error" that's really just
a convention mismatch. box_center() below returns bottom-center accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_MATCH_DISTANCE_PX = 150.0  # generous, but rules out matching to a clearly-wrong person
PERSON_CLASS_ID = 0  # COCO class id for "person"


@dataclass(frozen=True)
class FrameDetectionResult:
    frame_index: int
    yolo_boxes: list[tuple[float, float, float, float]]  # x1,y1,x2,y2, all detections this frame
    player_r_error_px: float | None  # None if player_r had no ground truth OR no match found
    player_l_error_px: float | None
    n_yolo_detections: int


def box_center(box) -> tuple[float, float]:
    """Returns the box's bottom-center (foot position), not the geometric center --
    see module docstring."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2


def match_ground_truth_to_detections(
    gt_points: dict[str, tuple[float, float]], detection_centers: list[tuple[float, float]]
) -> dict[str, float | None]:
    """Greedy nearest-neighbor, largest-distance-first-excluded: each ground-truth
    point claims its closest still-available detection, if within
    MAX_MATCH_DISTANCE_PX. Returns gt_key -> matched error in px, or None if
    unmatched. Deterministic given the fixed gt_points dict iteration order (Python
    3.7+ dicts preserve insertion order)."""
    available = list(range(len(detection_centers)))
    out: dict[str, float | None] = {}
    for gt_key, gt_pt in gt_points.items():
        if not available:
            out[gt_key] = None
            continue
        dists = [np.hypot(gt_pt[0] - detection_centers[i][0], gt_pt[1] - detection_centers[i][1])
                 for i in available]
        best_idx = int(np.argmin(dists))
        best_dist = dists[best_idx]
        if best_dist <= MAX_MATCH_DISTANCE_PX:
            out[gt_key] = float(best_dist)
            available.pop(best_idx)
        else:
            out[gt_key] = None
    return out


def run_frame_detection(model, frame, frame_index: int, ann) -> FrameDetectionResult:
    """ann: a cv_pipeline.annotations.FrameAnnotation for this frame."""
    results = model.predict(frame, classes=[PERSON_CLASS_ID], verbose=False)
    boxes = [tuple(b) for b in results[0].boxes.xyxy.cpu().numpy().tolist()] if len(results) else []
    centers = [box_center(b) for b in boxes]

    gt_points = {}
    if ann.player_r is not None:
        gt_points["player_r"] = ann.player_r
    if ann.player_l is not None:
        gt_points["player_l"] = ann.player_l

    matches = match_ground_truth_to_detections(gt_points, centers)

    return FrameDetectionResult(
        frame_index=frame_index,
        yolo_boxes=boxes,
        player_r_error_px=matches.get("player_r"),
        player_l_error_px=matches.get("player_l"),
        n_yolo_detections=len(boxes),
    )
