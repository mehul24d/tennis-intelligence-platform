"""recompute_hard_moments_top2.py — corrected hard-moment (crossing/occlusion) proxy.

FIX (2026-07-15): the original find_hard_moment_frames() in
run_tracking_validation[_all_clips].py flagged a frame as "hard" if ANY 2 detected
person-boxes were close together -- this is contaminated by background people
(spectators, officials, ball kids) in broadcast-style clips with visible crowds/stands
(confirmed visually on video8: a PlaySight broadcast angle with people in the stands
and sideline). video4 and video8 both dropped from 819/998 and 513/513 "hard moment"
frames to EXACTLY ZERO once restricted to the top-2-highest-confidence boxes per frame
(a reasonable proxy for "the two actual players", who are typically larger/more
prominent/higher-confidence than distant background people). This script recomputes
hard-moment counts for all 10 clips with the corrected proxy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations

CLIPS = [f"video{i}" for i in range(1, 11)]
HARD_MOMENT_PROXIMITY_PX = 200.0


def find_hard_moments_top2(model, clip: str, n_frames: int) -> set[int]:
    cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{clip}.mp4"))
    hard = set()
    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        confs = results[0].boxes.conf.cpu().numpy().tolist()
        if len(boxes) < 2:
            continue
        order = np.argsort(confs)[::-1][:2]
        top2 = [boxes[i] for i in order]
        c1 = ((top2[0][0] + top2[0][2]) / 2, top2[0][3])
        c2 = ((top2[1][0] + top2[1][2]) / 2, top2[1][3])
        d = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
        if d <= HARD_MOMENT_PROXIMITY_PX:
            hard.add(idx)
    return hard


def main():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    print(f"{'clip':<9} {'n_frames':>9} {'hard_moments (corrected)':>26}")
    for clip in CLIPS:
        ann = load_clip_annotations(clip)
        hard = find_hard_moments_top2(model, clip, len(ann))
        print(f"{clip:<9} {len(ann):>9} {len(hard):>26}")


if __name__ == "__main__":
    main()
