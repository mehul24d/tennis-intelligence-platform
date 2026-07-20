"""ball_motion_diff_stress_clips.py -- same motion-diff experiment as
ball_detection_experiments.py, applied to the two stress-test clips
(data/tennis_clip.mp4, data/match_tennis.mp4), which have NO ground-truth ball
annotations (unlike the amateur dataset). Reports only a candidate-rate (does
motion-diff find something where YOLO found nothing), same "unvalidated
candidate" caveat already used for these two clips' YOLO ball numbers in
STRESS_TEST_REPORT.md / STRESS_TEST_2_REPORT.md -- not a recall number, since
there's no ground truth to check against.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ball_detection_experiments import court_mask, motion_diff_candidate, OUT_DIR

TENNIS_CLIP = Path(__file__).resolve().parents[2] / "data" / "tennis_clip.mp4"
MATCH_TENNIS = Path(__file__).resolve().parents[2] / "data" / "match_tennis.mp4"

# Corners already calibrated and visually verified in the earlier stress tests.
TENNIS_CLIP_CORNERS = {"BL": (65, 793), "BR": (1855, 780), "TL": (395, 430), "TR": (1490, 425)}
MATCH_TENNIS_CORNERS = {"BL": (218, 543), "BR": (1050, 543), "TL": (395, 218), "TR": (890, 218)}


def run(video_path: Path, corners: dict, start_frame: int, n_frames: int, label: str):
    from ultralytics import YOLO

    yolo = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    mask = None
    n_yolo_hit = 0
    n_motion_candidate_on_miss = 0
    n_total = 0
    prev_gray = None
    saved_examples = 0

    t0 = time.time()
    for offset in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        n_total += 1
        if mask is None:
            mask = court_mask(frame.shape, corners)

        res = yolo.predict(frame, classes=[32], verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().tolist()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if boxes:
            n_yolo_hit += 1
        elif prev_gray is not None:
            candidates = motion_diff_candidate(prev_gray, gray, mask)
            if candidates:
                n_motion_candidate_on_miss += 1
                if saved_examples < 4:
                    annotated = frame.copy()
                    for x, y, area in candidates[:5]:
                        cv2.circle(annotated, (int(x), int(y)), 8, (0, 255, 0), 2)
                    out = OUT_DIR / f"{label}_motioncand_offset{start_frame + offset}.jpg"
                    cv2.imwrite(str(out), annotated)
                    saved_examples += 1
        prev_gray = gray

    elapsed = time.time() - t0
    n_yolo_miss = n_total - n_yolo_hit
    print(f"\n=== {label} ({n_total} frames) ===")
    print(f"  YOLO candidate rate: {n_yolo_hit}/{n_total} ({n_yolo_hit/n_total*100:.1f}%)")
    print(f"  YOLO misses: {n_yolo_miss}")
    print(f"  motion-diff found a candidate on {n_motion_candidate_on_miss}/{n_yolo_miss} "
          f"of those misses ({n_motion_candidate_on_miss/n_yolo_miss*100:.1f}%) -- UNVALIDATED, no ground truth")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  saved {saved_examples} example frames for visual spot-check")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run(TENNIS_CLIP, TENNIS_CLIP_CORNERS, start_frame=3600, n_frames=900, label="tennis_clip")
    run(MATCH_TENNIS, MATCH_TENNIS_CORNERS, start_frame=5 * 60 * 25, n_frames=300, label="match_tennis_wideshot")
