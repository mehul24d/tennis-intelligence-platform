"""ball_finetuned_combined_stress_clips.py -- applies the static-rejection filter
+ combined-with-motion-diff approach (validated on the amateur dataset's real
ground truth in ball_finetuned_combined_eval.py: 58.15% pooled recall, the best
number found so far) to the two stress-test clips.

NO GROUND TRUTH EXISTS for these two clips -- so, same as every other stress-clip
number in this project, this reports an UNVALIDATED CANDIDATE RATE, not a recall,
and is only trusted after visual spot-check of real example frames -- never taken
at face value just because the number looks good.

Specifically checks:
1. Does the static-rejection filter (reject a top detection whose position stays
   within STATIC_RADIUS_PX for STATIC_WINDOW consecutive frames) actually kill the
   two known artifacts found by direct pixel-coordinate inspection in the prior
   spot-check: (1442, 778) in tennis_clip, (412, 442) in match_tennis.
2. Manually verified example frames of the combined method's surviving
   detections, to check whether they look like genuine ball positions or new,
   different artifacts.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ball_detection_experiments import court_mask, motion_diff_candidate, OUT_DIR
from ball_finetuned_eval import MODEL_PATH, TENNIS_CLIP, MATCH_TENNIS
from ball_motion_diff_stress_clips import TENNIS_CLIP_CORNERS, MATCH_TENNIS_CORNERS

STATIC_RADIUS_PX = 3.0
STATIC_WINDOW = 10

KNOWN_ARTIFACTS = {
    "tennis_clip": (1442, 778),
    "match_tennis": (412, 442),
}
ARTIFACT_MATCH_RADIUS = 15.0


def run(model, video_path: Path, corners: dict, start_frame: int, n_frames: int, label: str):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    mask = None
    recent_positions = deque(maxlen=STATIC_WINDOW)
    prev_gray = None

    n_total = n_raw_hit = n_filtered_hit = n_combined_hit = 0
    n_artifact_seen_raw = n_artifact_seen_filtered = 0
    artifact_xy = KNOWN_ARTIFACTS[label]

    examples_dir = OUT_DIR / "finetuned_combined_stress_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for offset in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        n_total += 1
        if mask is None:
            mask = court_mask(frame.shape, corners)

        results = model.predict(frame, verbose=False, conf=0.25)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        confs = results[0].boxes.conf.cpu().numpy().tolist() if len(results) else []

        top_center = None
        if boxes:
            best_i = int(np.argmax(confs))
            b = boxes[best_i]
            top_center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            n_raw_hit += 1
            if np.hypot(top_center[0] - artifact_xy[0], top_center[1] - artifact_xy[1]) <= ARTIFACT_MATCH_RADIUS:
                n_artifact_seen_raw += 1

        is_static = False
        if top_center is not None:
            recent_positions.append(top_center)
            if len(recent_positions) == STATIC_WINDOW:
                xs = [p[0] for p in recent_positions]
                ys = [p[1] for p in recent_positions]
                spread = max(np.hypot(x - xs[0], y - ys[0]) for x, y in zip(xs, ys))
                if spread <= STATIC_RADIUS_PX:
                    is_static = True
        else:
            recent_positions.clear()

        yolo_filtered_center = None if is_static else top_center
        if yolo_filtered_center is not None:
            n_filtered_hit += 1
            if np.hypot(yolo_filtered_center[0] - artifact_xy[0], yolo_filtered_center[1] - artifact_xy[1]) <= ARTIFACT_MATCH_RADIUS:
                n_artifact_seen_filtered += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_center = None
        if prev_gray is not None and yolo_filtered_center is None:
            candidates = motion_diff_candidate(prev_gray, gray, mask)
            if candidates:
                # no ground truth -- just take the highest-area candidate as "the" motion pick
                motion_center = max(candidates, key=lambda c: c[2])[:2]
        prev_gray = gray

        combined_center = yolo_filtered_center or motion_center
        if combined_center is not None:
            n_combined_hit += 1
            if saved < 8 and offset % 15 == 0:
                annotated = frame.copy()
                cx, cy = combined_center
                source = "yolo" if yolo_filtered_center is not None else "motion"
                color = (0, 255, 0) if source == "yolo" else (255, 0, 255)
                cv2.circle(annotated, (int(cx), int(cy)), 10, color, 2)
                cv2.putText(annotated, source, (int(cx) + 12, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                out = examples_dir / f"{label}_offset{start_frame+offset}_{source}.jpg"
                cv2.imwrite(str(out), annotated)
                saved += 1

    print(f"\n=== {label} ({n_total} frames) ===")
    print(f"  raw candidate rate: {n_raw_hit}/{n_total} ({n_raw_hit/n_total*100:.1f}%) -- UNVALIDATED")
    print(f"  static-filtered candidate rate: {n_filtered_hit}/{n_total} ({n_filtered_hit/n_total*100:.1f}%) -- UNVALIDATED")
    print(f"  combined (filtered YOLO OR motion-diff) candidate rate: {n_combined_hit}/{n_total} ({n_combined_hit/n_total*100:.1f}%) -- UNVALIDATED")
    print(f"  known artifact {artifact_xy} present in RAW detections: {n_artifact_seen_raw}/{n_raw_hit if n_raw_hit else 1} of raw hits")
    print(f"  known artifact {artifact_xy} present AFTER static filter: {n_artifact_seen_filtered}/{n_filtered_hit if n_filtered_hit else 1} of filtered hits")
    print(f"  saved {saved} example frames to {examples_dir}")


if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    run(model, TENNIS_CLIP, TENNIS_CLIP_CORNERS, start_frame=3600, n_frames=900, label="tennis_clip")
    run(model, MATCH_TENNIS, MATCH_TENNIS_CORNERS, start_frame=5 * 60 * 25, n_frames=300, label="match_tennis")
