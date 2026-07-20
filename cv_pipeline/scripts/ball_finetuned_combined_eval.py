"""ball_finetuned_combined_eval.py -- tests two fixes on top of the fine-tuned
ball model, against the amateur dataset's REAL ground truth only (per
instruction: validate against real numbers before touching the unvalidated
stress clips):

1. STATIC-DETECTION REJECTION: the fine-tuned model was found (via visual
   spot-check on the stress clips) to hallucinate a persistent, near-zero-
   confidence-change detection at a FIXED pixel location for 15+ consecutive
   frames -- almost certainly a memorized court blemish, not the ball. A real
   ball cannot be static for a sustained window (it's either being struck,
   in flight, or briefly at rest for well under this window at this frame
   rate). Rule: if the fine-tuned model's top detection stays within
   STATIC_RADIUS_PX of its own position for STATIC_WINDOW consecutive frames,
   reject it as a static artifact (this frame's fine-tuned prediction counts
   as "no detection", not as a hit).
2. COMBINED (OR) with motion-diff: reuses the already-validated
   court-region-masked frame-differencing approach from
   ball_detection_experiments.py (57.62% pooled recall alone on this same
   dataset). A frame counts as detected if EITHER the (static-filtered)
   fine-tuned YOLO detection OR the motion-diff candidate matches ground
   truth within the project's existing 100px threshold.

Reuses court_mask/motion_diff_candidate directly from
ball_detection_experiments.py rather than reimplementing them.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ball_detection_experiments import court_mask, motion_diff_candidate
from ball_finetuned_eval import MODEL_PATH, AMATEUR_CLIPS
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations
from cv_pipeline.ball_detection import MAX_BALL_MATCH_DISTANCE_PX

STATIC_RADIUS_PX = 3.0
STATIC_WINDOW = 10


def run_clip(model, clip_name: str, video_path: Path):
    ann = load_clip_annotations(clip_name)
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    recent_positions = deque(maxlen=STATIC_WINDOW)
    prev_gray = None
    mask_cache = {}

    n_gt = 0
    n_hit_yolo_raw = 0       # fine-tuned YOLO, no static filter
    n_hit_yolo_filtered = 0  # fine-tuned YOLO, static-rejected
    n_hit_combined = 0       # static-filtered YOLO OR motion-diff

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        fa = ann.get(frame_idx)
        gt = fa.ball if (fa and not fa.ball_is_sentinel and not fa.ball_row_missing) else None

        results = model.predict(frame, verbose=False, conf=0.25)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        confs = results[0].boxes.conf.cpu().numpy().tolist() if len(results) else []

        top_center = None
        if boxes:
            best_i = int(np.argmax(confs))
            b = boxes[best_i]
            top_center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

        # static-artifact rejection
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_center = None
        if prev_gray is not None and fa and fa.court_corners:
            key = tuple(sorted(fa.court_corners.items()))
            if key not in mask_cache:
                mask_cache[key] = court_mask(frame.shape, fa.court_corners)
            mask = mask_cache[key]
            # Only bother running motion-diff on frames where the filtered
            # YOLO signal is absent -- matches how motion-diff was used
            # earlier (a miss-recovery step, not a competing primary signal).
            if yolo_filtered_center is None:
                candidates = motion_diff_candidate(prev_gray, gray, mask)
                if gt is not None and candidates:
                    dists = [np.hypot(gt[0] - c[0], gt[1] - c[1]) for c in candidates]
                    bi = int(np.argmin(dists))
                    if dists[bi] <= MAX_BALL_MATCH_DISTANCE_PX:
                        motion_center = (candidates[bi][0], candidates[bi][1])
        prev_gray = gray

        if gt is not None:
            n_gt += 1
            if top_center is not None and np.hypot(gt[0] - top_center[0], gt[1] - top_center[1]) <= MAX_BALL_MATCH_DISTANCE_PX:
                n_hit_yolo_raw += 1
            yolo_filtered_hit = (yolo_filtered_center is not None
                                  and np.hypot(gt[0] - yolo_filtered_center[0], gt[1] - yolo_filtered_center[1]) <= MAX_BALL_MATCH_DISTANCE_PX)
            if yolo_filtered_hit:
                n_hit_yolo_filtered += 1
            if yolo_filtered_hit or motion_center is not None:
                n_hit_combined += 1

    return {
        "clip": clip_name, "n_gt": n_gt,
        "yolo_raw": n_hit_yolo_raw, "yolo_filtered": n_hit_yolo_filtered, "combined": n_hit_combined,
    }


def main():
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

    results = []
    for clip in AMATEUR_CLIPS:
        r = run_clip(model, clip, DEFAULT_VIDEOS_DIR / f"{clip}.mp4")
        rate = lambda n: n / r["n_gt"] * 100 if r["n_gt"] else 0.0
        print(f"{clip}: raw={r['yolo_raw']}/{r['n_gt']} ({rate(r['yolo_raw']):.1f}%)  "
              f"static-filtered={r['yolo_filtered']}/{r['n_gt']} ({rate(r['yolo_filtered']):.1f}%)  "
              f"combined={r['combined']}/{r['n_gt']} ({rate(r['combined']):.1f}%)")
        results.append(r)

    tot_gt = sum(r["n_gt"] for r in results)
    tot_raw = sum(r["yolo_raw"] for r in results)
    tot_filtered = sum(r["yolo_filtered"] for r in results)
    tot_combined = sum(r["combined"] for r in results)
    print(f"\nPOOLED ({tot_gt} gt frames):")
    print(f"  fine-tuned YOLO raw:            {tot_raw}/{tot_gt} = {tot_raw/tot_gt*100:.2f}%")
    print(f"  fine-tuned YOLO static-filtered: {tot_filtered}/{tot_gt} = {tot_filtered/tot_gt*100:.2f}%")
    print(f"  combined (filtered YOLO OR motion-diff): {tot_combined}/{tot_gt} = {tot_combined/tot_gt*100:.2f}%")
    print(f"\n  (reference: stock-YOLO baseline 7.81%, motion-diff-alone 57.62%, fine-tuned-alone (earlier, unfiltered) 47.06%)")


if __name__ == "__main__":
    main()
