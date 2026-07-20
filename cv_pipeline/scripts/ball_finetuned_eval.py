"""ball_finetuned_eval.py -- validates the fine-tuned tennis-ball YOLOv8n model
(30 epochs on Viren Dhanwani's CC-BY-4.0-licensed 578-image Roboflow dataset,
cv_pipeline/scratch_output/ball_finetune/full_30ep/weights/best.pt) against the
exact same clips and ground truth already used for the stock-YOLO baseline
(ball_detection_experiments.py) and the stock-YOLO candidate-rate check on the
two stress-test clips (ball_motion_diff_stress_clips.py) -- a direct,
apples-to-apples comparison, not a new methodology.

The fine-tuned model has a single class ("tennis ball" at index 0), unlike stock
YOLOv8n's 80-class COCO checkpoint (where "sports ball" is class 32) -- so this
script calls model.predict(frame, verbose=False) with no class filter, since
there's only one class to detect. Everything else (match-distance threshold,
sentinel filtering, ground-truth loading) is imported directly from the existing
modules -- not reimplemented.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations
from cv_pipeline.ball_detection import MAX_BALL_MATCH_DISTANCE_PX, box_center

# ultralytics prepended its own "runs/detect/" root to the relative project path
# passed to model.train() -- actual location confirmed via `find` before use.
MODEL_PATH = (Path(__file__).resolve().parents[2] / "runs" / "detect" / "cv_pipeline"
              / "scratch_output" / "ball_finetune" / "full_30ep" / "weights" / "best.pt")
AMATEUR_CLIPS = ["video1", "video2", "video4", "video5", "video6", "video7", "video8", "video9", "video10"]

TENNIS_CLIP = Path(__file__).resolve().parents[2] / "data" / "tennis_clip.mp4"
MATCH_TENNIS = Path(__file__).resolve().parents[2] / "data" / "match_tennis.mp4"


def eval_amateur_clip(model, clip_name: str, video_path: Path):
    ann = load_clip_annotations(clip_name)
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_gt = n_hit = 0
    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        fa = ann.get(frame_idx)
        gt = fa.ball if (fa and not fa.ball_is_sentinel and not fa.ball_row_missing) else None
        if gt is None:
            continue
        n_gt += 1
        results = model.predict(frame, verbose=False, conf=0.25)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        if not boxes:
            continue
        centers = [box_center(b) for b in boxes]
        dists = [np.hypot(gt[0] - c[0], gt[1] - c[1]) for c in centers]
        if min(dists) <= MAX_BALL_MATCH_DISTANCE_PX:
            n_hit += 1

    return n_gt, n_hit


def eval_stress_clip(model, video_path: Path, start_frame: int, n_frames: int, label: str):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n_hit = n_total = 0
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        n_total += 1
        results = model.predict(frame, verbose=False, conf=0.25)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        if boxes:
            n_hit += 1
    print(f"  {label}: candidate rate {n_hit}/{n_total} ({n_hit/n_total*100:.1f}%) -- UNVALIDATED, no ground truth")


def main():
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))

    print("=== Amateur dataset (real ground-truth recall, same clips/threshold as stock-YOLO baseline) ===")
    total_gt = total_hit = 0
    t0 = time.time()
    for clip in AMATEUR_CLIPS:
        video_path = DEFAULT_VIDEOS_DIR / f"{clip}.mp4"
        n_gt, n_hit = eval_amateur_clip(model, clip, video_path)
        rate = n_hit / n_gt * 100 if n_gt else 0.0
        print(f"  {clip}: {n_hit}/{n_gt} ({rate:.1f}%)")
        total_gt += n_gt
        total_hit += n_hit
    elapsed = time.time() - t0
    print(f"\n  POOLED: {total_hit}/{total_gt} = {total_hit/total_gt*100:.2f}% "
          f"(stock-YOLO baseline was 7.81%, motion-diff was 57.62%)")
    print(f"  elapsed: {elapsed:.1f}s")

    print("\n=== Stress-test clips (candidate rate only, no ground truth -- same caveat as before) ===")
    eval_stress_clip(model, TENNIS_CLIP, start_frame=3600, n_frames=900, label="tennis_clip (stock-YOLO baseline: 14.3%)")
    eval_stress_clip(model, MATCH_TENNIS, start_frame=5 * 60 * 25, n_frames=300, label="match_tennis wide-shot (stock-YOLO baseline: 18.0%)")


if __name__ == "__main__":
    main()
