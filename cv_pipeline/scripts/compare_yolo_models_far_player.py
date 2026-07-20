"""compare_yolo_models_far_player.py — compares yolov8n vs yolov8s specifically on
the "clean" far-player subset of video1: frames where player_l is a REAL ground-truth
position (not a corner-sentinel placeholder) and genuinely far from player_r (>=200px),
confirmed via low-confidence (conf=0.01) inspection to have NO nearby YOLO candidate at
all -- so this is a fair test of "does a bigger model see something the small one
doesn't," not contaminated by sentinel/duplicate-label artifacts.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations
from cv_pipeline.player_detection import run_frame_detection

SEPARATED_THRESHOLD_PX = 200.0


def get_clean_separated_frames(clip: str) -> list[int]:
    annotations = load_clip_annotations(clip)
    out = []
    for idx, a in annotations.items():
        if a.player_r is None or a.player_l is None:  # excludes sentinel-affected frames
            continue
        rl_dist = np.hypot(a.player_r[0] - a.player_l[0], a.player_r[1] - a.player_l[1])
        if rl_dist >= SEPARATED_THRESHOLD_PX:
            out.append(idx)
    return out


def evaluate_model(model_name: str, clip: str, frame_idxs: set[int]):
    from ultralytics import YOLO

    model = YOLO(model_name)
    annotations = load_clip_annotations(clip)
    cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{clip}.mp4"))

    errors_l, errors_r = [], []
    n_matched_l = 0
    t0 = time.time()
    n_processed = 0
    for idx in range(max(frame_idxs) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if idx not in frame_idxs:
            continue
        n_processed += 1
        ann = annotations[idx]
        result = run_frame_detection(model, frame, idx, ann)
        if result.player_l_error_px is not None:
            n_matched_l += 1
            errors_l.append(result.player_l_error_px)
        if result.player_r_error_px is not None:
            errors_r.append(result.player_r_error_px)
    elapsed = time.time() - t0

    print(f"\n=== {model_name} on {clip}, clean-separated subset (n={len(frame_idxs)}) ===")
    print(f"  player_l detection_rate={n_matched_l/len(frame_idxs):.1%} ({n_matched_l}/{len(frame_idxs)}), "
          f"median_err={np.median(errors_l) if errors_l else float('nan'):.1f}px")
    print(f"  player_r detection_rate={len(errors_r)/len(frame_idxs):.1%}, "
          f"median_err={np.median(errors_r) if errors_r else float('nan'):.1f}px")
    print(f"  speed: {n_processed/elapsed:.1f} fps ({elapsed:.1f}s for {n_processed} frames)")
    return {"model": model_name, "detection_rate_l": n_matched_l/len(frame_idxs), "fps": n_processed/elapsed}


def main():
    clip = "video1"
    clean_frames = set(get_clean_separated_frames(clip))
    print(f"Clean, non-sentinel, genuinely-separated frames in {clip}: {len(clean_frames)}")

    results = []
    for model_name in ["yolov8n.pt", "yolov8s.pt"]:
        results.append(evaluate_model(model_name, clip, clean_frames))

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['model']}: player_l detection_rate={r['detection_rate_l']:.1%}, speed={r['fps']:.1f} fps")


if __name__ == "__main__":
    main()
