"""run_tracking_validation_all_clips.py — Step 6 across all 10 clips: same hard-moment
+ ID-consistency check as run_tracking_validation.py (video1), generalized. Reports,
per clip, how many hard-moment (crossing/close-proximity) frames actually occurred --
a clip with zero hard moments provides NO real evidence about ID-swap behavior under
crossing/occlusion, regardless of how "clean" its segment count looks.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations
from cv_pipeline.tracking import run_clip_tracking, box_bottom_center

CLIPS = [f"video{i}" for i in range(1, 11)]
HARD_MOMENT_PROXIMITY_PX = 200.0
NEAR_SWAP_WINDOW = 20


def find_hard_moment_frames(model, clip: str, n_frames: int) -> set[int]:
    cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{clip}.mp4"))
    hard_frames = set()
    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        if len(boxes) < 2:
            continue
        centers = [box_bottom_center(b) for b in boxes]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                if d <= HARD_MOMENT_PROXIMITY_PX:
                    hard_frames.add(idx)
    return hard_frames


def main():
    from ultralytics import YOLO

    reports = []
    for clip in CLIPS:
        annotations = load_clip_annotations(clip)
        n_frames = len(annotations)
        t0 = time.time()

        detect_model = YOLO("yolov8n.pt")
        hard_frames = find_hard_moment_frames(detect_model, clip, n_frames)

        track_model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{clip}.mp4"))
        result = run_clip_tracking(track_model, clip, annotations, cap)

        all_transitions = result.player_r_swap_frames + result.player_l_swap_frames
        n_near_hard = sum(
            1 for t in all_transitions if any(abs(t - h) <= NEAR_SWAP_WINDOW for h in hard_frames)
        )
        elapsed = time.time() - t0

        reports.append({
            "clip": clip, "n_frames": n_frames, "elapsed_s": elapsed,
            "n_hard_moments": len(hard_frames),
            "player_r_segments": result.player_r_n_segments,
            "player_l_segments": result.player_l_n_segments,
            "n_transitions": len(all_transitions),
            "n_transitions_near_hard": n_near_hard,
        })
        print(f"{clip}: {n_frames} frames, {elapsed:.0f}s | hard_moments={len(hard_frames)} | "
              f"player_r_segments={result.player_r_n_segments} | player_l_segments={result.player_l_n_segments} | "
              f"transitions={len(all_transitions)} (near hard moment: {n_near_hard})")

    print("\n=== SUMMARY ===")
    n_zero_hard = sum(1 for r in reports if r["n_hard_moments"] == 0)
    n_any_swap = sum(1 for r in reports if r["n_transitions"] > 0)
    total_hard = sum(r["n_hard_moments"] for r in reports)
    print(f"  clips with ZERO hard-moment frames (no real crossing/occlusion test occurred): {n_zero_hard}/10")
    print(f"  clips with at least one detected ID swap: {n_any_swap}/10")
    print(f"  total hard-moment frames across all 10 clips: {total_hard}")


if __name__ == "__main__":
    main()
