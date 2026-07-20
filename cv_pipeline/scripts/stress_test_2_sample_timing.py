"""stress_test_2_sample_timing.py -- Stress Test #2, Step 1 ONLY.

Times the existing frame-extraction step (cv2.VideoCapture + sequential cap.read(),
same pattern as stress_test_pro_clip.py) plus the same YOLOv8n person/ball detection
call used there, on a 60-90s sample of data/match_tennis.mp4 (a new ~35.6min, 25fps,
1280x720 professional highlight reel -- confirmed via cv2 metadata check, distinct
from the already-used data/tennis_clip.mp4).

Purpose: measure real per-frame cost (extraction alone, and extraction+detection) and
extrapolate the cost of processing the full file, BEFORE deciding how much of it to
actually process for the rest of Stress Test #2. Also captures every Nth frame as a
JPEG so the cut-heavy/highlight-reel question (many hard cuts between clipped points)
can be visually confirmed afterward.

Does not touch Phase 3 pipeline code -- read-only characterization.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VIDEO_PATH = Path(__file__).resolve().parents[2] / "data" / "match_tennis.mp4"
OUT_DIR = Path(__file__).resolve().parents[1] / "scratch_output" / "stress_test_2"

FPS = 25
START_SECONDS = 5 * 60  # start 5 min in, past any intro/graphics
SAMPLE_SECONDS = 75  # within the requested 60-90s window
START_FRAME = START_SECONDS * FPS
N_FRAMES = SAMPLE_SECONDS * FPS

CUT_SAMPLE_STRIDE = 5  # save every 5th frame + compute hist diff for cut detection


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video: {VIDEO_PATH.name}  total_frames={total_frames:.0f}  fps={fps}  "
          f"duration_min={total_frames/fps/60:.2f}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    # --- Phase A: pure extraction timing (decode + read only) ---
    t0 = time.time()
    frames = []
    for offset in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            print(f"  ran out of frames at offset {offset}")
            break
        frames.append(frame)
    extract_elapsed = time.time() - t0
    n_extracted = len(frames)
    print(f"\n=== Phase A: extraction only ===")
    print(f"  extracted {n_extracted} frames in {extract_elapsed:.1f}s "
          f"({extract_elapsed/n_extracted*1000:.1f}ms/frame, {n_extracted/extract_elapsed:.1f} fps)")

    # --- cut detection: grayscale histogram correlation between consecutive sampled frames ---
    print(f"\n=== Cut detection (every {CUT_SAMPLE_STRIDE}th frame, hist correlation) ===")
    prev_hist = None
    cut_candidates = []
    saved = 0
    for i in range(0, n_extracted, CUT_SAMPLE_STRIDE):
        frame = frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < 0.7:
                cut_candidates.append((START_FRAME + i, corr))
        prev_hist = hist
        if saved < 12:
            out_path = OUT_DIR / f"sample_offset{i}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
    print(f"  saved {saved} sample frames to {OUT_DIR}")
    print(f"  candidate hard cuts (hist correlation < 0.7) in {SAMPLE_SECONDS}s sample: {len(cut_candidates)}")
    for fidx, corr in cut_candidates[:20]:
        print(f"    frame {fidx} (t={fidx/fps:.1f}s): correlation={corr:.3f}")

    # --- Phase B: extraction + YOLOv8n person/ball detection (same call as stress_test_pro_clip.py) ---
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")

    detect_sample = frames[: min(300, n_extracted)]  # 12s worth, enough for a stable per-frame estimate
    t0 = time.time()
    for frame in detect_sample:
        _ = yolo.predict(frame, classes=[0, 32], verbose=False)
    detect_elapsed = time.time() - t0
    n_detect = len(detect_sample)
    print(f"\n=== Phase B: extraction (already done) + YOLOv8n person+ball detection ===")
    print(f"  ran detection on {n_detect} frames in {detect_elapsed:.1f}s "
          f"({detect_elapsed/n_detect*1000:.1f}ms/frame, {n_detect/detect_elapsed:.1f} fps)")

    # --- Extrapolation to full file ---
    full_duration_s = total_frames / fps
    extract_fps_rate = n_extracted / extract_elapsed
    detect_fps_rate = n_detect / detect_elapsed
    print(f"\n=== Extrapolation to full file ({full_duration_s/60:.1f} min, {total_frames:.0f} frames) ===")
    print(f"  extraction-only: {total_frames/extract_fps_rate/60:.1f} min")
    print(f"  extraction+detection (person+ball only, no tracking/pose/homography): "
          f"{total_frames/detect_fps_rate/60:.1f} min")


if __name__ == "__main__":
    main()
