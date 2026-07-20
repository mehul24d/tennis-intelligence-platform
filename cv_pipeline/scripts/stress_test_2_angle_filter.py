"""stress_test_2_angle_filter.py -- Stress Test #2, Step 2.

Heuristic camera-angle/shot-type filter: classifies each frame of
data/match_tennis.mp4 as 'valid' (full-court rally view, the only kind Phase 3's
pipeline was built/validated against) or 'other' (closeup/replay/graphic/crowd/
changeover), using two cheap signals calibrated against Step 1's confirmed
good/bad example frames -- no trained classifier, per the explicit preference for
a simple/heuristic approach given time constraints:

1. court-blue color fraction: % of pixels (below the top 25% crowd/stand band)
   matching the HSV range sampled from a known-good full-court frame's court
   surface. Necessary but NOT sufficient alone -- some closeups (e.g. a
   telephoto shot against a blurred blue backdrop) also score high on this.
2. long-line count: number of Hough-detected line segments >=180px long. Full-
   court broadcast views show many long straight court lines/net/sponsor-board
   edges (~50-70 in the calibration frames); closeups/graphics do not
   (single-digit to ~30 in the calibration frames), which is what actually
   separates the "blurry blue closeup" false positive from a real court view.

A frame is 'valid' only if BOTH thresholds are met. Thresholds were calibrated
directly against Step 1's already-confirmed labels (3 known-valid frames, 5
known-other frames spanning closeup/replay-graphic/changeover), not guessed.

Does not touch Phase 3 pipeline code. This filter is a pre-processing gate only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

VIDEO_PATH = Path(__file__).resolve().parents[2] / "data" / "match_tennis.mp4"
OUT_DIR = Path(__file__).resolve().parents[1] / "scratch_output" / "stress_test_2"

FPS = 25
START_SECONDS = 5 * 60
SAMPLE_SECONDS = 75
START_FRAME = START_SECONDS * FPS
N_FRAMES = SAMPLE_SECONDS * FPS

COURT_HSV_LOWER = np.array([95, 60, 90])
COURT_HSV_UPPER = np.array([122, 255, 255])
COURT_FRAC_THRESHOLD = 0.55
LINE_COUNT_THRESHOLD = 45


def classify(frame) -> tuple[bool, float, int]:
    h = frame.shape[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COURT_HSV_LOWER, COURT_HSV_UPPER)
    court_frac = mask[int(h * 0.25):, :].mean() / 255

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=180, maxLineGap=15)
    n_lines = 0 if lines is None else len(lines)

    is_valid = court_frac > COURT_FRAC_THRESHOLD and n_lines >= LINE_COUNT_THRESHOLD
    return is_valid, court_frac, n_lines


def run_on_sample():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    results = []
    for offset in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        is_valid, court_frac, n_lines = classify(frame)
        results.append((offset, is_valid, court_frac, n_lines, frame))

    n_valid = sum(1 for r in results if r[1])
    print(f"=== 75s sample ({len(results)} frames) ===")
    print(f"  valid: {n_valid} ({n_valid/len(results)*100:.1f}%)")
    print(f"  other: {len(results)-n_valid} ({(len(results)-n_valid)/len(results)*100:.1f}%)")

    # Save example frames: clear-valid, clear-other, and borderline (score near threshold)
    valid_sorted = sorted([r for r in results if r[1]], key=lambda r: -r[2])
    other_sorted = sorted([r for r in results if not r[1]], key=lambda r: r[2])
    borderline = sorted(results, key=lambda r: abs(r[2] - COURT_FRAC_THRESHOLD) + abs(r[3] - LINE_COUNT_THRESHOLD) / 100)

    examples_dir = OUT_DIR / "angle_filter_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    def save(tag, rows, n=4):
        for i, (offset, is_valid, cf, nl, frame) in enumerate(rows[:n]):
            label = "valid" if is_valid else "other"
            out = examples_dir / f"{tag}_{i}_offset{offset}_{label}_cf{cf:.2f}_nl{nl}.jpg"
            cv2.imwrite(str(out), frame)

    save("clear_valid", valid_sorted, n=4)
    save("clear_other", other_sorted, n=4)
    save("borderline", borderline, n=6)

    print(f"  saved example frames to {examples_dir}")
    return results


def run_on_full_file():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    t0 = time.time()
    n_valid = 0
    n_total = 0
    STRIDE = 5  # classify every 5th frame (5fps effective) -- filter is a pre-gate, doesn't need every frame
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % STRIDE == 0:
            is_valid, _, _ = classify(frame)
            n_total += 1
            if is_valid:
                n_valid += 1
        idx += 1
    elapsed = time.time() - t0

    print(f"\n=== Full file ({total_frames} frames, classified every {STRIDE}th = {n_total} sampled) ===")
    print(f"  valid: {n_valid} ({n_valid/n_total*100:.1f}%)")
    print(f"  other: {n_total-n_valid} ({(n_total-n_valid)/n_total*100:.1f}%)")
    print(f"  classification wall time: {elapsed:.1f}s ({elapsed/n_total*1000:.2f}ms/classified frame)")


if __name__ == "__main__":
    run_on_sample()
    run_on_full_file()
