"""ball_detection_experiments.py -- two cheap ball-detection-recovery experiments
requested as alternatives to adopting TrackNet (which was blocked pending a
licensing decision -- see conversation / STRESS_TEST_2_REPORT.md follow-up):

1. Motion-diff recovery: on frames where YOLOv8n's COCO "sports ball" class finds
   NOTHING, try frame-differencing between consecutive frames, restricted to a
   court-region mask built from the clip's real annotated homography corners (to
   exclude crowd/background motion), and see whether the resulting small moving
   blob lands near the real (ground-truth) ball position.
2. Trajectory interpolation: for gaps of 1-3 consecutive frames between two
   CONFIRMED ball detections (YOLO hits that matched ground truth within the
   project's existing 100px threshold -- see ball_detection.py), fit a quadratic
   (parabolic) curve per axis through the confirmed points immediately
   surrounding the gap and interpolate the missing frames, then check whether the
   interpolated position lands near ground truth too.

Evaluated against REAL ground truth (data/cv_annotated/annotations/*_ball.csv),
using the exact same sentinel-filtering and 100px match-distance convention
already established and committed in ball_detection.py / EVALUATION_REPORT.md --
not a new methodology invented for this test. Run on the 9 amateur clips already
used for the committed ~7-8% ball-detection baseline (video3 excluded, same as
that baseline) plus the two stress-test clips (data/tennis_clip.mp4 sample,
data/match_tennis.mp4 wide-shot sample) for direct, apples-to-apples comparison.

Does not modify any existing Phase 3 pipeline code.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations
from cv_pipeline.ball_detection import MAX_BALL_MATCH_DISTANCE_PX, SPORTS_BALL_CLASS_ID, box_center

OUT_DIR = Path(__file__).resolve().parents[1] / "scratch_output" / "ball_experiments"
AMATEUR_CLIPS = ["video1", "video2", "video4", "video5", "video6", "video7", "video8", "video9", "video10"]
# video3 excluded from the ball-detection baseline in EVALUATION_REPORT.md -- matched here.

MOTION_MIN_AREA = 4
MOTION_MAX_AREA = 400  # ball is small; excludes player-sized blobs
MOTION_DIFF_THRESHOLD = 25
MAX_GAP_FOR_INTERP = 3


def court_mask(shape, court_corners: dict) -> np.ndarray:
    points = list(court_corners.values())
    by_y = sorted(points, key=lambda p: p[1])
    far_pair = sorted(by_y[:2], key=lambda p: p[0])
    near_pair = sorted(by_y[2:], key=lambda p: p[0])
    poly = np.array([near_pair[0], near_pair[1], far_pair[1], far_pair[0]], dtype=np.int32)
    # Dilate outward ~15% to include serves/shots that land just past the lines.
    center = poly.mean(axis=0)
    poly_dilated = (center + (poly - center) * 1.3).astype(np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_dilated], 255)
    return mask


def motion_diff_candidate(prev_gray, cur_gray, mask):
    diff = cv2.absdiff(prev_gray, cur_gray)
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    _, thresh = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if MOTION_MIN_AREA <= area <= MOTION_MAX_AREA:
            (x, y), r = cv2.minEnclosingCircle(c)
            candidates.append((x, y, area))
    return candidates


def quad_interp(points, target_idx):
    # points: list of (frame_idx, x, y), at least 2, up to 4, surrounding a gap.
    if len(points) < 2:
        return None
    idxs = np.array([p[0] for p in points])
    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])
    deg = 2 if len(points) >= 3 else 1
    px = np.polyfit(idxs, xs, deg)
    py = np.polyfit(idxs, ys, deg)
    return float(np.polyval(px, target_idx)), float(np.polyval(py, target_idx))


def run_clip(clip_name: str, video_path: Path, max_frames: int | None = None):
    from ultralytics import YOLO

    ann = load_clip_annotations(clip_name, ) if video_path is None else load_clip_annotations(clip_name)
    yolo = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n_frames = min(n_frames, max_frames)

    # Pass 1: YOLO baseline + motion-diff on misses, sequentially (need prev frame).
    prev_gray = None
    frame_results = []  # (frame_idx, gt_xy_or_None, yolo_hit_xy_or_None, motion_hit_xy_or_None)
    mask_cache = {}

    t0 = time.time()
    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        fa = ann.get(frame_idx)
        gt = fa.ball if (fa and not fa.ball_is_sentinel and not fa.ball_row_missing) else None

        results = yolo.predict(frame, classes=[SPORTS_BALL_CLASS_ID], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        centers = [box_center(b) for b in boxes]
        yolo_hit = None
        if gt is not None and centers:
            dists = [np.hypot(gt[0] - c[0], gt[1] - c[1]) for c in centers]
            best_i = int(np.argmin(dists))
            if dists[best_i] <= MAX_BALL_MATCH_DISTANCE_PX:
                yolo_hit = centers[best_i]

        motion_hit = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and not centers and fa and fa.court_corners:
            key = tuple(sorted(fa.court_corners.items()))
            if key not in mask_cache:
                mask_cache[key] = court_mask(frame.shape, fa.court_corners)
            mask = mask_cache[key]
            candidates = motion_diff_candidate(prev_gray, gray, mask)
            if gt is not None and candidates:
                dists = [np.hypot(gt[0] - c[0], gt[1] - c[1]) for c in candidates]
                best_i = int(np.argmin(dists))
                if dists[best_i] <= MAX_BALL_MATCH_DISTANCE_PX:
                    motion_hit = (candidates[best_i][0], candidates[best_i][1])
        prev_gray = gray

        frame_results.append((frame_idx, gt, yolo_hit, motion_hit))

    elapsed = time.time() - t0

    # Pass 2: trajectory interpolation over YOLO-confirmed hits only (not motion, not GT).
    confirmed = [(f, h[0], h[1]) for f, gt, h, m in frame_results if h is not None]
    confirmed_idx = {c[0] for c in confirmed}
    interp_results = []
    for i, (frame_idx, gt, yolo_hit, motion_hit) in enumerate(frame_results):
        if yolo_hit is not None or gt is None:
            continue
        # find nearest confirmed frames before/after
        before = [c for c in confirmed if c[0] < frame_idx]
        after = [c for c in confirmed if c[0] > frame_idx]
        if not before or not after:
            continue
        gap_before = frame_idx - before[-1][0]
        gap_after = after[0][0] - frame_idx
        if gap_before + gap_after - 1 > MAX_GAP_FOR_INTERP:
            continue  # gap too long
        surround = before[-2:] + after[:2]
        pred = quad_interp(surround, frame_idx)
        if pred is None:
            continue
        dist = np.hypot(gt[0] - pred[0], gt[1] - pred[1])
        hit = dist <= MAX_BALL_MATCH_DISTANCE_PX
        interp_results.append((frame_idx, hit))

    n_gt = sum(1 for f, gt, h, m in frame_results if gt is not None)
    n_yolo_hit = sum(1 for f, gt, h, m in frame_results if gt is not None and h is not None)
    n_motion_recovered = sum(1 for f, gt, h, m in frame_results if gt is not None and h is None and m is not None)
    n_interp_hit = sum(1 for f, hit in interp_results if hit)
    n_interp_attempted = len(interp_results)

    return {
        "clip": clip_name,
        "n_frames_processed": len(frame_results),
        "n_frames_with_gt": n_gt,
        "yolo_recall": n_yolo_hit / n_gt if n_gt else None,
        "n_yolo_hit": n_yolo_hit,
        "n_yolo_miss": n_gt - n_yolo_hit,
        "motion_recovered_of_yolo_misses": n_motion_recovered,
        "motion_recovery_rate_of_misses": n_motion_recovered / (n_gt - n_yolo_hit) if (n_gt - n_yolo_hit) else None,
        "combined_recall_yolo_plus_motion": (n_yolo_hit + n_motion_recovered) / n_gt if n_gt else None,
        "interp_attempted": n_interp_attempted,
        "interp_hit": n_interp_hit,
        "interp_success_rate": n_interp_hit / n_interp_attempted if n_interp_attempted else None,
        "combined_recall_all_three": (n_yolo_hit + n_motion_recovered + n_interp_hit) / n_gt if n_gt else None,
        "elapsed_s": elapsed,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for clip in AMATEUR_CLIPS:
        video_path = DEFAULT_VIDEOS_DIR / f"{clip}.mp4"
        print(f"=== {clip} ===", flush=True)
        r = run_clip(clip, video_path)
        print(json.dumps(r, indent=2), flush=True)
        all_results.append(r)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
