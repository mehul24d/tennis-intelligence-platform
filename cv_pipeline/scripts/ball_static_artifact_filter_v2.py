"""ball_static_artifact_filter_v2.py -- redesigned static-artifact rejection for
the fine-tuned ball model, replacing the failed "10 consecutive top-confidence
frames near-static" rule from ball_finetuned_combined_eval.py /
ball_finetuned_combined_stress_clips.py.

WHY THE FIRST DESIGN FAILED (confirmed by direct spot-check): both known
artifacts -- (1442,778) in tennis_clip, (412,442) in match_tennis -- recur
frequently across a clip (12.1% and 5.3% of ALL frames respectively) but rarely
win "top confidence" 10 frames in a row, because other detections (real ball,
other noise) are interleaved. A rule that only looks at the single top-confidence
box per frame, and only within a short consecutive run, never accumulates enough
evidence to flag them.

NEW DESIGN -- two-pass, frequency-over-full-history, ALL candidate boxes (not
just top-1):
  Pass 1: run the model over every frame, collect every candidate box's center
  (at whatever confidence >= conf threshold), regardless of whether it was the
  frame's top pick. Bin centers into a coarse grid (ARTIFACT_BIN_PX) and count
  how many DISTINCT frames each bin was hit in, across the WHOLE clip.
  Any bin hit in >= ARTIFACT_FREQ_THRESHOLD of all processed frames is flagged
  as a static artifact location (real ball trajectories vary shot-to-shot and do
  not repeatedly return to the same few pixels across a large fraction of an
  entire clip).
  Pass 2: re-run detection, this time discarding any candidate box whose center
  falls within ARTIFACT_REJECT_RADIUS_PX of a flagged bin, then picking the
  highest-confidence SURVIVING box as that frame's detection (falling back to
  motion-diff on frames with no surviving box, exactly as before).

This is an offline, two-pass batch design (deliberately -- these are recorded
clips, not a live stream, so there is no causality constraint requiring a
single forward pass).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ball_detection_experiments import court_mask, motion_diff_candidate, OUT_DIR
from ball_finetuned_eval import MODEL_PATH, AMATEUR_CLIPS, TENNIS_CLIP, MATCH_TENNIS
from ball_motion_diff_stress_clips import TENNIS_CLIP_CORNERS, MATCH_TENNIS_CORNERS
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations
from cv_pipeline.ball_detection import MAX_BALL_MATCH_DISTANCE_PX

ARTIFACT_BIN_PX = 10.0
ARTIFACT_FREQ_THRESHOLD = 0.03  # a bin hit in >=3% of all frames is flagged
ARTIFACT_REJECT_RADIUS_PX = 15.0
CONF_THRESHOLD = 0.25


def _bin_key(x: float, y: float) -> tuple[int, int]:
    return (round(x / ARTIFACT_BIN_PX), round(y / ARTIFACT_BIN_PX))


def find_artifact_bins(model, video_path: Path, start_frame: int = 0, n_frames: int | None = None):
    """Pass 1: collect every candidate box center across the clip, flag frequent bins."""
    cap = cv2.VideoCapture(str(video_path))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_frames = n_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bin_counts: Counter = Counter()
    n_processed = 0
    for _ in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        n_processed += 1
        results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        seen_bins_this_frame = set()
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            key = _bin_key(cx, cy)
            seen_bins_this_frame.add(key)
        # count each bin at most once per frame (frequency = "how many frames
        # touched this bin", not "how many boxes")
        for key in seen_bins_this_frame:
            bin_counts[key] += 1

    flagged = {
        key: count / n_processed
        for key, count in bin_counts.items()
        if count / n_processed >= ARTIFACT_FREQ_THRESHOLD
    }
    return flagged, n_processed


def is_near_flagged_bin(x: float, y: float, flagged_bins: dict) -> bool:
    for (bx, by) in flagged_bins:
        px, py = bx * ARTIFACT_BIN_PX, by * ARTIFACT_BIN_PX
        if np.hypot(x - px, y - py) <= ARTIFACT_REJECT_RADIUS_PX:
            return True
    return False


def run_filtered_amateur_clip(model, clip_name: str, video_path: Path, flagged_bins: dict):
    ann = load_clip_annotations(clip_name)
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray = None
    mask_cache = {}

    n_gt = n_hit_filtered = n_hit_combined = 0
    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        fa = ann.get(frame_idx)
        gt = fa.ball if (fa and not fa.ball_is_sentinel and not fa.ball_row_missing) else None

        results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        confs = results[0].boxes.conf.cpu().numpy().tolist() if len(results) else []

        surviving = [(b, c) for b, c in zip(boxes, confs)
                     if not is_near_flagged_bin((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, flagged_bins)]
        top_center = None
        if surviving:
            best = max(surviving, key=lambda bc: bc[1])
            b = best[0]
            top_center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_center = None
        if prev_gray is not None and top_center is None and fa and fa.court_corners:
            key = tuple(sorted(fa.court_corners.items()))
            if key not in mask_cache:
                mask_cache[key] = court_mask(frame.shape, fa.court_corners)
            candidates = motion_diff_candidate(prev_gray, gray, mask_cache[key])
            if gt is not None and candidates:
                dists = [np.hypot(gt[0] - c[0], gt[1] - c[1]) for c in candidates]
                bi = int(np.argmin(dists))
                if dists[bi] <= MAX_BALL_MATCH_DISTANCE_PX:
                    motion_center = (candidates[bi][0], candidates[bi][1])
        prev_gray = gray

        if gt is not None:
            n_gt += 1
            filtered_hit = top_center is not None and np.hypot(gt[0] - top_center[0], gt[1] - top_center[1]) <= MAX_BALL_MATCH_DISTANCE_PX
            if filtered_hit:
                n_hit_filtered += 1
            if filtered_hit or motion_center is not None:
                n_hit_combined += 1

    return n_gt, n_hit_filtered, n_hit_combined


def main():
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

    print("=== PASS 1: flagging frequent-artifact bins on the two stress clips ===")
    known_artifacts = {"tennis_clip": (1442, 778), "match_tennis": (412, 442)}
    stress_clip_bins = {}
    for label, video_path, start, n in [
        ("tennis_clip", TENNIS_CLIP, 3600, 900),
        ("match_tennis", MATCH_TENNIS, 5 * 60 * 25, 300),
    ]:
        flagged, n_processed = find_artifact_bins(model, video_path, start_frame=start, n_frames=n)
        stress_clip_bins[label] = flagged
        artifact_xy = known_artifacts[label]
        artifact_bin = _bin_key(*artifact_xy)
        caught = artifact_bin in flagged or any(
            np.hypot((k[0] - artifact_bin[0]) * ARTIFACT_BIN_PX, (k[1] - artifact_bin[1]) * ARTIFACT_BIN_PX) <= ARTIFACT_REJECT_RADIUS_PX
            for k in flagged
        )
        print(f"  {label}: {n_processed} frames processed, {len(flagged)} bins flagged "
              f"(threshold {ARTIFACT_FREQ_THRESHOLD*100:.0f}% of frames)")
        print(f"    known artifact {artifact_xy} (bin {artifact_bin}) CAUGHT: {caught}")
        if flagged:
            top5 = sorted(flagged.items(), key=lambda kv: -kv[1])[:5]
            for (bx, by), freq in top5:
                print(f"      bin center ~({bx*ARTIFACT_BIN_PX:.0f},{by*ARTIFACT_BIN_PX:.0f}) freq={freq*100:.1f}%")

    print("\n=== PASS 2: re-validate on amateur dataset ground truth (no stress-clip artifact bins applied here -- ")
    print("    each amateur clip gets its OWN pass-1 artifact flagging, since artifacts are clip-specific) ===")
    total_gt = total_filtered = total_combined = 0
    for clip in AMATEUR_CLIPS:
        video_path = DEFAULT_VIDEOS_DIR / f"{clip}.mp4"
        flagged, n_processed = find_artifact_bins(model, video_path)
        n_gt, n_hit_filtered, n_hit_combined = run_filtered_amateur_clip(model, clip, video_path, flagged)
        rate = lambda n: n / n_gt * 100 if n_gt else 0.0
        print(f"  {clip}: {len(flagged)} artifact bins flagged | "
              f"filtered={n_hit_filtered}/{n_gt} ({rate(n_hit_filtered):.1f}%) | "
              f"combined={n_hit_combined}/{n_gt} ({rate(n_hit_combined):.1f}%)")
        total_gt += n_gt
        total_filtered += n_hit_filtered
        total_combined += n_hit_combined

    print(f"\nPOOLED ({total_gt} gt frames):")
    print(f"  v2-filtered fine-tuned YOLO: {total_filtered}/{total_gt} = {total_filtered/total_gt*100:.2f}%")
    print(f"  v2 combined (filtered YOLO OR motion-diff): {total_combined}/{total_gt} = {total_combined/total_gt*100:.2f}%")
    print(f"  (reference: v1 static-filtered was 44.02%, v1 combined was 58.15%, motion-diff-alone 57.62%)")


if __name__ == "__main__":
    main()
