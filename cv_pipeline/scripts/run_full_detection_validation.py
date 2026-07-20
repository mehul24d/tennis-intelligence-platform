"""run_full_detection_validation.py — Step 4+5 across all 10 clips: player detection
(yolov8n, person class, bottom-center matching, sentinel-aware ground truth) and ball
detection (yolov8n, sports-ball class, sentinel-aware ground truth). Same methodology
as the video1 validation (see PROGRESS.md), applied consistently to every clip, with
per-clip AND aggregate reporting -- outliers are flagged explicitly, not folded silently
into a blended mean.
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
from cv_pipeline.ball_detection import run_ball_frame_detection

CLIPS = [f"video{i}" for i in range(1, 11)]
SEPARATED_THRESHOLD_PX = 200.0


def run_clip(model, clip: str):
    annotations = load_clip_annotations(clip)
    cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{clip}.mp4"))
    n_frames = len(annotations)

    errors_r, errors_l_sep, errors_l_amb = [], [], []
    n_r_gt = n_r_matched = 0
    n_l_sep_gt = n_l_sep_matched = 0
    n_l_amb_gt = n_l_amb_matched = 0
    ball_errors = []
    n_ball_gt = n_ball_matched = 0

    t0 = time.time()
    n_processed = 0
    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        n_processed += 1
        ann = annotations[idx]

        result = run_frame_detection(model, frame, idx, ann)
        if ann.player_r is not None:
            n_r_gt += 1
            if result.player_r_error_px is not None:
                n_r_matched += 1
                errors_r.append(result.player_r_error_px)
        if ann.player_l is not None:
            is_separated = (
                ann.player_r is not None
                and np.hypot(ann.player_r[0] - ann.player_l[0], ann.player_r[1] - ann.player_l[1])
                >= SEPARATED_THRESHOLD_PX
            )
            if is_separated:
                n_l_sep_gt += 1
                if result.player_l_error_px is not None:
                    n_l_sep_matched += 1
                    errors_l_sep.append(result.player_l_error_px)
            else:
                n_l_amb_gt += 1
                if result.player_l_error_px is not None:
                    n_l_amb_matched += 1
                    errors_l_amb.append(result.player_l_error_px)

        ball_result = run_ball_frame_detection(model, frame, idx, ann)
        if ann.ball is not None:
            n_ball_gt += 1
            if ball_result.error_px is not None:
                n_ball_matched += 1
                ball_errors.append(ball_result.error_px)

    elapsed = time.time() - t0

    return {
        "clip": clip, "n_frames": n_processed, "elapsed_s": elapsed, "fps": n_processed / elapsed,
        "player_r_rate": n_r_matched / n_r_gt if n_r_gt else float("nan"),
        "player_r_n": n_r_gt,
        "player_r_median_err": float(np.median(errors_r)) if errors_r else float("nan"),
        "player_l_sep_rate": n_l_sep_matched / n_l_sep_gt if n_l_sep_gt else float("nan"),
        "player_l_sep_n": n_l_sep_gt,
        "player_l_sep_median_err": float(np.median(errors_l_sep)) if errors_l_sep else float("nan"),
        "player_l_amb_rate": n_l_amb_matched / n_l_amb_gt if n_l_amb_gt else float("nan"),
        "player_l_amb_n": n_l_amb_gt,
        "ball_rate": n_ball_matched / n_ball_gt if n_ball_gt else float("nan"),
        "ball_n": n_ball_gt,
        "ball_median_err": float(np.median(ball_errors)) if ball_errors else float("nan"),
        "ball_mean_err": float(np.mean(ball_errors)) if ball_errors else float("nan"),
    }


def flag_outliers(values: list[float], labels: list[str], metric_name: str, z_thresh: float = 1.5):
    """Simple outlier flag: values more than z_thresh std devs from the mean (ignoring
    NaNs). Not a rigorous statistical test (n=10 is small) -- a practical screen to
    surface anything worth a second look before trusting a blended average."""
    arr = np.array(values, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return []
    mean, std = arr[valid].mean(), arr[valid].std()
    if std == 0:
        return []
    flagged = []
    for label, v in zip(labels, values):
        if not np.isnan(v) and abs(v - mean) > z_thresh * std:
            flagged.append((label, v, mean))
    return flagged


def main():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    reports = [run_clip(model, clip) for clip in CLIPS]

    print(f"\n{'clip':<9} {'fps':>6} {'p_r_rate':>9} {'p_r_n':>6} {'p_r_err':>8} "
          f"{'p_l_sep_rate':>13} {'p_l_sep_n':>10} {'p_l_sep_err':>12} "
          f"{'p_l_amb_rate':>13} {'p_l_amb_n':>10} "
          f"{'ball_rate':>10} {'ball_n':>7} {'ball_err':>9}")
    for r in reports:
        print(f"{r['clip']:<9} {r['fps']:>6.1f} {r['player_r_rate']:>9.1%} {r['player_r_n']:>6} "
              f"{r['player_r_median_err']:>8.1f} "
              f"{r['player_l_sep_rate']:>13.1%} {r['player_l_sep_n']:>10} {r['player_l_sep_median_err']:>12.1f} "
              f"{r['player_l_amb_rate']:>13.1%} {r['player_l_amb_n']:>10} "
              f"{r['ball_rate']:>10.1%} {r['ball_n']:>7} {r['ball_median_err']:>9.1f}")

    print("\n=== OUTLIER CHECK (>1.5 std dev from cross-clip mean) ===")
    metrics = [
        ("player_r_rate", "player_r detection rate"),
        ("player_l_sep_rate", "player_l (separated) detection rate"),
        ("ball_rate", "ball detection rate"),
        ("ball_median_err", "ball median error (px)"),
        ("player_r_median_err", "player_r median error (px)"),
    ]
    any_flagged = False
    for key, name in metrics:
        vals = [r[key] for r in reports]
        labels = [r["clip"] for r in reports]
        flagged = flag_outliers(vals, labels, name)
        if flagged:
            any_flagged = True
            for label, v, mean in flagged:
                print(f"  FLAG: {label} — {name}={v:.3f} vs cross-clip mean={mean:.3f}")
    if not any_flagged:
        print("  none flagged")

    print("\n=== AGGREGATE (simple mean across clips; see outlier flags above first) ===")
    for key, name in metrics:
        vals = np.array([r[key] for r in reports], dtype=float)
        valid = vals[~np.isnan(vals)]
        print(f"  {name}: mean={valid.mean():.3f}, median={np.median(valid):.3f}, "
              f"min={valid.min():.3f}, max={valid.max():.3f}")


if __name__ == "__main__":
    main()
