"""run_full_detection_validation_combined_ball.py -- re-runs Step 4+5's ball-
detection half (see run_full_detection_validation.py, left untouched as the
exact source of EVALUATION_REPORT.md's committed stock-YOLO baseline numbers)
across all 10 amateur clips, but through the ACTUAL PRODUCTION combined-method
code path (ball_detection_combined.run_combined_ball_detection_for_clip -- the
same function v2_serving/video_pipeline.py now calls by default), not a side
experiment script.

WHY THIS EXISTS SEPARATELY from the earlier ball_detection_experiments.py /
ball_finetuned_combined_eval.py / ball_static_artifact_filter_v2.py scripts
that originally reported a 70.40% pooled-recall number: those scripts
prototyped the combined method's DESIGN. This script re-measures the SAME
thing through the actual wired-in production function, as a final end-to-end
regression check before/after switching cv_pipeline's default ball-detection
call sites over to it -- confirming the real integration behaves the same as
the standalone experiments that justified adopting it, not just trusting that
"it should be the same code."

THIS CHECK IS WHAT ACTUALLY CAUGHT A REAL BUG: the first run through this
script (2026-07-16) came back at 46.24%, far below the prototyped 70.40% --
which led to finding a ground-truth leak in the prototype's motion-diff
candidate-picking logic (it used ground truth to select which of several
candidate blobs to trust, something no real inference-time system can do).
Fixed in ball_detection_combined.py (picking the largest-area candidate
instead, a legitimate non-cheating heuristic) and re-run here again, landing
at the final, corrected, honest figure: **53.91% pooled recall** (video3
excluded, 9-clip scope) -- still a real ~6.9x improvement over stock YOLO's
7.81%, just not the originally-claimed 70.40%. See PROGRESS.md for the full
writeup, including a project-wide audit for the same failure category in
other trusted validation scripts (none found elsewhere).

Player detection is UNCHANGED (reuses run_frame_detection from
run_full_detection_validation.py directly) -- only ball detection differs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations
from cv_pipeline.ball_detection import MAX_BALL_MATCH_DISTANCE_PX
from cv_pipeline.ball_detection_combined import FINE_TUNED_MODEL_PATH, run_combined_ball_detection_for_clip
from cv_pipeline.homography import CourtHomography
from cv_pipeline.player_detection import run_frame_detection

CLIPS = [f"video{i}" for i in range(1, 11)]
SEPARATED_THRESHOLD_PX = 200.0


def run_clip(player_model, fine_tuned_model, clip: str):
    annotations = load_clip_annotations(clip)
    cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{clip}.mp4"))
    n_frames = len(annotations)

    first_court_ann = next((a for a in annotations.values() if a.court_corners), None)
    homography = CourtHomography(first_court_ann.court_corners)
    combined_by_index = {
        r.frame_index: r
        for r in run_combined_ball_detection_for_clip(fine_tuned_model, DEFAULT_VIDEOS_DIR / f"{clip}.mp4", homography)
    }
    n_homography_inapplicable = sum(1 for r in combined_by_index.values() if not r.homography_applicable)

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

        result = run_frame_detection(player_model, frame, idx, ann)
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

        combined_result = combined_by_index.get(idx)
        if ann.ball is not None:
            n_ball_gt += 1
            if combined_result is not None and combined_result.center is not None:
                dist = np.hypot(ann.ball[0] - combined_result.center[0], ann.ball[1] - combined_result.center[1])
                if dist <= MAX_BALL_MATCH_DISTANCE_PX:
                    n_ball_matched += 1
                    ball_errors.append(dist)

    elapsed = time.time() - t0

    return {
        "clip": clip, "n_frames": n_processed, "elapsed_s": elapsed, "fps": n_processed / elapsed,
        "player_r_rate": n_r_matched / n_r_gt if n_r_gt else float("nan"),
        "player_r_n": n_r_gt,
        "player_r_median_err": float(np.median(errors_r)) if errors_r else float("nan"),
        "player_l_sep_rate": n_l_sep_matched / n_l_sep_gt if n_l_sep_gt else float("nan"),
        "player_l_sep_n": n_l_sep_gt,
        "player_l_amb_rate": n_l_amb_matched / n_l_amb_gt if n_l_amb_gt else float("nan"),
        "player_l_amb_n": n_l_amb_gt,
        "ball_rate": n_ball_matched / n_ball_gt if n_ball_gt else float("nan"),
        "ball_n": n_ball_gt,
        "ball_median_err": float(np.median(ball_errors)) if ball_errors else float("nan"),
        "n_homography_inapplicable_frames": n_homography_inapplicable,
    }


def main():
    from ultralytics import YOLO
    player_model = YOLO("yolov8n.pt")
    fine_tuned_model = YOLO(str(FINE_TUNED_MODEL_PATH))

    reports = [run_clip(player_model, fine_tuned_model, clip) for clip in CLIPS]

    print(f"\n{'clip':<9} {'fps':>6} {'p_r_rate':>9} {'p_r_n':>6} "
          f"{'ball_rate':>10} {'ball_n':>7} {'ball_err':>9} {'homog_bad':>10}")
    for r in reports:
        print(f"{r['clip']:<9} {r['fps']:>6.2f} {r['player_r_rate']:>9.1%} {r['player_r_n']:>6} "
              f"{r['ball_rate']:>10.1%} {r['ball_n']:>7} {r['ball_median_err']:>9.1f} {r['n_homography_inapplicable_frames']:>10}")

    tot_ball_gt = sum(r["ball_n"] for r in reports)
    tot_ball_matched = sum(round(r["ball_rate"] * r["ball_n"]) if r["ball_n"] else 0 for r in reports)
    print(f"\nPOOLED ball recall (sum matched / sum gt across all 10 clips): "
          f"{tot_ball_matched}/{tot_ball_gt} = {tot_ball_matched/tot_ball_gt*100:.2f}%")
    print("(reference: stock-YOLO baseline (video3 excluded, 9 clips) 7.81%, "
          "combined method as originally prototyped 70.40% (video3 excluded, 9 clips))")

    # Also report with video3 excluded, matching the exact scope the 70.40%
    # figure was originally measured against, for a direct apples-to-apples check.
    reports_no_v3 = [r for r in reports if r["clip"] != "video3"]
    tot_gt_9 = sum(r["ball_n"] for r in reports_no_v3)
    tot_matched_9 = sum(round(r["ball_rate"] * r["ball_n"]) if r["ball_n"] else 0 for r in reports_no_v3)
    print(f"POOLED ball recall (video3 excluded, matching original 9-clip scope): "
          f"{tot_matched_9}/{tot_gt_9} = {tot_matched_9/tot_gt_9*100:.2f}%")


if __name__ == "__main__":
    main()
