"""build_clip_reports.py — Step 8+9: populates the ClipReport schema for all 10
clips and writes both per-clip JSON and one combined JSON/Parquet.

PROVENANCE: the detection/tracking/homography/pose numbers below are copied in from
today's already-completed, already-verified validation runs (steps 3-7 of this
session — see PROGRESS.md for the full derivation of each number, including the two
real bugs found and fixed along the way: the player-position sentinel contamination
fix, and the hard-moment background-people contamination fix). This script is NOT a
redundant live rerun of ~40 minutes of detection+tracking passes that would reproduce
identical numbers -- it is the structured-output step, turning today's verified
results into the schema. The underlying scripts (run_full_detection_validation.py,
run_tracking_validation_all_clips.py, recompute_hard_moments_top2.py,
verify_homography.py, run_pose_spot_check.py) are all independently re-runnable from
scratch if these numbers ever need to be reproduced or extended to new clips.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.schema import ClipReport, HomographyReport, PoseReport, RateMetric, Status, TrackingReport

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "clip_reports"

# n_frames, fps -- from annotations.py's load_clip_annotations() / OpenCV frame counts
CLIP_META = {
    "video1": (689, 60.09), "video2": (802, 60.0), "video3": (867, 60.0),
    "video4": (998, 60.0), "video5": (357, 60.0), "video6": (956, 60.0),
    "video7": (773, 60.0), "video8": (513, 60.0), "video9": (627, 60.0),
    "video10": (226, 60.0),
}

# player_r: (rate, n, median_err_px)
PLAYER_R = {
    "video1": (0.918, 668, 75.8), "video2": (0.995, 633, 61.3), "video3": (0.998, 858, 57.8),
    "video4": (0.994, 980, 67.5), "video5": (0.913, 356, 64.2), "video6": (0.952, 747, 91.8),
    "video7": (0.988, 773, 68.7), "video8": (0.988, 498, 58.2), "video9": (0.914, 395, 80.9),
    "video10": (0.978, 226, 58.5),
}

# player_l separated: (rate, n, median_err_px or None)
PLAYER_L_SEP = {
    "video1": (0.208, 96, 134.4), "video2": (0.010, 100, 97.2), "video3": (0.000, 63, None),
    "video4": (0.400, 15, 132.6), "video5": (0.341, 41, 121.8), "video6": (0.020, 152, 95.4),
    "video7": (0.000, 45, None), "video8": (0.056, 71, 75.4), "video9": (0.204, 152, 128.0),
    "video10": (0.444, 9, 129.0),
}

# player_l ambiguous: (rate, n)
PLAYER_L_AMB = {
    "video1": (0.078, 565), "video2": (0.005, 397), "video3": (0.022, 777),
    "video4": (0.001, 944), "video5": (0.025, 315), "video6": (0.063, 542),
    "video7": (0.022, 728), "video8": (0.017, 421), "video9": (0.027, 187),
    "video10": (0.005, 217),
}

# ball: (rate, n, median_err_px)
BALL = {
    "video1": (0.028, 250, 3.8), "video2": (0.004, 272, 3.4), "video3": (0.361, 202, 2.3),
    "video4": (0.159, 396, 2.8), "video5": (0.049, 164, 3.0), "video6": (0.123, 310, 4.1),
    "video7": (0.026, 269, 3.7), "video8": (0.011, 179, 3.1), "video9": (0.159, 170, 4.0),
    "video10": (0.141, 64, 3.3),
}

# tracking: (player_r_segments, player_l_segments, n_id_swaps)
TRACKING_SEGMENTS = {
    "video1": (1, 1, 0), "video2": (2, 1, 1), "video3": (1, 3, 2),
    "video4": (1, 1, 0), "video5": (1, 1, 0), "video6": (2, 2, 2),
    "video7": (1, 2, 1), "video8": (1, 1, 0), "video9": (1, 1, 0),
    "video10": (1, 1, 0),
}

# corrected (top-2-confidence) hard-moment frame counts
HARD_MOMENTS = {
    "video1": 1, "video2": 6, "video3": 30, "video4": 0, "video5": 0,
    "video6": 0, "video7": 4, "video8": 0, "video9": 0, "video10": 0,
}

# clips visually spot-checked for pose in step 7, and what was found
POSE_SPOT_CHECKS = {
    "video1": ("good", "not_attempted"),  # near: clean; far: YOLO found no far-player box in the tested frame
    "video7": ("good", None),  # far player not sampled for this clip
    "video3": ("good_hard_case", None),  # near player mid-serve, arm extended overhead -- accurate
    "video6": ("good_minor_imprecision", None),  # near player post-break, slight wrist/racket-grip ambiguity
    "video9": (None, "not_detected"),  # far player correctly boxed by YOLO, MediaPipe still produced zero landmarks
}


def rate_metric(rate, n, median_err, min_n=RateMetric.MIN_TRUSTED_N) -> RateMetric:
    if n == 0:
        return RateMetric(status=Status.NOT_APPLICABLE, note="no ground-truth frames of this type in this clip")
    status = Status.MEASURED if n >= min_n else Status.INSUFFICIENT_SAMPLE
    note = None if n >= min_n else f"n={n} < {min_n}; rate computed but not trustworthy at this sample size"
    return RateMetric(status=status, rate=rate, n=n, median_error_px=median_err, note=note)


def build_report(clip: str) -> ClipReport:
    n_frames, fps = CLIP_META[clip]

    if clip == "video1":
        homography = HomographyReport(
            geometric_sanity_status=Status.MEASURED,
            real_world_scale_status=Status.MEASURED,
            real_world_distance_metrics_usable=True,
            note="Independently validated against the baseline center hash mark "
                 "(a landmark never used in calibration): predicted vs. measured "
                 "position agreed to ~13px (~8cm real-world). Safe to use for "
                 "real-world-distance-derived metrics.",
        )
    elif clip == "video7":
        homography = HomographyReport(
            geometric_sanity_status=Status.MEASURED,
            real_world_scale_status=Status.EXCLUDED_KNOWN_ISSUE,
            real_world_distance_metrics_usable=False,
            note="Root-caused: annotated court corners span only the near "
                 "half-court (baseline to net, ~11.9m), not the full doubles "
                 "court (23.77m) -- confirmed by matching the implied real-world "
                 "span to baseline-to-net distance within 2.8%. Geometrically "
                 "self-consistent (0px reprojection error) but WRONG SCALE. "
                 "Excluded from all real-world-distance metrics until a per-clip "
                 "length-correction is implemented.",
        )
    else:
        homography = HomographyReport(
            geometric_sanity_status=Status.MEASURED,
            real_world_scale_status=Status.UNVALIDATED,
            real_world_distance_metrics_usable=False,
            note="Passes geometric self-consistency checks (0px reprojection "
                 "error, near-baseline pixel span > far-baseline span as "
                 "perspective requires) but was NOT independently checked "
                 "against a real-world landmark the way video1 and video7 were. "
                 "Given video7 was found to have a real, non-obvious scale error "
                 "despite passing the same self-consistency checks, this clip's "
                 "real-world scale should be treated as unconfirmed, not assumed "
                 "correct, until individually validated.",
        )

    r_rate, r_n, r_err = PLAYER_R[clip]
    player_r_detection = rate_metric(r_rate, r_n, r_err)

    l_sep_rate, l_sep_n, l_sep_err = PLAYER_L_SEP[clip]
    player_l_sep = rate_metric(l_sep_rate, l_sep_n, l_sep_err)

    l_amb_rate, l_amb_n = PLAYER_L_AMB[clip]
    player_l_amb = RateMetric(
        status=Status.MEASURED, rate=l_amb_rate, n=l_amb_n, median_error_px=None,
        note="'Ambiguous' bucket: player_r/player_l ground truth were <200px apart, "
             "likely both pointing near the same physical player (see steps 4-5) -- "
             "a low match rate here is largely a ground-truth artifact, not a pure "
             "detection miss. Not a trustworthy far-player accuracy estimate either way.",
    )

    b_rate, b_n, b_err = BALL[clip]
    ball_detection = rate_metric(b_rate, b_n, b_err)
    if clip == "video3":
        ball_detection = RateMetric(
            status=ball_detection.status, rate=ball_detection.rate, n=ball_detection.n,
            median_error_px=ball_detection.median_error_px,
            note="Real, verified outlier (not a bug) -- this clip has visibly higher "
                 "video/broadcast quality (bright ball, sharp resolution) than the "
                 "other 9. Do not treat as representative of typical performance; "
                 "see the ~7-9% figure (video3 excluded) for that.",
        )

    r_seg, l_seg, n_swaps = TRACKING_SEGMENTS[clip]
    hard_moments = HARD_MOMENTS[clip]
    if hard_moments == 0:
        coverage_status = Status.NOT_APPLICABLE
        note = ("Zero genuine crossing/proximity frames occurred in this clip "
                "(corrected, top-2-confidence-box proxy). Any 'no ID swap' result "
                "here is NOT evidence the tracker handles crossings/occlusion well -- "
                "this clip never tested that.")
    elif hard_moments < 10:
        coverage_status = Status.INSUFFICIENT_SAMPLE
        note = f"Only {hard_moments} hard-moment frame(s) -- thin coverage, treat any conclusion as weak."
    else:
        coverage_status = Status.MEASURED
        note = f"{hard_moments} genuine hard-moment frames -- meaningful test coverage."
    tracking = TrackingReport(
        player_r_n_segments=r_seg, player_l_n_segments=l_seg, n_id_swaps=n_swaps,
        hard_moment_frame_count=hard_moments, hard_moment_coverage_status=coverage_status, note=note,
    )

    near_status, far_status = POSE_SPOT_CHECKS.get(clip, (None, None))
    near_map = {
        "good": Status.MEASURED, "good_hard_case": Status.MEASURED,
        "good_minor_imprecision": Status.MEASURED, None: Status.NOT_APPLICABLE,
    }
    far_map = {
        "not_attempted": Status.NOT_ATTEMPTED, "not_detected": Status.NOT_DETECTED,
        None: Status.NOT_APPLICABLE,
    }
    pose_notes = {
        "video1": "Near: clean, accurate landmarks on a frontal ready-stance. "
                  "Far: YOLO found no far-player box at all in the tested frame -- "
                  "pose couldn't be attempted (cascading from detection weakness).",
        "video7": "Near: clean, accurate landmarks (ready stance, back to camera).",
        "video3": "Near: accurate landmarks even on a hard case -- mid-serve, arm "
                  "fully extended overhead, correctly tracked to the raised hand.",
        "video6": "Near: mostly accurate; minor ambiguity in 2 low-visibility "
                  "landmarks near the wrist/racket-grip during a post-break shot.",
        "video9": "Far: YOLO DID correctly detect the far player (after fixing a "
                  "box-selection bug in the spot-check script itself, which had "
                  "first picked a sideline bystander) -- but MediaPipe produced "
                  "ZERO landmarks on the correctly-boxed ~55x66px, motion-blurred crop.",
    }
    pose = PoseReport(
        near_player_status=near_map.get(near_status, Status.NOT_APPLICABLE),
        far_player_status=far_map.get(far_status, Status.NOT_APPLICABLE),
        note=pose_notes.get(clip, "Not one of the 6 hand-picked spot-check cases in step 7 -- "
                                   "no pose data for this clip (NOT_APPLICABLE, not silently absent)."),
    )

    return ClipReport(
        clip=clip, n_frames=n_frames, fps=fps, homography=homography,
        player_r_detection=player_r_detection, player_l_detection_separated=player_l_sep,
        player_l_detection_ambiguous=player_l_amb, ball_detection=ball_detection,
        tracking=tracking, pose=pose,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = {}
    for clip in CLIP_META:
        report = build_report(clip)
        reports[clip] = report.to_dict()
        with open(OUT_DIR / f"{clip}.json", "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    with open(OUT_DIR / "all_clips.json", "w") as f:
        json.dump(reports, f, indent=2)

    try:
        import pandas as pd
        rows = []
        for clip, r in reports.items():
            row = {"clip": clip, "n_frames": r["n_frames"], "fps": r["fps"]}
            for section in ["player_r_detection", "player_l_detection_separated",
                             "player_l_detection_ambiguous", "ball_detection"]:
                for k, v in r[section].items():
                    row[f"{section}.{k}"] = v
            for k, v in r["homography"].items():
                row[f"homography.{k}"] = v
            for k, v in r["tracking"].items():
                row[f"tracking.{k}"] = v
            for k, v in r["pose"].items():
                row[f"pose.{k}"] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_parquet(OUT_DIR / "all_clips.parquet", index=False)
        print(f"Wrote {OUT_DIR / 'all_clips.parquet'} ({len(df)} rows, {len(df.columns)} columns)")
    except ImportError:
        print("pandas/pyarrow not available in this env -- skipped Parquet export, JSON still written.")

    print(f"Wrote {len(reports)} per-clip JSON files + combined JSON to {OUT_DIR}")


if __name__ == "__main__":
    main()
