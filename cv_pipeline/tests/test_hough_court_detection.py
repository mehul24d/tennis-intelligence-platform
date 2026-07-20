"""test_hough_court_detection.py — real-data regression test for the
EXPERIMENTAL automated Hough-based court-corner detector (see
hough_court_detection.py's own docstring for what this is and is NOT --
not wired into the pipeline, not through the mandatory calibration
verification gate). Compares detected corners on frame 0 of data/tennis/
1.mp4 and 3.mp4 directly against this project's existing manually-traced
ground truth (the exact pixel values in reference_video1_calibration.py /
reference_video3_calibration.py) -- a regression test against the actual
measured numbers in PROGRESS.md's "Automated Hough-Line-Based Homography
Calibration" entry, not an arbitrary threshold.

Also covers the multi-frame, multi-clip follow-up (all 5 reference clips,
8 evenly-spread frames per camera-stable segment, averaged via
detect_court_corners_multi_frame) -- see PROGRESS.md's "Multi-Frame,
Multi-Clip Hough Evaluation" entry for the full investigation. Segment
windows and their ground truth are read directly from the checked-in
reference_videoN_calibration.py modules, not re-typed by hand, so this
test can't silently drift from the actual calibration values.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from cv_pipeline.hough_court_detection import (
    detect_court_corners,
    detect_court_corners_multi_frame,
)
from cv_pipeline.reference_video1_calibration import VIDEO1_CALIBRATION_POINTS
from cv_pipeline.reference_video2_calibration import VIDEO2_CALIBRATION_POINTS
from cv_pipeline.reference_video2_postpan_calibration import VIDEO2_POSTPAN_CALIBRATION_POINTS
from cv_pipeline.reference_video3_calibration import VIDEO3_CALIBRATION_POINTS
from cv_pipeline.reference_video4_calibration import VIDEO4_CALIBRATION_POINTS
from cv_pipeline.reference_video5_calibration import (
    VIDEO5_SEGMENT_A_POINTS,
    VIDEO5_SEGMENT_B_POINTS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH = {
    "1": {"bl": (200.0, 866.0), "br": (1718.0, 866.0), "tr": (1330.0, 300.0), "tl": (598.0, 300.0)},
    "3": {"bl": (193.0, 841.0), "br": (1719.0, 842.0), "tr": (1336.0, 268.0), "tl": (584.0, 268.0)},
}

# Measured, not guessed -- see PROGRESS.md/hough_court_detection.py's
# module docstring. Updated after coverage-weighted cluster selection +
# scoreboard segment-filtering (the current approach -- an earlier
# pixel-masking version of the scoreboard fix was replaced after it was
# found to occasionally destabilize unrelated Hough detections elsewhere
# in the frame; see PROGRESS.md's "Segment-Filtering Instead Of
# Pixel-Masking" entry): video1 mean 6.0px (was 11.8px unfixed), video3
# mean 4.2px (was 4.1px -- within normal frame-to-frame noise). Generous
# margins around the measured values, not tight equality -- this guards
# against a real regression (e.g. a future edit reintroducing the
# net-cord-as-baseline bug), not against normal floating-point noise.
MAX_MEAN_ERROR_PX = {"1": 12.0, "3": 9.0}
MAX_PER_CORNER_ERROR_PX = {"1": 18.0, "3": 9.0}


def _distance(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


@pytest.mark.parametrize("clip", ["1", "3"])
def test_detected_corners_match_existing_ground_truth(clip):
    video_path = REPO_ROOT / "data" / "tennis" / f"{clip}.mp4"
    if not video_path.exists():
        pytest.skip(f"{video_path} not present in this environment")

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read frame 0 of {video_path}"

    result = detect_court_corners(frame)
    detected = {"bl": result.bl, "br": result.br, "tr": result.tr, "tl": result.tl}

    errors = []
    for key, gt in GROUND_TRUTH[clip].items():
        det = detected[key]
        assert det is not None, f"clip {clip}: corner {key} was not detected at all"
        err = _distance(det, gt)
        errors.append(err)
        assert err < MAX_PER_CORNER_ERROR_PX[clip], (
            f"clip {clip} corner {key}: detected {det}, ground truth {gt}, error {err:.1f}px "
            f"exceeds {MAX_PER_CORNER_ERROR_PX[clip]}px -- see hough_court_detection.py's docstring "
            f"for the two real clustering bugs this kind of regression previously looked like"
        )

    mean_error = sum(errors) / len(errors)
    assert mean_error < MAX_MEAN_ERROR_PX[clip], (
        f"clip {clip}: mean corner error {mean_error:.1f}px exceeds {MAX_MEAN_ERROR_PX[clip]}px"
    )


def _gt_corners(points) -> dict[str, tuple[float, float]]:
    # each reference_videoN_calibration module lists BL, BR, TR, TL as the
    # first 4 (world_xy, pixel_xy) entries, in that order.
    keys = ["bl", "br", "tr", "tl"]
    return {k: points[i][1] for i, k in enumerate(keys)}


def _evenly_spaced(lo: int, hi: int, n: int) -> list[int]:
    if n == 1:
        return [(lo + hi) // 2]
    step = (hi - 1 - lo) / (n - 1)
    return sorted({round(lo + i * step) for i in range(n)})


def _read_frames_sequential(video_path: Path, wanted: set[int]) -> dict[int, "cv2.typing.MatLike"]:
    """Sequential .read() only, never cap.set()/seek -- per the documented
    cv2.CAP_PROP_POS_FRAMES seek-inaccuracy bug found on 5.mp4 (see
    reference_video5_calibration.py's docstring)."""
    cap = cv2.VideoCapture(str(video_path))
    out = {}
    max_wanted = max(wanted)
    idx = 0
    while idx <= max_wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            out[idx] = frame
        idx += 1
    cap.release()
    return out


# clip -> [(segment_label, frame_lo, frame_hi_exclusive, ground_truth), ...].
# Frame ranges are the camera's documented stable windows for each clip (see
# each reference_videoN_calibration.py's own camera-motion-check docstring)
# -- pans/ramps are deliberately excluded, matching how the manual
# calibrations themselves are scoped.
MULTI_FRAME_SEGMENTS = {
    "1": [("static", 0, 2020, _gt_corners(VIDEO1_CALIBRATION_POINTS))],
    "2": [
        ("pre-pan", 0, 400, _gt_corners(VIDEO2_CALIBRATION_POINTS)),
        ("post-pan", 560, 1344, _gt_corners(VIDEO2_POSTPAN_CALIBRATION_POINTS)),
    ],
    "3": [("static", 0, 933, _gt_corners(VIDEO3_CALIBRATION_POINTS))],
    "4": [
        ("stable-a", 0, 420, _gt_corners(VIDEO4_CALIBRATION_POINTS)),
        ("stable-b", 740, 1240, _gt_corners(VIDEO4_CALIBRATION_POINTS)),
    ],
    "5": [
        ("segment-a", 0, 136, _gt_corners(VIDEO5_SEGMENT_A_POINTS)),
        ("segment-b", 400, 940, _gt_corners(VIDEO5_SEGMENT_B_POINTS)),
    ],
}

N_FRAMES_PER_SEGMENT = 8

# Measured, not guessed -- see PROGRESS.md's "Segment-Filtering Instead Of
# Pixel-Masking" and "Median Aggregation" entries. Generous margins around
# the measured per-segment mean error, guarding against a real regression,
# not floating-point noise. Current pipeline: coverage-weighted baseline
# selection + scoreboard exclusion applied as a POST-HOC SEGMENT FILTER
# (not a pixel mask -- an earlier pixel-masking version destabilized
# unrelated far-baseline detection on 5.mp4 frame 0 via cv2.HoughLinesP's
# non-local probabilistic sensitivity to edge-pixel-population changes;
# segment-filtering leaves Hough's input untouched, avoiding that failure
# mode entirely) + median (not mean) aggregation across frames, which
# further guards against ordinary per-frame Hough noise (a minority of
# frames being unusually wrong, as opposed to the scoreboard-specific
# mechanism above).
MULTI_FRAME_MAX_MEAN_ERROR_PX = {
    ("1", "static"): 10.0,
    ("2", "pre-pan"): 14.0,
    ("2", "post-pan"): 12.0,
    ("3", "static"): 9.0,
    ("4", "stable-a"): 9.0,
    ("4", "stable-b"): 9.0,
    ("5", "segment-a"): 14.0,
    ("5", "segment-b"): 13.0,
}


_MULTI_FRAME_CASES = [
    (clip, label, lo, hi, gt) for clip, segs in MULTI_FRAME_SEGMENTS.items() for label, lo, hi, gt in segs
]


@pytest.mark.parametrize(
    "clip, label, lo, hi, gt",
    _MULTI_FRAME_CASES,
    ids=[f"{clip}-{label}" for clip, label, _, _, _ in _MULTI_FRAME_CASES],
)
def test_multi_frame_detection_matches_measured_results(clip, label, lo, hi, gt):
    video_path = REPO_ROOT / "data" / "tennis" / f"{clip}.mp4"
    if not video_path.exists():
        pytest.skip(f"{video_path} not present in this environment")

    idxs = _evenly_spaced(lo, hi, N_FRAMES_PER_SEGMENT)
    frames_by_idx = _read_frames_sequential(video_path, set(idxs))
    frames = [frames_by_idx[i] for i in idxs if i in frames_by_idx]
    assert len(frames) == len(idxs), f"clip {clip} [{label}]: could not read all wanted frames"

    result = detect_court_corners_multi_frame(frames)
    detected = {"bl": result.bl, "br": result.br, "tr": result.tr, "tl": result.tl}

    errors = []
    for key, gt_pt in gt.items():
        det = detected[key]
        assert det is not None, f"clip {clip} [{label}]: corner {key} was not detected in ANY sampled frame"
        errors.append(_distance(det, gt_pt))

    mean_error = sum(errors) / len(errors)
    max_allowed = MULTI_FRAME_MAX_MEAN_ERROR_PX[(clip, label)]
    assert mean_error < max_allowed, (
        f"clip {clip} [{label}]: mean corner error {mean_error:.1f}px exceeds {max_allowed}px"
    )
