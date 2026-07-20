"""test_ball_trajectory_kalman.py -- synthetic sanity checks for the
Kalman/ballistic gap-filling module. This module was built, verified against
real clip data, and explicitly NOT SHIPPED (see PROGRESS.md's "Kalman/
Ballistic Trajectory Filter Phase" entry -- a 125-sample manual visual audit
found only 33.6% accuracy on the newly-filled frames, well below B's 92%).
These tests do not re-litigate that decision; they just pin the two
behavioral claims the design relies on, on clean synthetic data where the
correct answer is known exactly, so a future refactor can't silently break
either one without a real ground truth to check against.
"""

from __future__ import annotations

import numpy as np

from cv_pipeline.ball_detection_combined import CombinedBallDetectionResult
from cv_pipeline.ball_trajectory_kalman import fit_trajectory_for_clip
from cv_pipeline.homography import CourtHomography

CORNERS = {"BL": (300.0, 850.0), "BR": (1600.0, 850.0), "TL": (650.0, 300.0), "TR": (1250.0, 300.0)}


def _homography() -> CourtHomography:
    return CourtHomography(CORNERS)


def _make_detections(n_frames: int, gap_range: range, position_fn) -> tuple[list[CombinedBallDetectionResult], dict]:
    homography = _homography()
    true_positions = {}
    for f in range(n_frames):
        X, Y = position_fn(f)
        px, py = homography.world_to_pixel(X, Y)
        true_positions[f] = (px, py)

    gap_set = set(gap_range)
    dets = []
    for f in range(n_frames):
        if f in gap_set:
            dets.append(CombinedBallDetectionResult(
                frame_index=f, center=None, source="none",
                homography_applicable=True, reference_match_correlation=1.0,
            ))
        else:
            dets.append(CombinedBallDetectionResult(
                frame_index=f, center=true_positions[f], source="fine_tuned_yolo",
                homography_applicable=True, reference_match_correlation=1.0,
            ))
    return dets, true_positions


def test_fills_gap_accurately_on_a_clean_continuous_arc():
    """A single continuous flight segment (no direction reversal) should be
    filled close to the true position -- this is the module's core claim."""
    def position_fn(f):
        return 2.0 + 0.15 * f, 5.0 + 0.25 * f

    dets, true_positions = _make_detections(30, range(10, 20), position_fn)
    homography = _homography()

    # The pixel-y residual needs a real arc for the residual filter to have
    # something non-trivial to fit -- add it uniformly to EVERY frame's true
    # position (including gap frames), so the "ground truth" used for the
    # error check below matches what the real detections actually encode.
    def arc_at(f):
        return -40 * np.sin(np.pi * f / 29)

    true_positions = {f: (x, y + arc_at(f)) for f, (x, y) in true_positions.items()}
    arced = []
    for d in dets:
        if d.center is None:
            arced.append(d)
            continue
        cx, cy = d.center
        arced.append(CombinedBallDetectionResult(
            frame_index=d.frame_index, center=(cx, cy + arc_at(d.frame_index)), source=d.source,
            homography_applicable=True, reference_match_correlation=1.0,
        ))

    results = fit_trajectory_for_clip(arced, homography)
    filled = [r for r in results if r.source == "kalman_filled"]
    assert len(filled) == 10  # frames 10..19, all bounded by real detections at frame 9 and 20

    for r in filled:
        tx, ty = true_positions[r.frame_index]
        err = float(np.hypot(tx - r.center[0], ty - r.center[1]))
        assert err < 5.0, f"frame {r.frame_index}: {err:.2f}px error, expected a tight fit on a clean synthetic arc"


def test_refuses_to_fill_across_a_direction_reversal():
    """A real discontinuity (e.g. a shot back) inside a gap must NOT be
    smoothed through -- the reset/skip mechanism is what's supposed to catch
    this, and its failure to catch INSIDE-gap discontinuities is exactly why
    this module was not shipped (see PROGRESS.md). This test only pins the
    boundary case it DOES catch: a reversal visible at the gap's edge."""
    def position_fn(f):
        if f <= 14:
            return 2.0 + 0.15 * f, 5.0 + 0.25 * f
        return 2.0 + 0.15 * 14 - 0.20 * (f - 14), 5.0 + 0.25 * 14 - 0.30 * (f - 14)

    dets, _ = _make_detections(30, range(10, 20), position_fn)
    homography = _homography()

    results = fit_trajectory_for_clip(dets, homography)
    gap_results = [r for r in results if 10 <= r.frame_index <= 19]
    assert all(r.source == "none" for r in gap_results)
    assert all(r.skip_reason == "segment_boundary_in_gap" for r in gap_results)


def test_unbounded_edge_gaps_are_never_filled():
    """No real detection before frame 5 or after frame 24 -- those edge gaps
    must be left as genuine, explicitly-reasoned gaps, never extrapolated."""
    def position_fn(f):
        return 2.0 + 0.1 * f, 5.0 + 0.2 * f

    dets, _ = _make_detections(30, list(range(0, 5)) + list(range(25, 30)), position_fn)
    homography = _homography()

    results = fit_trajectory_for_clip(dets, homography)
    edge_results = [r for r in results if r.frame_index < 5 or r.frame_index >= 25]
    assert all(r.source == "none" for r in edge_results)
    assert all(r.skip_reason == "clip_edge_unbounded" for r in edge_results)
