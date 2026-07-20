"""reference_video2_calibration.py — least-squares 8-point court calibration for
data/tennis/2.mp4 (Miami Open 2023, Alcaraz vs Sinner, same match as
data/tennis/1.mp4 but a different, later point). Calibrated 2026-07-19,
corrected same day.

MANDATORY for any new calibration module (added 2026-07-19, see
calibration_verification.py and PROGRESS.md): a checked-in
CalibrationVerificationManifest confirming all 4 doubles corners were visually
verified on >=3 frames (start/middle/end) is required and enforced by
cv_pipeline/tests/test_calibration_verification.py -- this module's own BL bug
below is exactly why a held-out-landmark numeric error alone isn't sufficient.

Independently calibrated from scratch, NOT copied from reference_video1_calibration.py
-- per instruction, this clip's camera framing was not assumed to match 1.mp4's
pixel-for-pixel even though it's the same physical camera/setup, since a "locked"
camera can still differ in exact zoom/crop between points. Confirmed different on
inspection: this clip's court corners sit at different pixel coordinates than
1.mp4's (e.g. near-baseline-left here is (200,879) vs 1.mp4's (200,866)).

CORRECTED 2026-07-19: the original BL point was itself mismeasured -- clicked at
(249,878), ~49px off from the true corner at (200,879), a plain reading error, not
a rendering/coordinate-system bug. That was checked and ruled out FIRST, per
instruction, before touching the calibration: pixel values passed through the
drawing code unchanged (compared the exact JSON-served coordinates against a
fresh crop), corner labels/ordering were correct (no doubles/singles swap), and
the camera does not drift within this clip (confirmed by comparing frame 0 and
frame 670's corner pixel position directly -- identical). Found when the
rendered overlay visibly missed the true left doubles sideline; re-measured all
4 corners individually afterward to confirm only BL was wrong (BR/TL/TR all
matched their original values exactly, confirmed via BL_wide_check.jpg/
BR_wide_check.jpg/TL_remeasure.jpg/TR_remeasure.jpg in
cv_pipeline/scratch_output/tennis2/).

LESSON: the wide-enough-crop check from 1.mp4 prevents a *mislabeling* error
(clicking the wrong line entirely) but does not prevent a plain misreading of
one point's coordinates on the correct line. A held-out-landmark error that
looks reasonable is not sufficient evidence a calibration is correct -- the
least-squares fit can partially absorb one badly-wrong point's error into the
other 7, keeping the aggregate number deceptively low. Verify every calibration
point's rendered position against a real frame directly, not just the
aggregate held-out numeric error, before trusting a calibration.

Held-out-landmark error (near-T, net-base -- same two landmarks used for 1.mp4,
NOT used in this calibration), corrected: near-T 6.98px, net-base 1.1px -- both
comfortably under 1.mp4's corrected ~4.4px/2.0px and the ~13px benchmark.
Per-point reprojection residuals are uniform (4.86-11.53px across all 8 points).
"""

from __future__ import annotations

from cv_pipeline.homography import COURT_LENGTH_M, DOUBLES_WIDTH_M, SINGLES_INSET_M
from cv_pipeline.homography import CourtHomography

_NEAR_SVC_Y = COURT_LENGTH_M / 2 - 6.4
_FAR_SVC_Y = COURT_LENGTH_M / 2 + 6.4

# (world_xy_meters, pixel_xy) -- frame 0 (and last frame, for points obscured by
# a player at frame 0) of data/tennis/2.mp4, 1920x1080.
VIDEO2_CALIBRATION_POINTS = [
    ((0.0, 0.0), (200.0, 879.0)),                                    # BL -- doubles corner, near-left (corrected)
    ((DOUBLES_WIDTH_M, 0.0), (1720.0, 878.0)),                       # BR -- doubles corner, near-right
    ((DOUBLES_WIDTH_M, COURT_LENGTH_M), (1327.0, 298.0)),            # TR -- doubles corner, far-right
    ((0.0, COURT_LENGTH_M), (578.0, 297.0)),                         # TL -- doubles corner, far-left
    ((SINGLES_INSET_M, _NEAR_SVC_Y), (508.0, 645.0)),                # near_svc_L -- singles sideline x near service line
    ((DOUBLES_WIDTH_M - SINGLES_INSET_M, _NEAR_SVC_Y), (1400.0, 645.0)),  # near_svc_R -- singles sideline x near service line
    ((DOUBLES_WIDTH_M / 2, 0.0), (950.0, 861.0)),                    # baseline_center -- doubles baseline center mark
    ((DOUBLES_WIDTH_M / 2, _FAR_SVC_Y), (952.0, 362.0)),             # far-T -- center line x far service line
]

VIDEO2_HELD_OUT_NEAR_T_PX = (952.0, 643.0)
VIDEO2_HELD_OUT_NET_PX = (952.0, 481.0)


def build_video2_homography() -> CourtHomography:
    return CourtHomography.from_point_correspondences(VIDEO2_CALIBRATION_POINTS)
