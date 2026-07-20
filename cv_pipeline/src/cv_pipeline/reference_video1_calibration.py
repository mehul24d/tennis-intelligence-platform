"""reference_video1_calibration.py — the least-squares 8-point court calibration
for data/tennis/1.mp4 (Miami Open 2023, Alcaraz vs Sinner), adopted 2026-07-17 for
the Master Build Prompt reference pipeline's Phase 6 render, corrected 2026-07-18.

MANDATORY for any new calibration module (added 2026-07-19, see
calibration_verification.py and PROGRESS.md): a checked-in
CalibrationVerificationManifest confirming all 4 doubles corners were visually
verified on >=3 frames (start/middle/end) is required and enforced by
cv_pipeline/tests/test_calibration_verification.py -- a held-out-landmark
numeric error alone is not sufficient (see this module's own BL/BR history and
reference_video2_calibration.py's BL history for why).

CORRECTED 2026-07-18: the original version of this calibration had BL/BR (the
near-baseline doubles corners) mislabeled -- the pixel points actually clicked,
(400,863)/(1518,863), are where the SINGLES sideline crosses the near baseline,
not the doubles corner. The real doubles corners are further out, at
(200,866)/(1718,866) -- confirmed by widening the crop and finding a second,
more-outer diagonal line at the near baseline that the original click missed.
This single mislabeling was the root cause of TWO separately-reported symptoms
that turned out to be one bug: (1) the rendered court-outline overlay showing
doubles width at the far baseline but only singles width at the near baseline
(a real geometric inconsistency -- both baselines must use the same corner type),
and (2) the near_svc_L/near_svc_R residual-asymmetry finding logged as an open
question on 2026-07-17 (50-60px reprojection residual vs 7-32px for the other 6
points) -- those two points were themselves correctly identified as singles-line
intersections all along; their high residual was the least-squares fit straining
against the corrupted BL/BR anchors, not an error of their own. After the fix,
their residuals drop to 7-18px, in line with every other point.

Held-out-landmark error after the fix: **near-T 4.4px, net-base 2.0px** -- versus
27.4px/24.9px before the fix, and the original 4-point calibration's 74.9px/45.7px.
This is now BETTER than video1(the other dev clip)'s own ~13px benchmark, not
just closer to it -- the "residual gap" reported on 2026-07-17 was itself an
artifact of this bug, not a real limitation of the least-squares technique. See
PROGRESS.md's "Court-Outline Rendering Bug" entry for the full investigation.
"""

from __future__ import annotations

from cv_pipeline.homography import COURT_LENGTH_M, DOUBLES_WIDTH_M, SINGLES_INSET_M
from cv_pipeline.homography import CourtHomography

_NEAR_SVC_Y = COURT_LENGTH_M / 2 - 6.4  # ITF: service line 6.4m (21ft) from net
_FAR_SVC_Y = COURT_LENGTH_M / 2 + 6.4

# (world_xy_meters, pixel_xy) -- frame 0 of data/tennis/1.mp4, 1920x1080.
# Labels record which real-world line each pixel point was verified against
# (doubles sideline corner vs. singles sideline crossing) -- see module
# docstring for why this distinction matters and was previously gotten wrong
# for BL/BR specifically.
VIDEO1_CALIBRATION_POINTS = [
    ((0.0, 0.0), (200.0, 866.0)),                                    # BL -- doubles corner, near-left
    ((DOUBLES_WIDTH_M, 0.0), (1718.0, 866.0)),                       # BR -- doubles corner, near-right
    ((DOUBLES_WIDTH_M, COURT_LENGTH_M), (1330.0, 300.0)),            # TR -- doubles corner, far-right
    ((0.0, COURT_LENGTH_M), (598.0, 300.0)),                         # TL -- doubles corner, far-left
    ((SINGLES_INSET_M, _NEAR_SVC_Y), (528.0, 649.0)),                # near_svc_L -- singles sideline x near service line
    ((DOUBLES_WIDTH_M - SINGLES_INSET_M, _NEAR_SVC_Y), (1408.0, 650.0)),  # near_svc_R -- singles sideline x near service line
    ((DOUBLES_WIDTH_M / 2, 0.0), (958.0, 862.0)),                    # baseline_center -- doubles baseline center mark
    ((DOUBLES_WIDTH_M / 2, _FAR_SVC_Y), (963.0, 373.0)),             # far-T -- center line x far service line
]

# Held out of calibration, used only to validate: near-T (center-service-line x
# near-service-line intersection) and net-base (center of net at ground level).
VIDEO1_HELD_OUT_NEAR_T_PX = (966.0, 645.0)
VIDEO1_HELD_OUT_NET_PX = (966.0, 484.0)


def build_video1_homography() -> CourtHomography:
    return CourtHomography.from_point_correspondences(VIDEO1_CALIBRATION_POINTS)
