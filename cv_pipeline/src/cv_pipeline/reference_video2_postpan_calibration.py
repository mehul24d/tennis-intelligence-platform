"""reference_video2_postpan_calibration.py — second least-squares 8-point court
calibration for data/tennis/2.mp4, covering frames ~560-1343 (the post-pan
segment), added 2026-07-19.

WHY THIS MODULE EXISTS: re-verifying reference_video2_calibration.py's manifest
against the new stricter per-corner-coordinate schema surfaced a real, confirmed
(direct pixel-intensity measurement, not eyeballing) mid-clip camera pan in
data/tennis/2.mp4 -- a gradual ~9-13px vertical shift of both baselines between
frames ~400-560, stable before and after. See PROGRESS.md's "data/tennis/2.mp4
Has a Genuine Mid-Clip Camera Pan" entry for the full measurement. A single
static homography (reference_video2_calibration.py, built from frame-0 corners)
is only accurate for frames 0-~560; this module covers frames ~560-1343.

Calibrated from frame 960 (middle of the stable post-pan window), independently
re-measured from scratch via wide-enough crops (not derived by applying an
assumed uniform pixel offset to the pre-pan points -- the wide-crop check on BR
found a second, more-inner singles-sideline diagonal that the naive
offset-BL/BR-corners approach could have confused with the real doubles corner,
same failure mode this project has hit before).

Held-out-landmark error (near-T, net-base -- not used in this calibration):
near-T 4.30px, net-base 1.69px -- both under the ~13px benchmark and
comparable to reference_video1_calibration.py's 4.4px/2.0px and this clip's
own pre-pan calibration's 6.98px/1.1px. Per-point reprojection residuals
across all 8 calibration points are uniform (3.8-10.67px, no outliers).

The ~400-560 ramp itself is deliberately NOT covered by either this module or
reference_video2_calibration.py -- see PROGRESS.md for why (continuous drift
during the ramp means no single frame in it has an unambiguous "correct"
homography, and it's a small fraction of the clip, ~2.7s of 22.4s). Frames in
that range should be treated as calibration-inapplicable for confident overlay
rendering, not silently assigned to whichever segment's homography is "closer."
"""

from __future__ import annotations

from cv_pipeline.homography import COURT_LENGTH_M, DOUBLES_WIDTH_M, SINGLES_INSET_M
from cv_pipeline.homography import CourtHomography

_NEAR_SVC_Y = COURT_LENGTH_M / 2 - 6.4
_FAR_SVC_Y = COURT_LENGTH_M / 2 + 6.4

# (world_xy_meters, pixel_xy) -- frame 960 of data/tennis/2.mp4, 1920x1080,
# post-pan stable framing (frames ~560-1343).
VIDEO2_POSTPAN_CALIBRATION_POINTS = [
    ((0.0, 0.0), (197.0, 864.0)),                                    # BL -- doubles corner, near-left
    ((DOUBLES_WIDTH_M, 0.0), (1718.0, 863.0)),                       # BR -- doubles corner, near-right
    ((DOUBLES_WIDTH_M, COURT_LENGTH_M), (1327.0, 289.0)),            # TR -- doubles corner, far-right
    ((0.0, COURT_LENGTH_M), (578.0, 289.0)),                         # TL -- doubles corner, far-left
    ((SINGLES_INSET_M, _NEAR_SVC_Y), (500.0, 636.0)),                # near_svc_L -- singles sideline x near service line
    ((DOUBLES_WIDTH_M - SINGLES_INSET_M, _NEAR_SVC_Y), (1400.0, 636.0)),  # near_svc_R -- singles sideline x near service line
    ((DOUBLES_WIDTH_M / 2, 0.0), (950.0, 855.0)),                    # baseline_center -- doubles baseline center mark
    ((DOUBLES_WIDTH_M / 2, _FAR_SVC_Y), (952.0, 353.0)),             # far-T -- center line x far service line
]

VIDEO2_POSTPAN_HELD_OUT_NEAR_T_PX = (952.0, 636.0)
VIDEO2_POSTPAN_HELD_OUT_NET_PX = (952.0, 472.0)


def build_video2_postpan_homography() -> CourtHomography:
    return CourtHomography.from_point_correspondences(VIDEO2_POSTPAN_CALIBRATION_POINTS)
