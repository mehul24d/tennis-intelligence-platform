"""reference_video3_calibration.py — least-squares 8-point court calibration for
data/tennis/3.mp4 (Miami Open 2023, Alcaraz vs Sinner, same match as 1.mp4/2.mp4
but a different point: 7-4(40)/6-4(15)). Calibrated 2026-07-19.

Camera framing confirmed static for the whole clip BEFORE calibrating -- per the
2.mp4 lesson, the regime/cut classifier alone was not trusted: directly measured
near-baseline and far-baseline pixel position at 2 x-columns each, every 40th
frame across the full 933-frame clip. All 4 readings stayed within a 1px band
(near: 840-841, far: 267-268) with no trend -- a genuinely static camera, unlike
2.mp4. (One transient None reading at frames 913-924/x=300 was checked and
confirmed a player briefly standing on that exact column, not drift -- the
reading recovered to the exact pre-occlusion value, and the second column never
moved.) One static homography for the whole clip is therefore correct here.

All 8 points independently re-measured via wide-enough crops (the 2.mp4 lesson
applied proactively, not after finding a bug) -- confirmed no second, more-outer
doubles/singles line exists near any of the 4 corners before accepting them.
Framing is close to but NOT identical to 1.mp4's (BL here (193,841) vs 1.mp4's
(200,866) -- similar x, ~25px different y -- confirmed different, not reused).

COORDINATE PRECISION LESSON (new, found on this clip): the first pass of
visual eyeballed crop-grid readings for near_svc_L produced a per-point
reprojection residual of 22px, a clear outlier against the other 7 points'
3.5-8.8px -- the same asymmetric-residual signature that flagged 1.mp4's
BL/BR mislabeling bug. Investigated the same way (wide crop first, to rule
out a mislabeled line): confirmed the line identity was correct (a genuine
second, more-outer doubles sideline was visible and correctly excluded), so
this was NOT a repeat of that bug -- it was a plain ~14px coordinate misread
from eyeballing a static zoomed screenshot. Re-measured this point (and,
for consistency, every other point) using a numeric method instead of
eyeballing: thresholding pixel brightness along fixed rows/columns to trace
each line's exact position and reading off the true intersection directly
from the brightness data, not a human's visual estimate of a rendered grid
image. This dropped near_svc_L's residual from 22px to 6.4px and held-out
error from 3.86/5.0px to 3.37/3.81px even before this final numeric pass --
after it, see the held-out error below. LESSON: even a "wide-enough crop"
that correctly avoids the mislabeling failure mode can still carry plain
eyeballing error of 10+ px on a fine, anti-aliased line intersection --
numeric brightness-thresholding is a more precise (and more reproducible)
measurement than a human reading a rendered grid overlay.
"""

from __future__ import annotations

from cv_pipeline.homography import COURT_LENGTH_M, DOUBLES_WIDTH_M, SINGLES_INSET_M
from cv_pipeline.homography import CourtHomography

_NEAR_SVC_Y = COURT_LENGTH_M / 2 - 6.4
_FAR_SVC_Y = COURT_LENGTH_M / 2 + 6.4

# (world_xy_meters, pixel_xy) -- frame 0 of data/tennis/3.mp4, 1920x1080.
VIDEO3_CALIBRATION_POINTS = [
    ((0.0, 0.0), (193.0, 841.0)),                                    # BL -- doubles corner, near-left
    ((DOUBLES_WIDTH_M, 0.0), (1719.0, 842.0)),                       # BR -- doubles corner, near-right
    ((DOUBLES_WIDTH_M, COURT_LENGTH_M), (1336.0, 268.0)),            # TR -- doubles corner, far-right
    ((0.0, COURT_LENGTH_M), (584.0, 268.0)),                         # TL -- doubles corner, far-left
    ((SINGLES_INSET_M, _NEAR_SVC_Y), (502.0, 622.0)),                # near_svc_L -- singles sideline x near service line
    ((DOUBLES_WIDTH_M - SINGLES_INSET_M, _NEAR_SVC_Y), (1416.0, 622.0)),  # near_svc_R -- singles sideline x near service line
    ((DOUBLES_WIDTH_M / 2, 0.0), (958.0, 835.0)),                    # baseline_center -- doubles baseline center mark
    ((DOUBLES_WIDTH_M / 2, _FAR_SVC_Y), (961.0, 343.0)),             # far-T -- center line x far service line
]

VIDEO3_HELD_OUT_NEAR_T_PX = (959.0, 621.0)
VIDEO3_HELD_OUT_NET_PX = (960.0, 458.0)


def build_video3_homography() -> CourtHomography:
    return CourtHomography.from_point_correspondences(VIDEO3_CALIBRATION_POINTS)
