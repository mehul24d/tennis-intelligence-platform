"""reference_video4_calibration.py — least-squares 8-point court calibration for
data/tennis/4.mp4 (Miami Open 2023, same match as 1.mp4/2.mp4/3.mp4, a different
point: 7-4(15)/6-5(15), players have switched ends vs 3.mp4). Calibrated
2026-07-19.

CAMERA MOTION CHECK, revised after a materially wrong first pass (left in
the git history / PROGRESS.md as the record of that correction, not
scrubbed): an initial vertical-only, fixed-narrow-window pixel scan found
what looked like a small ~4-5px drift and this module was first written
assuming a single mid-clip reference frame would cover the whole clip with
acceptable error. That was WRONG -- the narrow scan window was clipping at
some frames (a real bug in the diagnostic script, not the clip), and the
scan never checked horizontal (x) position at all. A corrected, wide,
clipping-guarded scan of 3 independent corners (BL, TL, BR) confirmed
this camera is NOT locked-off: a real, correlated, whole-frame horizontal
motion of up to ~45-68px (3-5x larger than 2.mp4's ~9-13px pan), with a
non-monotonic shape -- stable, then a smooth dip-and-return, stable again,
then a smooth rise, overshooting back past the original position and
continuing to drift without re-stabilizing before the clip ends.

Full characterization (BL corner x-position, representative of all 3
points checked): frames 0-420 stable at ~195.5-196px; a smooth ramp down
to a brief (~30-frame) flat bottom at ~150.5px around frames 485-515, then
a smooth ramp back up; frames 740-1240 stable again at ~195.0-196px --
STATISTICALLY THE SAME POSITION as frames 0-420 (confirmed directly: this
module's calibration, built from frame 900, validates against frame 0 at
0.2px held-out error); frames ~1250-1320 ramp up to a brief (~30-frame)
flat peak at ~218.5-219px around 1280-1310; then a smooth decline
continuing past the original baseline down to a genuinely flat ~162-164px
for the final ~44 frames (1500-1543, the clip's last frame) -- this tail
does settle, it is not still mid-drift when the clip ends, but 44 frames
is short and butts against the clip boundary with no trailing margin.

DECISION: only ONE homography is built here, covering the two genuinely
long, stable, position-matched windows (frames 0-420 and 740-1240, ~921
frames combined, ~60% of the clip). The two ~30-frame turning points (dip
bottom, peak) and the ~44-frame end tail are deliberately NOT given their
own calibrations -- each is real and genuinely flat (not noise), but too
brief relative to this project's verification standard (which calls for
checking 3+ meaningfully-separated frames spanning a segment) to produce a
calibration that could be verified with the same rigor as every other one
in this project. Frames 421-739 and 1241-1543 (the two ramps, including
the brief turning points and end tail within them) are excluded from
confident court-line overlay rendering rather than force-calibrated or
approximated -- same exclusion principle as 2.mp4's ramp, applied to two
regions here. See PROGRESS.md for the full investigation, including the
diagnostic-script bug that produced the wrong initial conclusion and how
it was caught.

All 8 points measured via numeric pixel-brightness thresholding (the 3.mp4
lesson), not eyeballed grid crops, from frame 900 (within the 740-1240
stable window).
"""

from __future__ import annotations

from cv_pipeline.homography import COURT_LENGTH_M, DOUBLES_WIDTH_M, SINGLES_INSET_M
from cv_pipeline.homography import CourtHomography

_NEAR_SVC_Y = COURT_LENGTH_M / 2 - 6.4
_FAR_SVC_Y = COURT_LENGTH_M / 2 + 6.4

# (world_xy_meters, pixel_xy) -- frame 900 (middle of the observed small
# drift window) of data/tennis/4.mp4, 1920x1080.
VIDEO4_CALIBRATION_POINTS = [
    ((0.0, 0.0), (189.0, 829.0)),                                    # BL -- doubles corner, near-left
    ((DOUBLES_WIDTH_M, 0.0), (1719.0, 829.0)),                       # BR -- doubles corner, near-right
    ((DOUBLES_WIDTH_M, COURT_LENGTH_M), (1338.0, 254.0)),            # TR -- doubles corner, far-right
    ((0.0, COURT_LENGTH_M), (582.0, 254.0)),                         # TL -- doubles corner, far-left
    ((SINGLES_INSET_M, _NEAR_SVC_Y), (501.0, 610.0)),                # near_svc_L -- singles sideline x near service line
    ((DOUBLES_WIDTH_M - SINGLES_INSET_M, _NEAR_SVC_Y), (1417.0, 610.0)),  # near_svc_R -- singles sideline x near service line
    ((DOUBLES_WIDTH_M / 2, 0.0), (958.0, 823.0)),                    # baseline_center -- doubles baseline center mark
    ((DOUBLES_WIDTH_M / 2, _FAR_SVC_Y), (960.0, 329.0)),             # far-T -- center line x far service line
]

VIDEO4_HELD_OUT_NEAR_T_PX = (959.0, 608.0)
VIDEO4_HELD_OUT_NET_PX = (960.0, 445.0)


def build_video4_homography() -> CourtHomography:
    return CourtHomography.from_point_correspondences(VIDEO4_CALIBRATION_POINTS)
