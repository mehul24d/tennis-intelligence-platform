"""reference_video5_calibration.py — least-squares 8-point court calibration for
data/tennis/5.mp4 (Miami Open 2023, same match as 1.mp4-4.mp4, a different point:
Alcaraz 7-4(0-15) / Sinner 6-6(0-40)). Calibrated 2026-07-19.

CAMERA MOTION CHECK (mandatory per project methodology -- the regime/cut
classifier reported "validated", 0 cuts, which as established on 2.mp4/4.mp4
says nothing about smooth in-shot pans): a direct pixel-brightness scan of 3
independent points (near-left corner x, far-left corner x, far-right corner x)
across the full 940-frame clip found a real, one-way pan-and-settle -- a
DIFFERENT shape from both 2.mp4 (small there-and-back pan) and 4.mp4 (complex
multi-segment there-and-back-and-overshoot). Here the camera holds one
position for ~135 frames, then transitions smoothly and monotonically over
~264 frames to a new position, and holds that new position for the rest of
the clip (~540 frames) -- it never returns to the original framing.

Full characterization: frames 0-~135 stable (near-left-corner x ~137-141,
far-left-corner x ~523-526, far-right-corner x ~1247-1250); frames ~136-399 a
real, sustained, monotonic transition (confirmed visually via a red/green
pixel overlay of frame 0 vs frame 320 vs frame 900 -- unambiguous corner
displacement); frames ~400-939 stable at a new position (near-left-corner x
~212-218, far-left-corner x ~592-594, far-right-corner x ~1316-1317) through
the end of the clip.

METHODOLOGY BUG FOUND AND FIXED (new, on this clip): cv2.CAP_PROP_POS_FRAMES
seeking proved frame-INACCURATE for some indices in this file -- one
seek-based diagnostic scan asked for frame 120 and silently returned frame
~178's content instead (~60-frame error), caught only because a coarser
seek-based scan gave a different, correct value at the same nominal index.
All final measurements in this module are cross-verified via zero-seek
sequential decoding (reading every frame via .read() in order, never
cap.set()), not trusted from any single seek.

SECOND BUG FOUND AND FIXED: an initial quick 8-point pass measured the far
corners (TL) by taking the mean of the first bright pixel-cluster in a scan
row. This is correct for a narrow, isolated corner blob but silently WRONG
when the scan row intersects a long, fully-painted, continuously-bright line
(the far baseline) -- the "cluster" spans from the true corner all the way to
wherever the next line crosses it (here, the singles sideline), and the mean
lands on a meaningless midpoint. This produced a first-pass TL of (611,
273.5) for Segment A that was off by ~85px in x from the true corner. Caught
via cross-validation: building a homography from the OTHER 7 points and
checking the predicted TL position (537, 282) disagreed with the measured
(611, 273.5) by 74px, while every other point was internally consistent to
within a few px. Re-measured TL using the correct method (find the row's true
ENDPOINT via explicit min/max tracking, not cluster-mean) and got (526, 275)
-- held-out error then dropped from 10.22/17.98px to 0.75/0.30px. LESSON:
mean-of-cluster is only valid for point-like blobs; corner-finding on a row
that also intersects an extended line must use the row's endpoint.

Segment A calibrated from frame 120 (within the clean 100-135 sub-window;
frames 0-~100 have the near-right corner (BR) occluded by a player standing
at that exact spot for the first ~100 frames of the segment -- confirmed
visually, not assumed, via corner crops at frames 0/40/80 (occluded) and
105/120 (clear)). Segment B calibrated from frame 600 (well within the
400-939 stable window, avoiding several single-frame/multi-frame
occlusion artifacts independently identified during the pan-check: near_xL
spikes at frames ~650/690/700/870/880, far_xR wobble at ~790-860/870-910 --
all confirmed as local occlusions, not camera motion, because the other 2 of
3 independently-tracked points stayed flat at those same frames).

All 8 points per segment measured via numeric pixel-brightness thresholding
(the 3.mp4 lesson), independently for each segment (not copied/offset from
one another).
"""

from __future__ import annotations

from cv_pipeline.homography import COURT_LENGTH_M, DOUBLES_WIDTH_M, SINGLES_INSET_M
from cv_pipeline.homography import CourtHomography

_NEAR_SVC_Y = COURT_LENGTH_M / 2 - 6.4
_FAR_SVC_Y = COURT_LENGTH_M / 2 + 6.4

# Segment A: frames [0, 136) -- original camera position, calibrated from
# frame 120 of data/tennis/5.mp4, 1920x1080.
VIDEO5_SEGMENT_A_POINTS = [
    ((0.0, 0.0), (141.0, 837.0)),                                     # BL
    ((DOUBLES_WIDTH_M, 0.0), (1621.0, 821.0)),                        # BR
    ((DOUBLES_WIDTH_M, COURT_LENGTH_M), (1247.0, 273.5)),             # TR
    ((0.0, COURT_LENGTH_M), (526.0, 275.0)),                          # TL
    ((SINGLES_INSET_M, _NEAR_SVC_Y), (442.0, 618.0)),                 # near_svc_L
    ((DOUBLES_WIDTH_M - SINGLES_INSET_M, _NEAR_SVC_Y), (1329.0, 612.0)),  # near_svc_R
    ((DOUBLES_WIDTH_M / 2, 0.0), (890.0, 822.0)),                     # baseline_center
    ((DOUBLES_WIDTH_M / 2, _FAR_SVC_Y), (888.0, 346.0)),              # far-T
]
VIDEO5_SEGMENT_A_HELD_OUT_NEAR_T_PX = (889.0, 615.0)
VIDEO5_SEGMENT_A_HELD_OUT_NET_PX = (889.0, 455.0)

# Segment B: frames [400, 940) -- shifted camera position, calibrated from
# frame 600 of data/tennis/5.mp4, 1920x1080.
VIDEO5_SEGMENT_B_POINTS = [
    ((0.0, 0.0), (216.5, 830.0)),                                     # BL
    ((DOUBLES_WIDTH_M, 0.0), (1693.0, 828.0)),                        # BR
    ((DOUBLES_WIDTH_M, COURT_LENGTH_M), (1317.0, 273.5)),             # TR
    ((0.0, COURT_LENGTH_M), (593.0, 273.5)),                          # TL
    ((SINGLES_INSET_M, _NEAR_SVC_Y), (515.5, 612.0)),                 # near_svc_L
    ((DOUBLES_WIDTH_M - SINGLES_INSET_M, _NEAR_SVC_Y), (1392.0, 612.0)),  # near_svc_R
    ((DOUBLES_WIDTH_M / 2, 0.0), (955.0, 822.0)),                     # baseline_center
    ((DOUBLES_WIDTH_M / 2, _FAR_SVC_Y), (955.0, 346.0)),              # far-T
]
VIDEO5_SEGMENT_B_HELD_OUT_NEAR_T_PX = (955.0, 613.0)
VIDEO5_SEGMENT_B_HELD_OUT_NET_PX = (955.0, 456.0)

# (start inclusive, end exclusive). Frames [136, 400) are the transition --
# real, sustained camera motion, excluded from confident calibration, same
# treatment as 2.mp4's ramp and 4.mp4's two ramps.
VIDEO5_EXCLUDED_RANGES = [(136, 400)]


def build_video5_segment_a_homography() -> CourtHomography:
    return CourtHomography.from_point_correspondences(VIDEO5_SEGMENT_A_POINTS)


def build_video5_segment_b_homography() -> CourtHomography:
    return CourtHomography.from_point_correspondences(VIDEO5_SEGMENT_B_POINTS)
