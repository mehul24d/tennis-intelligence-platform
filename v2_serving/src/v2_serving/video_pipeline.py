"""video_pipeline.py — orchestrates cv_pipeline's existing building blocks
(CourtHomography, select_players_by_court_position, pose_estimation, YOLO+ByteTrack)
into a LIVE INFERENCE run over an arbitrary video, for POST /analyze-video's
background task.

This is NOT the same thing as cv_pipeline's own validation scripts
(run_full_detection_validation.py etc.), which compare detections against the 10
amateur clips' ground-truth annotations and report accuracy. There is no ground
truth for an arbitrary uploaded/pathed video -- this module reports what was
OBSERVED (detection rates, pose success, track-id churn), not accuracy against a
known-correct answer. The result JSON says this explicitly (`"ground_truth"` field)
so a caller can't mistake a live inference rate for a validated accuracy figure the
way Phase 3's `EVALUATION_REPORT.md` numbers were.

PER-FRAME DATA (added in Phase 5, step 4 -- the video-player overlay): the
original Phase 4 version of this module returned ONLY aggregate rates
(near_player_detection_live_estimate.rate, etc.) -- there was no per-frame box
data anywhere in the result, and no way for a frontend overlay to draw a real
detection box at a real video timestamp. This was discovered as a real gap while
building Phase 5's overlay feature (the API's actual shape didn't match what
the overlay needed), not assumed away -- fixed here by also recording a
`frames` array (one entry per processed frame: near/far player boxes, ball box,
near/far track IDs, ALL OR NOTHING -- a frame with no detection has an explicit
None/null there, never an omitted entry, so "nothing detected" and "this frame
wasn't processed" stay distinguishable on the frontend). `homography.court_corners`
and top-level `video_width`/`video_height` were added for the same reason (the
overlay needs real pixel coordinates for the court quadrilateral, not just a
Status string).

INTEGRATION FRICTION, reported plainly rather than hidden (per Phase 4's own
verification discipline): cv_pipeline's `player_selection.select_players_by_court_position`
-- the fix that correctly distinguishes real players from courtside bystanders --
REQUIRES a homography, which in turn requires either annotated court corners (only
exist for the 10 known amateur dev clips) or a manual per-clip calibration (not
automated anywhere in cv_pipeline). For a genuinely new/arbitrary video with
neither, there is currently no automated way to get a homography, so player
selection falls back to a simple largest-box-is-near / smallest-remaining-box-is-far
heuristic -- explicitly the SAME heuristic already confirmed unreliable in Phase 3
(picks bystanders on video9 and the stress-test clip). This is flagged in the
result JSON (`"player_selection_method"`) rather than silently used as if it were
the fixed, validated approach. Closing this gap (an automated court-corner
detector, or a required manual-calibration step in the API) is future work, not
solved in this phase.

BALL DETECTION (switched to the combined method as the default, 2026-07-16):
previously this module always ran stock COCO-class YOLO for ball_box, the same
method whose ~7.8% recall baseline is documented in EVALUATION_REPORT.md. Now,
whenever a real homography is available (the 10 known amateur clips) AND
`classify_ball_detection_regime` classifies the clip's camera framing as
"validated" (see ball_detection_combined.py), this module runs the validated
combined method instead (fine-tuned YOLOv8n + frequency-based artifact filter +
court-region motion-diff fallback -- 53.91% pooled recall on the amateur
ground-truth dataset, vs 7.81% for stock YOLO; corrected 2026-07-16 from an
initially-reported 70.40% that turned out to reflect a ground-truth leak in
the prototyping script, not real production performance -- see
ball_detection_combined.py's GROUND-TRUTH LEAK note and PROGRESS.md). Every OTHER case (no
homography, or a "best_effort"-classified clip) still falls back to stock YOLO
exactly as before -- this is a regime-GATED switch, not a blanket replacement,
because the combined method's real recall was only ever validated on
locked-camera, single-continuous-shot amateur footage (see
ball_detection_combined.py's own docstring for why). Which method actually ran
is surfaced in the result JSON's `ball_detection_live_estimate.method` field
("combined_v2" or "stock_yolo"), not silently used interchangeably. Per-frame
`homography_applicable` (added alongside this switch -- see PROGRESS.md) is
also now threaded into each frame record when the combined method runs, so the
dashboard overlay can correctly suppress the court-line drawing on any
individual frame whose camera angle doesn't match the calibrated homography,
even within an overall "validated"-regime clip.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

CV_PIPELINE_SRC = Path(__file__).resolve().parents[3] / "cv_pipeline" / "src"
import sys
if str(CV_PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(CV_PIPELINE_SRC))

from cv_pipeline.annotations import DEFAULT_ANNOTATIONS_DIR, load_clip_annotations
from cv_pipeline.ball_detection_combined import (
    FINE_TUNED_MODEL_PATH, classify_ball_detection_regime, run_combined_ball_detection_for_clip,
)
from cv_pipeline.homography import CourtHomography
from cv_pipeline.reference_video1_calibration import (
    VIDEO1_HELD_OUT_NEAR_T_PX, VIDEO1_HELD_OUT_NET_PX, build_video1_homography,
)
from cv_pipeline.reference_video2_calibration import (
    VIDEO2_HELD_OUT_NEAR_T_PX, VIDEO2_HELD_OUT_NET_PX, build_video2_homography,
)
from cv_pipeline.reference_video2_postpan_calibration import (
    VIDEO2_POSTPAN_HELD_OUT_NEAR_T_PX, VIDEO2_POSTPAN_HELD_OUT_NET_PX, build_video2_postpan_homography,
)
from cv_pipeline.reference_video3_calibration import (
    VIDEO3_HELD_OUT_NEAR_T_PX, VIDEO3_HELD_OUT_NET_PX, build_video3_homography,
)
from cv_pipeline.reference_video4_calibration import (
    VIDEO4_HELD_OUT_NEAR_T_PX, VIDEO4_HELD_OUT_NET_PX, build_video4_homography,
)
from cv_pipeline.reference_video5_calibration import (
    VIDEO5_SEGMENT_A_HELD_OUT_NEAR_T_PX, VIDEO5_SEGMENT_A_HELD_OUT_NET_PX,
    VIDEO5_SEGMENT_B_HELD_OUT_NEAR_T_PX, VIDEO5_SEGMENT_B_HELD_OUT_NET_PX,
    build_video5_segment_a_homography, build_video5_segment_b_homography,
)
from cv_pipeline.player_selection import PlayerContinuityTracker, select_players_by_court_position
from cv_pipeline.pose_estimation import make_landmarker, run_pose_on_box
from cv_pipeline.schema import Status
from cv_pipeline.shot_classification import find_shot_events, flag_first_event_as_probable_serve

# The 10 amateur clips are the only ones with annotated court corners available --
# used to build a real homography when the input happens to be one of them.
KNOWN_ANNOTATED_CLIPS = {f"video{i}" for i in range(1, 11)}
# video7's homography is confirmed to have a real scale issue (see PROGRESS.md /
# EVALUATION_REPORT.md) -- excluded from real-world-distance use, surfaced here too.
VIDEO7_KNOWN_ISSUE_NOTE = (
    "video7's annotated court corners span only the near half-court "
    "(baseline-to-net), not the full doubles court -- confirmed root cause in "
    "Phase 3 (see EVALUATION_REPORT.md). Homography excluded from real-world-"
    "distance use for this clip specifically."
)

PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32
POSE_CROP_MIN_SIDE_PX = 15  # sanity floor, avoids feeding a degenerate crop to MediaPipe

# shot_classification's peak-prominence check needs real signal on BOTH sides
# of a candidate swing to confirm it -- if `frame_limit` cuts off the
# trailing side, a real event near the boundary can be silently dropped with
# no error (confirmed directly: video1's real serve at frame 107 computed a
# prominence of 0.282, below MIN_SHOT_PROMINENCE=0.35, when only frames 0-199
# were available, vs. 0.545 -- correctly above threshold -- once frames
# 200-299 were included too; see PROGRESS.md's "Serve-Exclusion Heuristic"
# entry). This constant pads landmark/box/ball collection PAST frame_limit
# purely to give boundary events that missing right-side context -- these
# extra frames are never added to the returned `frames` array, never counted
# in n_frames_processed/detection rates, and never fed to the person tracker
# (only pose+box+ball are needed for shot classification). 300 frames (~5s at
# this project's ~60fps clips) is a generous empirical margin over the one
# real case measured (100 frames was enough there) -- NOT a mathematically
# guaranteed bound, since scipy's prominence search is theoretically
# unbounded (a confirming valley could in principle be arbitrarily far away;
# the real audit data has inter-shot gaps up to 596 frames). See
# shot_classification_live_estimate's own note in the result for the
# residual-risk disclosure this doesn't eliminate.
SHOT_CLASSIFICATION_PADDING_FRAMES = 300

# data/tennis/2.mp4 has a confirmed real mid-clip camera pan (direct pixel-
# intensity measurement, not eyeballing -- see PROGRESS.md's "data/tennis/2.mp4
# Has a Genuine Mid-Clip Camera Pan" entry). Both baselines are stable at
# frames 0-360, ramp smoothly through frames ~400-560, and are stable again at
# frames 560-1343 (~9-13px total vertical shift, no horizontal component). Two
# separate homographies exist -- reference_video2_calibration.py (pre-pan) and
# reference_video2_postpan_calibration.py (post-pan) -- rather than one clip-
# wide static calibration.
VIDEO2_PAN_RAMP_START_FRAME = 400
VIDEO2_PAN_RAMP_END_FRAME = 560  # exclusive: frame 560 is the first stable post-pan frame
# Ramp frames (400-559) have no single frame with an unambiguously "correct"
# homography -- the shift is continuous throughout, not a discrete jump. For
# player *selection* (a coarse, meter-scale-tolerance geometric check), the
# ramp is split at its midpoint and each half uses the nearer segment's
# homography, since a 9-13px pixel error is negligible next to the ~1-3m
# margins select_players_by_court_position operates on. For court-line
# *overlay rendering* (a pixel-precision visual claim), ramp frames are
# excluded entirely (homography_applicable=False) rather than approximated --
# see this module's run_video_analysis for where each of these is applied.
VIDEO2_PAN_SELECTION_BOUNDARY_FRAME = (VIDEO2_PAN_RAMP_START_FRAME + VIDEO2_PAN_RAMP_END_FRAME) // 2

# data/tennis/4.mp4's camera is not locked-off (confirmed with 3 independent
# corners moving together -- see reference_video4_calibration.py and
# PROGRESS.md). Two long, stable, POSITION-MATCHED windows exist (frames
# 0-420 and 740-1240 -- confirmed the same camera position, not two separate
# calibrations) and are covered by ONE homography. Everything else -- two
# brief (~30-frame) turning points and a ~44-frame end tail, none long/stable
# enough to independently calibrate to this project's verification standard
# -- falls inside these two excluded ranges and is marked
# homography_applicable=False for court-line overlay rendering, the same
# exclusion principle as 2.mp4's ramp, applied to two regions instead of one.
# Unlike 2.mp4, there is no second homography to switch to for player
# *selection* during the excluded ranges -- the single homography is used
# throughout for that coarser, meter-scale-tolerance purpose (the up-to-68px
# pixel shift is still small next to select_players_by_court_position's
# multi-meter margins).
VIDEO4_EXCLUDED_RANGES = [(421, 740), (1241, 1544)]  # (start inclusive, end exclusive)


def _video4_frame_excluded(frame_idx: int) -> bool:
    return any(start <= frame_idx < end for start, end in VIDEO4_EXCLUDED_RANGES)


# data/tennis/5.mp4 has a real, one-way camera pan-and-settle -- a DIFFERENT
# shape from both 2.mp4 (small there-and-back pan) and 4.mp4 (complex
# multi-segment there-and-back-and-overshoot). Confirmed via 3 independent
# pixel-tracked points plus a visual frame-overlay check (see
# reference_video5_calibration.py's docstring). Segment A (frames 0-135,
# original camera position) and Segment B (frames 400-939, shifted ~75-100px)
# are two long, stable, genuinely DIFFERENT positions -- unlike 4.mp4's two
# stable windows, which turned out to be the same position and share one
# homography. Frames 136-399 are the real, sustained transition and are
# excluded from confident court-line overlay rendering, same principle as
# 2.mp4's ramp and 4.mp4's two ramps.
VIDEO5_TRANSITION_START_FRAME = 136
VIDEO5_TRANSITION_END_FRAME = 400  # exclusive: frame 400 is the first stable Segment-B frame
# Same rationale as VIDEO2_PAN_SELECTION_BOUNDARY_FRAME: player *selection* is
# a coarse, meter-scale-tolerance check, so the brief transition is split at
# its midpoint and each half uses the nearer segment's homography rather than
# being left without one.
VIDEO5_SELECTION_BOUNDARY_FRAME = (VIDEO5_TRANSITION_START_FRAME + VIDEO5_TRANSITION_END_FRAME) // 2


def _clip_stem(video_path: Path) -> str:
    return video_path.stem


def _is_reference_video1(video_path: Path) -> bool:
    """data/tennis/1.mp4 (Miami Open 2023, the Master Build Prompt reference
    clip) -- matched on the resolved path's last two parts, not just the bare
    stem '1', since that alone is too generic to safely key off of (a
    differently-located clip could coincidentally also be named 1.mp4)."""
    parts = video_path.resolve().parts[-2:]
    return parts == ("tennis", "1.mp4")


def _is_reference_video2(video_path: Path) -> bool:
    """data/tennis/2.mp4 (Miami Open 2023, same match as 1.mp4, a different
    point) -- see _is_reference_video1 for why path-matching, not bare stem."""
    parts = video_path.resolve().parts[-2:]
    return parts == ("tennis", "2.mp4")


def _is_reference_video3(video_path: Path) -> bool:
    """data/tennis/3.mp4 (Miami Open 2023, same match as 1.mp4/2.mp4, a
    different point) -- see _is_reference_video1 for why path-matching, not
    bare stem."""
    parts = video_path.resolve().parts[-2:]
    return parts == ("tennis", "3.mp4")


def _is_reference_video4(video_path: Path) -> bool:
    """data/tennis/4.mp4 (Miami Open 2023, same match as 1.mp4/2.mp4/3.mp4,
    a different point) -- see _is_reference_video1 for why path-matching,
    not bare stem."""
    parts = video_path.resolve().parts[-2:]
    return parts == ("tennis", "4.mp4")


def _is_reference_video5(video_path: Path) -> bool:
    """data/tennis/5.mp4 (Miami Open 2023, same match as 1.mp4-4.mp4, a
    different point) -- see _is_reference_video1 for why path-matching, not
    bare stem."""
    parts = video_path.resolve().parts[-2:]
    return parts == ("tennis", "5.mp4")


def _build_homography_if_available(video_path: Path) -> tuple[CourtHomography | None, dict]:
    if _is_reference_video5(video_path):
        # Segment A (frames 0-135) is the "default" homography returned here
        # -- used for ball-detection regime classification and as the
        # court_corners/singles_corners fallback for any frame without a
        # per-frame override. Segment B is built and applied separately,
        # per-frame, inside run_video_analysis -- same pattern as 2.mp4's
        # pre/post-pan split.
        homography = build_video5_segment_a_homography()
        segment_b_homography = build_video5_segment_b_homography()
        return homography, {
            "status": Status.MEASURED.value,
            "note": (
                "TWO homographies for this clip, not one -- a real one-way camera "
                "pan-and-settle was found and confirmed by direct pixel measurement "
                "plus a visual frame-overlay check (see reference_video5_calibration.py "
                "docstring, 2026-07-19). Unlike 4.mp4, this clip's two long stable "
                "windows are NOT the same camera position (~75-100px apart). Segment A "
                "(frames 0-135, this entry's corners below): least-squares 8-point "
                "calibration, held-out error 0.75px "
                f"({VIDEO5_SEGMENT_A_HELD_OUT_NEAR_T_PX}) at near-T, 0.30px "
                f"({VIDEO5_SEGMENT_A_HELD_OUT_NET_PX}) at net-base -- the best result "
                "of any reference clip so far, after fixing a mean-of-cluster "
                "corner-measurement bug (see docstring). Segment B (frames 400-939): "
                "separate calibration, held-out error 1.96px "
                f"({VIDEO5_SEGMENT_B_HELD_OUT_NEAR_T_PX}) at near-T, 1.92px "
                f"({VIDEO5_SEGMENT_B_HELD_OUT_NET_PX}) at net-base -- see each frame's "
                "own court_corners/singles_corners for the segment-B shape when "
                "present. Frames 136-399 (the transition itself) are excluded from "
                "confident court-line overlay rendering (homography_applicable=False), "
                "same principle as 2.mp4's ramp. Both segments safe for "
                "real-world-distance use within their own frame range."
            ),
            "court_corners": {
                "BL": [141.0, 837.0], "BR": [1621.0, 821.0],
                "TR": [1247.0, 273.5], "TL": [526.0, 275.0],
            },
            "singles_corners": {k: list(v) for k, v in homography.singles_corners_pixels().items()},
            "segmentb_court_corners": {
                "BL": [216.5, 830.0], "BR": [1693.0, 828.0],
                "TR": [1317.0, 273.5], "TL": [593.0, 273.5],
            },
            "segmentb_singles_corners": {k: list(v) for k, v in segment_b_homography.singles_corners_pixels().items()},
        }

    if _is_reference_video4(video_path):
        homography = build_video4_homography()
        return homography, {
            "status": Status.MEASURED.value,
            "note": (
                "Least-squares 8-point calibration (cv_pipeline/src/cv_pipeline/"
                "reference_video4_calibration.py), 2026-07-19. Unlike 1.mp4/3.mp4, "
                "this camera is NOT locked-off -- a real, correlated, whole-frame "
                "horizontal sway of up to ~45-68px was found and confirmed with 3 "
                "independent corner points moving together. ONE homography covers "
                "two long, stable, position-matched windows (frames 0-420 and "
                "740-1240, confirmed the same camera position via 4 independent "
                "numeric re-measurements clustering within ~1.5px). Frames 421-739 "
                "and 1241-1543 (two brief turning points plus a short end tail, "
                "none long/stable enough to independently calibrate) are excluded "
                "from confident court-line overlay rendering per-frame -- see "
                "VIDEO4_EXCLUDED_RANGES. Held-out-landmark error (within the "
                f"stable window): 2.2px at near-T ({VIDEO4_HELD_OUT_NEAR_T_PX}), "
                f"1.71px at net-base ({VIDEO4_HELD_OUT_NET_PX}). Safe for "
                "real-world-distance use within the stable windows only."
            ),
            "court_corners": {
                "BL": [189.0, 829.0], "BR": [1719.0, 829.0],
                "TR": [1338.0, 254.0], "TL": [582.0, 254.0],
            },
            "singles_corners": {k: list(v) for k, v in homography.singles_corners_pixels().items()},
        }

    if _is_reference_video3(video_path):
        homography = build_video3_homography()
        return homography, {
            "status": Status.MEASURED.value,
            "note": (
                "Least-squares 8-point calibration (cv_pipeline/src/cv_pipeline/"
                "reference_video3_calibration.py), 2026-07-19. Camera confirmed "
                "static for the whole clip before calibrating (full-clip 2-column "
                "pixel scan, no drift found), so a single homography covers all "
                "933 frames -- unlike 2.mp4, no segment split needed. All 8 "
                "calibration points measured via numeric pixel-brightness "
                "thresholding, not eyeballed grid crops (an eyeballed first pass "
                "had a 22px outlier residual on one point; the numeric re-measure "
                "fixed it) -- held-out-landmark error: 1.68px at near-T "
                f"({VIDEO3_HELD_OUT_NEAR_T_PX}), 1.68px at net-base "
                f"({VIDEO3_HELD_OUT_NET_PX}), the best result of any reference "
                "clip so far. Safe for real-world-distance use."
            ),
            "court_corners": {
                "BL": [193.0, 841.0], "BR": [1719.0, 842.0],
                "TR": [1336.0, 268.0], "TL": [584.0, 268.0],
            },
            "singles_corners": {k: list(v) for k, v in homography.singles_corners_pixels().items()},
        }

    if _is_reference_video2(video_path):
        # Pre-pan homography (frames 0-399) is still the "default" returned
        # here -- used for ball-detection regime classification, and as the
        # court_corners/singles_corners the frontend falls back to for any
        # frame without a per-frame override (frames 0-399, plus ramp frames,
        # which are marked homography_applicable=False regardless of which
        # corners would be shown). The post-pan homography is built and
        # applied separately, per-frame, inside run_video_analysis.
        homography = build_video2_homography()
        postpan_homography = build_video2_postpan_homography()
        return homography, {
            "status": Status.MEASURED.value,
            "note": (
                "TWO homographies for this clip, not one -- a real mid-clip camera "
                "pan was found and confirmed by direct pixel measurement (see "
                "PROGRESS.md's 'data/tennis/2.mp4 Has a Genuine Mid-Clip Camera "
                "Pan' entry, 2026-07-19). Segment 1 (frames 0-399, this entry's "
                "corners below): least-squares 8-point calibration "
                "(reference_video2_calibration.py), held-out error 6.98px "
                f"({VIDEO2_HELD_OUT_NEAR_T_PX}) at near-T, 1.1px "
                f"({VIDEO2_HELD_OUT_NET_PX}) at net-base. Segment 2 (frames "
                "560-1343): separate calibration (reference_video2_postpan_"
                "calibration.py), held-out error 4.30px "
                f"({VIDEO2_POSTPAN_HELD_OUT_NEAR_T_PX}) at near-T, 1.69px "
                f"({VIDEO2_POSTPAN_HELD_OUT_NET_PX}) at net-base -- see each "
                "frame's own court_corners/singles_corners for the segment-2 "
                "shape when present. Frames 400-559 (the pan ramp itself) are "
                "excluded from confident court-line overlay rendering "
                "(homography_applicable=False) since no single frame in a "
                "continuous ramp has an unambiguous 'correct' calibration -- "
                "see PROGRESS.md. Both segments safe for real-world-distance "
                "use within their own frame range."
            ),
            "court_corners": {
                "BL": [249.0, 878.0], "BR": [1720.0, 878.0],
                "TR": [1327.0, 298.0], "TL": [578.0, 297.0],
            },
            "singles_corners": {k: list(v) for k, v in homography.singles_corners_pixels().items()},
            "postpan_court_corners": {
                "BL": [197.0, 864.0], "BR": [1718.0, 863.0],
                "TR": [1327.0, 289.0], "TL": [578.0, 289.0],
            },
            "postpan_singles_corners": {k: list(v) for k, v in postpan_homography.singles_corners_pixels().items()},
        }

    if _is_reference_video1(video_path):
        homography = build_video1_homography()
        return homography, {
            "status": Status.MEASURED.value,
            "note": (
                "Least-squares 8-point calibration (cv_pipeline/src/cv_pipeline/"
                "reference_video1_calibration.py), corrected 2026-07-18 after an "
                "earlier version had the near-baseline doubles corners (BL/BR) "
                "mislabeled as the singles-sideline crossing instead -- see "
                "PROGRESS.md's 'Court-Outline Rendering Bug' entry. Held-out-"
                f"landmark error: 4.4px at near-T ({VIDEO1_HELD_OUT_NEAR_T_PX}), "
                f"2.0px at net-base ({VIDEO1_HELD_OUT_NET_PX}) -- better than the "
                "~13px benchmark this clip's calibration was originally being "
                "compared against. Safe for real-world-distance use."
            ),
            "court_corners": {
                "BL": [200.0, 866.0], "BR": [1718.0, 866.0],
                "TR": [1330.0, 300.0], "TL": [598.0, 300.0],
            },
            "singles_corners": {k: list(v) for k, v in homography.singles_corners_pixels().items()},
        }

    stem = _clip_stem(video_path)
    if stem not in KNOWN_ANNOTATED_CLIPS:
        return None, {
            "status": Status.NOT_ATTEMPTED.value,
            "note": f"No annotated court corners available for '{stem}' -- only the 10 "
                    f"amateur dev clips (video1..video10) have them. No automated "
                    f"court-corner detector exists in cv_pipeline yet (see this module's "
                    f"docstring). Player selection falls back to a size-based heuristic.",
        }

    annotations = load_clip_annotations(stem, DEFAULT_ANNOTATIONS_DIR)
    first_court_ann = next((a for a in annotations.values() if a.court_corners), None)
    if first_court_ann is None:
        return None, {"status": Status.NOT_ATTEMPTED.value, "note": "clip is known but has no court rows"}

    homography = CourtHomography(first_court_ann.court_corners)
    # Raw pixel corners, for the frontend to draw the court quadrilateral --
    # tuples become JSON arrays automatically, keys stay plain strings (BL/BR/TL/TR).
    corners = {k: list(v) for k, v in first_court_ann.court_corners.items()}
    if stem == "video7":
        return homography, {"status": Status.EXCLUDED_KNOWN_ISSUE.value, "note": VIDEO7_KNOWN_ISSUE_NOTE, "court_corners": corners}
    if stem == "video1":
        return homography, {
            "status": Status.MEASURED.value,
            "note": "Independently validated in Phase 3 against the baseline center hash "
                    "mark (~13px / ~8cm real-world error). Safe for real-world-distance use.",
            "court_corners": corners,
        }
    return homography, {
        "status": Status.UNVALIDATED.value,
        "note": "Geometrically self-consistent (built the same way as video1's) but never "
                "independently checked against a real landmark -- see Phase 3's "
                "EVALUATION_REPORT.md. Do not trust for real-world-distance metrics.",
        "court_corners": corners,
    }


def _select_boxes(boxes: list, homography: CourtHomography | None, continuity_tracker=None, y_upper_bound_m=None):
    """Returns (near_box, far_box, method_used). Falls back to the known-unreliable
    size heuristic when no homography is available -- see module docstring.

    continuity_tracker (opt-in, default None): a PlayerContinuityTracker,
    threaded in by the caller's per-frame loop for clips where crowd-selection
    bugs were found that a pure court-position rule can't cleanly fix on its
    own -- see PlayerContinuityTracker's docstring and PROGRESS.md's
    2026-07-19 entry (data/tennis/1.mp4's front-row-spectator bug). Ignored
    when None, which preserves every other clip's existing behavior exactly.

    y_upper_bound_m (opt-in, default None): passed straight through to
    select_players_by_court_position for clips with their own validated
    back-wall-staff bound (e.g. data/tennis/2.mp4's 32.0m) but no continuity
    tracker -- see PROGRESS.md's "Generalization Test: data/tennis/2.mp4" entry."""
    if continuity_tracker is not None:
        selection = continuity_tracker.select(boxes)
        return selection.near_box, selection.far_box, "court_position_plausibility_with_temporal_continuity"

    if homography is not None:
        selection = select_players_by_court_position(boxes, homography, y_upper_bound_m=y_upper_bound_m)
        return selection.near_box, selection.far_box, "court_position_plausibility"

    if not boxes:
        return None, None, "size_based_fallback_no_homography"
    boxes_by_size = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    near = boxes_by_size[-1]
    far = boxes_by_size[0] if len(boxes_by_size) > 1 else None
    return near, far, "size_based_fallback_no_homography"


def _rate_entry(n_success: int, n_total: int, interpretation_note: str | None = None) -> dict:
    if n_total == 0:
        return {"status": Status.NOT_APPLICABLE.value, "rate": None, "n": 0}
    entry = {"status": Status.MEASURED.value, "rate": round(n_success / n_total, 4), "n": n_total}
    if interpretation_note:
        entry["note"] = interpretation_note
    return entry


COMBINED_BALL_METHOD_NOTE = (
    "combined method (fine-tuned YOLOv8n + frequency-based static-artifact filter), "
    "used because this clip's homography is available and its camera framing was "
    "classified 'validated' (see ball_detection_combined.classify_ball_detection_regime). "
    "The court-region motion-diff fallback is DISABLED as of 2026-07-19 "
    "(use_motion_diff_fallback=False) -- a 5-clip/375-sample manual visual audit "
    "found it correct only 60-72% of the time per clip on this Miami Open broadcast "
    "footage (vs. 84-100% for fine-tuned-YOLO-alone detections), producing false "
    "positives on player-limb motion; disabling it roughly triples how often a shown "
    "marker is wrong at the cost of lower raw coverage -- see PROGRESS.md's 'Ball "
    "Detection: Coverage vs. Real Accuracy Under 3 Conditions' entry. Pooled "
    "validation on the 9-clip amateur ground-truth dataset (with motion-diff "
    "enabled): 53.91% recall, vs 7.81% for stock COCO YOLO (an initially-reported "
    "70.40% was corrected 2026-07-16 after a ground-truth leak was found in the "
    "prototyping script's candidate-picking logic -- see ball_detection_combined.py's "
    "GROUND-TRUTH LEAK note) -- that figure does not directly apply now that "
    "motion-diff is off, and THIS clip/segment has no ground truth of its own "
    "either way, so its own rate below is still an unvalidated live estimate, same "
    "caveat as every other *_live_estimate field in this result."
)

STOCK_BALL_METHOD_NOTE = (
    "stock COCO-class YOLO ball detection -- best-effort. Either this clip has no "
    "calibrated homography, or its camera framing was classified 'best_effort' "
    "(broadcast/multi-camera-angle footage, where the combined method's motion-diff "
    "component was directly spot-checked and found to produce false positives on "
    "player-limb motion). Known baseline: ~7.8% recall on the (dissimilar) amateur "
    "dataset."
)


def _ball_rate_entry(n_success: int, n_total: int, method: str, regime_diagnostics: dict) -> dict:
    note = COMBINED_BALL_METHOD_NOTE if method == "combined_v2" else STOCK_BALL_METHOD_NOTE
    if regime_diagnostics:
        note = f"{note} regime diagnostics: {regime_diagnostics}"
    entry = _rate_entry(n_success, n_total, note)
    entry["method"] = method
    return entry


FAR_PLAYER_INTERPRETATION_NOTE = (
    "This rate can look meaningfully HIGHER than Phase 3's ground-truth-validated "
    "far-player detection figures (e.g. video1's whole-clip average was 20.8%, "
    "n=96) for two independent, confirmed reasons, not a pipeline improvement: "
    "(1) a short frame_limit segment starting at frame 0 can be a genuinely easier "
    "stretch than the clip's whole-video average -- verified directly for this "
    "exact case: frames 0-120 of video1 score 83.3% (n=18) under Phase 3's own "
    "ground-truth matching, vs. 20.8% (n=96) for the full 689-frame clip; and "
    "(2) this live metric counts ANY court-plausible second detection as 'far "
    "player', whereas Phase 3's figure required matching within 150px of a known, "
    "ground-truth-labeled point -- a strictly looser bar with no ground truth to "
    "hold it to. Do not read this rate as an improved or more representative "
    "far-player detection accuracy."
)


SHOT_CLASSIFICATION_UNAVAILABLE_NOTE = (
    "Shot classification requires condition-B ball detections for its ball-anchoring "
    "step (a pose-only contact-frame proxy was tried and falsified -- see "
    "shot_classification.py's docstring), so it only runs when this clip's ball "
    "detection used the combined_v2 method. This clip/segment used '{method}' instead."
)

SHOT_CLASSIFICATION_NOTE = (
    "Forehand/backhand classification anchored to condition-B ball detections, with a "
    "probable_serve flag for the single earliest event per clip (the only "
    "serve-exclusion signal found reliable after three others were tested and "
    "falsified -- see shot_classification.py's docstring). Measured on this project's "
    "5 reference clips (52 real events, manually audited, plus 4 more found and audited "
    "while re-verifying video1 under production sampling -- see below): 87.5% confident "
    "forehand/backhand accuracy (35/40, excluding serves/overheads and misfires), 23.9% "
    "serve/overhead contamination of raw 'forehand' predictions before the "
    "probable_serve flag, 14.6% after it (these figures supersede the original 86.5%/ "
    "13.2%, both moved by roughly a point and a half in the re-audit below, a wash not a "
    "meaningful correction) -- see PROGRESS.md's 'Shot-Type Detection' and "
    "'Serve-Exclusion Heuristic' entries. Left-handed players are not supported "
    "(dominant_hand='right' is hardcoded -- see shot_classification.py). This "
    "clip/segment's own event count below is unvalidated live inference, same caveat "
    "as every other *_live_estimate field in this result -- it is NOT re-measured "
    "accuracy for this specific video. THREE INTEGRATION CAVEATS found while wiring "
    "this in, not present during the original audit -- all three now investigated and "
    "resolved or closed out, not left as open guesses: (1) a short `frame_limit` can "
    "under-detect real events near the segment's tail -- the peak-prominence check "
    "needs enough signal on both sides of a candidate swing to confirm it (confirmed "
    "directly: video1's real serve at frame 107 was missed with frame_limit=200 and "
    "found correctly with frame_limit=500, same code, same video, only the window "
    "differed). MITIGATED (not eliminated): landmark/box/ball collection now reads "
    "SHOT_CLASSIFICATION_PADDING_FRAMES extra frames past frame_limit purely for this "
    "context, without adding those frames to the returned `frames` array or any "
    "detection-rate counter -- see `boundary_note`/`boundary_padding_frames_read`/"
    "`video_truncated_by_frame_limit` in shot_classification_live_estimate below for "
    "this specific run's residual risk, which is real but smaller (the prominence search "
    "is unbounded in principle, so an extremely distant confirming valley -- this "
    "project's own audit data has one real 596-frame inter-shot gap -- could still, in a "
    "pathological case, fall outside the padded window). (2) video1's near-player "
    "selection in this pipeline uses PlayerContinuityTracker (a front-row-spectator "
    "fix -- see this module's own docstring), which the original 52-event manual audit "
    "did NOT use. RE-AUDITED, holds up: all 4 previously-confirmed-correct near-player "
    "events (frames 476/491/788/1556) are byte-for-byte unchanged; 3 of the original 4 "
    "misfires persist unchanged (not worsened); one misfire (frame 235) no longer "
    "registers as a candidate at all; the one new event this configuration surfaces "
    "(frame 324) was independently re-verified by hand and is correctly classified. (3) "
    "The original 52-event audit sampled landmarks/boxes every 2nd frame (STEP=2, for "
    "speed); this live pipeline processes every frame (STEP=1). RE-AUDITED, holds up: "
    "checked video1's far player specifically (unaffected by (2)'s tracker difference) "
    "-- 3 additional real candidate events under STEP=1 (frames 382, 1490, 1680: two "
    "correctly-classified real shots plus one overhead smash mislabeled forehand, the "
    "SAME already-quantified contamination category, not a new failure mode) and 3 "
    "shifted anchor frames (860->859, 1198->1199, 1333->1324, each still correctly "
    "classified, same real shot as originally audited). These 4 new events are folded "
    "into the 87.5%/14.6% figures above. SCOPE, stated plainly: only video1 (both "
    "roles) was directly re-verified under STEP=1 and PlayerContinuityTracker; clips "
    "2-5 have not been independently re-sampled under either. video1 showed no "
    "meaningful divergence from the original audit, reassuring evidence this isn't "
    "systemic, but that is not the same claim as having checked all 5 clips -- see "
    "PROGRESS.md's 'Serve-Exclusion Heuristic' entry for the full comparison tables."
)


def run_video_analysis(video_path_str: str, frame_limit: int) -> dict:
    video_path = Path(video_path_str)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path_str}")

    from ultralytics import YOLO

    t0 = time.time()
    homography, homography_report = _build_homography_if_available(video_path)

    # y_upper_bound_m=29.0 and temporal continuity are both specific,
    # validated fixes for data/tennis/1.mp4's two independent crowd-selection
    # bugs (back-wall staff, front-row spectator -- see PROGRESS.md's
    # 2026-07-19 entry and PlayerContinuityTracker's docstring). Not applied
    # to any other clip -- their homography scale/framing hasn't been checked
    # against this same bound, so applying it blindly elsewhere risks a
    # different, unverified failure mode.
    continuity_tracker = None
    y_upper_bound_m = None
    if _is_reference_video1(video_path):
        continuity_tracker = PlayerContinuityTracker(homography, y_upper_bound_m=29.0)
    # video2_postpan_homography is None for every clip except data/tennis/2.mp4
    # -- when set, the per-frame loop below picks between `homography` (pre-pan,
    # frames 0-399) and this (post-pan, frames 560+) instead of using a single
    # homography for the whole clip. See VIDEO2_PAN_RAMP_START_FRAME's docstring.
    video2_postpan_homography = None
    if _is_reference_video2(video_path):
        # y_upper_bound_m=32.0: this clip's OWN validated bound (real far-player
        # world_y tops out at 30.56m, two back-wall staff sit at 34-35m, a clean
        # ~3.5m gap) -- different from 1.mp4's 29.0 because it's a different
        # homography scale, not reused. Temporal continuity checked and found
        # unnecessary here (no stationary near-side bystander found) -- see
        # PROGRESS.md's "Generalization Test: data/tennis/2.mp4" entry. This
        # bound was validated against the pre-pan homography's scale; the
        # post-pan homography's scale is close enough (~1% held-out-error
        # difference) that reusing the same bound is not expected to introduce
        # a new failure mode, but it has not been independently re-checked
        # against the post-pan segment specifically.
        y_upper_bound_m = 32.0
        video2_postpan_homography = build_video2_postpan_homography()
    elif _is_reference_video3(video_path):
        # y_upper_bound_m=30.0: this clip's OWN validated bound. Found via a
        # full-clip scan (933 frames): every frame's far-side candidates
        # cluster in two clean, well-separated groups -- real player
        # positions at 14.5-27.8m (near the 23.77m far baseline, as
        # expected) and a stationary pair of back-wall staff at 33.7-34.0m,
        # a ~6m gap between them. Without this bound, the back-wall staff
        # was selected as "far player" in effectively every frame (mean
        # far-side world_y was a near-constant ~34.0m across all 933 frames
        # before the fix). Near side checked too (no front-row-spectator
        # bug like 1.mp4's -- the -4m to +6m near-side range is real player
        # movement, visually confirmed at its most extreme frame). Temporal
        # continuity not needed, same as 2.mp4 -- see PROGRESS.md.
        y_upper_bound_m = 30.0
    elif _is_reference_video4(video_path):
        # y_upper_bound_m=30.0: this clip's OWN validated bound, derived the
        # same way as 3.mp4's (which happens to land on the same number --
        # coincidence, not reuse; independently re-derived from this clip's
        # own full-clip scan, restricted to the two homography-valid stable
        # windows -- frames 0-420 and 740-1240 -- since world coordinates
        # are meaningless outside them). Real far-player positions cluster
        # 23.4-27.8m (near the 23.77m baseline); back-wall staff cluster
        # 33.1-35m; a clean ~5.3m gap. Near side checked and clean (no
        # front-row-spectator pattern). Temporal continuity not needed.
        # NOT applied to the excluded ranges (421-739, 1241-1543) -- player
        # *selection* there still uses this same homography per
        # VIDEO4_EXCLUDED_RANGES' docstring (coarse tolerance), so the bound
        # is reused there too rather than left unset, since leaving it unset
        # would just reintroduce the back-wall-staff bug in those frames.
        y_upper_bound_m = 30.0

    # video5_segmentb_homography is None for every clip except data/tennis/5.mp4
    # -- when set, the per-frame loop below picks between `homography`
    # (Segment A, frames 0-135) and this (Segment B, frames 400+) instead of
    # using a single homography for the whole clip. See
    # VIDEO5_TRANSITION_START_FRAME's docstring.
    video5_segmentb_homography = None
    if _is_reference_video5(video_path):
        # y_upper_bound_m=30.0: this clip's OWN validated bound, independently
        # derived from a full-clip person-detection scan restricted to
        # homography-valid frames (Segments A and B only -- the transition
        # frames 136-399 have no valid world coordinates to check). Real
        # far-player x-plausible positions cluster 11.4-26.91m; back-wall
        # staff cluster 33.83-34.29m; a clean 6.92m gap between them, so 30.0
        # (coincidentally the same number as 3.mp4/4.mp4, independently
        # re-derived, not reused) sits with ~3m margin on both sides. A
        # SECOND stationary-object cluster was also found near the net
        # (world_y ~12.3-13.26m, at world_x -3.3/14.93 -- camera operators or
        # ball-kids standing off to the sides near the net posts, not on
        # court) but it's already fully rejected by the existing
        # X_PLAUSIBILITY_MARGIN_M check in player_selection.py (0 of 1223
        # detections in that cluster have a plausible x), so no additional
        # fix was needed for it. Near side checked too and is clean: a smooth
        # -3.95 to 7.41m spread with no gap, no front-row-spectator pattern
        # like 1.mp4's. Reused for player *selection* during the transition
        # too (coarse, meter-scale tolerance), same reasoning as 4.mp4's
        # excluded ranges.
        y_upper_bound_m = 30.0
        video5_segmentb_homography = build_video5_segment_b_homography()

    yolo = YOLO("yolov8n.pt")
    track_model = YOLO("yolov8n.pt")
    landmarker = make_landmarker()

    # Ball detection: regime-gated switch to the combined method -- see this
    # module's docstring. Precomputed as a batch up front (the combined
    # method's own two-pass design needs the full frame range at once, unlike
    # the streaming per-frame loop below) and looked up by frame index inside
    # the loop, rather than re-running stock YOLO's ball class per frame.
    ball_detection_method = "stock_yolo"
    ball_regime_diagnostics: dict = {}
    combined_ball_results_by_index: dict[int, object] = {}
    if homography is not None:
        regime, regime_diagnostics = classify_ball_detection_regime(video_path)
        ball_regime_diagnostics = {"regime": regime, **regime_diagnostics}
        if regime == "validated" and FINE_TUNED_MODEL_PATH.exists():
            fine_tuned_model = YOLO(str(FINE_TUNED_MODEL_PATH))
            # use_motion_diff_fallback=False explicitly, not relying on the
            # function's own default -- see ball_detection_combined.py's
            # docstring and PROGRESS.md's "Ball Detection: Coverage vs. Real
            # Accuracy" entry for the 5-clip/375-sample audit this decision
            # is based on. The regime gate above (validated vs. best_effort)
            # is unchanged by this -- it still decides whether fine-tuned
            # YOLO runs at all, independent of the motion-diff sub-decision.
            # n_frames extends SHOT_CLASSIFICATION_PADDING_FRAMES past
            # frame_limit -- see that constant's docstring. This only grows
            # the precomputed ball-detection dict; the main per-frame loop
            # below still only ever looks up indices < frame_limit for the
            # returned `frames` array and ball_detection_live_estimate, so
            # this has no effect on any existing field's values.
            combined_results = run_combined_ball_detection_for_clip(
                fine_tuned_model, video_path, homography, start_frame=0,
                n_frames=frame_limit + SHOT_CLASSIFICATION_PADDING_FRAMES,
                use_motion_diff_fallback=False,
            )
            combined_ball_results_by_index = {r.frame_index: r for r in combined_results}
            ball_detection_method = "combined_v2"

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    # OpenCV's frame count can be an unreliable estimate for some codecs, but
    # it's only used here to tell "frame_limit truncated real remaining
    # video" apart from "frame_limit already covers the whole clip" for the
    # shot-classification boundary note below -- a rough number is fine for
    # that purpose.
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: list[dict] = []
    n_frames_processed = 0
    n_near_detected = 0
    n_far_detected = 0
    n_ball_detected = 0
    n_near_pose_attempted = 0
    n_near_pose_success = 0
    n_far_pose_attempted = 0
    n_far_pose_success = 0
    near_track_ids: set[int] = set()
    far_track_ids: set[int] = set()
    selection_methods_seen: set[str] = set()
    # Accumulated for shot_classification.find_shot_events, called once after
    # this loop (it needs the whole clip's landmarks/boxes at once, same
    # batch-after-streaming pattern as the combined ball-detection method
    # above) -- see the shot-classification block after the loop for why this
    # only ever produces real events when ball_detection_method=="combined_v2".
    near_landmarks_by_frame: dict[int, list | None] = {}
    far_landmarks_by_frame: dict[int, list | None] = {}
    near_box_by_frame: dict[int, tuple] = {}
    far_box_by_frame: dict[int, tuple] = {}

    for frame_idx in range(frame_limit):
        ok, frame = cap.read()
        if not ok:
            break
        n_frames_processed += 1

        person_results = yolo.predict(frame, classes=[PERSON_CLASS_ID], verbose=False)
        boxes = person_results[0].boxes.xyxy.cpu().numpy().tolist() if len(person_results) else []

        # For data/tennis/2.mp4, player *selection* (coarse, meter-scale
        # tolerance) uses whichever of the two homographies is nearer for this
        # frame -- the pixel-level pre/post-pan difference is negligible next
        # to select_players_by_court_position's margins. Every other clip's
        # `homography` is unchanged. See VIDEO2_PAN_RAMP_START_FRAME's docstring.
        frame_homography = homography
        if video2_postpan_homography is not None and frame_idx >= VIDEO2_PAN_SELECTION_BOUNDARY_FRAME:
            frame_homography = video2_postpan_homography
        # data/tennis/5.mp4: same nearer-segment selection reasoning as
        # 2.mp4's post-pan switch above -- see VIDEO5_SELECTION_BOUNDARY_FRAME.
        if video5_segmentb_homography is not None and frame_idx >= VIDEO5_SELECTION_BOUNDARY_FRAME:
            frame_homography = video5_segmentb_homography

        near_box, far_box, method = _select_boxes(
            boxes, frame_homography, continuity_tracker=continuity_tracker, y_upper_bound_m=y_upper_bound_m,
        )
        selection_methods_seen.add(method)
        frame_record = {
            "index": frame_idx,
            "near_box": near_box,  # [x1,y1,x2,y2] or None -- explicit null, not an omitted key
            "far_box": far_box,
            "ball_box": None,
            "near_track_id": None,
            "far_track_id": None,
            # explicit empty list (never omitted, never a bare single object)
            # -- filled in after the loop. A list because two distinct
            # candidate swings can legitimately anchor to the SAME frame
            # (confirmed on real data) -- see the shot-classification block
            # below for why this can't just be a single nullable object.
            "shot_events": [],
        }

        # Court-line *overlay rendering* (a pixel-precision visual claim, not a
        # coarse geometric check) is handled separately and more strictly than
        # player selection above: post-pan frames get their own court_corners/
        # singles_corners so the dashboard draws the correct shape. Ramp-frame
        # suppression (homography_applicable=False for frames 400-559) is
        # applied AFTER the ball-detection block below, not here -- that block
        # unconditionally overwrites frame_record["homography_applicable"]
        # with its own per-frame framing check when the combined method runs,
        # which would otherwise silently clobber this override (the framing
        # check's histogram-correlation approach is exactly the mechanism
        # confirmed blind to this pan -- see PROGRESS.md -- so it would almost
        # always say "applicable" for ramp frames if left to decide alone).
        if video2_postpan_homography is not None and frame_idx >= VIDEO2_PAN_RAMP_END_FRAME:
            # court_polygon_pixels() returns BL, BR, TR, TL in that fixed
            # order (see homography.py's from_point_correspondences /
            # __init__ corner ordering) -- matches CORNER_ORDER used
            # elsewhere (e.g. calibration_verification.py).
            postpan_quad = video2_postpan_homography.court_polygon_pixels()
            frame_record["court_corners"] = {
                label: [float(x), float(y)]
                for label, (x, y) in zip(["BL", "BR", "TR", "TL"], postpan_quad)
            }
            frame_record["singles_corners"] = {
                k: list(v) for k, v in video2_postpan_homography.singles_corners_pixels().items()
            }

        # data/tennis/5.mp4: same per-frame court_corners/singles_corners
        # override reasoning as 2.mp4's post-pan block above -- Segment B
        # frames (400+) get their own shape. See VIDEO5_TRANSITION_END_FRAME.
        if video5_segmentb_homography is not None and frame_idx >= VIDEO5_TRANSITION_END_FRAME:
            segmentb_quad = video5_segmentb_homography.court_polygon_pixels()
            frame_record["court_corners"] = {
                label: [float(x), float(y)]
                for label, (x, y) in zip(["BL", "BR", "TR", "TL"], segmentb_quad)
            }
            frame_record["singles_corners"] = {
                k: list(v) for k, v in video5_segmentb_homography.singles_corners_pixels().items()
            }

        if near_box is not None:
            n_near_detected += 1
            n_near_pose_attempted += 1
            near_box_by_frame[frame_idx] = tuple(near_box)
            x1, y1, x2, y2 = near_box
            if (x2 - x1) >= POSE_CROP_MIN_SIDE_PX and (y2 - y1) >= POSE_CROP_MIN_SIDE_PX:
                pose_result = run_pose_on_box(landmarker, frame, near_box)
                if pose_result.landmarks is not None:
                    n_near_pose_success += 1
                near_landmarks_by_frame[frame_idx] = pose_result.landmarks

        if far_box is not None:
            n_far_detected += 1
            n_far_pose_attempted += 1
            far_box_by_frame[frame_idx] = tuple(far_box)
            x1, y1, x2, y2 = far_box
            if (x2 - x1) >= POSE_CROP_MIN_SIDE_PX and (y2 - y1) >= POSE_CROP_MIN_SIDE_PX:
                pose_result = run_pose_on_box(landmarker, frame, far_box)
                if pose_result.landmarks is not None:
                    n_far_pose_success += 1
                far_landmarks_by_frame[frame_idx] = pose_result.landmarks

        if ball_detection_method == "combined_v2":
            combined_result = combined_ball_results_by_index.get(frame_idx)
            if combined_result is not None:
                frame_record["homography_applicable"] = combined_result.homography_applicable
                if combined_result.center is not None:
                    n_ball_detected += 1
                    cx, cy = combined_result.center
                    frame_record["ball_box"] = [cx - 6, cy - 6, cx + 6, cy + 6]
        else:
            ball_results = yolo.predict(frame, classes=[SPORTS_BALL_CLASS_ID], verbose=False)
            ball_boxes = ball_results[0].boxes.xyxy.cpu().numpy().tolist() if len(ball_results) else []
            if ball_boxes:
                n_ball_detected += 1
                frame_record["ball_box"] = ball_boxes[0]  # first/highest-confidence candidate

        # Applied AFTER the ball-detection block above so this always wins for
        # ramp frames, regardless of what combined_result.homography_applicable
        # said -- see the comment where court_corners is set for why order
        # matters here.
        if (
            video2_postpan_homography is not None
            and VIDEO2_PAN_RAMP_START_FRAME <= frame_idx < VIDEO2_PAN_RAMP_END_FRAME
        ):
            frame_record["homography_applicable"] = False

        # data/tennis/4.mp4: same "applied after ball detection so it always
        # wins" reasoning as 2.mp4's ramp above -- see VIDEO4_EXCLUDED_RANGES.
        if _is_reference_video4(video_path) and _video4_frame_excluded(frame_idx):
            frame_record["homography_applicable"] = False

        # data/tennis/5.mp4: same "applied after ball detection so it always
        # wins" reasoning -- see VIDEO5_TRANSITION_START_FRAME/_END_FRAME.
        if (
            video5_segmentb_homography is not None
            and VIDEO5_TRANSITION_START_FRAME <= frame_idx < VIDEO5_TRANSITION_END_FRAME
        ):
            frame_record["homography_applicable"] = False

        track_results = track_model.track(frame, classes=[PERSON_CLASS_ID], persist=True,
                                            tracker="bytetrack.yaml", verbose=False)
        t_boxes = track_results[0].boxes.xyxy.cpu().numpy().tolist() if len(track_results) else []
        t_ids = (track_results[0].boxes.id.cpu().numpy().tolist()
                 if (len(track_results) and track_results[0].boxes.id is not None) else [])
        if near_box is not None and t_boxes:
            nb = np.array(near_box[:2])
            dists = [np.hypot(*(np.array(b[:2]) - nb)) for b in t_boxes]
            if dists:
                tid = int(t_ids[int(np.argmin(dists))])
                near_track_ids.add(tid)
                frame_record["near_track_id"] = tid
        if far_box is not None and t_boxes:
            fb = np.array(far_box[:2])
            dists = [np.hypot(*(np.array(b[:2]) - fb)) for b in t_boxes]
            if dists:
                tid = int(t_ids[int(np.argmin(dists))])
                far_track_ids.add(tid)
                frame_record["far_track_id"] = tid

        frames.append(frame_record)

    # Boundary padding: extra frames read PAST frame_limit, feeding only
    # near/far landmarks_by_frame and box_by_frame (never `frames`, never a
    # detection-rate counter, never the person tracker -- none of those are
    # needed just to give a boundary shot-classification event real
    # right-side context) -- see SHOT_CLASSIFICATION_PADDING_FRAMES's
    # docstring for why this exists. `cap` continues reading right where the
    # main loop left off (frame_limit), so this needs no seek. Skipped
    # entirely when shot classification won't run anyway.
    n_padding_frames_read = 0
    if ball_detection_method == "combined_v2":
        for pad_offset in range(SHOT_CLASSIFICATION_PADDING_FRAMES):
            ok, frame = cap.read()
            if not ok:
                break
            pad_frame_idx = frame_limit + pad_offset
            n_padding_frames_read += 1

            person_results = yolo.predict(frame, classes=[PERSON_CLASS_ID], verbose=False)
            boxes = person_results[0].boxes.xyxy.cpu().numpy().tolist() if len(person_results) else []

            frame_homography = homography
            if video2_postpan_homography is not None and pad_frame_idx >= VIDEO2_PAN_SELECTION_BOUNDARY_FRAME:
                frame_homography = video2_postpan_homography
            if video5_segmentb_homography is not None and pad_frame_idx >= VIDEO5_SELECTION_BOUNDARY_FRAME:
                frame_homography = video5_segmentb_homography

            pad_near_box, pad_far_box, _ = _select_boxes(
                boxes, frame_homography, continuity_tracker=continuity_tracker, y_upper_bound_m=y_upper_bound_m,
            )
            if pad_near_box is not None:
                near_box_by_frame[pad_frame_idx] = tuple(pad_near_box)
                x1, y1, x2, y2 = pad_near_box
                if (x2 - x1) >= POSE_CROP_MIN_SIDE_PX and (y2 - y1) >= POSE_CROP_MIN_SIDE_PX:
                    pose_result = run_pose_on_box(landmarker, frame, pad_near_box)
                    near_landmarks_by_frame[pad_frame_idx] = pose_result.landmarks
            if pad_far_box is not None:
                far_box_by_frame[pad_frame_idx] = tuple(pad_far_box)
                x1, y1, x2, y2 = pad_far_box
                if (x2 - x1) >= POSE_CROP_MIN_SIDE_PX and (y2 - y1) >= POSE_CROP_MIN_SIDE_PX:
                    pose_result = run_pose_on_box(landmarker, frame, pad_far_box)
                    far_landmarks_by_frame[pad_frame_idx] = pose_result.landmarks

    # None (unknown, e.g. an unreliable frame-count estimate for this codec)
    # vs. True/False is preserved deliberately -- the boundary note below
    # reads differently depending on which of the three it is.
    video_truncated_by_frame_limit = (total_frame_count > frame_limit) if total_frame_count > 0 else None

    # Shot classification (forehand/backhand + probable_serve), run once over
    # the whole clip now that the per-frame landmarks/boxes are collected --
    # same batch-after-streaming pattern as the combined ball-detection
    # method above, and for the same reason: find_shot_events needs to look
    # ahead/behind across frames (peak-finding, ball-anchoring windows), not
    # decide anything from a single frame in isolation. Gated on
    # ball_detection_method == "combined_v2" because ball-anchoring requires
    # condition-B (fine-tuned-YOLO) detections specifically -- see
    # shot_classification.py's docstring for why a pose-only proxy was tried
    # and falsified. Every other clip gets an explicit NOT_ATTEMPTED status
    # below, never a silently-empty result indistinguishable from "ran and
    # found nothing."
    if ball_detection_method == "combined_v2":
        ball_by_frame = {
            r.frame_index: r.center for r in combined_ball_results_by_index.values()
            if r.source == "fine_tuned_yolo" and r.center is not None
        }
        raw_shot_events_by_role = {
            "near": find_shot_events(near_landmarks_by_frame, ball_by_frame, near_box_by_frame),
            "far": find_shot_events(far_landmarks_by_frame, ball_by_frame, far_box_by_frame),
        }
        # Flag probable-serve across ALL found events, including any found
        # only because of the padding region -- "which event is earliest" is
        # unaffected by whether it happens to land inside or outside
        # frame_limit. The frame_limit filter happens AFTER, below.
        flagged_shot_events_by_role = flag_first_event_as_probable_serve(raw_shot_events_by_role)
        # Events anchored at or past frame_limit exist only to give BOUNDARY
        # events (anchored before frame_limit) real right-side context -- no
        # frame_record exists for them (frames only covers 0..frame_limit-1),
        # and they aren't part of what this segment's caller actually
        # requested, so they're dropped here rather than reported.
        shot_events_by_role = {
            role: [ev for ev in evs if ev.frame_index < frame_limit]
            for role, evs in flagged_shot_events_by_role.items()
        }

        # dict-of-LISTS, not dict-of-single-event: shot_classification.py can
        # legitimately anchor two distinct candidate peaks (e.g. a backswing
        # peak and a follow-through peak from the same real swing) to the
        # IDENTICAL ball-contact frame -- confirmed on real data (video1's
        # near player has two events both anchored to frame 788). A
        # single-event dict here would let the second silently overwrite the
        # first in frame_record, while n_events below still counted both --
        # a real count/output mismatch, found while re-auditing this
        # integration, not a hypothetical. Every event is now preserved.
        events_by_frame_role: dict[tuple[str, int], list] = {}
        for role, evs in shot_events_by_role.items():
            for ev in evs:
                events_by_frame_role.setdefault((role, ev.frame_index), []).append(ev)
        for frame_record in frames:
            for role in ("near", "far"):
                evs_here = events_by_frame_role.get((role, frame_record["index"]))
                if evs_here:
                    frame_record["shot_events"].extend(
                        {"role": role, "classification": ev.classification, "probable_serve": ev.probable_serve}
                        for ev in evs_here
                    )

        if video_truncated_by_frame_limit is False:
            boundary_note = (
                "frame_limit already reached this video's true end (or the estimate says so), "
                "so no boundary-truncation risk applies to this result."
            )
        elif n_padding_frames_read == 0:
            boundary_note = (
                "No padding frames were available past frame_limit (video ended right there or "
                "before) -- any residual under-detection very near the tail reflects the video's "
                "own true end, not a still-fixable frame_limit choice."
            )
        else:
            boundary_note = (
                f"Read {n_padding_frames_read}/{SHOT_CLASSIFICATION_PADDING_FRAMES} requested "
                "extra frames past frame_limit purely for peak-detection context (see "
                "SHOT_CLASSIFICATION_PADDING_FRAMES) -- not part of this segment's own "
                "frames/detection-rate output. This substantially reduces (the one real "
                "boundary-miss case found needed only ~100 padding frames to resolve) but does "
                "NOT mathematically eliminate under-detection risk for events very close to "
                "frame_limit -- the peak-prominence search is unbounded in principle (this "
                "project's own audit data has a real 596-frame inter-shot gap). Treat events in "
                "the last few dozen frames before frame_limit as the highest remaining residual "
                "risk, same caveat as the rest of this note, now smaller rather than absent."
            )

        n_shot_events = sum(len(evs) for evs in shot_events_by_role.values())
        shot_classification_status = {
            "status": Status.MEASURED.value if n_shot_events else Status.NOT_DETECTED.value,
            "n_events": n_shot_events,
            "n_events_by_role": {role: len(evs) for role, evs in shot_events_by_role.items()},
            "boundary_padding_frames_requested": SHOT_CLASSIFICATION_PADDING_FRAMES,
            "boundary_padding_frames_read": n_padding_frames_read,
            "video_truncated_by_frame_limit": video_truncated_by_frame_limit,
            "boundary_note": boundary_note,
            "note": SHOT_CLASSIFICATION_NOTE,
        }
    else:
        shot_classification_status = {
            "status": Status.NOT_ATTEMPTED.value,
            "note": SHOT_CLASSIFICATION_UNAVAILABLE_NOTE.format(method=ball_detection_method),
        }

    elapsed = time.time() - t0

    def _pose_status(n_attempted: int, n_success: int) -> dict:
        if n_attempted == 0:
            return {"status": Status.NOT_ATTEMPTED.value, "note": "player box never available this segment"}
        if n_success == 0:
            return {"status": Status.NOT_DETECTED.value,
                    "note": f"pose attempted on {n_attempted} frame(s), landmarks found on 0"}
        return {"status": Status.MEASURED.value,
                "success_rate": round(n_success / n_attempted, 4), "n_attempted": n_attempted}

    return {
        "ground_truth": "NONE -- this is live inference output on an unannotated video, "
                         "not a validated accuracy figure. Compare against Phase 3's "
                         "EVALUATION_REPORT.md numbers only qualitatively, never directly.",
        "clip": video_path.name,
        "n_frames_processed": n_frames_processed,
        "source_fps": fps,
        "video_width": video_width,
        "video_height": video_height,
        "processing_time_s": round(elapsed, 1),
        "player_selection_method": sorted(selection_methods_seen),
        "homography": homography_report,
        "frames": frames,
        # Field names carry "_live_estimate" explicitly -- Phase 3's ground-truth-
        # validated figures (video1's whole-clip far-player rate: 20.8%, n=96) live
        # in EVALUATION_REPORT.md under plain names with no such suffix. A name that
        # can't be confused is more robust than a name plus a caveat, especially once
        # this flows into a dashboard or gets summarized by the LLM agent -- a reader
        # (human or model) skimming field names alone should not be able to mistake
        # one for the other.
        "near_player_detection_live_estimate": _rate_entry(n_near_detected, n_frames_processed),
        "far_player_detection_live_estimate": _rate_entry(n_far_detected, n_frames_processed, FAR_PLAYER_INTERPRETATION_NOTE),
        "ball_detection_live_estimate": _ball_rate_entry(n_ball_detected, n_frames_processed, ball_detection_method, ball_regime_diagnostics),
        "near_player_pose_live_estimate": _pose_status(n_near_pose_attempted, n_near_pose_success),
        "far_player_pose_live_estimate": _pose_status(n_far_pose_attempted, n_far_pose_success),
        "shot_classification_live_estimate": shot_classification_status,
        "tracking": {
            "near_player_distinct_track_ids": sorted(near_track_ids),
            "far_player_distinct_track_ids": sorted(far_track_ids),
            "note": "distinct track IDs seen in the near/far role across the segment -- "
                    "more than 1 suggests an ID swap/track-loss event occurred (see Phase "
                    "3's tracking findings for known causes: scene cuts, fast-motion "
                    "detection dropouts).",
        },
    }
