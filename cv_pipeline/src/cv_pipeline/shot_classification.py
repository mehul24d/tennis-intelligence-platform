"""shot_classification.py -- forehand/backhand classification from Phase 3's
MediaPipe pose output, anchored to condition B's ball detections. Own
module, does not modify pose_estimation.py or ball_detection_combined.py.

HANDEDNESS, VERIFIED NOT ASSUMED: both players across all 5 Miami clips
(Alcaraz, Sinner -- confirmed via on-screen scoreboard labels) were visually
confirmed right-handed from multiple unambiguous frontal frames (serve-toss
hand vs. racket hand; forehand strikes facing the camera directly) before
this module was written. `dominant_hand` below defaults to "right" for that
reason -- it is a real, checked fact for this dataset, not a silent
convention. A clip with a left-handed player would need `dominant_hand`
passed explicitly per player; this module does not detect handedness itself.

WHY BALL-ANCHORED, NOT A POSE-ONLY PROXY (history, not hypothetical -- see
PROGRESS.md's "Shot-Type Detection" entry for the full investigation): two
pose-only contact-frame proxies were built and tested against real,
visually-confirmed examples, and both failed in the same underlying way.
Peak elbow-extension often lands in the follow-through (which frequently
crosses to the wrong anatomical side on a big topspin finish). The
"ascending-edge" fix (first frame extension rises above threshold) just
moved the failure to the OTHER end of the swing arc: it lands in the
backswing (which frequently crosses to the wrong side loading up). Real
measured errors on 3 ball-verified examples: 6, 15, and 21 frames off, 2 of
3 misclassified. Neither the minimum, maximum, nor first-threshold-crossing
of the raw extension curve reliably locates contact -- only a brief window
right at contact reliably has the correct side, and pose alone (elbow
position/timing) can't find that window. Condition B's real ball
detections can: the ball is nearest the player right at contact, which is
exactly the moment neither pose proxy could reliably find.

NO SILENT POSE-ONLY FALLBACK, ON PURPOSE: if a candidate swing has no
usable ball detection nearby, this module reports NO shot for it -- it does
NOT fall back to classifying at the peak or ascending-edge frame. That
would quietly reintroduce the exact failure mode ball-anchoring was built
to fix into an unflagged fraction of the output. An honest gap beats a
wrong classification, same principle as the motion-diff decision
(use_motion_diff_fallback=False in ball_detection_combined.py) and the
Kalman-filter rejection.

WHY ELBOW, NOT WRIST, FOR THE SIDE TEST ITSELF: wrist-landmark visibility
collapses exactly when it matters most -- near-player wrist usable in only
~16% of frames on average (7-25% per clip), because fast swing motion this
close to camera blurs the wrist in the source broadcast footage itself
(visually confirmed). Elbow visibility is far more robust (~55%
near-player average, ~80% far-player).

COVERAGE, MEASURED BEFORE BUILDING THIS, NOT ASSUMED: candidate events (from
elbow-extension peaks, which already require elbow+shoulder+hip visibility)
have a usable condition-B ball detection nearby -- within
BALL_ANCHOR_WINDOW_FRAMES frames, spatially within the player's box expanded
by BALL_ANCHOR_BOX_MARGIN_FRAC -- 87.1% of the time for the near player
(27/31 measured candidate events) and 61.7% for the far player (29/47),
71.8% pooled. This is real, joint (pose AND ball) coverage measured against
this exact dataset before implementation, not an assumption -- see
PROGRESS.md. It does NOT account for the peak-finder's own unmeasured
recall against true real shots (a separate, still-open question), so true
end-to-end coverage against every real shot in a clip is likely somewhat
lower than 71.8%, not higher.

SERVES ARE NOT DISTINGUISHED FROM FOREHANDS FOR MOST EVENTS, ON PURPOSE:
a serve is geometrically a large elbow extension on the dominant-hand
side, same as a groundstroke forehand, and will have a real ball detection
near the player at toss/contact too, so ball-anchoring does not fix this.
This module labels serves "forehand" -- a known, explicit scope decision.
The required manual audit classified 3-way (forehand/backhand/serve) on
all 52 real events across all 5 clips and measured 23.3% (10/43)
serve/overhead contamination of "forehand" predictions -- large enough to
act on, per the pre-agreed threshold (see PROGRESS.md).

PROBABLE-SERVE FLAG, WHAT WAS TRIED AND WHY MOST OF IT FAILED (real
measurement, not guessing -- see PROGRESS.md's "Serve-Exclusion Heuristic"
entry for the full investigation): the obvious idea -- flag a shot as a
probable serve when it follows a "motion lull" -- was tested three
different concrete ways against 15 real, visually-confirmed reference
events (5 clips: known serves, known overhead smashes, known real
groundstrokes) before writing any classification logic, and every general
mid-clip version of it failed:
  - Gap since the previous detected shot event: falsified by a direct
    counterexample. clip3's near-player backhand at frame 676 is a real
    backhand (visually confirmed) following a 596-frame gap; clip2's
    near-player serve at frame 1207 is a real serve following only a
    121-frame gap. No single threshold separates these -- the gap that
    would exclude the false-positive risk is larger than the gap on a
    confirmed real serve.
  - Ball-detection DENSITY in the window before the event: falsified.
    Condition B's fine-tuned YOLO detects the ball just as densely during
    pre-serve ball-bouncing as during live rally flight (e.g. 46
    detections in the 60 frames before the real serve at clip1 f107, vs.
    52 before the real overhead at clip1 f1035 -- no contrast).
  - Ball spatial spread / path-length in the window before the event:
    falsified. The two confirmed real serves in the reference set
    disagreed with EACH OTHER (path_len 744px at clip1 f107 vs. 1857px at
    clip5 f84, window=60 frames), let alone separated cleanly from the
    real non-serve counterexample (621px at clip3 f676).
  - (Bonus check, not itself a "motion lull" signal, but cheap given data
    already on hand): elbow-extension-ratio magnitude at the anchor frame.
    Also falsified -- serves/overheads (0.53-0.94) heavily overlap real
    forehands (0.71-0.82) and even a real backhand (0.66) in the same
    reference set.

ONE THING DID hold up, cleanly, with zero counterexamples in the reference
set: the very FIRST candidate event in a clip's merged (near+far) timeline
was a real serve in all 5 clips checked (clip1 f107, clip2 n34, clip3 n80,
clip4 f406, clip5 f84 -- each visually confirmed: racket raised overhead,
jumping extension). This is narrow and structural, not a general
serve detector -- it only catches the clip-opening shot (each of these 5
clips happens to start near a point boundary), not a serve occurring later
in a clip (e.g. clip2's real serve at n1207, correctly NOT caught) or an
overhead smash mid-rally (clip1 f1035, clip4 f546, correctly NOT caught --
neither has a preceding lull to detect in the first place). Measured
effect on the 52-event audit set: catches 5 of the 10 known contaminating
events, dropping "forehand"-prediction contamination from 23.3% (10/43) to
13.2% (5/38) -- see `flag_first_event_as_probable_serve` and PROGRESS.md.
The remaining ~13.2% (mid-clip serves, overhead smashes) is a documented,
known limitation, same treatment as every other tried-and-fell-short
approach in this project -- not silently left unmentioned.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.signal import find_peaks

SHOULDER_L, SHOULDER_R = 11, 12
ELBOW_L, ELBOW_R = 13, 14
HIP_L, HIP_R = 23, 24
VIS_THRESHOLD = 0.5

# Peak-finding parameters on the (scale-normalized) elbow-extension signal --
# used only to find CANDIDATE swing windows now, not to classify at the peak
# frame itself (see module docstring for why that failed). Provisional, not
# validated against ground truth (none exists for shot events), tunable
# against the manual audit like every other threshold in this project's
# ball-detection work.
MIN_SHOT_PROMINENCE = 0.35  # in torso-length-normalized extension-ratio units
MIN_SHOT_GAP_FRAMES = 15  # ~0.25s at 59.94fps -- real shots shouldn't repeat faster than this in these rally clips

# Ball-anchoring parameters -- these EXACT values are what the 87.1%/61.7%
# coverage measurement in the module docstring was measured with. Changing
# them changes real coverage; re-measure, don't just assume, if tuned.
BALL_ANCHOR_WINDOW_FRAMES = 20
BALL_ANCHOR_BOX_MARGIN_FRAC = 0.6
# Small tolerance for finding usable pose landmarks near the ball-anchor
# frame itself, since the exact anchor frame may momentarily lack elbow
# visibility even if nearby frames have it.
POSE_NEAR_ANCHOR_TOLERANCE_FRAMES = 3


@dataclass(frozen=True)
class ShotEvent:
    frame_index: int  # the ball-anchored contact frame, NOT the pose peak frame
    classification: Literal["forehand", "backhand"]
    peak_frame_index: int  # diagnostic: which candidate-event peak this was anchored from
    ball_distance_to_peak_frames: int  # diagnostic: how far the anchor moved from the raw peak
    # True only for the single earliest event across all roles in a clip --
    # the one signal found reliable for probable-serve detection (see module
    # docstring). Never drops or reclassifies an event; callers who want
    # serve-excluded forehand/backhand counts filter on this themselves.
    probable_serve: bool = False


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _cross_z(v, w):
    return v[0] * w[1] - v[1] * w[0]


def _landmarks_ok(lm, indices) -> bool:
    return all(lm[i][2] >= VIS_THRESHOLD for i in indices)


def _elbow_extension_ratio(lm) -> float | None:
    """Right-elbow distance from the shoulder-midpoint, normalized by torso
    length (shoulder-midpoint to hip-midpoint distance) -- scale-invariant,
    so the SAME threshold applies to the near player (large in frame) and
    far player (small in frame) without separate calibration."""
    if not _landmarks_ok(lm, (SHOULDER_L, SHOULDER_R, HIP_L, HIP_R, ELBOW_R)):
        return None
    shoulder_mid = _mid(lm[SHOULDER_L][:2], lm[SHOULDER_R][:2])
    hip_mid = _mid(lm[HIP_L][:2], lm[HIP_R][:2])
    torso_len = float(np.hypot(hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]))
    if torso_len < 1e-6:
        return None
    elbow = lm[ELBOW_R][:2]
    dist = float(np.hypot(elbow[0] - shoulder_mid[0], elbow[1] - shoulder_mid[1]))
    return dist / torso_len


def _anatomical_right_side(lm, point: tuple[float, float]) -> bool | None:
    """Self-calibrating side test: does `point` fall on the anatomical RIGHT
    side of the body's vertical axis (shoulder-midpoint -> hip-midpoint), in
    THIS frame's own image-plane orientation? Uses the right shoulder itself
    as the reference for "which cross-product sign means right side" --
    deliberately avoids hardcoding a near-player-faces-away /
    far-player-faces-camera convention, which would silently break if a
    clip's camera setup ever differed."""
    if not _landmarks_ok(lm, (SHOULDER_L, SHOULDER_R, HIP_L, HIP_R)):
        return None
    shoulder_mid = _mid(lm[SHOULDER_L][:2], lm[SHOULDER_R][:2])
    hip_mid = _mid(lm[HIP_L][:2], lm[HIP_R][:2])
    midline = (hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])
    right_shoulder_vec = (lm[SHOULDER_R][0] - shoulder_mid[0], lm[SHOULDER_R][1] - shoulder_mid[1])
    point_vec = (point[0] - shoulder_mid[0], point[1] - shoulder_mid[1])
    ref = _cross_z(midline, right_shoulder_vec)
    val = _cross_z(midline, point_vec)
    if abs(ref) < 1e-6:
        return None
    return (val > 0) == (ref > 0)


def _distance_to_box(box: tuple[float, float, float, float], pt: tuple[float, float]) -> float:
    """0 if pt is inside box, else Euclidean distance to the nearest edge."""
    x1, y1, x2, y2 = box
    dx = max(x1 - pt[0], 0.0, pt[0] - x2)
    dy = max(y1 - pt[1], 0.0, pt[1] - y2)
    return float(np.hypot(dx, dy))


def _find_extension_peaks(
    landmarks_by_frame: dict[int, list[tuple[float, float, float]] | None],
    min_prominence: float, min_gap_frames: int,
) -> list[tuple[int, float]]:
    """Candidate swing windows -- local peaks in elbow-extension. Returns
    (frame_index, extension_ratio) pairs. This step alone is NOT what
    classifies a shot (see module docstring); it only proposes WHERE to look
    for a ball-anchored contact frame."""
    frames = sorted(f for f, lm in landmarks_by_frame.items() if lm is not None)
    ratios, valid_frames = [], []
    for f in frames:
        r = _elbow_extension_ratio(landmarks_by_frame[f])
        if r is not None:
            ratios.append(r)
            valid_frames.append(f)

    if len(ratios) < 3:
        return []

    ratios_arr = np.array(ratios)
    peak_idxs, _ = find_peaks(ratios_arr, prominence=min_prominence)

    # De-duplicate peaks closer than min_gap_frames apart (in real
    # frame_index terms, not array-position terms), keeping the highest.
    peak_idxs = sorted(peak_idxs, key=lambda i: -ratios_arr[i])
    kept: list[int] = []
    for i in peak_idxs:
        if all(abs(valid_frames[i] - valid_frames[j]) >= min_gap_frames for j in kept):
            kept.append(i)
    kept.sort()

    return [(valid_frames[i], float(ratios_arr[i])) for i in kept]


def _find_ball_anchor_frame(
    peak_frame: int, ball_by_frame: dict[int, tuple[float, float]],
    box_by_frame: dict[int, tuple[float, float, float, float]],
    window: int, margin_frac: float,
) -> int | None:
    """Within `window` frames of `peak_frame`, finds the ball detection
    spatially nearest the player and returns its frame_index -- the
    ball-anchored contact-frame estimate. `margin_frac` only QUALIFIES a
    frame as usable (ball within the expanded box at all); among qualifying
    frames, the one with the SMALLEST distance to the real, unexpanded box
    is picked, as the actual closest-approach proxy for contact. Returns
    None if no usable ball detection exists nearby (caller must then DROP
    this event, per this module's no-silent-fallback design).

    BUG FOUND AND FIXED (see PROGRESS.md's "Shot-Type Detection" entry): an
    earlier version picked the FIRST frame where the ball entered the
    margined region (an artifact of `dist == 0.0 and dist < best_dist`
    never being true twice), not the frame of closest approach. On real
    clip 1 data this anchored to the ball's early approach -- often still
    during the backswing -- and produced a systematic bias (every single
    "backhand" prediction in a first audit pass was visually confirmed to
    actually be a forehand, 0/6). Fixed by ranking qualifying frames by
    distance to the REAL (unexpanded) box instead of a same-value tie that
    could never update."""
    if not box_by_frame:
        return None
    box_frames = sorted(box_by_frame)
    best_frame, best_dist = None, None
    for f in range(peak_frame - window, peak_frame + window + 1):
        if f not in ball_by_frame:
            continue
        nearest_box_frame = min(box_frames, key=lambda k: abs(k - f))
        if abs(nearest_box_frame - f) > window:
            continue
        box = box_by_frame[nearest_box_frame]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        expanded = (x1 - w * margin_frac, y1 - h * margin_frac, x2 + w * margin_frac, y2 + h * margin_frac)
        if _distance_to_box(expanded, ball_by_frame[f]) > 0.0:
            continue  # not within the usable margin at all
        real_dist = _distance_to_box(box, ball_by_frame[f])
        if best_dist is None or real_dist < best_dist:
            best_frame, best_dist = f, real_dist
    return best_frame


def _nearest_usable_pose_frame(
    anchor_frame: int, landmarks_by_frame: dict[int, list[tuple[float, float, float]] | None],
    tolerance: int,
) -> int | None:
    for offset in range(tolerance + 1):
        for f in ({anchor_frame} if offset == 0 else {anchor_frame - offset, anchor_frame + offset}):
            lm = landmarks_by_frame.get(f)
            if lm is not None and _landmarks_ok(lm, (SHOULDER_L, SHOULDER_R, HIP_L, HIP_R, ELBOW_R)):
                return f
    return None


def find_shot_events(
    landmarks_by_frame: dict[int, list[tuple[float, float, float]] | None],
    ball_by_frame: dict[int, tuple[float, float]],
    box_by_frame: dict[int, tuple[float, float, float, float]],
    dominant_hand: Literal["right", "left"] = "right",
    min_prominence: float = MIN_SHOT_PROMINENCE,
    min_gap_frames: int = MIN_SHOT_GAP_FRAMES,
    ball_anchor_window: int = BALL_ANCHOR_WINDOW_FRAMES,
    ball_anchor_margin_frac: float = BALL_ANCHOR_BOX_MARGIN_FRAC,
) -> list[ShotEvent]:
    """Finds candidate swing windows (elbow-extension peaks), anchors each to
    the nearest real ball detection near the player (condition B only --
    `ball_by_frame` should already be filtered to source=='fine_tuned_yolo'),
    and classifies forehand/backhand by which side of the body's own
    vertical axis the dominant-hand elbow is on AT THE BALL-ANCHORED FRAME
    (not the peak frame -- see module docstring for why that failed).
    Candidate events with no usable ball anchor, or no usable pose within
    POSE_NEAR_ANCHOR_TOLERANCE_FRAMES of the anchor, are DROPPED, not
    fallen back on -- this function will simply return fewer events than
    candidate peaks found, which is the intended, honest behavior."""
    if dominant_hand != "right":
        raise NotImplementedError(
            "dominant_hand='left' not implemented -- no left-handed player was found in "
            "this dataset's handedness check (see PROGRESS.md), so this path is untested. "
            "Implementing it would mean mirroring the elbow index (LEFT instead of RIGHT) "
            "and the forehand/backhand side mapping, not just flipping a sign blindly."
        )

    peaks = _find_extension_peaks(landmarks_by_frame, min_prominence, min_gap_frames)

    events: list[ShotEvent] = []
    for peak_frame, _ratio in peaks:
        anchor_frame = _find_ball_anchor_frame(
            peak_frame, ball_by_frame, box_by_frame, ball_anchor_window, ball_anchor_margin_frac,
        )
        if anchor_frame is None:
            continue  # no usable ball nearby -- dropped, not classified via a fallback proxy

        pose_frame = _nearest_usable_pose_frame(anchor_frame, landmarks_by_frame, POSE_NEAR_ANCHOR_TOLERANCE_FRAMES)
        if pose_frame is None:
            continue  # ball anchor found, but no usable pose near it -- dropped

        lm = landmarks_by_frame[pose_frame]
        is_right = _anatomical_right_side(lm, lm[ELBOW_R][:2])
        if is_right is None:
            continue

        classification: Literal["forehand", "backhand"] = "forehand" if is_right else "backhand"
        events.append(ShotEvent(
            frame_index=anchor_frame, classification=classification,
            peak_frame_index=peak_frame, ball_distance_to_peak_frames=abs(anchor_frame - peak_frame),
        ))

    return events


def flag_first_event_as_probable_serve(
    events_by_role: dict[str, list[ShotEvent]],
) -> dict[str, list[ShotEvent]]:
    """Marks the single earliest event (by anchored frame_index) across ALL
    roles combined as `probable_serve=True` -- the only serve signal this
    module found reliable after testing three different "motion lull"
    operationalizations against real data and falsifying all three (see
    module docstring). Call this once per clip, after `find_shot_events` has
    been run separately for each role (this module's find_shot_events is
    single-role by design -- merging happens here, not there, so the
    per-role contact-detection logic stays simple and its existing tests
    stay valid).

    Returns NEW per-role lists (ShotEvent is frozen) with every event
    unchanged except the single earliest one. Does not drop or reclassify
    anything -- a caller building serve-excluded forehand/backhand counts
    filters on `probable_serve` themselves. `events_by_role` with no events
    at all (empty lists throughout) is returned unchanged.

    SCOPE, STATED PLAINLY: this rule is validated specifically on this
    project's 5 reference clips, each of which happens to begin near a real
    point boundary (5/5 checked, all confirmed real serves by eye). It does
    NOT generalize to catching serves later in a clip, or overhead smashes
    mid-rally -- neither has a preceding gap for this rule to key off, and
    no reliable general signal for those was found (see module docstring).
    An arbitrary clip that does NOT begin near a point boundary would have
    its first event flagged incorrectly by this same rule; that has not
    been checked because no such clip exists in this project's dataset."""
    all_events = [(role, i, ev) for role, evs in events_by_role.items() for i, ev in enumerate(evs)]
    if not all_events:
        return events_by_role

    earliest_role, earliest_i, _ = min(all_events, key=lambda t: t[2].frame_index)

    result: dict[str, list[ShotEvent]] = {}
    for role, evs in events_by_role.items():
        if role == earliest_role:
            result[role] = [
                replace(ev, probable_serve=True) if i == earliest_i else ev
                for i, ev in enumerate(evs)
            ]
        else:
            result[role] = list(evs)
    return result
