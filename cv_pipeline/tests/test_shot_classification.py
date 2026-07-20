"""test_shot_classification.py -- synthetic sanity checks for the
ball-anchored forehand/backhand heuristic. Confirms the specific behavior
this module exists for: re-anchoring classification away from a misleading
elbow-extension peak (which real testing found lands in the follow-through,
often on the wrong anatomical side -- see PROGRESS.md) onto the real
ball-proximity frame, and dropping events cleanly (never falling back to
the peak-frame proxy) when no usable ball/pose data is nearby. Real-clip
accuracy is a separate, manual-audit question, also in PROGRESS.md.
"""

from __future__ import annotations

from cv_pipeline.shot_classification import ShotEvent, find_shot_events, flag_first_event_as_probable_serve

WIDE_BOX = (-10.0, -10.0, 10.0, 10.0)


def _lm_at(elbow_r_xy, shoulder_l=(-1, 0), shoulder_r=(1, 0), hip_l=(-1, 2), hip_r=(1, 2)):
    lm = [(0.0, 0.0, 1.0)] * 33
    lm[11] = (shoulder_l[0], shoulder_l[1], 1.0)
    lm[12] = (shoulder_r[0], shoulder_r[1], 1.0)
    lm[14] = (elbow_r_xy[0], elbow_r_xy[1], 1.0)
    lm[23] = (hip_l[0], hip_l[1], 1.0)
    lm[24] = (hip_r[0], hip_r[1], 1.0)
    return lm


def _clip_with_misleading_follow_through():
    """A single forehand: true contact at frame 20 (moderate extension,
    correct/right side), but the raw extension signal peaks higher at frame
    24, during a follow-through that has crossed to the wrong/left side --
    reproducing the exact real failure mode found on real footage."""
    landmarks_by_frame = {f: _lm_at((0.0, 0.1)) for f in range(40)}
    ramp = {
        15: (0.2, 0.2), 16: (0.5, 0.2), 17: (0.8, 0.3), 18: (1.0, 0.3), 19: (1.0, 0.35),
        20: (1.0, 0.4),  # true contact -- correct/right side, moderate extension
        21: (0.5, 0.6), 22: (-0.2, 0.8), 23: (-0.9, 0.9),
        24: (-1.5, 1.0),  # raw peak -- higher extension, WRONG/left side
        25: (-1.3, 0.8), 26: (-0.8, 0.5), 27: (-0.3, 0.3),
    }
    for f, xy in ramp.items():
        landmarks_by_frame[f] = _lm_at(xy)
    return landmarks_by_frame


def test_ball_anchor_overrides_misleading_peak_frame():
    landmarks_by_frame = _clip_with_misleading_follow_through()
    box_by_frame = {f: WIDE_BOX for f in range(40)}
    ball_by_frame = {20: (0.0, 0.0)}  # ball only near true contact, nowhere near the raw peak

    events = find_shot_events(landmarks_by_frame, ball_by_frame, box_by_frame)
    assert len(events) == 1
    assert events[0].frame_index == 20  # anchored to true contact, not the peak at 24
    assert events[0].peak_frame_index == 24
    assert events[0].classification == "forehand"  # correct -- would have been "backhand" at the raw peak


def test_event_dropped_when_no_ball_nearby():
    landmarks_by_frame = _clip_with_misleading_follow_through()
    box_by_frame = {f: WIDE_BOX for f in range(40)}
    events = find_shot_events(landmarks_by_frame, ball_by_frame={}, box_by_frame=box_by_frame)
    assert events == []  # no silent fallback to peak-frame classification


def test_event_dropped_when_ball_found_but_no_pose_nearby():
    landmarks_by_frame = _clip_with_misleading_follow_through()
    # Remove all pose data in a wide window around the true contact frame.
    for f in range(15, 26):
        landmarks_by_frame[f] = None
    box_by_frame = {f: WIDE_BOX for f in range(40)}
    ball_by_frame = {20: (0.0, 0.0)}
    events = find_shot_events(landmarks_by_frame, ball_by_frame, box_by_frame)
    assert events == []


def test_no_events_on_a_flat_ready_stance_signal():
    landmarks_by_frame = {f: _lm_at((0.0, 0.1)) for f in range(40)}
    box_by_frame = {f: WIDE_BOX for f in range(40)}
    events = find_shot_events(landmarks_by_frame, ball_by_frame={20: (0.0, 0.0)}, box_by_frame=box_by_frame)
    assert events == []


def test_left_handed_not_silently_supported():
    import pytest
    landmarks_by_frame = _clip_with_misleading_follow_through()
    box_by_frame = {f: WIDE_BOX for f in range(40)}
    with pytest.raises(NotImplementedError):
        find_shot_events(landmarks_by_frame, {20: (0.0, 0.0)}, box_by_frame, dominant_hand="left")


def _ev(frame_index: int, classification: str = "forehand") -> ShotEvent:
    return ShotEvent(frame_index=frame_index, classification=classification,
                      peak_frame_index=frame_index, ball_distance_to_peak_frames=0)


def test_probable_serve_flags_only_the_single_earliest_event_across_roles():
    events_by_role = {
        "near": [_ev(50), _ev(300)],
        "far": [_ev(10), _ev(200)],  # far's frame 10 is the overall earliest
    }
    result = flag_first_event_as_probable_serve(events_by_role)
    assert [e.probable_serve for e in result["near"]] == [False, False]
    assert [e.probable_serve for e in result["far"]] == [True, False]
    # non-flagged events are otherwise untouched
    assert result["near"][0] == events_by_role["near"][0]
    assert result["far"][1] == events_by_role["far"][1]


def test_probable_serve_earliest_within_a_single_role():
    events_by_role = {"near": [_ev(80), _ev(5)], "far": []}
    result = flag_first_event_as_probable_serve(events_by_role)
    assert result["near"][0].probable_serve is False  # frame 80
    assert result["near"][1].probable_serve is True   # frame 5, the earliest
    assert result["far"] == []


def test_probable_serve_no_events_returns_unchanged():
    events_by_role = {"near": [], "far": []}
    assert flag_first_event_as_probable_serve(events_by_role) == events_by_role


def test_probable_serve_does_not_mutate_input_events():
    original = _ev(10)
    events_by_role = {"near": [original], "far": []}
    result = flag_first_event_as_probable_serve(events_by_role)
    assert original.probable_serve is False  # frozen, unchanged
    assert result["near"][0].probable_serve is True
    assert result["near"][0] is not original
