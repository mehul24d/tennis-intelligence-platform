"""tracking.py — runs YOLOv8 + ByteTrack (via ultralytics' built-in model.track())
across a clip and validates ID consistency against the ground truth's 2 labeled
players: does the tracker keep the SAME track ID on the SAME physical player
throughout the clip, or does it swap IDs (e.g. on occlusion or players crossing near
the net)?

METHOD: for each frame, match each ground-truth point (player_r, player_l) to its
nearest tracked box (same bottom-center convention as player_detection.py, and same
sentinel-aware ground truth from annotations.py). Build two sequences: the track ID
assigned to "whichever box is closest to player_r this frame" and same for player_l.
An ID SWAP is detected when the dominant (most common) track ID for a ground-truth
slot changes partway through the clip -- i.e. the slot was mostly ID=3 for a stretch,
then mostly ID=5 for a later stretch, with more than one such dominant-ID segment.

This does NOT attempt to identify WHICH frame the swap happened on with sub-frame
precision -- ByteTrack assigns IDs frame-by-frame and matching is done independently
per frame, so a swap is inferred from a change in the majority-vote ID over a rolling
window, not claimed as an exact event timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_MATCH_DISTANCE_PX = 150.0
ROLLING_WINDOW = 15  # frames, for majority-vote ID smoothing


@dataclass(frozen=True)
class ClipTrackingResult:
    clip: str
    n_frames: int
    player_r_id_sequence: list[int | None]  # None = no matched track this frame
    player_l_id_sequence: list[int | None]
    player_r_n_segments: int  # distinct dominant-ID segments (1 = no swap detected)
    player_l_n_segments: int
    player_r_swap_frames: list[int]  # approximate frame indices where the dominant ID changed
    player_l_swap_frames: list[int]


def box_bottom_center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2


def match_frame_ids(gt_r, gt_l, boxes, ids) -> tuple[int | None, int | None]:
    """Exclusive matching for player_r/player_l in the SAME frame -- both cannot claim
    the same box. Without this, when only one real detection exists near both
    ground-truth points (the same contamination found in steps 4-5: player_r/player_l
    frequently sit near the same physical player), both slots would spuriously match
    the same track ID, producing a trivially "perfect" ID-consistency result that
    doesn't actually test 2-distinct-player tracking at all. player_r matched first
    (arbitrary but fixed priority, consistent with player_detection.py)."""
    if not len(boxes) or ids is None:
        return None, None
    centers = [box_bottom_center(b) for b in boxes]
    available = list(range(len(centers)))

    def claim(gt_point):
        if gt_point is None or not available:
            return None
        dists = [np.hypot(gt_point[0] - centers[i][0], gt_point[1] - centers[i][1]) for i in available]
        best_pos = int(np.argmin(dists))
        if dists[best_pos] <= MAX_MATCH_DISTANCE_PX:
            box_idx = available.pop(best_pos)
            return int(ids[box_idx])
        return None

    r_id = claim(gt_r)
    l_id = claim(gt_l)
    return r_id, l_id


def _rolling_majority_segments(id_sequence: list[int | None]) -> tuple[int, list[int]]:
    """Smooths the raw per-frame ID sequence with a rolling-window majority vote (so a
    single missed/misassigned frame doesn't count as a full "segment"), then counts
    how many distinct dominant-ID segments result and where each transition happens.
    """
    n = len(id_sequence)
    smoothed: list[int | None] = []
    for i in range(n):
        lo, hi = max(0, i - ROLLING_WINDOW // 2), min(n, i + ROLLING_WINDOW // 2 + 1)
        window = [v for v in id_sequence[lo:hi] if v is not None]
        if not window:
            smoothed.append(None)
            continue
        vals, counts = np.unique(window, return_counts=True)
        smoothed.append(int(vals[np.argmax(counts)]))

    segments = []
    transitions = []
    current = None
    for i, v in enumerate(smoothed):
        if v is None:
            continue
        if current is None:
            current = v
            segments.append(v)
        elif v != current:
            current = v
            segments.append(v)
            transitions.append(i)
    # collapse immediate back-and-forth flicker: only count a segment as "new" if it
    # persists on its own (already enforced by majority-vote smoothing above).
    n_segments = len(set(segments)) if segments else 0
    # more precisely: count actual segment RUNS, not unique ids (an ID could return
    # later, still counts as a swap event at both boundaries)
    n_segment_runs = 1 if segments else 0
    for i in range(1, len(segments)):
        if segments[i] != segments[i - 1]:
            n_segment_runs += 1
    return n_segment_runs, transitions


def run_clip_tracking(model, clip: str, annotations: dict, cap) -> ClipTrackingResult:
    n_frames = len(annotations)
    r_ids: list[int | None] = []
    l_ids: list[int | None] = []

    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        ann = annotations[idx]
        results = model.track(frame, classes=[0], persist=True, tracker="bytetrack.yaml", verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        ids = results[0].boxes.id.cpu().numpy().tolist() if (len(results) and results[0].boxes.id is not None) else None

        r_id, l_id = match_frame_ids(ann.player_r, ann.player_l, boxes, ids)
        r_ids.append(r_id)
        l_ids.append(l_id)

    r_segments, r_transitions = _rolling_majority_segments(r_ids)
    l_segments, l_transitions = _rolling_majority_segments(l_ids)

    return ClipTrackingResult(
        clip=clip, n_frames=n_frames,
        player_r_id_sequence=r_ids, player_l_id_sequence=l_ids,
        player_r_n_segments=r_segments, player_l_n_segments=l_segments,
        player_r_swap_frames=r_transitions, player_l_swap_frames=l_transitions,
    )
