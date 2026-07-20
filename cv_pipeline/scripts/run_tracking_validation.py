"""run_tracking_validation.py — Step 6: ByteTrack ID-consistency validation.

Per the project's own discipline: a high overall ID-consistency number is meaningless
if it's dominated by easy stretches (one player alone in frame, well-separated) and
never actually tested during hard moments (players close together / crossing near the
net, or one briefly occluded). This script explicitly separates the two: "hard
moment" frames are frame where 2+ YOLO person boxes exist with bottom-centers within
HARD_MOMENT_PROXIMITY_PX of each other (a tracker-centric proxy for crossing/
occlusion risk, not dependent on ground-truth semantics, which step 4/5 already found
to be an unreliable signal of "two distinct players" on its own).

Reports segment counts (>1 = an ID swap was detected) for the whole clip, AND swap
locations relative to hard-moment frames -- if swaps cluster in/near hard moments,
that's the tracker failing exactly where it's supposed to be tested; if a clip has
zero hard moments at all, its "perfect" ID-consistency number is not meaningfully
validated and is reported as such.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations
from cv_pipeline.tracking import run_clip_tracking, box_bottom_center

HARD_MOMENT_PROXIMITY_PX = 200.0
NEAR_SWAP_WINDOW = 20  # frames -- how close a swap needs to be to a hard moment to count as "near" it


def find_hard_moment_frames(model, clip: str, annotations: dict) -> set[int]:
    """Re-runs plain detection (not tracking) to find frames with 2+ close-together
    person boxes -- proximity/crossing/occlusion-risk frames, independent of ground
    truth quality issues found in steps 4-5."""
    cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{clip}.mp4"))
    hard_frames = set()
    for idx in range(len(annotations)):
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        if len(boxes) < 2:
            continue
        centers = [box_bottom_center(b) for b in boxes]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                if d <= HARD_MOMENT_PROXIMITY_PX:
                    hard_frames.add(idx)
    return hard_frames


def main():
    from ultralytics import YOLO

    clip = "video1"
    annotations = load_clip_annotations(clip)

    print(f"=== {clip}: finding hard-moment (crossing/close-proximity) frames ===")
    detect_model = YOLO("yolov8n.pt")
    hard_frames = find_hard_moment_frames(detect_model, clip, annotations)
    print(f"  {len(hard_frames)}/{len(annotations)} frames flagged as hard moments "
          f"(2+ person boxes within {HARD_MOMENT_PROXIMITY_PX:.0f}px)")

    print(f"\n=== {clip}: running ByteTrack ===")
    track_model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{clip}.mp4"))
    result = run_clip_tracking(track_model, clip, annotations, cap)

    print(f"  player_r: {result.player_r_n_segments} dominant-ID segment(s) "
          f"({'no swap detected' if result.player_r_n_segments <= 1 else 'SWAP(S) DETECTED'})")
    print(f"    transition frames (approx): {result.player_r_swap_frames}")
    print(f"  player_l: {result.player_l_n_segments} dominant-ID segment(s) "
          f"({'no swap detected' if result.player_l_n_segments <= 1 else 'SWAP(S) DETECTED'})")
    print(f"    transition frames (approx): {result.player_l_swap_frames}")

    print(f"\n=== Are swaps concentrated in hard moments, or scattered/absent? ===")
    all_transitions = result.player_r_swap_frames + result.player_l_swap_frames
    if not all_transitions:
        print("  No ID swaps detected at all in this clip.")
        if not hard_frames:
            print("  ALSO no hard-moment frames were found -- this clip never seriously tested the "
                  "tracker (players were never close together), so 'no swaps' here does not mean the "
                  "tracker handles crossings/occlusion well. It means this clip didn't test that.")
        else:
            print(f"  But {len(hard_frames)} hard-moment frames DID occur and were tracked through "
                  f"cleanly -- this is a real, meaningful pass, not just an easy clip.")
    else:
        for t in all_transitions:
            near_hard = any(abs(t - h) <= NEAR_SWAP_WINDOW for h in hard_frames)
            print(f"  transition near frame {t}: {'NEAR a hard moment' if near_hard else 'NOT near any flagged hard moment'}")


if __name__ == "__main__":
    main()
