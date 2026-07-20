"""run_pose_spot_check.py — Step 7: MediaPipe pose on YOLO player boxes, VISUAL
SPOT-CHECK ONLY. No ground truth exists for pose -- this script makes no accuracy
claim, computes no error rate. It saves annotated sample frames covering a deliberate
mix of easy and hard cases (frontal/still, mid-swing motion blur, the far/small
player where detection is already known to be weak from steps 4-5) so a human can
look at them and judge "does this look reasonable," which is the only standard that
applies here. Frames where pose clearly fails (no player detected, landmarks
obviously off the body) are saved and labeled as failures, not filtered out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations
from cv_pipeline.homography import CourtHomography
from cv_pipeline.pose_estimation import run_pose_on_box, make_landmarker
from cv_pipeline.player_selection import select_players_by_court_position

OUT_DIR = Path(__file__).resolve().parents[1] / "scratch_output" / "pose_spot_check"

# Hand-picked to cover: easy near-player frontal stance, a mid-swing/serve motion
# (harder: limbs extended, potential blur), and the far/small player specifically
# (already known from steps 4-5 to be a weak detection case -- worth seeing whether
# pose on top of a rare, likely-lower-quality box degrades further or is fine when it
# does get a box at all).
CASES = [
    ("video1", 400, "near player, clean frontal/ready stance (easy case)"),
    ("video7", 5, "near player mid-serve-toss, arm extended (harder: motion/pose)"),
    ("video3", 12, "near player mid-serve, arm raised, ball in frame (harder)"),
    ("video6", 700, "near player after break resumes, mid-shot (harder: motion blur)"),
    ("video1", 400, "far player, same frame as case 1 (harder: small/low-res box)"),
    ("video9", 300, "far player specifically (known-weak detection case)"),
]


def get_yolo_boxes(model, frame):
    results = model.predict(frame, classes=[0], verbose=False)
    return results[0].boxes.xyxy.cpu().numpy().tolist()


def build_clip_homography(clip: str) -> CourtHomography:
    annotations = load_clip_annotations(clip)
    first_court_ann = next(a for a in annotations.values() if a.court_corners)
    return CourtHomography(first_court_ann.court_corners)


def pick_box(boxes, prefer: str, homography: CourtHomography):
    """Selects by COURT-POSITION PLAUSIBILITY (via homography), not box size --
    size-based selection ("largest=near, smallest=far") was confirmed wrong on two
    independent clips (video9 here, and the Phase 3 stress-test clip): it can pick
    a bystander/official whose box happens to be a similar size to the real
    far-player box. See player_selection.py for the full rationale."""
    if not boxes:
        return None, None
    selection = select_players_by_court_position(boxes, homography)
    box = selection.far_box if prefer == "far" else selection.near_box
    return box, selection.note


def draw_pose(frame, pose_result):
    if pose_result.landmarks is None:
        cv2.putText(frame, "POSE: NO LANDMARKS DETECTED (failure)", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame
    for x, y, vis in pose_result.landmarks:
        color = (0, 255, 0) if vis > 0.5 else (0, 165, 255)  # orange = low-visibility landmark
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    x1, y1, x2, y2 = pose_result.box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
    return frame


def main():
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yolo = YOLO("yolov8n.pt")
    landmarker = make_landmarker()

    homography_cache: dict[str, CourtHomography] = {}

    for i, (clip, frame_idx, description) in enumerate(CASES):
        prefer = "far" if "far player" in description else "near"
        cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{clip}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[{i}] {clip} frame {frame_idx}: FRAME READ FAILED")
            continue

        if clip not in homography_cache:
            homography_cache[clip] = build_clip_homography(clip)
        homography = homography_cache[clip]

        boxes = get_yolo_boxes(yolo, frame)
        box, selection_note = pick_box(boxes, prefer, homography)
        print(f"    [selection] {selection_note}")
        if box is None:
            print(f"[{i}] {clip} frame {frame_idx} ({description}): "
                  f"NO PLAUSIBLE {prefer.upper()}-PLAYER BOX -- pose cannot be attempted. FAILURE CASE.")
            out_path = OUT_DIR / f"{i}_{clip}_f{frame_idx}_NO_DETECTION.png"
            cv2.putText(frame, f"NO PLAUSIBLE {prefer.upper()}-PLAYER BOX (failure)", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imwrite(str(out_path), frame)
            continue

        pose_result = run_pose_on_box(landmarker, frame, box)
        annotated = draw_pose(frame.copy(), pose_result)
        status = "FAILURE (no landmarks)" if pose_result.landmarks is None else "landmarks drawn"
        out_path = OUT_DIR / f"{i}_{clip}_f{frame_idx}.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"[{i}] {clip} frame {frame_idx} ({description}): {status} -> {out_path}")


if __name__ == "__main__":
    main()
