"""stress_test_pro_clip.py — Phase 3 stress test: out-of-dataset generalization check
on data/tennis_clip.mp4 (professional practice-court clip). NO GROUND TRUTH exists
for this clip -- everything reported is qualitative observation (detected/not,
plausible/not, stable/not), never an accuracy percentage. Scoped to a 900-frame
(~15s) segment of the ~13-minute source clip, matching the amateur dataset's
per-clip scale, rather than processing the full video.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.ball_detection import run_ball_frame_detection
from cv_pipeline.homography import CourtHomography
from cv_pipeline.pose_estimation import run_pose_on_box, make_landmarker
from cv_pipeline.player_selection import select_players_by_court_position

CLIP_PATH = Path(__file__).resolve().parents[2] / "data" / "tennis_clip.mp4"
OUT_DIR = Path(__file__).resolve().parents[1] / "scratch_output" / "stress_test"
START_FRAME = 3600  # 60s in, at 60fps
N_FRAMES = 900  # 15s segment

# Manually estimated from a representative frame (see STRESS_TEST_REPORT.md's
# homography section) -- not independently validated, but sufficient for
# near/far ORDERING via player_selection.py, which is what pose selection needs.
MANUAL_CORNERS = {
    "BL": (65, 793), "BR": (1855, 780),
    "TL": (395, 430), "TR": (1490, 425),
}


def box_bottom_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2


def main():
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yolo = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(CLIP_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    n_frames_with_2plus_boxes = 0
    n_frames_with_1_box = 0
    n_frames_with_0_boxes = 0
    max_boxes_seen = 0
    box_count_examples = {}  # n_boxes -> one representative frame index

    # ByteTrack
    track_model = YOLO("yolov8n.pt")
    id_sequences = []  # list of (frame_offset, [ (id, box, conf) ])

    ball_matches = 0
    ball_candidates_seen = 0

    for offset in range(N_FRAMES):
        idx = START_FRAME + offset
        ok, frame = cap.read()
        if not ok:
            break

        results = yolo.predict(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        confs = results[0].boxes.conf.cpu().numpy().tolist()
        n = len(boxes)
        max_boxes_seen = max(max_boxes_seen, n)
        if n == 0:
            n_frames_with_0_boxes += 1
        elif n == 1:
            n_frames_with_1_box += 1
        else:
            n_frames_with_2plus_boxes += 1
        if n not in box_count_examples:
            box_count_examples[n] = (offset, frame.copy(), boxes, confs)

        track_results = track_model.track(frame, classes=[0], persist=True, tracker="bytetrack.yaml", verbose=False)
        t_boxes = track_results[0].boxes.xyxy.cpu().numpy().tolist() if len(track_results) else []
        t_ids = (track_results[0].boxes.id.cpu().numpy().tolist()
                 if (len(track_results) and track_results[0].boxes.id is not None) else [])
        id_sequences.append((offset, list(zip(t_ids, t_boxes))))

        ball_res_boxes = yolo.predict(frame, classes=[32], verbose=False)[0].boxes.xyxy.cpu().numpy().tolist()
        if ball_res_boxes:
            ball_candidates_seen += 1

    print(f"=== Detection over {N_FRAMES} frames (segment {START_FRAME}-{START_FRAME+N_FRAMES}) ===")
    print(f"  frames with 0 person boxes: {n_frames_with_0_boxes}")
    print(f"  frames with 1 person box: {n_frames_with_1_box}")
    print(f"  frames with 2+ person boxes: {n_frames_with_2plus_boxes}")
    print(f"  max boxes seen in a single frame: {max_boxes_seen}")
    print(f"  distinct box-count example frames captured: {sorted(box_count_examples.keys())}")
    print()
    print(f"=== Ball (sports-ball class) candidate frames: {ball_candidates_seen}/{N_FRAMES} ===")

    # Save annotated examples for each distinct box-count case (covers 0,1,2+ boxes)
    for n, (offset, frame, boxes, confs) in sorted(box_count_examples.items()):
        annotated = frame.copy()
        for b, c in zip(boxes, confs):
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{c:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out_path = OUT_DIR / f"detect_nboxes{n}_offset{offset}.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"  saved {out_path} (n_boxes={n}, offset={offset})")

    # ID-sequence summary: which ids appear, and do any appear only briefly (bystander flicker)?
    from collections import Counter
    id_counts = Counter()
    for offset, id_box_list in id_sequences:
        for tid, box in id_box_list:
            id_counts[int(tid)] += 1
    print()
    print("=== ByteTrack ID frequency across segment ===")
    for tid, count in sorted(id_counts.items(), key=lambda x: -x[1]):
        print(f"  id={tid}: appeared in {count}/{N_FRAMES} frames")

    # Save a tracking-annotated sample frame near the middle of the segment
    mid_offset = N_FRAMES // 2
    mid_ids = id_sequences[mid_offset][1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME + mid_offset)
    ok, mid_frame = cap.read()
    if ok:
        for tid, box in mid_ids:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(mid_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(mid_frame, f"id={int(tid)}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.imwrite(str(OUT_DIR / f"tracking_mid_offset{mid_offset}.png"), mid_frame)
        print(f"  saved tracking_mid_offset{mid_offset}.png")

    # Pose on near + far player in a couple of frames (pick a frame with 2 boxes if available).
    # Selection uses court-position plausibility (via homography), NOT box size --
    # size-based selection was confirmed to pick bystanders/officials on this exact
    # clip (see STRESS_TEST_REPORT.md) and on the amateur dataset's video9. See
    # player_selection.py for the full rationale and the fix.
    landmarker = make_landmarker()
    homography = CourtHomography(MANUAL_CORNERS)
    two_box_frame = box_count_examples.get(2) or box_count_examples.get(max((k for k in box_count_examples if k >= 2), default=None))
    if two_box_frame:
        offset, frame, boxes, confs = two_box_frame
        selection = select_players_by_court_position(boxes, homography)
        print(f"  [selection] {selection.note}")
        for label, box in [("near", selection.near_box), ("far", selection.far_box)]:
            if box is None:
                print(f"  pose on {label} player (offset {offset}): NO PLAUSIBLE BOX -- skipped")
                continue
            pose_result = run_pose_on_box(landmarker, frame, box)
            annotated = frame.copy()
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            if pose_result.landmarks:
                for x, y, vis in pose_result.landmarks:
                    color = (0, 255, 0) if vis > 0.5 else (0, 165, 255)
                    cv2.circle(annotated, (int(x), int(y)), 4, color, -1)
                status = "landmarks found"
            else:
                cv2.putText(annotated, "POSE: NO LANDMARKS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                status = "NO LANDMARKS"
            out_path = OUT_DIR / f"pose_{label}_offset{offset}.png"
            cv2.imwrite(str(out_path), annotated)
            print(f"  pose on {label} player (offset {offset}): {status} -> {out_path}")


if __name__ == "__main__":
    main()
