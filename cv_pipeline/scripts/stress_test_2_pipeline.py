"""stress_test_2_pipeline.py -- Stress Test #2, Step 3.

Runs Phase 3's detection/tracking/pose stack (same YOLOv8n + ByteTrack + MediaPipe
calls as stress_test_pro_clip.py) on the frames from the 75s sample of
data/match_tennis.mp4 that Step 2's heuristic camera-angle filter classified as
'valid' (full-court rally view). Read-only characterization -- does not modify
any Phase 3 pipeline code.

Net-cam tagging: Step 2's visual spot-check found the 'valid' bucket contains a
specialty net-crossing camera angle (ball crossing the net, no players in frame)
that passes the court-color/line-count filter but isn't a normal wide rally shot.
Heuristically tagged here as: 0 person boxes detected AND court-line count >= 45
(the same line-count signature already used for the angle filter). This is
reported SEPARATELY from genuine wide-rally frames throughout, per instruction --
ball detection on an isolated, large, close net-crossing ball is not a valid proxy
for ball-detection quality on normal wide-shot rally play, and blending the two
would misattribute any apparent improvement.

Homography: only calibrated for the single dominant wide-broadcast camera setup
(the framing seen throughout the original 75s sample's main rally shots, corners
below). The video cuts between multiple distinct camera setups even within the
'valid' bucket (the net-cam angle is a different lens/framing entirely) -- a
single 4-corner calibration cannot cover all of them, and per-shot recalibration
is out of scope for this test. Homography is therefore applied ONLY to frames
whose person-detection layout is consistent with the calibrated wide shot, and
skipped (reported as skipped, not silently dropped) for others.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.homography import CourtHomography
from cv_pipeline.pose_estimation import run_pose_on_box, make_landmarker
from cv_pipeline.player_selection import select_players_by_court_position

VIDEO_PATH = Path(__file__).resolve().parents[2] / "data" / "match_tennis.mp4"
OUT_DIR = Path(__file__).resolve().parents[1] / "scratch_output" / "stress_test_2"

FPS = 25
START_SECONDS = 5 * 60
SAMPLE_SECONDS = 75
START_FRAME = START_SECONDS * FPS
N_FRAMES = SAMPLE_SECONDS * FPS

COURT_HSV_LOWER = np.array([95, 60, 90])
COURT_HSV_UPPER = np.array([122, 255, 255])
COURT_FRAC_THRESHOLD = 0.55
LINE_COUNT_THRESHOLD = 45

# Calibrated against sample_offset0.jpg (the dominant wide-broadcast framing).
MANUAL_CORNERS = {
    "BL": (230, 543), "BR": (1050, 543),
    "TL": (395, 218), "TR": (890, 218),
}


def classify_angle(frame):
    h = frame.shape[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COURT_HSV_LOWER, COURT_HSV_UPPER)
    court_frac = mask[int(h * 0.25):, :].mean() / 255
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=180, maxLineGap=15)
    n_lines = 0 if lines is None else len(lines)
    is_valid = court_frac > COURT_FRAC_THRESHOLD and n_lines >= LINE_COUNT_THRESHOLD
    return is_valid, n_lines


def main():
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    print("Extracting + classifying 75s sample...")
    valid_frames = []  # (offset, frame, n_lines)
    for offset in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        is_valid, n_lines = classify_angle(frame)
        if is_valid:
            valid_frames.append((offset, frame, n_lines))
    print(f"  {len(valid_frames)} valid frames out of {N_FRAMES}")

    yolo = YOLO("yolov8n.pt")
    track_model = YOLO("yolov8n.pt")
    landmarker = make_landmarker()
    homography = CourtHomography(MANUAL_CORNERS)

    n_person_0 = n_person_1 = n_person_2plus = 0
    n_ball_hit_all = 0
    n_ball_hit_netcam = 0
    n_ball_hit_wide = 0
    n_netcam = 0
    n_wide = 0
    id_counts = Counter()
    prev_ids = None
    id_changes = 0
    homography_applied = 0
    homography_skipped = 0
    pose_far_attempts = 0
    pose_far_success = 0
    pose_examples_saved = 0

    examples_dir = OUT_DIR / "step3_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for offset, frame, n_lines in valid_frames:
        person_res = yolo.predict(frame, classes=[0], verbose=False)[0]
        boxes = person_res.boxes.xyxy.cpu().numpy().tolist()
        n_person = len(boxes)
        if n_person == 0:
            n_person_0 += 1
        elif n_person == 1:
            n_person_1 += 1
        else:
            n_person_2plus += 1

        is_netcam = (n_person == 0 and n_lines >= LINE_COUNT_THRESHOLD)
        # also tag frames whose only detections look like a stray/near-net artifact
        # (kept simple per the 0-person signature, the dominant real case observed)
        if is_netcam:
            n_netcam += 1
        else:
            n_wide += 1

        ball_res = yolo.predict(frame, classes=[32], verbose=False)[0]
        ball_hit = len(ball_res.boxes.xyxy) > 0
        if ball_hit:
            n_ball_hit_all += 1
            if is_netcam:
                n_ball_hit_netcam += 1
            else:
                n_ball_hit_wide += 1

        track_res = track_model.track(frame, classes=[0], persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        t_ids = (track_res.boxes.id.cpu().numpy().tolist() if track_res.boxes.id is not None else [])
        cur_ids = set(int(i) for i in t_ids)
        for tid in cur_ids:
            id_counts[tid] += 1
        if prev_ids is not None and cur_ids and prev_ids and cur_ids != prev_ids and not is_netcam:
            id_changes += 1
        prev_ids = cur_ids if cur_ids else prev_ids

        # far-player pose + homography, only on genuine wide multi-person frames.
        # Pre-filter: on this footage YOLO fires small (<2000px^2) false-positive
        # "person" boxes on the on-screen clock/scoreboard graphic and line
        # officials, which the unfiltered court-position ranking sometimes ranks
        # as more plausible "far player" than the real (also fairly small,
        # low-confidence) far player -- confirmed on the reference frame: real far
        # player box was area=3042/conf=0.32, spurious clock-graphic box was
        # area=1214/conf=0.51, and unfiltered selection picked the graphic. This
        # is a pre-filter in THIS test script, not a change to
        # select_players_by_court_position() itself -- flagged, not silently
        # patched into Phase 3 code.
        MIN_BOX_AREA = 2000
        if n_person >= 2 and not is_netcam:
            sized_boxes = [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_BOX_AREA]
            pose_far_attempts += 1
            try:
                selection = select_players_by_court_position(sized_boxes, homography)
                homography_applied += 1
                far_box = selection.far_box
            except Exception:
                homography_skipped += 1
                far_box = min(sized_boxes, key=lambda b: (b[3] - b[1]) * (b[2] - b[0])) if sized_boxes else None

            if far_box is not None:
                pose_result = run_pose_on_box(landmarker, frame, far_box)
                if pose_result.landmarks:
                    pose_far_success += 1
                if pose_examples_saved < 6:
                    annotated = frame.copy()
                    x1, y1, x2, y2 = [int(v) for v in far_box]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    if pose_result.landmarks:
                        for x, y, vis in pose_result.landmarks:
                            cv2.circle(annotated, (int(x), int(y)), 3, (0, 255, 0), -1)
                    out_path = examples_dir / f"farpose_offset{offset}_{'landmarks' if pose_result.landmarks else 'nolandmarks'}.jpg"
                    cv2.imwrite(str(out_path), annotated)
                    pose_examples_saved += 1
        elif n_person == 0 and is_netcam and n_netcam <= 6:
            out_path = examples_dir / f"netcam_offset{offset}.jpg"
            cv2.imwrite(str(out_path), frame)

    elapsed = time.time() - t0
    n_total = len(valid_frames)

    print(f"\n=== Step 3 results over {n_total} valid frames ({elapsed:.1f}s, {elapsed/n_total*1000:.1f}ms/frame) ===")
    print(f"\n--- Net-cam tagging ---")
    print(f"  tagged net-cam (0 person boxes + line-count>=45): {n_netcam} ({n_netcam/n_total*100:.1f}%)")
    print(f"  genuine wide-shot frames: {n_wide} ({n_wide/n_total*100:.1f}%)")

    print(f"\n--- Person detection (all valid frames) ---")
    print(f"  0 boxes: {n_person_0} ({n_person_0/n_total*100:.1f}%)")
    print(f"  1 box:   {n_person_1} ({n_person_1/n_total*100:.1f}%)")
    print(f"  2+ boxes: {n_person_2plus} ({n_person_2plus/n_total*100:.1f}%)")

    print(f"\n--- Ball detection (COCO sports-ball class) ---")
    print(f"  all valid frames: {n_ball_hit_all}/{n_total} ({n_ball_hit_all/n_total*100:.1f}%)")
    if n_netcam:
        print(f"  net-cam frames only: {n_ball_hit_netcam}/{n_netcam} ({n_ball_hit_netcam/n_netcam*100:.1f}%)")
    if n_wide:
        print(f"  genuine wide-shot frames only: {n_ball_hit_wide}/{n_wide} ({n_ball_hit_wide/n_wide*100:.1f}%)")

    print(f"\n--- Tracking (ByteTrack, wide-shot frames only) ---")
    print(f"  distinct IDs seen: {len(id_counts)}")
    for tid, count in sorted(id_counts.items(), key=lambda x: -x[1])[:6]:
        print(f"    id={tid}: {count} frames")
    print(f"  ID changes across wide-shot frames: {id_changes}")

    print(f"\n--- Far-player pose (2+ person boxes, genuine wide-shot frames only) ---")
    print(f"  attempts: {pose_far_attempts}")
    print(f"  homography applied: {homography_applied}, homography skipped (fallback to box-size): {homography_skipped}")
    if pose_far_attempts:
        print(f"  landmarks found: {pose_far_success}/{pose_far_attempts} ({pose_far_success/pose_far_attempts*100:.1f}%)")
    print(f"\n  saved example frames to {examples_dir}")


if __name__ == "__main__":
    main()
