"""sanity_check_overlay.py — Step 2 of Phase 3: extract a handful of real video frames
and overlay all three ground-truth annotation types, so a human can visually confirm
the join logic and the ball-sentinel exclusion rule are correct before anything else in
the pipeline depends on them. Not a reusable library module -- a one-off manual-review
script, run directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations

OUT_DIR = Path(__file__).resolve().parents[1] / "scratch_output" / "sanity_check_overlay"

CLIP = "video1"
# Chosen to cover: a real ball position (400), a sentinel run near the start (150) and
# mid-clip (300), and one more real-ball frame for variety (450).
SAMPLE_FRAMES = [0, 150, 300, 400, 450, 620]


def draw_overlay(frame, ann):
    if ann.court_corners:
        pts = ann.court_corners
        order = ["BL", "BR", "TR", "TL"]
        for i in range(4):
            p1 = tuple(int(c) for c in pts[order[i]])
            p2 = tuple(int(c) for c in pts[order[(i + 1) % 4]])
            cv2.line(frame, p1, p2, (255, 255, 0), 2)
        for name in order:
            p = tuple(int(c) for c in pts[name])
            cv2.circle(frame, p, 8, (255, 255, 0), -1)
            cv2.putText(frame, name, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if ann.player_r:
        p = tuple(int(c) for c in ann.player_r)
        cv2.circle(frame, p, 12, (0, 165, 255), 3)
        cv2.putText(frame, "player_r", (p[0] + 14, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    if ann.player_l:
        p = tuple(int(c) for c in ann.player_l)
        cv2.circle(frame, p, 12, (255, 0, 255), 3)
        cv2.putText(frame, "player_l", (p[0] + 14, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    if ann.ball is not None:
        p = tuple(int(c) for c in ann.ball)
        cv2.circle(frame, p, 12, (0, 0, 255), 3)
        cv2.putText(frame, "ball", (p[0] + 14, p[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif ann.ball_is_sentinel:
        cv2.putText(frame, "BALL: no ground truth (sentinel excluded)", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif ann.ball_row_missing:
        cv2.putText(frame, "BALL: no ground truth (row missing)", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    label = f"frame_{ann.frame_index:03d}"
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotations = load_clip_annotations(CLIP)
    cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{CLIP}.mp4"))

    for idx in SAMPLE_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"frame {idx}: read failed")
            continue
        ann = annotations[idx]
        frame = draw_overlay(frame, ann)
        out_path = OUT_DIR / f"{CLIP}_frame{idx:03d}_overlay.png"
        cv2.imwrite(str(out_path), frame)
        ball_status = (
            "real" if ann.ball is not None
            else "SENTINEL-excluded" if ann.ball_is_sentinel
            else "row-missing" if ann.ball_row_missing
            else "?"
        )
        print(f"wrote {out_path}  (ball: {ball_status})")


if __name__ == "__main__":
    main()
