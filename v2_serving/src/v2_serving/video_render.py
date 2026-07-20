"""video_render.py — burns run_video_analysis's per-frame overlay data (court
lines, player boxes, ball, shot-classification labels) directly into the
source video's frames and writes a real .mp4 file. The server-side
counterpart to VideoOverlay.jsx's live in-browser canvas overlay -- same
overlay elements, same "only draw what's true for this exact frame, explicit
absence stays absent" rule that component's docstring establishes -- but as a
downloadable artifact rather than requiring the dashboard to be open.

Draws from an ALREADY-COMPUTED run_video_analysis result -- does not re-run
any detection/pose/ball/pose model itself, so this is cheap (pure video I/O +
cv2 drawing) relative to the analysis that produced `result`. A caller with a
completed /analyze-video job passes its stored (video_path, result) straight
through; see routers/render.py.

Frames beyond the analyzed range (i.e. index >= the caller's frame_limit) are
written UNANNOTATED, not dropped -- the output video's duration always
matches the source, with an honest "no overlay data past this point" rather
than a silently truncated file that could be mistaken for the whole clip
having been analyzed.

CODEC, A REAL BUG FOUND AFTER SHIPPING (not caught by any test here, since
none of them decode the output -- see the fix): the first version used
`cv2.VideoWriter_fourcc(*"mp4v")`, which encodes MPEG-4 Part 2 ("FMP4"/
"mp4v" -- confirmed via `cv2.VideoCapture(...).get(cv2.CAP_PROP_FOURCC)` on
the actual output file). That container+codec combination is NOT supported
by any mainstream browser's HTML5 `<video>` tag (Chrome/Safari/Firefox all
require H.264, VP8/VP9, or AV1) -- the file downloaded fine over HTTP
(confirmed: curl and the server's access log both showed clean 200/206
responses, including Range requests exactly like a `<video>` element
scrubbing) but silently failed to PLAY in a browser, with no error surfaced
anywhere in this pipeline. Fixed to `"avc1"` (H.264) -- verified empirically
on this machine's OpenCV/FFmpeg build (not assumed): `VideoWriter_fourcc(*
"H264")`/`"X264"`/`"h264"` all fail to open for an mp4 container on this
build and silently fall back to `"avc1"` anyway (a real FFmpeg stderr
message, not a guess), while `"avc1"` opens directly and a read-back of the
result reports fourcc `'h264'`, confirming real H.264 output. Regression
test: `tests/test_video_render.py::test_rendered_output_uses_a_browser_playable_codec`
writes a real (tiny) file and asserts its readback fourcc is NOT "mp4v"/
"FMP4", specifically to catch this exact failure mode recurring.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Rough BGR (OpenCV's channel order) approximations of VideoOverlay.jsx's
# COLORS -- exact hex match isn't the point, visual correspondence to the
# live dashboard overlay is.
COLORS_BGR = {
    "near": (129, 211, 52),
    "far": (189, 120, 56),
    "ball": (20, 202, 250),
    "court": (203, 120, 167),
    "singles": (212, 211, 34),
    "shot": (117, 113, 251),
}

# Matches CourtHomography.court_polygon_pixels()/singles_corners_pixels()'s
# fixed key order, used throughout cv_pipeline (see
# calibration_verification.py's CORNER_ORDER) -- corners dicts are always
# {"BL": [x,y], "BR": [x,y], "TR": [x,y], "TL": [x,y]}.
CORNER_ORDER = ["BL", "BR", "TR", "TL"]


def _is_homography_applicable(frame_record: dict) -> bool:
    # Same default-true-unless-explicit-False rule as VideoOverlay.jsx's
    # isHomographyApplicable, for the same reason (older/other result shapes
    # predating this field shouldn't suddenly suppress the court overlay).
    return frame_record.get("homography_applicable") is not False


def _draw_dashed_polygon(img: np.ndarray, pts: list[tuple[int, int]], color: tuple, dash_len=6, gap_len=4) -> None:
    for i in range(len(pts)):
        p1 = np.array(pts[i], dtype=float)
        p2 = np.array(pts[(i + 1) % len(pts)], dtype=float)
        seg_len = float(np.linalg.norm(p2 - p1))
        if seg_len < 1e-6:
            continue
        direction = (p2 - p1) / seg_len
        n_dashes = int(seg_len // (dash_len + gap_len)) + 1
        for d in range(n_dashes):
            start = p1 + direction * d * (dash_len + gap_len)
            end = start + direction * dash_len
            cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, 1, cv2.LINE_AA)


def _draw_quad(img: np.ndarray, corners: dict | None, color: tuple, dashed: bool) -> None:
    if not corners:
        return
    pts = [(int(corners[k][0]), int(corners[k][1])) for k in CORNER_ORDER]
    if dashed:
        _draw_dashed_polygon(img, pts, color)
    else:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)


def _draw_box(img: np.ndarray, box, color: tuple, label: str) -> None:
    if not box:
        return
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1 + 2, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_ball(img: np.ndarray, box, color: tuple) -> None:
    if not box:
        return
    x1, y1, x2, y2 = box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    cv2.circle(img, (cx, cy), 5, color, 2, cv2.LINE_AA)
    cv2.putText(img, "ball", (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_shot_events(img: np.ndarray, frame_record: dict, color: tuple) -> None:
    # A LIST, not a single label -- two distinct candidate swings can
    # legitimately anchor to the same frame (confirmed on real data, see
    # video_pipeline.py's shot-classification block) -- stacked here instead
    # of one overwriting the other, same reasoning as VideoOverlay.jsx's draw
    # loop over shot_events.
    for i, ev in enumerate(frame_record.get("shot_events") or []):
        box = frame_record.get("near_box") if ev["role"] == "near" else frame_record.get("far_box")
        if not box:
            continue
        x1, y1, x2, _y2 = box
        label = f"{ev['classification'].upper()}{' (probable serve)' if ev['probable_serve'] else ''}"
        (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cx = int((x1 + x2) / 2)
        y = int(y1) - 18 - i * 16
        cv2.putText(img, label, (cx - tw // 2, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def render_annotated_video(video_path: str, result: dict[str, Any], output_path: str) -> dict:
    """Writes `output_path` (.mp4) -- the source video with every element
    `run_video_analysis`'s `result` found drawn directly into the matching
    frame. Raises FileNotFoundError if `video_path` doesn't exist, RuntimeError
    if the output VideoWriter can't be opened (e.g. an unwritable directory or
    an unsupported fourcc on this platform). Returns a small summary dict --
    output_path, how many frames were written to the file total vs. how many
    of those actually had overlay data to draw (the two differ whenever the
    analysis that produced `result` used a frame_limit shorter than the full
    video -- see this module's docstring)."""
    t0 = time.time()
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(src))
    fps = result.get("source_fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(result.get("video_width") or cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(result.get("video_height") or cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # avc1 (H.264), not mp4v -- see this module's docstring for the real
    # browser-playback bug this fixes and how the choice was verified.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open VideoWriter for {out_path}")

    frames_by_index = {f["index"]: f for f in result.get("frames", [])}
    homography_report = result.get("homography") or {}
    clip_court_corners = homography_report.get("court_corners")
    clip_singles_corners = homography_report.get("singles_corners")

    n_frames_total = 0
    n_frames_annotated = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_record = frames_by_index.get(frame_idx)
        if frame_record is not None:
            if _is_homography_applicable(frame_record):
                frame_court_corners = frame_record.get("court_corners") or clip_court_corners
                frame_singles_corners = frame_record.get("singles_corners") or clip_singles_corners
                _draw_quad(frame, frame_court_corners, COLORS_BGR["court"], dashed=False)
                _draw_quad(frame, frame_singles_corners, COLORS_BGR["singles"], dashed=True)

            near_tid = frame_record.get("near_track_id")
            far_tid = frame_record.get("far_track_id")
            _draw_box(frame, frame_record.get("near_box"), COLORS_BGR["near"],
                      f"near{f' (id {near_tid})' if near_tid is not None else ''}")
            _draw_box(frame, frame_record.get("far_box"), COLORS_BGR["far"],
                      f"far{f' (id {far_tid})' if far_tid is not None else ''}")
            _draw_ball(frame, frame_record.get("ball_box"), COLORS_BGR["ball"])
            _draw_shot_events(frame, frame_record, COLORS_BGR["shot"])
            n_frames_annotated += 1
        # else: past the analyzed range -- written below unannotated, per
        # this module's docstring, not dropped.

        writer.write(frame)
        n_frames_total += 1
        frame_idx += 1

    cap.release()
    writer.release()

    return {
        "output_path": str(out_path),
        "n_frames_total": n_frames_total,
        "n_frames_annotated": n_frames_annotated,
        "elapsed_s": round(time.time() - t0, 1),
    }
