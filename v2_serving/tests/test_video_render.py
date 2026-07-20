"""test_video_render.py — real (not mocked) checks against
render_annotated_video's actual output file. Unlike test_render.py (which
mocks this function out for the API-layer tests), this module writes a real,
tiny synthetic source video and a real output file, because the bug this
guards against -- an unplayable-in-browser codec -- can ONLY be caught by
inspecting the actual bytes written, not by asserting on a return value. See
video_render.py's own docstring for the real bug this was written after
finding (mp4v/FMP4 output that downloaded fine over HTTP but silently failed
to play in every mainstream browser).
"""

from __future__ import annotations

import cv2
import numpy as np

from v2_serving.video_render import render_annotated_video


def _write_synthetic_source(path: str, n_frames: int = 5, size=(64, 48)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(path, fourcc, 10.0, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _read_fourcc(path: str) -> str:
    cap = cv2.VideoCapture(path)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))


def test_rendered_output_uses_a_browser_playable_codec(tmp_path):
    src = str(tmp_path / "source.mp4")
    out = str(tmp_path / "out.mp4")
    _write_synthetic_source(src)

    result = {
        "source_fps": 10.0, "video_width": 64, "video_height": 48,
        "frames": [], "homography": {},
    }
    render_annotated_video(src, result, out)

    fourcc = _read_fourcc(out)
    # The real bug: mp4v/FMP4 downloads fine over HTTP but no mainstream
    # browser's <video> tag can decode it. h264/avc1 is what we assert FOR;
    # mp4v/FMP4 is what we assert AGAINST, by name, so this test fails loudly
    # and specifically if the codec ever regresses back to it.
    assert fourcc not in ("mp4v", "FMP4"), f"regressed to a browser-unplayable codec: {fourcc!r}"
    assert fourcc in ("h264", "avc1"), f"unexpected codec: {fourcc!r}"


def test_rendered_output_frame_count_matches_source(tmp_path):
    src = str(tmp_path / "source.mp4")
    out = str(tmp_path / "out.mp4")
    _write_synthetic_source(src, n_frames=7)

    result = {"source_fps": 10.0, "video_width": 64, "video_height": 48, "frames": [], "homography": {}}
    summary = render_annotated_video(src, result, out)

    assert summary["n_frames_total"] == 7
    assert summary["n_frames_annotated"] == 0  # no per-frame data in `result["frames"]` this test

    cap = cv2.VideoCapture(out)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 7
    cap.release()


def test_rendered_output_draws_a_box_on_an_annotated_frame(tmp_path):
    """Not pixel-exact (compression + drawing anti-aliasing make that
    brittle) -- just confirms drawing a box actually changes the frame's
    pixels versus an unannotated run, i.e. the overlay path really executes."""
    src = str(tmp_path / "source.mp4")
    out_plain = str(tmp_path / "plain.mp4")
    out_annotated = str(tmp_path / "annotated.mp4")
    _write_synthetic_source(src, n_frames=3, size=(200, 150))

    base_result = {"source_fps": 10.0, "video_width": 200, "video_height": 150, "homography": {}}
    render_annotated_video(src, {**base_result, "frames": []}, out_plain)
    render_annotated_video(
        src,
        {**base_result, "frames": [
            {"index": 0, "near_box": [20, 20, 80, 100], "far_box": None, "ball_box": None,
             "near_track_id": None, "far_track_id": None, "shot_events": []},
        ]},
        out_annotated,
    )

    def _first_frame(path):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        assert ok
        return frame

    plain_frame = _first_frame(out_plain)
    annotated_frame = _first_frame(out_annotated)
    assert not np.array_equal(plain_frame, annotated_frame)
