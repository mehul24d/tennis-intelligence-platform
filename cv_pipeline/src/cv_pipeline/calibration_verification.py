"""calibration_verification.py — mandatory visual verification step for any new
court homography calibration, added 2026-07-19 after data/tennis/2.mp4's BL
corner was found to be mismeasured by ~49px despite a held-out-landmark error
(10.1px) that looked entirely reasonable. The held-out-landmark check alone did
NOT catch that bug -- the least-squares fit partially absorbed the one bad
point's error into the other 7, keeping the aggregate number deceptively low.
The bug was only found because the rendered overlay was inspected on a frame
(frame 670) that hadn't been checked before; frames 0 and the last frame had
both looked fine, because the fit happened to be closer to correct there.

THIS IS NOW A MANDATORY, NOT OPTIONAL, STEP: `render_verification_frames`
produces the human-reviewable images, `CalibrationVerificationManifest` records
a real per-frame, per-corner sign-off, and `test_calibration_verification.py`
fails the test suite for any `reference_video*_calibration.py` module that
doesn't have a complete, corresponding manifest checked in -- so this cannot be
silently skipped under time pressure the way it was the first time on 1.mp4
and 2.mp4 (both calibrations were originally accepted on the strength of the
held-out numeric error alone).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cv_pipeline.homography import CourtHomography

CORNER_ORDER = ["BL", "BR", "TR", "TL"]
CORNER_LABEL_COLOR = (255, 255, 255)
DOUBLES_COLOR = (255, 0, 255)  # magenta
SINGLES_COLOR = (0, 255, 255)  # yellow

VERIFICATION_DIR = Path(__file__).resolve().parents[2] / "data" / "calibration_verification"


def _frames_at(video_path: str, indices: list[int]) -> dict[int, "np.ndarray"]:
    """Reads the requested frame indices via pure sequential decoding (never
    cap.set(CAP_PROP_POS_FRAMES, ...)) -- added 2026-07-19 after finding that
    seeking silently returns the WRONG frame's content past a keyframe
    boundary on every reference clip checked (1.mp4-5.mp4 all reproduce it,
    e.g. video4.json's own frame 300 seeks to something that doesn't match a
    sequential read at all: maxdiff=255). A single, narrow, targeted seek
    each call is not safe on these files; only reading frames in order from
    the start is. See reference_video5_calibration.py's docstring for the
    investigation that found this."""
    cap = cv2.VideoCapture(str(video_path))
    wanted = set(indices)
    found: dict[int, "np.ndarray"] = {}
    fi = 0
    max_idx = max(wanted)
    while fi <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in wanted:
            found[fi] = frame.copy()
        fi += 1
    cap.release()
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"could not sequentially read frame(s) {sorted(missing)}")
    return found


def render_verification_frames(
    video_path: Path, homography: CourtHomography, out_dir: Path,
    frame_indices: list[int] | None = None,
) -> dict[int, Path]:
    """Draws the doubles (magenta, solid) and singles (yellow, dashed-by-color-only
    since cv2.polylines has no native dash) court outlines, plus a labeled marker
    at each of the 4 doubles corners, onto `frame_indices` (default: start/25%
    is not used -- start, middle, end of the clip, matching the "at least 3
    frames spanning the clip" requirement) and saves them to `out_dir`. Returns
    {frame_index: saved_path} so a caller/test can confirm the files exist.

    This does NOT itself verify correctness -- a human must look at the saved
    images and confirm all 4 corners land on the real court lines at all 3
    frames, per data/tennis/2.mp4's BL bug (a 49px corner error was NOT visible
    in the aggregate held-out-landmark number, only in a rendered image)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if frame_indices is None:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_indices = sorted({0, total // 2, max(0, total - 1)})

    doubles = homography.court_polygon_pixels()
    singles = homography.singles_corners_pixels()
    singles_pts = np.array([singles[k] for k in CORNER_ORDER], dtype=np.int32)

    frames = _frames_at(video_path, frame_indices)
    saved: dict[int, Path] = {}
    for idx in frame_indices:
        frame = frames[idx].copy()
        cv2.polylines(frame, [doubles.reshape(-1, 1, 2)], True, DOUBLES_COLOR, 2)
        cv2.polylines(frame, [singles_pts.reshape(-1, 1, 2)], True, SINGLES_COLOR, 2)
        for label, pt in zip(CORNER_ORDER, doubles):
            pt = tuple(int(v) for v in pt)
            cv2.drawMarker(frame, pt, DOUBLES_COLOR, cv2.MARKER_CROSS, 24, 2)
            cv2.putText(frame, label, (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORNER_LABEL_COLOR, 2)
        path = out_dir / f"calib_check_frame{idx}.jpg"
        cv2.imwrite(str(path), frame)
        saved[idx] = path
    return saved


@dataclass
class CalibrationVerificationManifest:
    """Records a real human sign-off that render_verification_frames' output was
    inspected and every corner landed on the true court line, at every checked
    frame.

    `corners_confirmed[frame_index][corner_label]` must be a struct, not a bare
    bool: `{"confirmed": bool, "pixel": [x, y], "matches": "<what it was checked
    against>"}`. This is deliberately stricter than a checkbox -- added
    2026-07-19 after a spot-check found the free-text `confirmed_note` field
    could sound specific (dates, frame numbers, root-cause narrative) without
    ever actually naming a single corner's pixel coordinates, which meant
    nothing forced the per-corner claim itself to be backed by a real
    measurement. Writing plausible-looking numbers without re-measuring from
    the actual frame defeats the entire point of this file -- see
    reference_video2_calibration.py's BL history for what "looked reasonable
    but wasn't independently checked" costs.
    """

    clip: str
    calibration_module: str
    frame_indices: list[int]
    corners_confirmed: dict[str, dict[str, dict]]  # {str(frame_idx): {corner: {confirmed, pixel, matches}}}
    confirmed_note: str
    held_out_error_px: dict[str, float] = field(default_factory=dict)

    def is_complete(self) -> tuple[bool, str]:
        if len(self.frame_indices) < 3:
            return False, f"only {len(self.frame_indices)} frame(s) checked, need >=3"

        seen_pixels_by_corner: dict[str, list[tuple]] = {c: [] for c in CORNER_ORDER}
        note_lower = self.confirmed_note.lower()
        any_coord_in_note = False

        for idx in self.frame_indices:
            per_frame = self.corners_confirmed.get(str(idx))
            if per_frame is None:
                return False, f"frame {idx} has no corner confirmations recorded"
            for corner in CORNER_ORDER:
                entry = per_frame.get(corner)
                if not isinstance(entry, dict):
                    return False, (
                        f"frame {idx} corner {corner} must be a struct "
                        f"{{confirmed, pixel, matches}}, not a bare bool/missing value"
                    )
                if not entry.get("confirmed"):
                    return False, f"frame {idx} corner {corner} not confirmed"
                pixel = entry.get("pixel")
                if (
                    not isinstance(pixel, (list, tuple))
                    or len(pixel) != 2
                    or not all(isinstance(v, (int, float)) for v in pixel)
                ):
                    return False, (
                        f"frame {idx} corner {corner} is missing a real pixel "
                        f"coordinate pair -- record what was actually read off the frame"
                    )
                if not str(entry.get("matches", "")).strip():
                    return False, (
                        f"frame {idx} corner {corner} has no 'matches' description "
                        f"of what real court feature it was checked against"
                    )
                seen_pixels_by_corner[corner].append(tuple(pixel))
                px_str = str(int(round(pixel[0])))
                py_str = str(int(round(pixel[1])))
                if px_str in self.confirmed_note or py_str in self.confirmed_note:
                    any_coord_in_note = True

        for corner, pixels in seen_pixels_by_corner.items():
            if len(pixels) >= 2 and len(set(pixels)) == 1:
                return False, (
                    f"corner {corner} has the exact identical pixel coordinate "
                    f"{pixels[0]} recorded across all {len(pixels)} checked frames -- "
                    f"this looks copy-pasted rather than independently re-measured "
                    f"per frame. Even on a static camera, an independent re-read "
                    f"should show at least sub-pixel human-estimation variance; "
                    f"if the camera is genuinely locked-off and the reads are "
                    f"intentionally exact, say so explicitly in confirmed_note"
                )

        if not self.confirmed_note.strip():
            return False, "confirmed_note is empty -- record what was actually checked"
        if not any_coord_in_note:
            return False, (
                "confirmed_note must reference at least one specific pixel "
                "coordinate that was actually recorded above"
            )
        return True, "complete"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CalibrationVerificationManifest":
        return cls(**json.loads(path.read_text()))


def manifest_path_for(clip_stem: str) -> Path:
    return VERIFICATION_DIR / f"{clip_stem}.json"
