"""verify_homography.py — Step 3 of Phase 3: build the court homography from video1's
annotated corners, then validate it with an INDEPENDENT check (not circular): predict
where the singles sidelines should appear in pixel space (a court marking that was
NOT used to calibrate the homography, since only the 4 doubles corners were used), and
compare against the singles line's actual pixel position, found by scanning for the
bright/white line in the real video frame near the baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import DEFAULT_VIDEOS_DIR, load_clip_annotations
from cv_pipeline.homography import CourtHomography, DOUBLES_WIDTH_M

OUT_DIR = Path(__file__).resolve().parents[1] / "scratch_output" / "homography_check"
CLIP = "video1"
CHECK_FRAME = 400


def find_bright_line_xs(frame, y: int, x_lo: int, x_hi: int, expect_two: bool = True):
    """Scans row `y` between x_lo/x_hi for bright (white-line) pixels, returns the
    x-positions of local intensity peaks -- a simple, inspectable line-finder, not a
    trained detector. Uses a row-band average (y-2..y+2) to reduce single-row noise."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    band = gray[max(0, y - 2):y + 3, x_lo:x_hi].astype(np.float32).mean(axis=0)
    threshold = band.mean() + 1.5 * band.std()
    above = band > threshold
    # group consecutive above-threshold pixels into peaks, take each group's center
    peaks = []
    start = None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        if not v and start is not None:
            peaks.append(x_lo + (start + i - 1) / 2)
            start = None
    if start is not None:
        peaks.append(x_lo + (start + len(above) - 1) / 2)
    return peaks, band, threshold


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotations = load_clip_annotations(CLIP)
    ann = annotations[CHECK_FRAME]
    homography = CourtHomography(ann.court_corners)

    print(f"Court corners (pixel): {ann.court_corners}")

    # Sanity check 1: re-projecting the 4 calibration corners should return ~0 error
    # (trivial by construction, but confirms the homography isn't degenerate/flipped).
    for name, px in ann.court_corners.items():
        wx, wy = homography.pixel_to_world(*px)
        print(f"  {name}: pixel={px} -> world=({wx:.2f}m, {wy:.2f}m)")

    # INDEPENDENT check: predict singles sideline pixel-x near the baseline (world_y
    # close to 0, but not exactly 0 -- pick world_y=1.0m so the predicted line is a
    # few pixels off the baseline itself, avoiding the baseline's own bright line
    # confusing the scan).
    left_x, right_x = homography.singles_sideline_pixel_xs_at_y(1.0)
    print(f"\nPredicted singles sideline pixel-x at world_y=1.0m: left={left_x:.1f}, right={right_x:.1f}")

    cap = cv2.VideoCapture(str(DEFAULT_VIDEOS_DIR / f"{CLIP}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, CHECK_FRAME)
    ok, frame = cap.read()
    assert ok

    # Scan pixel row near the baseline (baseline itself is ~y=870; world_y=1.0m maps
    # to a pixel-y slightly above that -- compute it from the homography directly).
    scan_y = int(round(homography.world_to_pixel(DOUBLES_WIDTH_M / 2, 1.0)[1]))
    bl_x = int(ann.court_corners["BL"][0])
    br_x = int(ann.court_corners["BR"][0])
    peaks, band, threshold = find_bright_line_xs(frame, scan_y, bl_x, br_x)
    print(f"Scanning pixel row y={scan_y} (between x={bl_x} and x={br_x}) for bright lines...")
    print(f"  found {len(peaks)} bright-line peak(s) at x={[f'{p:.1f}' for p in peaks]}")

    if len(peaks) >= 2:
        # Compare the two peaks closest to the predicted singles sideline positions.
        left_actual = min(peaks, key=lambda p: abs(p - left_x))
        right_actual = min(peaks, key=lambda p: abs(p - right_x))
        print(f"\n  LEFT  singles line: predicted={left_x:.1f}px, closest actual peak={left_actual:.1f}px, "
              f"error={abs(left_x - left_actual):.1f}px")
        print(f"  RIGHT singles line: predicted={right_x:.1f}px, closest actual peak={right_actual:.1f}px, "
              f"error={abs(right_x - right_actual):.1f}px")
    else:
        print("  Could not isolate two clear line peaks -- see saved debug image/plot.")

    # Save an annotated frame: predicted singles lines (yellow) vs all detected bright
    # peaks (red ticks) on the scan row, plus the homography-projected net polyline.
    debug = frame.copy()
    cv2.line(debug, (bl_x, scan_y), (br_x, scan_y), (255, 255, 255), 1)
    cv2.circle(debug, (int(left_x), scan_y), 8, (0, 255, 255), 3)
    cv2.circle(debug, (int(right_x), scan_y), 8, (0, 255, 255), 3)
    cv2.putText(debug, "yellow=predicted singles line", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for p in peaks:
        cv2.drawMarker(debug, (int(p), scan_y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.putText(debug, "red X=detected bright-line peak", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    net_poly = homography.net_pixel_polyline()
    for i in range(len(net_poly) - 1):
        p1 = tuple(int(c) for c in net_poly[i])
        p2 = tuple(int(c) for c in net_poly[i + 1])
        cv2.line(debug, p1, p2, (0, 255, 0), 2)
    cv2.putText(debug, "green=homography-projected net line", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = OUT_DIR / f"{CLIP}_frame{CHECK_FRAME:03d}_homography_check.png"
    cv2.imwrite(str(out_path), debug)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
