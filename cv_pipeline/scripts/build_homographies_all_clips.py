"""build_homographies_all_clips.py — Step 3, generalized: build a CourtHomography for
each of the 10 clips and run sanity/degeneracy checks on all of them (not just
video1, which verify_homography.py validated in detail with the real, independent
center-hash-mark check). This script is the broader, cheaper check: does each clip's
court-corner geometry look sane (near corners below far corners in pixel-y, roughly
trapezoidal, non-degenerate homography), so a single bad annotation row doesn't
silently break a later step across all 10 clips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import load_clip_annotations
from cv_pipeline.homography import CourtHomography, DOUBLES_WIDTH_M, COURT_LENGTH_M

CLIPS = [f"video{i}" for i in range(1, 11)]


def main():
    print(f"{'clip':<10} {'near_span_px':>13} {'far_span_px':>12} {'reproj_err_px':>14} {'status':>10}")
    for clip in CLIPS:
        annotations = load_clip_annotations(clip)
        first_court_ann = next((a for a in annotations.values() if a.court_corners), None)
        if first_court_ann is None:
            print(f"{clip:<10} NO COURT ROWS FOUND")
            continue
        corners = first_court_ann.court_corners
        homography = CourtHomography(corners)

        # Sanity 1: the near-baseline pixel span (left to right) should be WIDER than
        # the far-baseline span in any normal court-level camera shot (near objects
        # appear bigger) -- catches a geometrically-degenerate corner set (e.g. 3
        # points nearly collinear) that findHomography might still "succeed" on.
        points = list(corners.values())
        by_y = sorted(points, key=lambda p: p[1])
        far_span = abs(by_y[0][0] - by_y[1][0])
        near_span = abs(by_y[2][0] - by_y[3][0])

        # Sanity 2: reprojecting the 4 calibration corners pixel->world->pixel should
        # return ~0 error (catches a degenerate/singular homography).
        errs = []
        for px in points:
            wx, wy = homography.pixel_to_world(*px)
            back_px, back_py = homography.world_to_pixel(wx, wy)
            errs.append(np.hypot(back_px - px[0], back_py - px[1]))
        max_reproj_err = max(errs)

        ok = near_span > far_span and max_reproj_err < 1.0
        status = "OK" if ok else "CHECK"
        print(f"{clip:<10} {near_span:>13.1f} {far_span:>12.1f} {max_reproj_err:>14.4f} {status:>10}")


if __name__ == "__main__":
    main()
