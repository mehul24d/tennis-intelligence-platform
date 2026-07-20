"""hough_court_detection.py — EXPERIMENTAL automated court-corner detector
using classical Hough-transform line detection, tested 2026-07-19 as a
direct comparison against this project's existing numeric-traced (manually
measured) calibrations for data/tennis/1.mp4 and data/tennis/3.mp4.

NOT WIRED INTO video_pipeline.py OR ANY reference_videoN_calibration.py
MODULE. Deliberately named outside the `reference_video*_calibration`
pattern so `test_calibration_verification.py`'s mandatory-manifest gate
(added for exactly this class of module -- see that test's own docstring)
does not discover it: this has NOT gone through that verification process
(3+ frames, all 4 corners, human sign-off) and must not be treated as a
drop-in replacement for the existing manual calibrations until it has.

WHAT THIS DOES: given a single representative frame, isolates the court's
white line pixels (low-saturation, high-value HSV thresholding, measured
directly from real pixels -- line S~21/V~255 vs. court-surface S~106-146/
V~161-176), restricts the search to the court's own color-masked region
(excludes crowd/ad-boards/scoreboard from ever entering the Hough search),
runs `cv2.HoughLinesP` for candidate line segments, clusters segments that
lie along the same real line (see `_cluster_segments`), and picks the
near/far baseline and left/right doubles-sideline candidates using
REAL-WORLD court geometry rather than naive pixel heuristics -- see
`classify_and_fit`'s docstring for the two real failures this went through
and how each was fixed, found by testing against real ground truth, not
assumed correct on the first pass.

SINGLE-FRAME RESULTS (measured directly against this project's existing
manually-traced corner pixels, the actual ground truth used to build
`reference_video1_calibration.py`/`reference_video3_calibration.py` --
frame 0 of each clip, same frame the manual calibration used):

    video1: BL 7.5px, BR 32.7px, TR 5.5px, TL 1.6px -- mean 11.8px
    video3: BL 6.1px, BR 4.0px, TR 1.7px, TL 4.5px  -- mean 4.1px

video3's result is genuinely comparable to the manual method's own
held-out-landmark precision (1.68px/1.68px). video1's BR outlier was
inspected visually: a line judge/ball-person is crouched close to that
exact corner in this specific frame, plausibly interfering with the line
mask there -- a single-frame occlusion artifact.

MULTI-FRAME, MULTI-CLIP FOLLOW-UP (all 5 reference clips, 8 evenly-spread
frames per camera-stable segment -- pans/ramps excluded, same windows the
manual calibrations themselves are scoped to -- averaged via
`detect_court_corners_multi_frame`; see PROGRESS.md's "Multi-Frame,
Multi-Clip Hough Evaluation" entry for the full per-frame breakdown):

    clip 1: BL 17.3px BR  3.9px TR 3.8px TL 0.3px -- mean 6.3px
    clip 2: BL 14.0px BR 14.2px TR 4.1px TL 5.0px -- mean 9.3px
    clip 3: BL  8.4px BR  6.4px TR 2.3px TL 4.6px -- mean 5.4px
    clip 4: BL  3.2px BR 12.3px TR 3.4px TL 3.4px -- mean 5.5px
    clip 5: BL  7.4px BR 16.9px TR 2.8px TL 2.4px -- mean 7.4px
    overall mean across all 5 clips: 6.8px

Confirms the video1 BR occlusion hypothesis: averaging across 8 frames
drops that corner's error from 32.7px (frame 0 alone) to 3.9px. But this
is NOT a uniform win -- video3 (the other single-frame-tested clip) got
SLIGHTLY WORSE (4.1px -> 5.4px), pulled up by one bad frame (BL 53.5px at
frame 666) that simple unweighted averaging has no defense against. A
consistent pattern emerged across all 5 clips that single-frame testing
on 2 clips could not have shown: the two NEAR corners (BL, BR) are
volatile (0.3-17.3px) while the two FAR corners (TR, TL) are reliably
good in every clip (0.3-5.0px, no exceptions) -- NOT a left/right-specific
weakness (recomputed per clip: BL is worst in 2 of 5 clips, BR in the
other 3, a near-tie in one of those). Root-caused via visual + programmatic
tracing on 2 bad frames (see PROGRESS.md's follow-up entry for the full
diagnosis): `classify_and_fit` picked near/far-baseline by raw fitted-line
y-position alone, with no weighting for how much of the baseline's real
pixel width a candidate cluster actually covers -- a short, poorly-
supported cluster (anchored near just one corner, sometimes because a
broadcast scoreboard graphic's white text passed the line-color mask and
polluted it, sometimes just because Hough didn't find enough segments
across the full baseline that frame) could out-rank a long, well-supported
cluster by a razor-thin y margin, then got extrapolated +-3000px past
where it had any real support, producing large error specifically at
whichever corner was farthest from that cluster's actual data.

THREE FIXES APPLIED FOR THIS, IN SEQUENCE, EACH RE-MEASURED (see
PROGRESS.md's "Coverage-Weighted Cluster Selection + Scoreboard
Exclusion", "Segment-Filtering Instead Of Pixel-Masking", and "Median
Aggregation" entries for the full before/after of each):

1. `_best_covered_extremum` (used by `classify_and_fit` for near/far
   baseline selection) restricts the y-position comparison to a tolerance
   band around the true extremum, then breaks ties toward the
   better-covered (larger x_span) cluster within that band -- directly
   fixes the short-cluster-wins-by-a-few-px failure mode above.
2. Scoreboard handling, attempted twice: a first version masked the fixed
   on-screen "ALCARAZ .../SINNER ..." graphic region out of `line_mask`
   (a pixel mask) BEFORE `cv2.HoughLinesP` ran. This worked for the
   originally-diagnosed contamination case but was found, via the 5-clip
   re-evaluation below, to occasionally destabilize a real line detection
   FAR from the masked region (5.mp4 frame 0 lost its only far-baseline
   cluster, y~273, despite the mask sitting at y=890-1030) --
   `cv2.HoughLinesP`'s randomized-order internal voting is sensitive to
   the total edge-point population, not just local pixels, so removing
   points anywhere can change results elsewhere. Replacing the fill VALUE
   (tried a neutral court-color fill instead of zeroing) made no
   difference at all -- confirmed on the exact failing frame, byte-for-
   byte identical segments either way -- because the mechanism was never
   about the fill's appearance. Fixed properly by moving the exclusion to
   AFTER detection instead: `SCOREBOARD_EXCLUSION_REGION` is now used by
   `_segment_in_excluded_region` to drop only the individual Hough
   segments that overlap the scoreboard's box, post-hoc, in
   `detect_court_corners` -- `cv2.HoughLinesP` always sees the same full,
   unmasked edge population regardless of whether the scoreboard is
   present, so it cannot be perturbed elsewhere, while the scoreboard's
   own spurious segments are still discarded.
3. `detect_court_corners_multi_frame` aggregates per-corner detections
   across frames via MEDIAN rather than mean. This is a general
   robustness layer against ordinary per-frame Hough noise -- unrelated
   to the scoreboard mechanism above -- observed directly: several
   clip/corner combinations had one or two sampled frames with much
   higher error than the rest (e.g. 3.mp4 BL: seven frames under 8px, one
   at 53.5px) while others were uniformly elevated across all 8 sampled
   frames (a genuine systematic bias, e.g. 1.mp4's BL, or a real,
   documented occlusion spanning the whole window, e.g. 5.mp4 segment-a's
   BR). Median fixes the former (a minority of frames wrong) and, as
   expected, does nothing for the latter (median of consistently-bad
   values is itself still bad) -- both fix types were needed, neither
   alone would have been sufficient.

RESULT, same 5-clip, 8-frames-per-segment multi-frame evaluation, at each
stage:

    clip   unfixed   +coverage+pixel-mask   +segment-filter   +median (final)
    1.mp4   6.3px          4.3px                4.3px             4.5px
    2.mp4   9.3px          6.4px                6.8px             6.4px
    3.mp4   5.4px          4.1px                4.2px             3.9px
    4.mp4   5.5px          4.0px                4.2px             4.0px
    5.mp4   7.4px          9.4px (REGRESSED)    7.3px             6.8px
    overall 6.8px          5.65px               5.36px            5.14px

The pixel-masking version's 5.mp4 regression is fully resolved by
segment-filtering (7.4px -> 7.3px, back in line with the original), and
every clip is at or below its original unfixed baseline in the final
(segment-filter + median) configuration -- a clean improvement across all
5 clips, not just 4 of 5. Near-corner (BL/BR) error is now much closer to
far-corner (TR/TL) error in most clips, though not fully closed
everywhere: 2.mp4's BL (~12px) and 5.mp4's BR (~13px, the documented
player-occlusion window) remain clearly elevated -- both understood,
neither an unexplained mystery.

WHAT THIS DOES NOT YET PROVE, stated plainly, not silently generalized:
median aggregation is still not a full outlier-REJECTION scheme (no
explicit MAD/IQR-based filtering was implemented, just the switch from
mean to median) -- it happens to be sufficient for every case observed
here (never more than 2 of 8 sampled frames badly wrong for any
corner/segment) but has no guarantee against a worse ratio. No
held-out-landmark cross-check (near-T/net-base) was attempted -- only the
4 doubles corners themselves were compared. The scoreboard segment-filter
was verified to resolve the one specific instability found (5.mp4 frame
0) but was not exhaustively searched for other, unfound instances of the
same non-local HoughLinesP sensitivity under the OLD pixel-masking
approach -- moot now since that approach was replaced, but worth noting
the underlying algorithmic sensitivity itself hasn't been fully
characterized. This has NOT been run through
`test_calibration_verification.py`'s mandatory-manifest gate and must not
be treated as a drop-in replacement for the existing manual calibrations
-- treat it as a candidate to keep comparing against the manual method,
not an automatic replacement, until proven at least as reliable across
all 5 clips over a larger sample than 8 frames each. See PROGRESS.md for
the full investigation, including the original single-frame failures and
all three follow-up fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedCourtCorners:
    bl: tuple[float, float] | None
    br: tuple[float, float] | None
    tr: tuple[float, float] | None
    tl: tuple[float, float] | None
    n_segments: int


def _line_intersection(l1, l2) -> tuple[float, float] | None:
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


# Fixed on-screen scoreboard graphic region ("ALCARAZ ... / SINNER ..."),
# in (x0, y0, x1, y1) pixel coordinates. This is a broadcast overlay
# anchored to the OUTPUT SCREEN, not to the court -- measured directly by
# thresholding for its near-black background box across all 5 reference
# clips, 2 frames each: y-range was pixel-identical (914-1007) in every
# case, x0 was identical (180-181), x1 varied (454-563px) with the score's
# digit count at different points in the match. Region below uses a
# generous margin around the full measured extent (see PROGRESS.md's
# "Coverage-Weighted Cluster Selection + Scoreboard Exclusion" and
# "Segment-Filtering Instead Of Pixel-Masking" entries) -- its white/
# yellow text passes the exact same high-value/low-saturation HSV
# threshold used for court lines below, and was confirmed (1.mp4 frame
# 288) to corrupt the near-baseline line cluster when included.
#
# NOTE: this is applied as a POST-HOC FILTER on detected Hough segments
# (see `detect_court_corners`), not as a pixel/edge-map mask before
# `cv2.HoughLinesP` runs. An earlier version zeroed this region out of
# `line_mask` before detection; that was measured to occasionally make
# `cv2.HoughLinesP` (a probabilistic, randomized-order transform) lose an
# unrelated, real line segment elsewhere in the frame -- removing edge
# pixels anywhere changes the total population the algorithm samples from,
# not just locally. Filtering by segment overlap AFTER detection keeps
# Hough's input identical regardless of the scoreboard's presence, so it
# cannot perturb detection elsewhere, while still discarding the
# scoreboard's own spurious segments.
SCOREBOARD_EXCLUSION_REGION = (150, 890, 600, 1030)  # x0, y0, x1, y1


def _segment_in_excluded_region(seg, region: tuple[float, float, float, float]) -> bool:
    """True if `seg`'s (x1,y1,x2,y2) bounding box overlaps `region`
    (x0,y0,x1,y1) at all -- a simple axis-aligned-box overlap test, not
    exact segment/rectangle intersection, since any segment even partially
    inside the scoreboard region is suspect and should be dropped."""
    x1, y1, x2, y2 = seg
    rx0, ry0, rx1, ry1 = region
    seg_x0, seg_x1 = min(x1, x2), max(x1, x2)
    seg_y0, seg_y1 = min(y1, y2), max(y1, y2)
    return seg_x0 < rx1 and seg_x1 > rx0 and seg_y0 < ry1 and seg_y1 > ry0


def _detect_line_mask(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (line_mask, edges). Restricts the search to the court's own
    color-masked region first (blue/cyan hue range, measured directly from
    real pixels) so ad-board/crowd/scoreboard edges elsewhere in the frame
    never enter the Hough search -- without this, those regions' own strong,
    unrelated edges dominate the candidate-segment list. Does NOT mask out
    the on-screen scoreboard graphic here -- see SCOREBOARD_EXCLUSION_REGION's
    comment for why that is applied as a post-hoc segment filter in
    `detect_court_corners` instead of a pixel mask in this function."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    court_mask = cv2.inRange(hsv, (80, 60, 100), (130, 200, 220))
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    region_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(region_mask, [largest], -1, 255, thickness=-1)
    region_mask = cv2.erode(region_mask, np.ones((5, 5), np.uint8))  # avoid the mask's own hard boundary edge

    line_mask = cv2.inRange(hsv, (0, 0, 190), (180, 70, 255))
    line_mask = cv2.bitwise_and(line_mask, region_mask)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    edges = cv2.Canny(line_mask, 50, 150)
    return line_mask, edges


def _line_params(x1, y1, x2, y2) -> tuple[float, float]:
    """(theta_deg in [0,180), rho) in normal form x*cos+y*sin=rho."""
    dx, dy = x2 - x1, y2 - y1
    theta = np.degrees(np.arctan2(dy, dx)) % 180
    normal = np.radians(theta) + np.pi / 2
    rho = x1 * np.cos(normal) + y1 * np.sin(normal)
    return theta, rho


def _cluster_segments(segments, min_length=40.0, theta_tol=4.0, rho_tol=18.0) -> list[dict]:
    """Greedy single-link clustering on (theta, rho) -- groups multiple short
    Hough segments that lie along the SAME real court line (broken up by
    gaps, players crossing the line, etc.) before any fitting happens.
    Without this, unrelated lines that happen to share a rough angle bucket
    (e.g. a baseline and a service line, both "near-horizontal") get
    averaged together into one bad fit -- the first version of this detector
    did exactly that and produced 70-160px mean corner error; see
    PROGRESS.md for the visual debugging that found it."""
    items = []
    for x1, y1, x2, y2 in segments:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length:
            continue
        theta, rho = _line_params(x1, y1, x2, y2)
        items.append({"seg": (x1, y1, x2, y2), "length": length, "theta": theta, "rho": rho})

    clusters: list[dict] = []
    for it in sorted(items, key=lambda d: -d["length"]):
        placed = False
        for c in clusters:
            dtheta = min(abs(it["theta"] - c["theta"]), 180 - abs(it["theta"] - c["theta"]))
            if dtheta < theta_tol and abs(it["rho"] - c["rho"]) < rho_tol:
                c["members"].append(it)
                total = c["total_length"] + it["length"]
                c["theta"] = (c["theta"] * c["total_length"] + it["theta"] * it["length"]) / total
                c["rho"] = (c["rho"] * c["total_length"] + it["rho"] * it["length"]) / total
                c["total_length"] = total
                placed = True
                break
        if not placed:
            clusters.append({"members": [it], "theta": it["theta"], "rho": it["rho"], "total_length": it["length"]})
    return clusters


def _fit_cluster_line(cluster: dict):
    pts = np.array(
        [[x, y] for m in cluster["members"] for x1, y1, x2, y2 in [m["seg"]] for x, y in [(x1, y1), (x2, y2)]],
        dtype=np.float32,
    )
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    x_span = float(pts[:, 0].max() - pts[:, 0].min())
    y_span = float(pts[:, 1].max() - pts[:, 1].min())
    long_line = (float(x0 - vx * 3000), float(y0 - vy * 3000), float(x0 + vx * 3000), float(y0 + vy * 3000))
    return long_line, x_span, y_span


def _best_covered_extremum(candidates: list[dict], key, want_max: bool, tol: float = 30.0) -> dict | None:
    """Among `candidates`, finds the extremal `key(candidate)` value (max if
    want_max else min), then -- among only those candidates within `tol` of
    that extremum -- returns whichever has the largest x_span (best real
    coverage of the line it claims to represent).

    Added to fix a real, diagnosed failure: `classify_and_fit` used to pick
    near/far-baseline by raw extremal Y-position alone (`max`/`min` on
    `mid_y`), with no regard for how much of the baseline's actual pixel
    width a candidate cluster covers. On 4.mp4 frame 60, a short,
    3-segment cluster covering only x=181-541 (nowhere near the BR corner)
    beat a well-supported, 8-segment cluster covering x=496-1562 (nearly
    the full baseline) by a 2.1px mid_y margin -- and `_fit_cluster_line`'s
    +-3000px extrapolation then turned that short cluster's local fit into
    a 35.5px error at the far corner it was never actually measured near.
    Restricting the y-position comparison to a `tol`-px band around the
    true extremum, then breaking ties toward the best-covered cluster
    within that band, prevents a short cluster from winning outright while
    still tolerating normal small y-differences between real candidates.
    See PROGRESS.md's "Coverage-Weighted Cluster Selection + Scoreboard
    Exclusion" entry for the measured before/after effect. `tol=30.0` is
    generous enough to keep genuinely competing near-baseline clusters in
    play (the winning true-baseline cluster above was only 2.1px from the
    short cluster) while still being far short of the ~80-500px gap to an
    unrelated line (service line, net cord) in every case checked."""
    if not candidates:
        return None
    extremum = max(key(c) for c in candidates) if want_max else min(key(c) for c in candidates)
    band = [c for c in candidates if abs(key(c) - extremum) <= tol]
    return max(band, key=lambda c: c["x_span"])


def classify_and_fit(segments, w: int, h: int) -> dict:
    """Clusters raw Hough segments into real court lines, then picks the
    near/far baseline and left/right doubles-sideline candidates using
    REAL-WORLD geometry, not naive pixel heuristics -- three real failures
    were found and fixed here by testing against real ground truth:

    1. Baseline selection is NOT "the 2 widest horizontal clusters by pixel
       x-span", even though the baseline IS the real-world-widest horizontal
       line (it spans the full doubles width; service lines span only the
       narrower singles width). The net cord is ALSO a wide, bright,
       near-horizontal line spanning close to the full doubles width, and in
       practice out-competed the true far baseline on pixel x-span in
       testing (perspective foreshortening makes distant lines narrower in
       pixels even though they're wider in real-world meters) -- the first
       version of this function picked the net cord as "far_baseline".
       Fixed: the net sits at mid-court height, strictly between the near
       and far baselines in image-y, so picking the topmost and bottommost
       horizontal clusters by Y-POSITION instead correctly separates
       baseline from net regardless of which one is wider in pixels in any
       given frame.
    2. The doubles sideline is the MOST EXTREME (furthest from center) tall
       vertical-ish line on each side (the singles sideline runs parallel
       but closer to center) -- among vertical-ish clusters on a given side,
       the one furthest from center wins, filtered to require a real y-span
       (rules out short unrelated diagonal clutter). This one worked
       correctly from the first pass (see PROGRESS.md's overlay debugging).
    3. Y-POSITION ALONE is also not enough, on its own, to pick near/far
       baseline -- see `_best_covered_extremum`'s docstring for the
       diagnosed failure (a short, poorly-supported cluster beating a
       well-supported one by a couple of pixels, then getting extrapolated
       far past where it has any real support). Fixed by restricting the
       Y-position comparison to a tolerance band around the true extremum
       and breaking ties toward the better-covered cluster within it."""
    clusters = _cluster_segments(segments)
    fitted = []
    for c in clusters:
        line, x_span, y_span = _fit_cluster_line(c)
        theta = c["theta"] if c["theta"] <= 90 else 180 - c["theta"]
        fitted.append({"line": line, "x_span": x_span, "y_span": y_span, "theta": theta})

    horiz = [f for f in fitted if f["theta"] < 25 and f["x_span"] > 250]
    vert = [f for f in fitted if f["theta"] > 40 and f["y_span"] > 250]

    def mid_y(line):
        return (line[1] + line[3]) / 2

    def mid_x(line):
        return (line[0] + line[2]) / 2

    result: dict = {}
    if horiz:
        near = _best_covered_extremum(horiz, lambda f: mid_y(f["line"]), want_max=True)
        far = _best_covered_extremum(horiz, lambda f: mid_y(f["line"]), want_max=False)
        if near is not None:
            result["near_baseline"] = near["line"]
        if far is not None:
            result["far_baseline"] = far["line"]
    left_candidates = [f for f in vert if mid_x(f["line"]) < w / 2]
    right_candidates = [f for f in vert if mid_x(f["line"]) >= w / 2]
    if left_candidates:
        result["left_sideline"] = min(left_candidates, key=lambda f: mid_x(f["line"]))["line"]
    if right_candidates:
        result["right_sideline"] = max(right_candidates, key=lambda f: mid_x(f["line"]))["line"]
    return result


def detect_court_corners(img: np.ndarray) -> DetectedCourtCorners:
    """Top-level entry point -- see module docstring for the full method and
    its measured accuracy against this project's existing ground truth."""
    h, w = img.shape[:2]
    line_mask, edges = _detect_line_mask(img)
    raw_segments = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60, minLineLength=80, maxLineGap=15)
    all_segments = [] if raw_segments is None else [s[0] for s in raw_segments]
    segments = [s for s in all_segments if not _segment_in_excluded_region(s, SCOREBOARD_EXCLUSION_REGION)]
    lines = classify_and_fit(segments, w, h)

    corners: dict[str, tuple[float, float] | None] = {"bl": None, "br": None, "tr": None, "tl": None}
    if lines.get("left_sideline") and lines.get("near_baseline"):
        corners["bl"] = _line_intersection(lines["left_sideline"], lines["near_baseline"])
    if lines.get("right_sideline") and lines.get("near_baseline"):
        corners["br"] = _line_intersection(lines["right_sideline"], lines["near_baseline"])
    if lines.get("right_sideline") and lines.get("far_baseline"):
        corners["tr"] = _line_intersection(lines["right_sideline"], lines["far_baseline"])
    if lines.get("left_sideline") and lines.get("far_baseline"):
        corners["tl"] = _line_intersection(lines["left_sideline"], lines["far_baseline"])

    return DetectedCourtCorners(bl=corners["bl"], br=corners["br"], tr=corners["tr"], tl=corners["tl"],
                                 n_segments=len(segments))


def detect_court_corners_multi_frame(frames: list[np.ndarray]) -> DetectedCourtCorners:
    """Runs detect_court_corners independently on each frame, then takes the
    per-axis MEDIAN of the per-corner detections across frames that found
    that corner. Added to smooth out single-frame occlusion/noise artifacts
    (e.g. video1's BR-corner line-judge occlusion, see module docstring)
    without needing to hand-pick a clean frame -- see PROGRESS.md's
    multi-frame/multi-clip evaluation entries for the measured effect of
    this vs. single-frame detection.

    Uses median rather than mean specifically because ordinary per-frame
    Hough noise (unrelated to the scoreboard-masking mechanism fixed
    elsewhere in this module) was observed to occasionally produce a
    single bad frame -- or, in the 8-frame samples measured, sometimes two
    -- with much higher error than the rest of that segment's frames (e.g.
    3.mp4 frame 666's BL corner at 53.5px against seven other frames all
    under 8px). A plain mean lets one such frame drag the whole average;
    a median (with fewer than half the sampled frames affected, as in
    every case observed) ignores it outright. This is a general
    robustness layer against ordinary detection noise, independent of
    (and stacks with) the scoreboard-specific segment-filtering fix in
    `detect_court_corners`/`SCOREBOARD_EXCLUSION_REGION` -- it does
    nothing to fix a SYSTEMATIC bias present across most/all sampled
    frames (e.g. 5.mp4 segment-a's BR corner, elevated in all 8 frames
    from a real, documented player occlusion spanning that whole window),
    since a median of consistently-bad values is itself still bad; it
    only protects against a minority of frames being wrong.

    A corner is None only if it was not detected in ANY of the input
    frames."""
    all_results = [detect_court_corners(f) for f in frames]
    corners: dict[str, tuple[float, float] | None] = {}
    for key in ("bl", "br", "tr", "tl"):
        pts = [getattr(r, key) for r in all_results if getattr(r, key) is not None]
        corners[key] = (
            (float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts])))
            if pts
            else None
        )
    return DetectedCourtCorners(
        bl=corners["bl"],
        br=corners["br"],
        tr=corners["tr"],
        tl=corners["tl"],
        n_segments=sum(r.n_segments for r in all_results),
    )
