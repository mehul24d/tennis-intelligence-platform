"""ball_detection_combined.py — the validated "improved" ball-detection method:
a fine-tuned YOLOv8n (30 epochs on Viren Dhanwani's CC-BY-4.0 Roboflow
tennis-ball dataset) with frequency-based static-artifact rejection, combined
(OR) with court-region-masked motion-diff on frames with no surviving
detection.

VALIDATED, not assumed: pooled ground-truth recall across the 9-clip amateur
dataset (2,074 ground-truth ball frames) went 7.81% (stock COCO YOLO,
ball_detection.py) -> 57.62% (motion-diff alone) -> **53.91% (this combined
method, through the real production code path)** -- see PROGRESS.md for the
full investigation, including two prior failed static-artifact-filter designs
before this one, AND a ground-truth leak found in the original prototyping
that had produced an invalid 70.40% figure (the prototype used ground truth
to pick which motion-diff candidate to trust when several existed in a frame
-- not something a real inference-time system can do; fixed by picking the
largest-area candidate instead, a legitimate non-cheating heuristic). 53.91%
is the corrected, final, honest number -- still a real ~6.9x improvement over
stock YOLO, just not as large as first claimed.

REGIME-DEPENDENT, not universally adopted: this method's real recall was only
validated on locked-camera, single-continuous-shot amateur footage. On the two
broadcast/professional stress-test clips (multi-camera-angle highlight reels),
visual spot-check found motion-diff produces false positives on player-limb
motion in frames with two players in view, and no ground truth exists there to
measure real recall at all. `classify_ball_detection_regime` below makes that
distinction explicit and cheap, reusing the same histogram-correlation hard-cut
detector already built and validated for Stress Test #2's camera-angle filter
-- high cut-rate footage (broadcast-style) is NOT eligible for the "improved
method" claim and falls back to stock YOLO, reported as best-effort.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .ball_detection import MAX_BALL_MATCH_DISTANCE_PX, SPORTS_BALL_CLASS_ID, box_center
from .homography import CourtHomography

# The fine-tuned checkpoint itself lives under a scratch/training-run output
# directory (ultralytics' own `runs/detect/...` convention from the training
# invocation), not a proper packaged models/ directory -- a known, small
# deployment-hygiene gap (moving/copying it somewhere more permanent is a
# one-line change whenever this gets tidied up), called out here rather than
# silently referenced as if it were a normal package resource.
FINE_TUNED_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "runs" / "detect" / "cv_pipeline"
    / "scratch_output" / "ball_finetune" / "full_30ep" / "weights" / "best.pt"
)

ARTIFACT_BIN_PX = 10.0
ARTIFACT_FREQ_THRESHOLD = 0.03  # a pixel bin recurring in >=3% of all frames is flagged
ARTIFACT_REJECT_RADIUS_PX = 15.0
CONF_THRESHOLD = 0.25

# Regime classification thresholds -- reuses the same hard-cut detector
# (grayscale histogram correlation between consecutive frames) built for Stress
# Test #2's camera-angle filter (cv_pipeline/scripts/stress_test_2_sample_timing.py),
# not reimplemented differently here.
CUT_CORREL_THRESHOLD = 0.7
CUT_SAMPLE_STRIDE = 5
HIGH_CUT_RATE_FRACTION = 0.02  # >=1 cut per ~50 sampled frames -> broadcast-style


def _bin_key(x: float, y: float) -> tuple[int, int]:
    return (round(x / ARTIFACT_BIN_PX), round(y / ARTIFACT_BIN_PX))


def _is_near_flagged_bin(x: float, y: float, flagged_bins: dict) -> bool:
    for (bx, by) in flagged_bins:
        px, py = bx * ARTIFACT_BIN_PX, by * ARTIFACT_BIN_PX
        if np.hypot(x - px, y - py) <= ARTIFACT_REJECT_RADIUS_PX:
            return True
    return False


def find_artifact_bins(model, video_path: Path, start_frame: int = 0, n_frames: int | None = None) -> dict:
    """Pass 1 of the two-pass design: collects every candidate box center (not
    just the per-frame top pick -- that was the first design's mistake) across
    the clip and flags pixel bins that recur suspiciously often. A real ball's
    trajectory varies shot-to-shot; it does not return to the same few pixels
    across a large fraction of an entire clip."""
    cap = cv2.VideoCapture(str(video_path))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_frames = n_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bin_counts: Counter = Counter()
    n_processed = 0
    for _ in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        n_processed += 1
        results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        seen_bins_this_frame = {_bin_key((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes}
        for key in seen_bins_this_frame:
            bin_counts[key] += 1

    if n_processed == 0:
        return {}
    return {key: count / n_processed for key, count in bin_counts.items() if count / n_processed >= ARTIFACT_FREQ_THRESHOLD}


def court_mask(shape, homography: CourtHomography) -> np.ndarray:
    poly = homography.court_polygon_pixels(dilate=1.3)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def motion_diff_candidates(prev_gray, cur_gray, mask, min_area: float = 4, max_area: float = 400, diff_threshold: int = 25):
    """Returns (x, y, area) tuples -- area is included so callers can rank
    candidates by something other than cv2.findContours' arbitrary scan order
    (see the GROUND-TRUTH LEAK note in run_combined_ball_detection_for_clip's
    docstring for why "just pick candidates[0]" was wrong)."""
    diff = cv2.absdiff(prev_gray, cur_gray)
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            (x, y), _ = cv2.minEnclosingCircle(c)
            out.append((x, y, area))
    return out


FRAME_REFERENCE_MATCH_THRESHOLD = 0.5  # NOT the 0.7 cut-detection bar -- see
# frame_matches_reference_framing's docstring. Empirically set: video4.mp4's
# worst-case gradual lighting/exposure drift across its ENTIRE length (full
# scan, every 5th frame) never dropped below 0.6196 correlation to its own
# calibration frame; match_tennis.mp4's real hard cut drops to 0.09-0.2
# immediately and climbs back through 0.5 only partway into its gradual
# zoom-back-out (by frame ~7622, still visually a closeup, not yet the full
# wide shot). 0.5 sits with real margin below video4's true minimum while
# still unambiguously catching match_tennis's cut region -- not a threshold
# picked in isolation, but checked against both known real-clip distributions.


def _frame_histogram(frame) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def frame_matches_reference_framing(reference_hist: np.ndarray, frame, threshold: float = FRAME_REFERENCE_MATCH_THRESHOLD) -> tuple[bool, float]:
    """Per-FRAME gate, not the per-clip-window gate `classify_ball_detection_regime`
    provides. Compares the CURRENT frame's grayscale histogram directly
    against the histogram of the frame the homography corners were actually
    calibrated on (`reference_hist`) -- a direct per-frame check, not a
    stateful cut-tracker (a stateful "only re-check at detected cuts" design
    was tried and reverted the same day, 2026-07-16: it correctly stopped
    video4.mp4's false positives, but then incorrectly stayed flagged
    "inapplicable" through match_tennis.mp4's frames 7675/7750, which are
    visually normal wide-shot frames -- the real transition back to the wide
    shot there is a GRADUAL zoom, not a hard cut, so it never tripped the
    consecutive-frame cut detector and the frozen post-cut value never got
    re-evaluated. A live per-frame check naturally recovers as soon as
    correlation climbs back above threshold, which is what actually matches
    reality in both clips.)

    TWO REAL BUGS FOUND for this threshold specifically, both from checking
    against actual clips rather than assuming a value:
    1. Using the 0.7 cut-detection threshold here (this function used to
       share that constant) produced 183 false positives on video4.mp4 --
       gradual lighting drift within an unchanged, locked-off camera crossed
       0.7 even though consecutive-frame correlation stayed ~1.0000
       throughout (confirmed no actual cut ever occurs in that clip).
    2. FRAME_REFERENCE_MATCH_THRESHOLD is now 0.5, checked against a full
       scan of video4.mp4 (true minimum correlation 0.6196, comfortable
       margin above 0.5) and against match_tennis.mp4's real cut region
       (0.09-0.4 immediately post-cut, clearly below 0.5)."""
    hist = _frame_histogram(frame)
    corr = cv2.compareHist(reference_hist, hist, cv2.HISTCMP_CORREL)
    return corr >= threshold, corr


@dataclass(frozen=True)
class CombinedBallDetectionResult:
    frame_index: int
    center: tuple[float, float] | None
    source: Literal["fine_tuned_yolo", "motion_diff", "none"]
    homography_applicable: bool  # False if this specific frame's camera framing
    # doesn't match the one the homography was calibrated against (see
    # frame_matches_reference_framing) -- a consumer (e.g. the dashboard
    # overlay) should suppress court-line rendering for this frame when False,
    # rather than inheriting the clip-level regime label uncritically.
    reference_match_correlation: float


def run_combined_ball_detection_for_clip(
    fine_tuned_model, video_path: Path, homography: CourtHomography,
    start_frame: int = 0, n_frames: int | None = None,
    use_motion_diff_fallback: bool = False,
) -> list[CombinedBallDetectionResult]:
    """The validated combined method: pass-1 artifact-bin flagging, then pass-2
    detection with those bins' candidate boxes discarded, falling back (if
    `use_motion_diff_fallback`) to court-region-masked motion-diff on frames
    with no surviving detection. Requires a real, calibrated CourtHomography
    (used both for the motion-diff mask and implicitly as a precondition --
    see classify_ball_detection_regime, which should gate whether this
    function is even called for a given clip).

    `use_motion_diff_fallback` DEFAULTS TO FALSE as of 2026-07-19, a real
    reversal from this function's original design. A 5-clip, 375-sample
    manual visual audit (see PROGRESS.md's "Ball Detection: Coverage vs.
    Real Accuracy Under 3 Conditions" entry) found that on Miami Open
    broadcast footage specifically (as opposed to the amateur locked-camera
    dataset this method's 53.91% pooled-recall figure was validated on),
    motion-diff detections were correct only 60-72% of the time per clip
    (vs. 84-100% for fine-tuned-YOLO-alone detections) -- consistent with
    this module's own STOCK_BALL_METHOD_NOTE-adjacent finding that
    motion-diff produces false positives on player-limb motion in
    two-player broadcast frames. Averaged across all 5 clips, disabling it
    roughly triples how often a SHOWN ball marker is simply wrong (29% wrong
    with it on vs. 8% wrong with it off), at the cost of the marker being
    absent more often (raw coverage drops from ~100% to ~58-78% per clip).
    That tradeoff was judged worth it: a marker parked on a player's head or
    a court line is a worse user-facing failure mode than an honest gap.
    Pass True explicitly to restore the old always-fill behavior for
    research/comparison purposes -- this was NOT deleted, only defaulted
    off.

    PER-FRAME homography applicability (added 2026-07-16, threshold revised
    same day -- see frame_matches_reference_framing's docstring for the two
    real bugs found and fixed in this check specifically, including a
    same-day revert of a stateful cut-tracker design that traded one false
    positive for a false negative): the FIRST frame processed (start_frame)
    is treated as the calibration reference; every subsequent frame is
    compared directly to it via frame_matches_reference_framing. On a frame
    that doesn't match, the motion-diff fallback is SKIPPED (not just
    flagged) -- its court-region mask is built from the same homography and
    is equally inapplicable to a different camera angle, so running it there
    wouldn't just risk a wrong overlay, it would risk a wrong ball position
    too. Only the fine-tuned-YOLO-alone signal (which doesn't depend on the
    homography) is used for such frames.

    GROUND-TRUTH LEAK, FOUND AND FIXED (2026-07-16): this function's
    motion-diff fallback previously picked `candidates[0]` -- the first blob
    in cv2.findContours' arbitrary scan order. A separate PROTOTYPE script
    (since corrected, see PROGRESS.md) had instead used ground truth to
    select whichever candidate was closest to the real ball position, which
    is not something a real inference-time system can ever do, and produced
    an inflated, invalid 70.40% pooled-recall figure that was mistakenly
    carried into this "validated" production function's own docstrings.
    Re-running the actual production code path end-to-end (not the
    prototype) surfaced the gap directly: honest performance with an
    arbitrary-first-candidate pick was ~46.2% pooled recall, not 70.40%.
    Fixed here, without touching ground truth, by picking the LARGEST-AREA
    candidate instead of the first one -- a legitimate, inference-time-only
    heuristic (the real ball is a specific physical size; spurious small
    motion-diff blobs are more likely to be noise) already used successfully
    in an earlier, separate unvalidated spot-check. Re-measured after this
    fix: **53.91% pooled recall (video3 excluded, 9-clip scope)** -- the
    final, corrected, honest number. See PROGRESS.md for the full writeup."""
    flagged_bins = find_artifact_bins(fine_tuned_model, video_path, start_frame=start_frame, n_frames=n_frames)

    cap = cv2.VideoCapture(str(video_path))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_frames = n_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mask = None
    prev_gray = None
    reference_hist = None

    out: list[CombinedBallDetectionResult] = []
    for offset in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if mask is None:
            mask = court_mask(frame.shape, homography)
        if reference_hist is None:
            reference_hist = _frame_histogram(frame)
        homography_applicable, ref_corr = frame_matches_reference_framing(reference_hist, frame)

        results = fine_tuned_model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if len(results) else []
        confs = results[0].boxes.conf.cpu().numpy().tolist() if len(results) else []
        surviving = [(b, c) for b, c in zip(boxes, confs)
                     if not _is_near_flagged_bin((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, flagged_bins)]

        center = None
        source: Literal["fine_tuned_yolo", "motion_diff", "none"] = "none"
        if surviving:
            best_box, _ = max(surviving, key=lambda bc: bc[1])
            center = box_center(best_box)
            source = "fine_tuned_yolo"

        if use_motion_diff_fallback:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if center is None and prev_gray is not None and homography_applicable:
                candidates = motion_diff_candidates(prev_gray, gray, mask)
                if candidates:
                    # no ground truth available at inference time -- pick the
                    # LARGEST-AREA candidate (see this function's GROUND-TRUTH
                    # LEAK docstring note for why picking candidates[0] was wrong).
                    best = max(candidates, key=lambda c: c[2])
                    center = (best[0], best[1])
                    source = "motion_diff"
            prev_gray = gray

        out.append(CombinedBallDetectionResult(
            frame_index=start_frame + offset, center=center, source=source,
            homography_applicable=homography_applicable, reference_match_correlation=ref_corr,
        ))

    return out


def classify_ball_detection_regime(video_path: Path, sample_seconds: float = 30.0) -> tuple[Literal["validated", "best_effort"], dict]:
    """Cheap heuristic deciding whether a clip resembles the validated regime
    (locked-off single-camera, amateur-style footage -- what the combined
    method was actually measured against) or the best-effort regime
    (broadcast/multi-camera-angle footage, where motion-diff was directly
    spot-checked and found to produce false positives on player motion).
    Reuses the histogram-correlation hard-cut detector already built and
    validated for Stress Test #2's camera-angle filter, rather than inventing
    a new signal -- a high hard-cut rate is exactly the "many camera angles /
    highlight-reel-style editing" signature found there.
    Returns (regime, diagnostics) so callers/logs can see the actual cut rate,
    not just the binary decision.

    SAMPLES 3 WINDOWS spread across the clip (start/middle/end), not just the
    first `sample_seconds` -- a single-window-at-the-start design was tried
    first and found to misclassify match_tennis.mp4 (the AO-final highlight
    reel already confirmed cut-heavy in Stress Test #2): its opening ~5
    minutes are a single continuous shot before the cut-heavy editing begins,
    so sampling only frame 0 onward reported cut_rate=0.0 and wrongly
    classified genuinely cut-heavy footage as "validated". Real highlight
    reels/broadcasts commonly open with a continuous intro shot, so a single
    early window is not a reliable representative sample.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_frames = int(sample_seconds * fps)

    # 3 windows at 10%/50%/90% of the clip's duration -- skipped if the clip is
    # too short to fit 3 non-overlapping windows, in which case one window
    # (from frame 0) is used.
    if total_frames > window_frames * 4:
        starts = [int(total_frames * f) for f in (0.1, 0.5, 0.9)]
    else:
        starts = [0]

    n_cuts = 0
    n_sampled = 0
    for start in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        prev_hist = None
        for offset in range(window_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if offset % CUT_SAMPLE_STRIDE != 0:
                continue
            n_sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if corr < CUT_CORREL_THRESHOLD:
                    n_cuts += 1
            prev_hist = hist

    cut_rate = n_cuts / n_sampled if n_sampled else 0.0
    regime: Literal["validated", "best_effort"] = "best_effort" if cut_rate >= HIGH_CUT_RATE_FRACTION else "validated"
    return regime, {"cut_rate": cut_rate, "n_cuts": n_cuts, "n_sampled": n_sampled, "windows_sampled": starts}
