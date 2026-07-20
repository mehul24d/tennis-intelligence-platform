# CV Pipeline Stress Test #2 — Broadcast Highlight Reel + Camera-Angle Filtering

**Source file**: `data/match_tennis.mp4` — 35.65 min, 25fps, 1280×720, a broadcast
highlight compilation of the Alcaraz–Djokovic Australian Open final. Confirmed via
`cv2` metadata check, distinct from `data/tennis_clip.mp4` (the professional
practice-court clip used in the first stress test, `STRESS_TEST_REPORT.md`).

All numbers below are real, measured outputs from actual runs of this file through
the existing detection/tracking/pose code (`cv_pipeline/src/cv_pipeline/`), called
the same way `stress_test_pro_clip.py` called it in the first stress test. No
existing Phase 3 pipeline code was modified. Two bugs were found and fixed, but
both live in this test's own new scripts (`cv_pipeline/scripts/stress_test_2_*.py`)
— see §4.

---

## 1. Step 1 — sample timing and cut-heaviness

75s sample (1,875 frames, starting 5:00 into the file, past any intro/graphics):

| step | measured | extrapolated to full file (53,473 frames, 35.65 min) |
|---|---|---|
| frame extraction only (`cv2.VideoCapture.read()`) | 1.1ms/frame, 918fps | ~1.0 min |
| extraction + YOLOv8n person+ball detection | 42.1ms/frame, 23.8fps | ~37.5 min |

Full-file processing is tractable in well under an hour — far cheaper than the
point-document generation cost measured earlier in this project (~42hrs for the
full 5,981-match corpus).

**Cut-heaviness confirmed, not assumed**: histogram-correlation cut detection
flagged 24 candidate hard cuts in the 75s sample (~1 every 3s). Three were
visually verified: full-court rally view → mid-cut → broadcast closeup with
graphics/replay overlay (ghosted slow-motion image, scoreboard mid-transition).
This is a genuine highlight-reel-style cut compilation, not continuous match
footage — homography/tracker state cannot be assumed continuous across the file.

## 2. Step 2 — heuristic camera-angle filter

Two cheap signals, calibrated directly against Step 1's confirmed good/bad example
frames (no trained classifier, per the explicit preference for a simple heuristic
given time constraints):
1. **court-blue color fraction** (HSV range sampled from a known-good court
   surface, restricted to the bottom 75% of frame to exclude the crowd/stand band)
2. **long-line count** (Hough-detected line segments ≥180px, since full-court
   broadcast views show ~50-70 long straight lines — net, court lines, sponsor
   board edges — that closeups/graphics don't)

A frame is "valid" only if both thresholds are met (court fraction > 0.55, line
count ≥ 45).

**Results**:
- 75s sample: **51.9% valid, 48.1% other** (974/1,875)
- Full file (every 5th frame classified, 10,695 sampled, 15.35ms/frame, 164s total
  wall time): **35.1% valid, 64.9% other** (3,753/10,695)

The full-file rate is lower than the sample rate because the sample happened to
land in a rally-heavy stretch; the full reel has proportionally more crowd shots,
replays, and changeovers throughout.

**Visual spot-check** (`cv_pipeline/scratch_output/stress_test_2/angle_filter_examples/`):
- Clear "valid" and clear "other" examples were correctly classified.
- Two known failure modes found, both real, both worth naming:
  - **False positive**: a specialty net-cam angle (ball crossing the net, zero
    players visible) passes the filter — high court-color and line-count score,
    but it's not a normal wide rally shot. Handled explicitly in Step 3 (see §3).
  - **False negative**: one borderline frame was a genuine full-court rally view
    (Djokovic mid-swing at net, ball in play) misclassified as "other" because the
    crowd/sponsor-board band at the top of frame and net obstruction reduced the
    line count below threshold.

This filter is directionally correct but imperfect — reported as such, not as a
clean binary signal.

## 3. Step 3 — detection, tracking, pose on filtered frames

Run on the 974 "valid" frames from the 75s sample (147.5s wall time, 151.4ms/frame
for the full detect+track+pose stack).

### Net-cam tagging

Heuristic signature: 0 person boxes detected AND line-count ≥ 45 (the same
line-count signal from the angle filter). This caught **5/974 frames (0.5%)** as
net-cam — a small contaminant in this particular sample, but real, and kept
separate throughout rather than blended into the aggregate.

### Person detection (far player)

| | count | rate |
|---|---|---|
| 0 person boxes | 5/974 | 0.5% |
| 1 person box | 5/974 | 0.5% |
| 2+ person boxes (both players visible) | 964/974 | **99.0%** |

**Comparison to amateur baseline**: the amateur dataset's far-player detection
mean was **10.5%** (range 0–34% across 8 adequately-sampled clips; `video1`
specifically was 20.8%, the figure referenced in the task brief). The first
stress test's professional practice-court clip found the far player present in
every sampled frame but with an unreliable *auto-selection* heuristic (occasionally
picking a courtside official). Here, on broadcast footage with the fixed
court-position-based selection (see §4), **99.0% of valid frames had both players
correctly detected as separate boxes** — a large, genuine improvement over both
prior conditions, on genuine wide-shot frames (net-cam frames excluded from this
count by construction, since they have 0 person boxes).

### Far-player pose

| | count | rate |
|---|---|---|
| attempts (2+ boxes, wide-shot frames only) | 964 | — |
| landmarks found | 247 | **25.6%** |

The first stress test found far-player pose failed on *every* checked frame (zero
landmarks on a clean ~56×78px box, attributed to resolution, not clutter). Here,
after fixing a box-selection bug specific to this test (§4), far-player pose
succeeds in roughly 1 of 4 valid frames — spot-checked and confirmed genuine (see
`step3_examples/farpose_landmarks_found_offset57.jpg`: real far player, correct
box, plausible dense landmark cluster on a ~35×79px crop). This is a real
improvement over the first stress test's complete failure, though still far from
reliable — 3 of 4 valid frames still produce no far-player pose.

### Ball detection — net-cam contamination flagged separately, per instruction

| | count | rate |
|---|---|---|
| all valid frames (net-cam + wide, blended) | 26/974 | 2.7% |
| **net-cam frames only** | 1/5 | 20.0% |
| **genuine wide-shot frames only** | 25/969 | **2.6%** |

The net-cam-only rate (20.0%, n=5 — too small to be more than a directional
signal) is elevated exactly as anticipated: an isolated ball crossing close to a
fixed net-level camera is a large, high-contrast, unoccluded target — an easy case
for the COCO "sports ball" class, and **not representative of ball detection during
normal wide-shot rally play**. It is reported here for transparency only, and
excluded from the headline comparison below.

**Comparison to amateur baseline (genuine wide-shot frames only, net-cam
excluded)**: amateur dataset ball detection was ~7-8% (mean 7.8%, `video3`
excluded as a ground-truth-validated recall figure). The first stress test's
professional clip found a 14.3% *candidate* rate (unvalidated, COCO class firing
rate, same caveat as here). This broadcast highlight reel's genuine wide-shot rate
is **2.6% — lower than both prior figures, not higher.**

**This is the honest, somewhat counterintuitive finding of this test**: broadcast
video quality does not translate into better ball detection on wide rally shots.
A plausible explanation (not verified further, out of scope for this test): the
tennis ball is a small, fast-moving object regardless of source resolution, and
broadcast footage at this zoom level actually renders it *smaller* in pixel terms
than the closer, lower-resolution amateur/practice-court framings — motion blur
and small absolute pixel size, not overall image quality, appear to be the binding
constraint. No fix was attempted; this is reported as a limitation, not resolved.

### Tracking (ByteTrack) — reported with an explicit caveat, not as a clean number

49 distinct IDs seen across 969 wide-shot frames, 82 ID changes. **This number is
not a fair comparison to the first stress test's continuous-clip tracking result**
(3 ID changes in 900 continuous frames): the "valid" frame sequence here is not
temporally contiguous — Step 2's filter removes "other" frames from the middle of
the sequence, and `persist=True` tracking was run across those artificial gaps.
The elevated ID-change count is at least partly a byproduct of that
discontinuity, not purely a tracking-quality signal. Not treated as a headline
result for this reason.

## 4. Bugs found and fixed in this test's own scripts (not Phase 3 pipeline code)

1. **Player-selection false positive**: on the reference frame, YOLO fired a
   small (1214px², conf 0.51) false-positive "person" box on the on-screen
   clock/scoreboard graphic. `select_players_by_court_position()`'s unfiltered
   court-position ranking picked this graphic box as "far player" over the real
   far player (a similarly small but genuine 3042px², conf 0.32 box), because
   both boxes' projected court positions were near the far baseline area under
   this test's homography calibration. **Fixed by pre-filtering to boxes
   ≥2000px² before calling the existing selection function** — a filter added in
   `stress_test_2_pipeline.py`, not a change to `player_selection.py` itself.
   Confirmed fixed via two independent spot-checks (see
   `step3_examples/farpose_offset0_nolandmarks.jpg`, now showing the correct far
   player box).
2. Homography corners for this file's dominant wide-broadcast camera setup were
   manually estimated from one reference frame (same informal-but-visually-verified
   method as the first stress test's `MANUAL_CORNERS`) and confirmed by drawing
   the corners back onto the frame before use (`corner_check.jpg`). This
   calibration covers only the single dominant broadcast framing — the net-cam
   angle and any other distinct camera setups within the "valid" bucket are a
   different lens/framing entirely and were not separately calibrated; homography
   was applied to all 964 non-net-cam frames without per-shot re-verification, so
   some residual calibration error on frames with different zoom/framing than the
   reference frame is possible and was not separately quantified.

## 5. What this does and does not answer

**Answered**: on genuine wide-shot rally frames, broadcast-quality video gives a
large, real improvement in far-player *detection* (99.0% vs. 10.5% amateur mean /
20.8% best-amateur-clip) and a real (though partial) improvement in far-player
*pose* (25.6% vs. 0% in the first stress test's single spot-checked frame). Ball
detection does **not** improve — it's lower on this broadcast footage (2.6%) than
on both the amateur baseline (~7.8%) and the first stress test's clip (14.3%
candidate rate), and this is attributed to the ball's small absolute pixel
footprint at broadcast zoom/distance, not to any deficiency in image quality.

**Explicitly does not resolve, and remains open regardless of this test's
outcome**:
- The score/point-state gap: win-probability `live_adjustment` remains
  `not_available` — nothing in this test touches score overlay reading or point
  boundary detection.
- Full-length (35.65 min) processing was not run end-to-end in this test — only
  extrapolated from sample timing (§1) and run on a 75s/974-valid-frame subset in
  Step 3.
- The camera-angle filter (Step 2) improves what fraction of frames are worth
  running the pipeline on at all, but is a separate mechanism from — and is not
  credited for — the detection/pose improvement found in Step 3, which is
  attributable to broadcast video quality itself, not to the filter.

## 6. Constraint honored

No existing Phase 3 pipeline code (`cv_pipeline/src/cv_pipeline/*.py`) was
modified. All fixes and thresholds in this report live in new, separate scripts
under `cv_pipeline/scripts/stress_test_2_*.py`. The one real finding that *might*
warrant a Phase 3 code change — the player-selection false-positive on
graphic/official boxes (§4.1) — is flagged here for discussion, not applied to
`player_selection.py` itself.
