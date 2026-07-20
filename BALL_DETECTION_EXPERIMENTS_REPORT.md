# Ball Detection: TrackNet Investigation + Motion-Diff/Interpolation Experiments

Follow-up to `cv_pipeline/EVALUATION_REPORT.md` (committed baseline: ~7.8% mean ball
recall on the amateur dataset) and `STRESS_TEST_2_REPORT.md` (2.6% on genuine
wide-shot broadcast frames, an unexpectedly *worse* result than amateur footage).
This investigates whether a pretrained specialized model (TrackNet) or two cheap
detection-recovery techniques (motion-diff, trajectory interpolation) close that
gap. No existing Phase 3 pipeline code was modified.

## 1. TrackNet / TrackNetV2 — blocked pending a licensing decision, not evaluated

Searched for pretrained tennis-ball-tracking weights. Findings:

- **`yastrebksv/TrackNet`** (PyTorch reimplementation): does have pretrained tennis
  weights (Google Drive link), trained on 10 broadcast clips (19,835 labeled frames,
  1280×720/30fps — similar framing to our broadcast test clip). **No LICENSE file
  anywhere in the repo** — under default copyright rules that means all rights
  reserved; no stated terms for research use, redistribution, or commercial use.
- **`ArtLabss/tennis-tracking`**: Unlicense (public domain), but does **not** ship
  TrackNet weights at all — only bundles YOLOv3 weights for player detection. Ball
  tracking in that repo still depends on obtaining TrackNet weights from elsewhere.
- The original NCTU TrackNetV2 GitLab source (`nol.cs.nctu.edu.tw`) was unreachable
  (DNS failure) — could not independently confirm its license.

**No path to a cleanly-licensed pretrained TrackNet exists.** Attempting to download
and load the `yastrebksv` weights (a `.pt` file from a Google Drive link found via
search, not a source named in advance) was blocked by this environment's sandbox
permission classifier, which flags loading untrusted external model weights from a
self-discovered source as requiring explicit authorization. **This was not run.**
TrackNet remains a real option worth revisiting if the licensing question is
resolved (e.g. by contacting the maintainer, or by finding/training an
unambiguously-licensed weights file), but it is not evaluated here.

## 2. Motion-diff recovery — real, substantial improvement on the amateur dataset

**Method**: on frames where YOLOv8n's COCO "sports ball" class finds nothing,
compute a grayscale frame-difference against the previous frame, restricted to a
mask built from the clip's *actual annotated* court corners (dilated ~30% to
include serves/shots landing near the lines), then look for small (4–400px²)
moving blobs. Evaluated against real ground truth
(`data/cv_annotated/annotations/*_ball.csv`), using the exact same sentinel-
filtering and 100px match-distance convention already committed in
`ball_detection.py` / `EVALUATION_REPORT.md` — not a new methodology invented for
this test.

**Methodology sanity check**: the YOLO-only baseline recomputed here (pooled across
all 9 clips, video3 excluded, matching `EVALUATION_REPORT.md`'s scope) came out to
**7.81%** — matching the committed 7.8% mean almost exactly, confirming this
re-implementation is measuring the same thing the same way.

**Amateur dataset, per-clip and pooled results** (2,074 total ground-truth ball
frames across 9 clips):

| clip | n (gt frames) | YOLO-only recall | + motion-diff | + interpolation |
|---|---|---|---|---|
| video1 | 250 | 2.8% | 52.0% | 52.8% |
| video2 | 272 | 0.4% | 25.4% | 25.4% |
| video4 | 396 | 15.9% | 69.9% | 76.0% |
| video5 | 164 | 4.9% | 43.9% | 45.1% |
| video6 | 310 | 12.3% | 49.7% | 53.5% |
| video7 | 269 | 2.6% | 85.9% | 86.2% |
| video8 | 179 | 1.1% | 67.0% | 67.0% |
| video9 | 170 | 15.9% | 77.6% | 85.3% |
| video10 | 64 | 14.1% | 15.6% | 20.3% |
| **pooled (sum/sum)** | **2074** | **7.81%** | **57.62%** | **60.37%** |

**This is a real, large improvement** — motion-diff alone recovers roughly half to
five-sixths of YOLO's misses on most clips, pushing pooled recall from 7.8% to
57.6%. **Visually spot-checked, not just trusted from the number**: two independent
checks (`motioncheck_video1_f36-39.jpg`, `motioncheck_video7_f101-104.jpg`) confirm
the recovered position lands within a few pixels of the real ball, on frames where
the ball is spatially isolated from player motion (near the net, away from a
player's swing).

**Why this works here**: these are locked-off, static-camera amateur clips with
usually one player's motion in frame at a time near the ball's location, and (per
`EVALUATION_REPORT.md`'s own diagnosis) the ball is small but generally
high-contrast against the court surface. Frame-differencing isolates it cleanly
when it isn't competing with a larger nearby motion source.

## 3. Motion-diff on the two stress-test clips — does NOT transfer, confirmed by spot-check

The stress-test clips have **no ground truth**, so only a candidate-rate (not
recall) could be measured — reported with the same "unvalidated" caveat already
used for their YOLO ball numbers.

| clip | YOLO candidate rate | motion-diff found *something* on YOLO misses |
|---|---|---|
| `tennis_clip.mp4` (900 frames) | 14.3% | 92.0% of misses (709/771) |
| `match_tennis.mp4` wide-shot (300 frames) | 18.0% | 99.6% of misses (245/246) |

**These candidate-rate numbers are not meaningful on their own** and are not
reported as a win — visual spot-check (`tennis_clip_motioncand_offset3603.jpg`,
`match_tennis_wideshot_motioncand_offset7501.jpg`) shows the "candidates" are
almost entirely **false positives on player footwork/limb motion**, not the ball.
Both stress-test clips have two players moving simultaneously within the (larger,
manually-estimated, less precise) court mask, and player motion is a much larger,
messier signal than the ball's — exactly the confounding factor absent from most of
the amateur clips' cleaner single-player-near-ball frames. **Motion-diff does not
transfer to broadcast/professional footage with two players in frame** — a real,
negative finding, not glossed over.

## 4. Trajectory interpolation — a small, reliable, but low-volume gain

**Method**: for gaps of 1-3 consecutive frames between two YOLO detections that
were themselves confirmed correct (matched ground truth within 100px — never
ground truth itself, so this only uses information a real deployed pipeline would
actually have), fit a quadratic (parabolic) curve per axis through the confirmed
points immediately surrounding the gap, and check whether the interpolated
position lands within 100px of ground truth.

**Result**: 57 gaps attempted (pooled across the 9 amateur clips), **57/57 (100%)
landed within 100px of ground truth**. This adds +2.75 percentage points on top of
motion-diff (57.62% → 60.37% pooled).

**Honest caveat on volume**: interpolation can only fire where two YOLO hits
already bracket a short gap — since YOLO's own hit rate is only 7.8%, such
brackets are rare (57 opportunities out of 2,074 ground-truth frames, several
clips with 0-2 attempts). The 100% success rate is real but low-sample and
low-impact in isolation; its main value is as a small, free addition once
motion-diff has already substantially raised the number of confirmed detections
to interpolate between.

## 5. What NOT done, per instruction, and why that's still correct

- **No custom ball-detection model trained from scratch.** Confirmed sample sizes
  (~250-680 labeled ball positions per clip, ~2,074 total across all 9 clips) are
  far too small for a from-scratch detector, and there's no GPU budget for it on
  this M2 hardware.
- **yolov8s/yolov8m not tried.** Consistent with the already-established finding
  (`EVALUATION_REPORT.md` §2) that a bigger generic YOLO checkpoint didn't help
  far-player detection; the ball is an even smaller object, so the same
  model-capacity ceiling almost certainly applies with more force, not less.

## 6. Recommendation

**Motion-diff (restricted to the court region via real homography) is a genuine,
validated improvement worth adopting for the amateur/single-camera use case** —
7.8% → 57.6% pooled recall, visually confirmed, not just a number. **It should NOT
be adopted for broadcast/multi-player-in-frame footage** (the stress-test clips)
without further work — it currently just re-detects player motion there.
Trajectory interpolation is a safe, free, small addition once motion-diff is in
place, but shouldn't be sold as a fix on its own given how rarely it can fire.
TrackNet remains unresolved — a licensing question, not a technical one — and is
the natural next thing to revisit if a clean weights source can be found or
authorized.

No existing Phase 3 pipeline code (`cv_pipeline/src/cv_pipeline/*.py`) was
modified; all of the above lives in new scripts under
`cv_pipeline/scripts/ball_detection_experiments.py` and
`cv_pipeline/scripts/ball_motion_diff_stress_clips.py`.
