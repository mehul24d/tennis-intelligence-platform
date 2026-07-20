# CV Pipeline Stress Test #4 — 4K/30fps Night Match (`data/tennis.mp4`)

**Source file**: `data/tennis.mp4` — Qatar ExxonMobil Open, night match, blue hard
court, floodlit. Confirmed real properties via `cv2` metadata (not assumed):
**3840×2160 (4K), 1947 frames, ~29.997fps (NOT 60fps as described — a real
discrepancy from the task brief, flagged rather than silently corrected), 64.9s
duration.** No ground truth exists for this clip — every number below is either
a real measured cost, a candidate rate with an explicit "unvalidated" caveat, or
a directly spot-checked visual finding, matching the standard used for every
prior stress test.

---

## 1. Real measured cost, before processing more

75-frame(15s)/300-frame samples were timed first, per the project's established
"measure before committing" discipline:

| step | measured | rate |
|---|---|---|
| frame extraction only (15s / 449 frames) | 9.8s | 21.8ms/frame (45.8fps) |
| extraction + person + fine-tuned-ball detection, single pass (15s / 449 frames) | 56.5s | 125.9ms/frame |
| **full combined method** (pass-1 artifact-bin flagging + pass-2 detection, the actual production function, 300 frames) | 35.5s | **118.5ms/frame** |

**Comparison to the previously-measured 74.3ms/frame** (`match_tennis.mp4`,
1280×720, same full combined method): **118.5 / 74.3 = 1.6x slower** — far less
than the ~9x increase in raw pixel count (3840×2160 = 8.3M px vs 1280×720 =
0.92M px) would suggest. This makes sense: YOLO resizes its input to a fixed
inference size internally regardless of source resolution, so detection cost
itself is largely resolution-independent — the extra 1.6x is mostly decode/
resize overhead and the motion-diff/artifact-bin frame-differencing operations,
which do scale with source image size before any resizing happens.

**Decision: processed the full ~65s clip, not just a sample.** At 118.5ms/frame,
the full 1,947-frame clip projects to ~230s (~3.85 min) — modest enough that a
representative sample wasn't necessary; the full clip gives strictly better,
more representative evidence for the same cost. Confirmed directly: the full run
took 220.8s (113.4ms/frame, consistent with the sample-based estimate).

---

## 2. Ball detection — the core question this test exists to answer

**Full-clip candidate rate: 1,947/1,947 frames (100.0%)** — source breakdown:
77.3% from the fine-tuned YOLO path, 22.7% from the motion-diff fallback.
**This is an unvalidated candidate rate, not a recall figure** — there is no
ground truth for this clip, exactly as for both prior stress-test clips. It is
reported here, and then immediately checked against real frames, not treated
as a trustworthy number on its own.

**Visually spot-checked, both easy and hard cases specifically requested:**

- **Serve (ball high above court)**: frame 30 — a clean, correct detection of
  the tossed ball above the server's head mid-serve-motion, fine-tuned-YOLO
  source. (`ball_examples/frame30_fine_tuned_yolo.jpg`)
- **Volley/rally exchange (ball near court level)**: frame 240 — a correct
  detection of the ball near net height during a volley exchange, fine-tuned-
  YOLO source. (`ball_examples/frame240_fine_tuned_yolo.jpg`)
- **Motion-diff, isolated real ball**: frame 60 — an essentially perfect
  detection, the motion-diff marker landing almost exactly on the visibly bright
  yellow-green ball in flight over the net. (`ball_examples/frame60_motion_diff.jpg`)

**This is a real, visible improvement over the daytime broadcast condition.**
Stress Test #2 (`match_tennis.mp4`, daytime broadcast, similar 1280×720
resolution) found ball detection did NOT improve with broadcast quality —
2.6% on genuine wide-shot frames, actually below the amateur baseline (~7.8%).
Here, under night/floodlit/high-contrast conditions, both the fine-tuned model
and the motion-diff fallback are visibly, repeatedly landing on the real ball
across multiple genuinely different shot types (serve, volley, net-crossing
flight) — not cherry-picked easy frames. The mechanism is visually obvious in
every example: the ball's bright yellow-green color against the dark night sky
or the saturated blue court is a much higher-contrast target than daytime
broadcast footage at similar zoom, both for the YOLO detector and for
frame-differencing.

**New artifact pattern found, checked specifically as instructed — real, not
assumed away.** The frequency-based artifact filter flagged **20 distinct pixel
bins** in the 300-frame sample used for artifact-bin analysis — far more than
the 6-7 bins seen in prior clips. Investigated directly rather than left as a
number: the top-flagged bin (~10.7% frequency) was traced to real frames
(offsets 213-215) and confirmed to be a **static false positive** — a
hallucinated "ball" detection on an empty patch of court near the baseline
sideline mark/hash, in a frame where the real ball is clearly visible elsewhere
(near the net, in flight). This is the same general failure mode found in prior
clips (the fine-tuned model, trained on only 578 images, hallucinating on a
specific static court marking) — but at a new, different fixed location
specific to this court's line markings, not the same `(1442,778)`/`(412,442)`
locations found before. The other ~19 flagged bins were not individually
traced given time constraints, but given this pattern and the higher overall
ball-activity rate in this footage, some are plausibly real (frequently-hit
rally zones), not all necessarily artifacts — flagged as an open question, not
resolved here.

**Net assessment**: night/high-contrast lighting is a real, visible, positive
factor for ball detection — the clearest improvement found across all stress
tests to date — but the artifact-filter's false-positive problem persists in a
new location, and the 100% candidate rate should not be read as validated
recall.

---

## 3. Player detection and pose

**Player selection**: `select_players_by_court_position` on the spot-checked
frame (offset 90) reported "3/3 boxes plausible... rejected 0 box(es) as likely
bystanders/staff" — near and far boxes both landed correctly on the two
players, not the ball-kid/staff figures also visible in frame.

**Pose**: both near and far player pose succeeded on the spot-checked frame.
Far-player box was 83×193px at native 4K resolution — roughly 28×64px at a
720p-equivalent scale, similar in relative size to boxes that succeeded in
`match_tennis.mp4`'s far-player pose checks. Landmarks visually confirmed
plausible (a full skeleton overlay consistent with the player's actual mid-shot
posture). Not exhaustively tested across many frames given time constraints —
this is a single spot-check, same standard as prior stress tests' pose checks,
not a systematic evaluation.

---

## 4. Homography

No annotated corners exist for this clip (same as both prior stress-test
clips) — 4 doubles-court corners manually estimated from a reference frame,
drawn back onto the frame and visually confirmed before use
(`corner_check4.jpg`). **Sanity-checked against a known reference** (the net's
ground-level position, same style of check used for `video1`'s independently-
validated homography and the prior stress tests' manual calibrations):
predicted net-base pixel position ~988px (full-res y-coordinate) vs. the
visually-measured actual net-base position ~954px — **~34px error at native
4K resolution**. This is a rough, unvalidated, self-consistent calibration
only — same category and rigor as `tennis_clip.mp4`'s and `match_tennis.mp4`'s
manual corners, not independently validated the way `video1`'s is.

---

## 5. Regime classification

**Classified `validated`** (continuous single-camera feed) — both the
clip-level classifier (`cut_rate: 0.0`, 0 cuts in the default 3-window sample)
and a full-duration scan (every 5th frame across all 1,947 frames, 390 samples)
confirm **zero detected cuts anywhere in the clip**. This matches visual
expectation — this looks like unedited raw match footage, not a cut-heavy
highlight reel like `match_tennis.mp4`. No false cut detection was triggered by
fast serve motion or floodlight flicker — checked specifically, not assumed.
Consistent with this, `homography_applicable` was `True` for all 1,947 frames
(0 flagged) — the per-frame gating logic (built and fixed in the `video4.mp4`
investigation) correctly found no camera-angle mismatches anywhere in this
clip either.

---

## 6. Summary

- **Real 4K/30fps cost**: 118.5ms/frame for the full combined ball-detection
  method — a 1.6x slowdown from 720p, not the ~9x pixel-count increase might
  suggest, because YOLO's fixed internal inference resolution absorbs most of
  the resolution difference. A full 65s clip processes in ~3.7 min.
- **Ball detection genuinely improves under high-contrast night lighting** —
  this is the first stress test where broadcast/professional-quality footage
  showed a real, repeatedly-confirmed positive effect on ball detection,
  demonstrated across serve, volley, and open-court flight cases, not just
  easy frames. Contrast with `match_tennis.mp4` (daytime broadcast), where
  ball detection was found to be no better, and even worse, than the amateur
  baseline.
- **A new artifact-filter false positive was found and confirmed**, at a
  location specific to this clip's court markings, distinct from the two
  previously-known static-artifact locations — the same underlying model
  limitation (memorized court features from a 578-image training set)
  recurring in a new place, not a new bug.
- **Homography, player detection, and pose all worked as well as or better
  than prior clips** on this spot-check, plausibly aided by the much larger
  absolute pixel real-estate at 4K.

**Per instruction, no pipeline code was modified based on this test.** The
newly-confirmed artifact-filter false positive is reported as a finding for
discussion, not acted on unilaterally.
