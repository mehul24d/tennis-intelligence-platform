# CV Pipeline (Phase 3) — Evaluation Report

Generated from `cv_pipeline/data/clip_reports/all_clips.json` (structured schema,
`cv_pipeline/src/cv_pipeline/schema.py`), which in turn was populated from the
validated results of steps 3-7 run against all 10 annotated clips
(`data/cv_annotated/`). Full derivation and every bug found/fixed along the way is
in `PROGRESS.md`'s Phase 3 section — this document is the pulled-together summary,
not a replacement for that record.

**How to read every number below**: each metric in the structured schema carries an
explicit status (`measured`, `insufficient_sample`, `excluded_known_issue`,
`not_attempted`, `not_detected`, `unvalidated`, `not_applicable`) — never a bare
number. Don't quote a rate from `all_clips.json` without checking its status field
first; several of the numbers below are flagged as unreliable precisely because that
distinction matters.

---

## 1. Player detection (near player)

**Excellent and consistent across all 10 clips.**

| | value |
|---|---|
| detection rate | 91.3%–99.8% (mean 96.4%, median 98.3%) |
| position error | 57.8–91.8px median (tight, no clip is a major outlier) |

No clip-level caveats needed here — this is the one metric in the whole pipeline
that's simply solid everywhere.

## 2. Player detection (far player) — genuine, hardware-appropriate limitation

**Detection rate on the clean, non-sentinel, genuinely-separated subset**, across
the 8 clips with an adequate sample (n≥20 — `video4` n=15 and `video10` n=9 are
**excluded from this figure as statistically unreliable**, not zero):

`video1` 20.8% (n=96), `video2` 1.0% (n=100), `video3` 0.0% (n=63), `video5` 34.1%
(n=41), `video6` 2.0% (n=152), `video7` 0.0% (n=45), `video8` 5.6% (n=71), `video9`
20.4% (n=152) — **mean 10.5%, wide spread (0–34%)**.

**Two real bugs were found and fixed before trusting this number at all**:
1. **Ground-truth sentinel contamination** — `player_r`/`player_l` occasionally sit
   at the same corner placeholder used for "ball not tracked." Fixed in
   `annotations.py` (`player_r_is_sentinel`/`player_l_is_sentinel`).
2. **Same-player duplicate labeling** — ground truth frequently points at the same
   physical player for both `player_r` and `player_l` slots (especially when the far
   player is off-frame). The "ambiguous" bucket (r/l <200px apart) exists specifically
   to isolate this — its low match rate is a ground-truth artifact, not a detection
   failure, and should never be quoted as a far-player accuracy number.

**Confirmed via `conf=0.01` inspection**: this is not a confidence-threshold problem
— no candidate detection exists at any confidence level in the missed frames.
**Confirmed via a `yolov8n` vs `yolov8s` comparison** on the clean 96-frame subset:
`yolov8s` scored 18.8% (marginally worse) at 37% lower fps — not adopted. This is a
genuine small/distant-object detection limit for this camera setup, not something a
bigger model or a lower threshold fixes.

**Sample-size caveat, stated plainly**: far-player detection rate could not be
reliably estimated for `video4` (n=15) or `video10` (n=9) — a single match/miss
swings either rate by 7–11 percentage points. Aggregate far-player statistics above
are driven by the 8 adequately-sampled clips only.

## 3. Ball detection — a recall problem, not a precision problem

**Headline number, committed: ~7-8% is the representative ball-detection rate for
typical clip conditions** (9 of 10 clips, `video3` excluded — mean 7.8%, median
4.9%).

`video3`'s 36.1% is reported **separately**, as a demonstration that ball-detection
quality scales strongly with video/broadcast quality — not folded into the headline.
Spot-checked directly: `video3`'s matched frames have sub-3px errors (not
coincidental false positives), and the clip is visibly higher-quality/higher-contrast
(bright ball against plain sky, sharper resolution) than the other 9.

**Position error is tight and consistent everywhere** it does match: 2.3–4.1px
median across all 10 clips, including `video3`. **This is a recall problem, not a
precision problem** — the generic COCO "sports ball" class (used because it's
zero-extra-cost on the same YOLOv8n model already loaded for players) rarely finds
the ball, but is accurate to a few pixels on the rare frames it does. Matches the
plan's own expectation that generic YOLO struggles with small, fast-moving balls.

## 4. Homography — 1 of 10 clips independently validated, 1 known-bad, 8 unconfirmed

| status | clips | meaning |
|---|---|---|
| **Validated, usable for real-world-distance metrics** | `video1` | Independently confirmed against the baseline center hash mark (a landmark never used in calibration): ~13px (~8cm real-world) error. |
| **Known-bad, excluded** | `video7` | Annotated corners span only the near half-court (baseline-to-net, ~11.9m), not the full doubles court (23.77m) — root-caused by matching the implied real-world span to a known ITF dimension within 2.8%. Geometrically self-consistent (0px reprojection error) but wrong scale — the reprojection check alone could never have caught this. |
| **Unvalidated, not assumed correct** | `video2`–`6`, `8`–`10` (8 clips) | Pass the same geometric self-consistency checks `video7` passed, but were never independently checked against a real landmark. Given `video7` proves self-consistency doesn't guarantee correct scale, **these 8 clips' real-world-distance outputs should not be trusted until individually validated** the same way `video1` was. |

A separate, real bug was also found and fixed here: the `BL`/`BR`/`TL`/`TR` corner
label strings don't consistently mean the same physical corner across clips (`video7`
and `video9` had them effectively rotated/mixed). Fixed by deriving near/far/left/right
geometrically from pixel positions rather than trusting the label names — this fix
applies to and is baked into all 10 clips' homographies.

## 5. Tracking (ByteTrack ID consistency) — only 4 of 10 clips have any real test coverage

**A contaminated proxy was caught and fixed mid-validation.** The original "hard
moment" (crossing/occlusion-risk) definition — any 2 person-boxes within 200px —
was picking up background people (spectators, officials, ball kids) in
broadcast-style clips, not real player proximity. Confirmed visually on `video8`
(flagged 100% of frames as "hard," corrected to 0% once restricted to the top-2-
highest-confidence boxes per frame).

**Corrected hard-moment frame counts**: `video1`=1, `video2`=6, `video3`=30,
`video4`=0, `video5`=0, `video6`=0, `video7`=4, `video8`=0, `video9`=0, `video10`=0.

**6 of 10 clips (`video4`, `5`, `6`, `8`, `9`, `10`) had ZERO genuine crossing/
proximity test coverage.** Their "no ID swaps" results are not evidence the tracker
handles crossings well — those clips simply never produced a real crossing to test
against. Only `video1` (1 frame), `video2` (6), `video3` (30), and `video7` (4) have
any real coverage, and even those are thin.

**The one substantively meaningful result**: `video3`, with the most real coverage
(30 hard-moment frames), showed **2 ID swaps** — genuine evidence of tracking
instability under real crossing conditions.

**`video6` showed 2 swaps despite ZERO real hard moments** — investigated and
precisely diagnosed by pulling the actual frames: the near player physically walked
out of camera view for ~145 frames (~2.4s, an on-court break), and ByteTrack assigned
a brand-new ID on their return since the original track had timed out (expected
behavior — ByteTrack does short-term motion/IOU association only, no long-term
re-identification by appearance). The second (`player_l`) swap in the same clip could
not be pinned to a single clean mechanism — likely a mix of the same break event and
the pre-existing far-player detection weakness, reported honestly as ambiguous rather
than overclaimed.

## 6. Pose estimation — visual spot-check only, no ground truth exists, no accuracy claim

6 hand-picked frames across 4 clips (`video1`, `video3`, `video6`, `video7`, `video9`)
covering easy and hard cases. **No quantitative claim is made anywhere in this
section — there is no ground truth for pose in this dataset.**

- **Near player**: consistently good, including under real difficulty — clean on
  frontal ready-stances (`video1`, `video7`), and accurate even on a mid-serve with
  the arm fully extended overhead (`video3`, correctly tracked to the raised hand).
  One minor imprecision noted (`video6`: low-visibility wrist/racket-grip landmarks
  during a post-break shot) — not a hard failure.
- **Far player**: two distinct real failures, precisely diagnosed rather than
  glossed over. `video1`: YOLO found no far-player box at all in the tested frame —
  pose couldn't be attempted (cascades from the detection weakness in §2). `video9`:
  after catching and fixing a box-selection bug in the spot-check script itself
  (it had picked a sideline bystander over the real far player), pose was rerun on
  the correctly-identified far-player box — and produced **zero landmarks**, on a
  ~55×66px, motion-blurred crop. This compounds, rather than introduces, the
  far-player limitation already found in detection.

---

## Summary: what this pipeline is and isn't good for, today

**Solid, trustworthy**: near-player detection and pose (any camera framing), ball
position accuracy when detected, `video1`'s real-world-distance homography.

**Real, documented limitations, not swept under the rug**: far-player detection and
pose (a hardware/resolution limit of this camera setup, not a threshold or model-size
problem — tested and ruled out both), ball detection recall (works but rarely
triggers, except on higher-quality source video), tracking ID stability under
crossing/occlusion (only lightly tested — most clips never produced a real crossing
to test against; the one well-tested clip showed real swaps), and real-world-distance
metrics for 9 of 10 clips (only `video1` independently confirmed; `video7` confirmed
bad; the rest unconfirmed either way).
