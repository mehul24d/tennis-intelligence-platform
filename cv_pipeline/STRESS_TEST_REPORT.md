# Phase 3 Stress Test — Professional Clip (Out-of-Dataset Generalization)

**No ground truth exists for `data/tennis_clip.mp4`.** Every finding below is a
direct visual/quantitative observation (detected or not, plausible or not, stable
or not) — never an accuracy percentage, and nothing here should be quoted as a
validated benchmark. Single clip, single stress test — conclusions are suggestive,
not a new validated result. **No Phase 3 pipeline code was modified based on this
clip alone.**

**Update**: the box-selection bug found in §5 below (picking a courtside bystander
over the real far player) was subsequently confirmed on a second, independent
amateur-dataset clip (`video9`) and fixed properly in pipeline code —
`cv_pipeline/src/cv_pipeline/player_selection.py` (`select_players_by_court_position()`,
homography-projected court-position plausibility instead of box size). §5's
findings and the summary table below describe the ORIGINAL (pre-fix) behavior,
left as-is since that's what was actually observed during this stress test; see
`PROGRESS.md`'s "Box-selection fix" entry for the fix itself.

**Fix status, stated precisely — resolved vs. not, and why**:
- **On the amateur dataset (real, validated corner annotations): RESOLVED.**
  Re-run and confirmed on `video9` directly — the fix correctly rejects the
  sideline bystander and selects the real far player, matching the
  manually-established ground truth exactly. No regression on the other 5
  amateur spot-check cases.
- **On this clip specifically: NOT resolved, and the cause is not the selection
  logic.** The bystander's projected position falls genuinely inside the assumed
  court width under this clip's rough, single-frame, by-eye corner calibration
  (§3) — a homography-precision problem, not a flaw in
  `select_players_by_court_position()`'s logic. The same function, given a
  properly calibrated homography, works correctly (as the amateur-dataset
  result above demonstrates). **Next step if this clip is ever revisited**:
  re-click the four court corners more carefully (ideally cross-checked against
  a second visible landmark, the way `video1`'s amateur-dataset homography was
  validated) to get a tighter calibration — not a pipeline-code change. Not a
  priority now; this was a single stress test, not a clip in active use.

## Scope

`data/tennis_clip.mp4` is ~13 minutes (47,644 frames, 60fps, 1920x1080) —
far longer than any amateur dataset clip (~10s each). Processed a representative
**900-frame (15s) segment** (frames 3600-4500, chosen for continuous rally content
with both players and the ball visible), matching the amateur dataset's per-clip
scale, rather than the full 13 minutes.

## 1. Detection (YOLOv8n, unchanged from Phase 3 — no re-litigation of the model choice)

**Every one of the 900 sampled frames had 2+ person boxes** (up to 12 in one
frame) — this is a genuinely busier real scene than the amateur dataset, not a
false-positive bug. Visually confirmed: up to 7 real people are visible
simultaneously (2 players, a coach against the side wall, 2 officials near a net
post, 2 seated staff at a courtside table) — all correctly boxed at plausible
confidences (0.30–0.90).

- **Near player**: detected in every sampled frame (0 frames with a missing
  large/near-player box across the segment) — consistent with the amateur
  dataset's strong near-player result.
- **Far player**: also consistently present as a distinct small box in every frame
  checked — a real result, but **the automated "smallest box = far player"
  heuristic is unreliable here**, exactly as found in the amateur dataset's
  `video9`: it twice picked a courtside official instead of the real far player
  (confirmed and corrected before running pose — see §5).
- **Officials/staff mistaken for a player?** Checked specifically. They are
  correctly detected AS people (that's real, correct detection) but are NOT
  confused with the two actual players in a way that would corrupt a *properly
  disambiguated* near/far selection — the risk is entirely in naive
  smallest/largest-box selection heuristics, not in the underlying detector. Same
  failure class as `video8` in the amateur dataset, now confirmed to generalize.

## 2. Tracking (ByteTrack)

The near player's identity, tracked via the largest-box-per-frame across the full
900-frame segment, changed **3 times**. Each transition was individually
investigated by pulling exact frames and per-frame box data (not assumed) —
**precisely because Phase 3's own discipline (e.g. the `video6` ID-swap
investigation) requires this, not a description offered from pattern-matching**:

1. **Frame ~157: a real scene cut**, not a tracking failure. The player's shirt
   changed from black to white and shoe color changed within 2 frames — this
   clip is very likely an edited compilation of multiple separate rally moments,
   not one continuous take (unlike every amateur dataset clip, which were
   confirmed raw/unedited). ByteTrack correctly assigned a new ID after a hard
   cut; this is expected, correct behavior, not a limitation.
2. **Frame ~592: a genuine 1-frame detection dropout during fast motion.**
   Confirmed via exact per-frame box areas: the near player's box (area ~164,948px²)
   disappears entirely for exactly one frame, then reappears one frame later at
   roughly half the area (~92,775px², consistent with a fast lunge/crouch and
   motion blur) under a new ID. This is a real, if brief, tracking limitation —
   the same class of failure as the amateur dataset's `video6` case (short-term
   detection loss defeats ByteTrack's short-term-only association).
3. **Frame ~844: more ambiguous**, not cleanly attributed. A nearby overlapping
   detection (a second, smaller box) coexisted for 2 frames immediately before the
   near-player's original track vanished — plausibly a box-overlap/duplicate-
   detection artifact rather than a clean occlusion event. Reported as
   unresolved rather than assigned a mechanism that wasn't actually confirmed.

**Comparison to the amateur dataset**: this clip's near player changed identity 3
times in 15 seconds, markedly *more* than `video1`'s near player (0 changes across
its full 689-frame/~11.5s clip). This is not necessarily "worse tracking" — one of
the three transitions is a scene cut (not a tracking failure at all), and the
underlying per-frame detection quality is otherwise strong. But it's a genuinely
different regime (edited/compiled footage, continuous fast rally play) than the
amateur dataset's raw single-take clips, and the two aren't directly comparable
without that caveat.

## 3. Homography — single manual calibration, explicitly not validated

No annotated court corners exist for this clip. Manually estimated 4 corners from
a representative frame by eye (near baseline ≈ `(65,793)`–`(1855,780)`, far
baseline ≈ `(395,430)`–`(1490,425)`), built the homography the same way as the
Phase 3 pipeline, and sanity-checked the net's predicted pixel position against
where it's visually seen:

- **Predicted net-base position**: `(953, 564)`.
- **Visually measured net-base position**: ~`(950, 495–505)`.
- **Disagreement: ~60–70px** — notably worse than `video1`'s independently
  validated ~13px error.

**This is expected and stated plainly, not glossed over**: this is a single,
quick, by-eye calibration (no repeated measurement, no independent landmark
cross-check), unlike `video1`'s validated homography. Treat this clip's
homography as a rough sanity-check only — plausible general geometry, not
usable for any real-world-distance metric.

## 4. Pose estimation (MediaPipe) — the clip's most direct, useful comparison

- **Near player**: clean, accurate landmarks on a real mid-swing frame (arm
  extended toward the ball, correctly tracked shoulders/hips/knees/ankles) —
  comparable in quality to the amateur dataset's best near-player results.
- **Far player**: first attempt picked the wrong box (an official near the wall,
  not the real far player — the same box-selection bug found in the amateur
  dataset's `video9`, caught and corrected before drawing any conclusion).
  **Rerun on the correctly-identified far-player box (~56x78px, set against a
  clean, uncluttered dark-green background) still produced ZERO landmarks.**

**This is the clearest evidence gathered in Phase 3 on the size-vs-clutter
question**: this clip's far player has a much cleaner background than most
amateur far-player cases, yet pose still fails completely. **This points toward
crop size/resolution being the primary driver of far-player pose failure, not
background clutter** — though this is one data point on one clip, not a
controlled test, and should be treated as suggestive rather than conclusive.

## 5. Ball detection

The generic COCO "sports ball" class (same zero-extra-cost approach as Phase 3)
returned at least one ball candidate in **129/900 frames (14.3%)** of the segment.
This is qualitatively higher than the amateur dataset's typical ~7-8% (video3
excluded) — plausibly explained by the ball's better visible contrast in this
clip (as the task description anticipated), though **no ground truth exists here
to confirm these candidates are correctly *positioned*, only that the class fired**
— this is a candidate-detection rate, not a validated match rate the way Phase 3's
amateur-dataset ball numbers were (which had ground truth to check position error
against). Don't treat 14.3% as directly comparable to the amateur dataset's
validated ~7-8% recall figure; they're measuring different things.

---

## Summary: does this generalize?

| | amateur dataset | this clip | verdict |
|---|---|---|---|
| Near-player detection | 91-99.8% | consistent, no misses in 900 sampled frames | **generalizes well** |
| Far-player detection | 0-34% (wide spread) | present but unreliable auto-selection (same bug class as `video9`) | **same limitation replicates** |
| Far-player pose | often fails (small/blurred crops) | fails even with a clean background | **limitation confirmed to be size/resolution-driven, not clutter-driven** (suggestive, single data point) |
| Tracking ID stability | `video1`: 0 swaps/11.5s; `video6`: 2 swaps, 1 well-explained (break), 1 ambiguous | 3 changes/15s: 1 scene cut (not a failure), 1 well-explained (motion dropout), 1 ambiguous | **not directly comparable** — this clip is likely edited/compiled footage, a different regime than the amateur dataset's raw takes |
| Ball detection | ~7-8% typical, validated against ground truth | 14.3% candidate rate, no ground truth to validate position | **not directly comparable** — different metric definitions |

**Bottom line**: the near-player results and the box-selection bug both generalize
cleanly (same behavior, same known failure mode). The far-player pose test is this
clip's most valuable contribution — real evidence (not assumption) that resolution,
not clutter, is the dominant constraint. Tracking and ball-detection numbers from
this clip are **not apples-to-apples** with the amateur dataset and shouldn't be
merged into Phase 3's aggregate figures.
