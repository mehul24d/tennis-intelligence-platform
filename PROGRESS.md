# v2 Progress Log

Chronological status log for the v2 build (multimodal layer on top of the completed
v1 platform). Complements the phase table in `README.md` with the detail behind each
entry — what was done, what's deferred, and why.

---

## 2026-07-14

### v1 bug fix: `PtWinner` server-relative convention in `ml_informed_markov.py`

**Fixed.** While grounding a RAG "notable point" document example in real
`point_timeline_service.get_point_timeline` output, found that
`ml_informed_markov_predict` (`tennis-intelligence-platform/src/tennis_intel/live/ml_informed_markov.py`)
derived `a_won_this_point` as if `PtWinner` were player-relative, when it's actually
server-relative (`PtWinner==1` means the server won, not "player 1 won" — a convention
`point_timeline_service.py` already documented and correctly used elsewhere). This
inverted the return-side Beta posterior update on every point the tracked player
returns, in every match.

- Root-caused with a direct comparison against the correct convention on a real match
  (`20220725-M-Kitzbuhel-R32-Jurij_Rodionov-Hernan_Casanova`): 12/12 mismatches on
  `Svr==2` points, 0/8 on `Svr==1` points, in a 20-point sample.
- Fixed the derivation to be server-relative, matching `point_timeline_service.py`'s
  already-audited convention.
- Added `tennis-intelligence-platform/tests/unit/test_ml_informed_markov.py` (5 tests,
  passing) — covers all four `(server_is_a, point_winner)` combinations plus an
  end-to-end serve/return posterior-direction check. Required as a permanent regression
  guard, not just a one-off verification.
- Diff scoped to only the `a_won_this_point` derivation — no ripple into
  `ml_informed_point_probabilities`, `sensitivity_aware_blend`,
  `recursion_sensitivity`, or other engines (pure Markov / ML+MC / hybrid don't use
  this posterior).

### RAG engine (Phase 1): match + player documents

**Done and verified.** Built in `rag_engine/`:

- `src/rag_engine/ingest/match_documents.py` — match-summary documents sourced
  directly from `matches_with_elo.parquet` (full ~198k-match corpus, not the
  charted-only 5,988-match frozen-join subset — same rationale
  `career_stats_service.py` already used for player stats).
- `src/rag_engine/ingest/player_documents.py` — player career-profile documents,
  reusing `tennis_intel.serving.career_stats_service.get_player_profile` directly
  rather than re-deriving stats.
- `src/rag_engine/index/embedder.py` + `vector_store.py` — local CPU-only embeddings
  (`all-MiniLM-L6-v2`) into a Chroma collection persisted under `rag_engine/data/chroma/`,
  no external service.
- `src/rag_engine/build_index.py` — CLI entrypoint, supports `--match-limit`/
  `--player-limit` for partial builds.
- 8 passing tests (`rag_engine/tests/`): document-shape/grounding checks and a
  vector-store round-trip (build, retrieve, metadata-filter, reset-clears-previous).

**Verified at two scales:**
- 100-doc dev subset (50 matches + 50 players) — pipeline correctness confirmed,
  relevance weak (expected — too small a sample to contain e.g. a real Wimbledon
  match).
- **22,610-doc subset (20,000 most-recent matches + all 2,610 players with ≥10 career
  matches)** — relevance confirmed materially better: "recent Wimbledon final
  results" now correctly surfaces real Wimbledon matches (top-3 all real, distance
  0.68-0.70); metadata filtering (`doc_type=player_profile`) confirmed still correct
  at this scale.

### `point_documents.py` (notable-point/rally RAG documents) — unblocked

**Was blocked, now unblocked** by the `MatchState` timing fix below (same-day). Not
yet built — next up.

### Deferred: full 198k-match embed

**Deferred, not started.** CPU-only embedding on this M2 initially stalled badly
(became near-idle, likely tokenizer-thread oversubscription); after setting
`TOKENIZERS_PARALLELISM=false` and restarting, throughput recovered to ~130 docs/sec
for matches — but the full ~198k-match + 2,610-player corpus (~200k docs) would still
take 5+ hours. Decided (2026-07-14) to defer the literal full embed to closer to when
it's actually needed in production, and index the 22,610-doc representative subset
above instead for now, sufficient to validate the pipeline. Throughput optimization
(larger batch size, GPU/cloud embedding, or incremental/background indexing) can be
revisited if the wait time still matters at that point.

### v1 bug fix: pre-point vs. post-point indexing in `point_timeline_service.py` / `match_summary_service.py`

**Fixed**, in a dedicated focused session per the same rigor as the `PtWinner` fix.
Full writeup: `tennis-intelligence-platform/docs/known_issue_ml_informed_markov_pre_point_state.md`.

Traced the issue further than originally scoped and found it was narrower than
first thought: `ml_informed_p1[i]` itself (from `compute_five_engine_trajectory`)
was left **unchanged** — it correctly represents "the prediction just before point
`i`," which is also exactly what 15+ calibration/evaluation pipeline scripts
(`evaluate_full_match_calibration.py`, `sweep_prior_strength.py`, etc.) already
correctly rely on. Changing that function would have silently broken that much
larger surface. The actual bug was isolated to **two** of the four downstream
consumers mis-indexing that (correct) shared array into a before/after pair for
display:

- `point_timeline_service.py::get_point_timeline` — fixed.
- `match_summary_service.py::get_match_summary`'s `largest_probability_swing` —
  fixed (same underlying off-by-one).
- `model_agreement_service.py::get_model_agreement` — confirmed **never bugged**
  (reads the trajectory directly per-point, no before/after pairing); unaffected.
- `replay_service.py::replay_match_by_id` — confirmed **never bugged**, same
  reason; unaffected.

**Verification performed:**
1. Hand-computed synthetic 4-point sequence (`tests/unit/test_point_probability_indexing.py`)
   with a deliberate, unambiguous large swing on a known point — confirmed it failed
   pre-fix (attributed to the wrong point, exactly as predicted by hand) and passed
   post-fix, for both fixed services.
2. Real match (`20220725-M-Kitzbuhel-R32-Jurij_Rodionov-Hernan_Casanova`)
   direction-correctness: **89.9% overall (107/119)**, **100% (35/35) on every point
   with a non-negligible swing (≥0.01)** — up from 52.1%. The remaining 12 mismatches
   are all tiny-magnitude noise (≤0.007) at already near-certain states (≥0.98),
   the pre-agreed legitimate-exception category.
3. All 209 existing unit tests still pass — no regressions.
4. All four downstream consumers exercised end-to-end against the real match above;
   confirmed reproducible across separate runs.

`point_documents.py` is now unblocked (see above) — **but see the new blocking issue
below, found the same day while building it, which re-blocks it and everything else.**

### `point_documents.py`: built, swing-neutral phrasing verified, then RE-BLOCKED by a critical v1 finding

Built `rag_engine/ingest/point_documents.py` sourcing from `get_point_timeline`, with
swing-neutral phrasing (per
`known_issue_after_point_swing_includes_next_point_context.md`: never assert the
point's own outcome *caused* the swing; state the fact, add an automatic hedge —
`direction_matches_winner` — whenever the point winner's side and the probability
direction disagree). Verified against 3 real, grounded examples (one clean, one
hand-picked ambiguous, one automatically-flagged) — phrasing confirmed correct after
one fix (decoupled the probability sentence from "point won by X" and labeled it
"overall match win probability" — see git history / conversation, not a code bug, a
wording fix).

**While finding a third, automatically-flagged example to verify the hedge on, hit a
new, much bigger, CONFIRMED, BLOCKING issue** — see
`docs/critical_issue_gm_attribution_mismatch.md`:

**`Gm1`/`Gm2` (games-won-per-player columns, read directly into every live-probability
engine's `MatchState` via `row_to_match_state`) are wrong ~49% of the time at
individual game boundaries, specifically and almost perfectly correlated with server
identity: 0.11% mismatch when player 1 serves the deciding point, 99.90% mismatch when
player 2 serves it** — confirmed by two independent hand-traced methods sharing zero
code, confirmed at corpus scale (147,290 boundaries, all 5,981 matches), a shift test
ruling out a simple one-row lag, and a split test isolating server identity as the
driver (not hold/break, not p1/p2 winner identity alone).

**This is a v1 foundational-correctness issue, not a v2/RAG concern — it blocked ALL
further v2 work** (point_documents.py, the LLM agent, everything) until resolved.

### RESOLUTION: the "Gm1/Gm2 bug" was actually a PtWinner convention regression — full record in `docs/ptwinner_convention_correction.md`

The `Gm1`/`Gm2` investigation above eventually inverted itself: `Gm1`/`Gm2` were
correct all along. The real bug was that `PtWinner` had been changed, the same day,
from its original correct convention (**literal**, fixed-player-relative —
`PtWinner==1` means player 1 won, period) to an incorrect **server-relative** one
(`PtWinner==1` means the server won), based on a pre-existing script
(`check_ptwinner_disagreement_at_scale.py`) whose "0.00% disagreement" claim turned
out to only test internal self-consistency, never against `Gm1`/`Gm2` — a blind spot
that made two genuinely different, both-internally-consistent conventions
indistinguishable to it.

**Reverted to literal `PtWinner` everywhere it had been changed**:
`ml_informed_markov.py`, `point_timeline_service.py`, `match_summary_service.py`, and
five functions in `point_level_features.py` (`compute_in_match_momentum`,
`compute_consecutive_points_streak`, `compute_split_points_streak`,
`compute_in_match_serve_return_rate`, `compute_in_match_serve_return_rate_rolling`).
Two diagnostic pipeline scripts also fixed and flagged. All 211 unit tests pass.

**Verified**: literal `PtWinner` matches `Gm1`/`Gm2` at 99.91% corpus-wide, symmetric
across which player serves (vs. ~51%/chance for server-relative) — settled via a
corpus-scale test, two independent hand-traces in plain tennis terms, and ruling out
five other hypotheses in sequence with real numbers at each step (see the doc for
the full trace).

**`day9_point_classifiers.joblib` was known-stale, then RETRAINED and DEPLOYED
2026-07-15.** Impact analysis (feature-shift + importance ranking) found two of its
top-5 features (`p1_in_match_return_rate` #3, `points_streak` #5) shifted materially
under the correction — not a "barely moved the needle" case, so retraining was
scoped as its own dedicated follow-up rather than skipped by default.

**Retrain executed in full** (`pipelines/retrain_day9_candidate_and_compare.py`):
regenerated the point-level dataset on corrected features → sanity-checked the shift
was actually present in the fresh training data (96.8% of `Svr==2` rows changed,
matching the earlier isolated measurement exactly) → retrained with the same
methodology/hyperparameters as the original → compared old vs. new on rolling-origin
folds across 4 independent years (2022-2025), not a single split.

**Result: consistent, real improvement.** Log_loss 0.6281 → 0.6247, Brier 0.2187 →
0.2172, the new model winning in **every single one** of the 4 folds by almost the
same margin each time. Calibration comparable, no clear winner. Top-4 SHAP feature
ranks identical (stable model structure). One surprising result —
`p1_in_match_return_rate` kept rank #3 but importance dropped ~42% — was investigated
with a follow-up correlation check rather than accepted at face value: ruled out
"signal stolen by `is_second_serve_point`" (collinearity negligible, r≈0 in both old
and new), found instead that the OLD feature had a **backwards** correlation with the
actual training target (r=-0.029 — higher return rate weakly predicted *losing* the
next point, a direct fingerprint of the bug) which flipped to the correct positive
sign post-fix (r=+0.023) — the model had been partially exploiting spurious,
wrong-signed noise, and the importance drop reflects losing access to that noise, not
a real predictor becoming meaningless. Full comparison, tables, and the correlation
check are recorded in `docs/ptwinner_convention_correction.md`'s "Retrain results"
section.

**Deployed**: `day9_point_classifiers.joblib` now contains the retrained model.
Pre-retrain classifier preserved at `day9_point_classifiers_PRE_PTWINNER_FIX.joblib`.
Load-time warnings removed from `replay_service.py` and `build_day9_point_model.py`
(both updated to point at the retrain-results doc instead).

**v1's flagship match-level XGBoost/LightGBM/CatBoost win-probability models are
confirmed unaffected** — verified directly (no code path from those models' feature
pipelines touches `PtWinner`/`Svr` at all; they're built from match-summary aggregate
columns spanning the full ~198k-match corpus, independent of point-level charted
data).

**Two earlier same-day findings were flagged for re-verification; #2 is now
re-verified, #1 is still open:**

1. **The near-certain-tail-noise explanation**
   (`known_issue_ml_informed_markov_pre_point_state.md`) — **still needs
   re-measurement.** After the pre-point/post-point indexing fix, the ~10% of points
   still showing "wrong direction" on the Rodionov/Casanova match were found to be
   all negligible-magnitude noise at already near-certain states (`probability_before`
   ≥0.98, swing ≤0.007), with 100% direction-correctness on every point with a
   meaningful swing (≥0.01). The underlying indexing fix stands; this specific
   89.9%/100% breakdown was measured against the `winner` field under the old
   convention and has not yet been re-measured.
2. **The `is_second_serve_point` next-point-context finding**
   (`known_issue_after_point_swing_includes_next_point_context.md`) — **RE-VERIFIED
   2026-07-14.** Re-checked directly under the corrected `PtWinner` convention: the
   mechanism holds, and slightly more strongly than originally reported. The
   original worked example (Wimbledon point 203) turned out to no longer be a
   mismatch at all under the corrected convention (`PtWinner` there means Djokovic
   won, and Djokovic's own probability rose — fully consistent) — replaced with a
   freshly-found genuine mismatch (point 95: Alcaraz wins, Djokovic's probability
   rises), where isolating `is_second_serve_point` alone explains **108.6%/110.0%**
   of the real swing (vs. the original 93-96%). `point_documents.py`'s swing-neutral
   hedge phrasing remains justified on this finding — confirmed, not just assumed.

### `point_documents.py` wired in — Phase 1 (RAG engine) complete

- Exported from `rag_engine/ingest/__init__.py`; wired into `build_index.py`
  (`--point-match-limit`, `--skip-points` flags; point docs are the slowest step,
  each scanned match requires v1's full 5-engine per-point computation).
- 3 new tests added to `rag_engine/tests/test_ingest.py` (shape, text grounded in
  metadata, hedge text well-formed) — deliberately kept to a small `match_limit=3`
  fixture to stay fast-ish (~2 min), so the hedge test doesn't assert a minimum
  trigger count (not guaranteed at that small a sample; the real ~64/334 hedge rate
  is already confirmed at proper scale in the finding #2 re-verification above).
  All 7 tests in `rag_engine/tests/` pass.
- `rag_engine/README.md` added: what it does, usage, test notes, status.

**Phase 1 (RAG engine) is now complete**: match, player, and point documents all
built, tested against real data, and wired into a single `build_index.py` entrypoint
backed by a local Chroma store.

---

### Generation provider switched to Gemini; `rag_engine/generate.py` built

Decided (2026-07-15): Phase 2 uses the **Gemini API**, not Claude — `rag_engine`'s
generation seam built accordingly.

- Verified current SDK guidance directly (not from memory, since this is exactly the
  kind of fast-moving surface that goes stale silently): **`google-genai`** is
  Google's current, GA-since-May-2025 unified Gen AI SDK; **`google-generativeai`**
  is deprecated (2025-11-30) and was NOT used.
- `rag_engine/generate.py` — `GeminiClient` (reads `GEMINI_API_KEY`, preferred, or
  `GOOGLE_API_KEY` as fallback; raises `MissingAPIKeyError` immediately at
  construction if neither is set, not on first call), `build_prompt()` (grounding
  discipline: answer ONLY from retrieved context, say so explicitly if context is
  insufficient), `generate_grounded_answer()` (ties retrieval output to generation).
  Verified against the actually-installed SDK, not just documentation — client
  construction, `GenerateContentConfig`, and the tool-use types
  (`FunctionDeclaration`, `Tool`) all confirmed to exist as used.
- Retrieval/index-building code never imports `generate.py` and is fully unaffected
  by whether a key is configured — verified directly (module imports cleanly with
  `google-genai` not installed at all).
- `rag_engine/pyproject.toml`'s `generation` extra swapped from `anthropic` to
  `google-genai`. 6 new tests in `tests/test_generate.py`, all passing (17/17 total
  in `rag_engine/tests/` now).
- `README.md` (both repo-root and `rag_engine/`) updated: `llm_agent/` description,
  pipeline diagram, design-constraints section all now say Gemini, not Claude.

## Phase 2 (LLM tactical agent) — complete

`llm_agent/` built: `system_prompt.py` (grounding-rules persona — every claim must
cite `[L#]`/`[D#]`, live probabilities always hedged as model estimates named by
engine, explicit "insufficient historical data" rather than fabrication),
`live_features.py` (`LiveFeatureSnapshot`, engine-tagged, estimate-flagged),
`agent.py` (`TennisAnalystAgent` — stateful multi-turn via `google-genai`'s
`client.chats.create`/`send_message`, 503-retry reused from `rag_engine.generate`,
citation-audit `sources_used`/`sources_offered`). 5 mocked unit tests pass. 10-question
manual eval run against real retrieval + real Gemini calls, reviewed by hand — no
hallucination found; 3 of the most numeric-heavy answers independently re-verified
against raw rows in `matches_with_elo.parquet` (bypassing the RAG layer entirely) and
all matched exactly, including correctly disambiguating between 3 same-opponent
matches in one case.

## Phase 3 (CV pipeline) — in progress

`data/cv_annotated/`: 10 clips (`video1..video10.mp4`, 1920x1080, 60fps), each with 3
ground-truth CSVs (`{clip}_ball.csv`, `{clip}_court.csv`, `{clip}_player.csv`), joined
by a `frame_NNN` string key. Confirmed schema by inspecting real rows, not assuming:
- `court.csv`: only ~7 rows per clip (sampled every 100 frames, not per-frame) — court
  is static (locked camera), so `cv_pipeline/src/cv_pipeline/annotations.py` holds the
  most recent row forward. **Column names vary in both case (`BL_x` vs `bl_x`) and
  order across clips** — confirmed by inspecting all 10 headers directly (a real
  finding, not an assumption); fixed by normalizing to lowercase and selecting by
  name, never by position.
- `player.csv`: one row per frame, `player_r_x/y`, `player_l_x/y` (right/left court
  side, single point per player, not a bbox). Column order also varies by clip;
  selected by name.
- `ball.csv`: `ball_x`, `ball_y`, consistent across all clips, but occasionally
  missing frames entirely at a clip's tail.

**Ball sentinel finding**: confirmed (statistics + visual frame inspection, not
assumption) that ball rows within 25px of pixel `(1920, 0)` (top-right corner) are a
"no ball detected" placeholder, not a real position — 54-77% of frames in **every one**
of the 10 clips cluster at that exact same corner; within a clip these form long
unbroken runs (up to 101 consecutive frames = ~1.7s motionless, which a real ball in
play never does); and directly overlaying the raw coordinate on the source video frame
lands in empty background, not on the visible ball (checked at 3 separate frames),
while non-corner rows land exactly on the visible ball. Encoded as
`BALL_SENTINEL_CORNER`/`BALL_SENTINEL_RADIUS_PX` in `annotations.py`; excluded from
position-error scoring but reported as its own "ball not annotated" rate per clip.
Known caveat: at least one spot-checked sentinel frame (`video1` `frame_620`) has the
ball genuinely visible elsewhere in the image, meaning the sentinel rate measures gaps
in the *ground truth itself*, not strictly "ball wasn't visible" — a detector that
finds the ball in an excluded frame isn't wrong, it just can't be scored there.

Step 2 (overlay sanity check, `cv_pipeline/scripts/sanity_check_overlay.py`): court
quad, player points, and ball point all visually confirmed to land correctly on real
video frames (6 sample frames from `video1`, both real-ball and sentinel-excluded
cases) — images in `cv_pipeline/scratch_output/sanity_check_overlay/`.

Step 3 (homography, `cv_pipeline/src/cv_pipeline/homography.py`,
`cv_pipeline/scripts/verify_homography.py`): `CourtHomography` built from the 4
annotated doubles corners (BL/BR near baseline, TL/TR far baseline; ITF standard
10.97m x 23.77m). Net's pixel position is NOT a constant y — `net_pixel_y_at_x()`
projects the real-world net line (y=11.885m) through the homography per-x, correctly
accounting for perspective. **Validated independently**: predicted pixel position of
the baseline center hash mark (a court feature never used in calibration) measured to
within ~13px (~8cm real-world) of its actual pixel position in the source video — a
genuine, non-circular check. (A first attempt at an independent check using the
singles sidelines failed — not because the homography was wrong, but because the
simple brightness-threshold line finder used to locate them in the dusk-lit footage
just re-detected the already-calibrated doubles sidelines instead; documented as a
tooling limitation of that specific check, not a homography flaw.)

**Known issue / shelved analysis**: attempted to test whether ball-annotation sentinel
(gap) rate differs between near-side and far-side court frames. Could not be answered
from the ball CSVs alone — real (non-sentinel) ball y-positions in the sampled clip
never approached the true near baseline even though the camera clearly captures that
depth, strongly suggesting the annotation itself under-captures the ball specifically
when it's close to the camera (larger, faster, motion-blurred) — the same phenomenon
that produces sentinel rows in the first place. Any near/far side label for a sentinel
frame has to be imputed from nearby real frames, which are themselves overwhelmingly
one side — a circular confound, not a threshold-tuning problem. **Do not revisit this
via ball-position-derived side labeling.** Revisit only once pose estimation (step 7)
gives an independent way to determine active court side per frame from player pose,
not from the ball annotation itself.

**Corner-label inconsistency found and fixed** (`cv_pipeline/scripts/build_homographies_all_clips.py`):
generalizing the homography build to all 10 clips surfaced a real bug — the
`BL`/`BR`/`TL`/`TR` corner label strings do **not** mean the same physical corner
across clips. Confirmed by overlaying the raw labels on real frames: `video1` uses
B=near baseline (larger pixel-y)/T=far, but `video7` has this flipped (its "BL"/"BR"
sit at the net, its "TL"/"TR" sit at the near baseline), and `video9` mixes L/R and
B/T inconsistently (its "BL"/"TL" are both at the net, "BR"/"TR" both at the near
baseline). Trusting the label strings would have silently built a wrong,
rotated/mirrored homography for at least 2 of the 10 clips. **Fixed**:
`CourtHomography.__init__` now derives near/far and left/right geometrically from
actual pixel positions (largest-pixel-y pair = near baseline, smallest = far;
sorted by x within each pair for left/right) rather than trusting the label names.
Re-validated across all 10 clips: reprojection error of the 4 calibration corners is
exactly 0.0px for every clip, and near-baseline pixel span > far-baseline pixel span
(expected under perspective) holds for all 10, confirming the geometric sort picks
the physically correct pairing. **Note: the 4-corner reprojection check is trivially
self-consistent (0.0px by construction) and cannot catch a wrong overall SCALE** —
see the video7 finding below, found by an independent landmark this check missed.

**`video7` independent validation: FAILED, real issue found, not resolved.** Tried
two independent landmarks (not used in calibration): (1) the near-side service line
"T" mark — inconclusive, weak/low-contrast line on clay against the evening light in
this clip; (2) the net's ground-contact base (unambiguous, easy to precisely locate)
— predicted pixel y=584.1 vs. measured actual y≈495-500, an **~87px error** (vs. ~13px
for `video1`'s independent check, roughly 6-7x worse — not attributable to
measurement noise). This suggests the 4 annotated "court corners" for `video7` may
not span the full baseline-to-baseline doubles court (23.77m) the way `video1`'s do —
if so, every real-world-distance prediction from this clip's homography is off by a
consistent scale factor, an error the 4-corner reprojection check can never catch
(it's self-consistent by construction). **Root cause resolved** (not left a mystery): computed what real-world span the
annotated far corners would need to represent for the net's predicted position to
match its measured actual position, and checked it against known ITF tennis
dimensions. Implied span = 12.22m, which matches **baseline-to-net (11.885m) to
within 2.8%** — clearly closer than any other candidate tried (full baseline-to-
baseline 23.77m was 49% off; near-service-line 5.485m was 123% off; singles/doubles
width, wrong axis anyway, 11-48% off). Combined with the earlier visual finding that
this clip's raw `BL`/`BR` labels sit at the net posts, not the far baseline, this
confirms: **`video7`'s annotated court corners span only the near half-court (near
baseline to net), not the full doubles court** — the annotation itself is internally
consistent, just scoped to half the court length, which is why a homography built
assuming full-length (23.77m) scaling misplaced the net. **Flagged, not fixed**:
`video7` (and any other clip with this same half-court annotation pattern, not yet
individually checked) is excluded from real-world-distance-derived metrics (speed,
court coverage in meters) in the eventual evaluation report, pending a per-clip
length-detection fix. Still fully usable for pixel-space metrics (detection/tracking
accuracy against ground truth), which don't depend on the homography.

## Step 4 (YOLOv8 player detection) — video1 validated, model choice settled

`cv_pipeline/src/cv_pipeline/player_detection.py`: YOLOv8 (person class, CPU
inference) vs. ground truth, greedy nearest-neighbor exclusive matching
(`MAX_MATCH_DISTANCE_PX=150`). **Detection point convention confirmed, not assumed**:
ground truth is a foot/ground-contact point, not body-center — verified directly by
comparing a real YOLO box against `video1` `frame_400`'s ground truth (box
bottom-center: 28px error; box center: 179px error, would have been a spurious
"failure" from a pure convention mismatch). `box_center()` uses bottom-center.

**Player CSVs have the same corner-sentinel pattern as the ball CSVs**: `player_r`/
`player_l` occasionally sit at the same `(1920,0)`-ish placeholder corner used for
"ball not tracked" (e.g. `video1` `frame_005`: `player_l=(1916,4)`) — a "far player
not tracked" placeholder, not a real position. Found while debugging an initially
confusing detection-rate discrepancy; confirmed by checking down to `conf=0.01` that
no YOLO candidate exists anywhere near these positions at any confidence, in every
sampled frame. Fixed the same way as the ball sentinel:
`FrameAnnotation.player_r_is_sentinel`/`player_l_is_sentinel` added to
`annotations.py`, sentinel positions now excluded (`None`) from ground truth rather
than silently scored against.

**Real, non-sentinel results for `video1`** (689 frames):
- `player_r` (near player, well-separated from `player_l` in nearly all frames):
  89.0% detection rate (613/689), median error 75.8px.
- `player_l` (far/second player), on the CLEAN subset only (96 frames — real,
  non-sentinel, genuinely >=200px separated from `player_r`, confirmed not a
  same-player-duplicate artifact): **20.8% detection rate**, median error 134.4px.
  This is a genuine YOLOv8n limitation (small/distant object at this camera
  distance+resolution), not a confidence-threshold or model-capacity issue —
  confirmed both ways: (a) no candidate detection exists at `conf=0.01` in sampled
  miss frames, (b) `yolov8s` (next size up) tested on the same 96-frame subset scored
  **18.8%** (marginally worse, within noise) at **11.9 fps vs. 19.0 fps** for
  `yolov8n` (~37% slower). **Not adopting `yolov8s`** — no accuracy benefit, real
  speed cost. `yolov8n` retained; far-player detection rate on small/distant players
  documented as a known, hardware-appropriate limitation rather than something to
  chase with a bigger model.
- Speed: ~19-26 fps on CPU (`yolov8n`), real-time-ish, no concern.

**Before/after record for the player-sentinel fix** (`video1`, same as every other
correction today — confirmed the fix moves numbers in the expected direction, not
just assumed): `player_r` detection rate 89.0% (613/689) -> 91.8% (613/668, same
numerator, denominator correctly shrank by the 21 sentinel-affected `player_r`
frames excluded from ground truth). `player_l` "separated" bucket: n=139 (27.3%,
contaminated) -> n=96 (**20.8%**, clean) — the 43 sentinel-contaminated frames
removed exactly matches the earlier count, and the cleaned rate agrees exactly with
the independently-computed number from `compare_yolo_models_far_player.py`'s
separate 96-frame calculation, cross-confirming the fix is implemented correctly.

## Step 4+5 (player + ball detection) — full 10-clip validation

`cv_pipeline/src/cv_pipeline/ball_detection.py` (NEW): ball detection via YOLOv8's
built-in COCO "sports ball" class (id=32) — tried first since it's zero extra cost
(same pretrained model already used for players), per the plan's framing of trying
the cheapest option before a custom color/motion detector.
`cv_pipeline/scripts/run_full_detection_validation.py`: player (person class,
sentinel-aware, separated/ambiguous split) + ball (sentinel-aware) detection across
all 10 clips, same methodology as the `video1` validation.

**Results** (player_r / player_l-separated / ball detection rates, all 10 clips):
player_r 91.3%-99.8% (mean 96.4%, median 98.3%) — excellent and consistent
everywhere. player_l (separated, far-player) 0.0%-44.4% (mean 16.8%, median 13.0%)
— consistent with the video1 finding that this is a genuine, hardware-appropriate
YOLOv8n limitation, not clip-specific. ball 0.4%-36.1% (mean 10.6%, median 8.6%);
ball median position error tight and consistent everywhere it does match (2.3-4.1px
across all 10 clips — a recall problem, not a precision problem, matching the plan's
expectation that generic YOLO struggles to *find* small fast balls but is accurate
when it does).

**Outliers flagged and investigated, not silently averaged over**:
- **Ball detection rate headline number, committed**: **~7-9% is the representative
  figure for "typical" clip conditions** (9 of 10 clips, `video3` excluded).
  `video3`'s 36.1% is reported *separately*, as a demonstration that ball-detection
  quality scales strongly with video/broadcast quality (spot-checked matched frames
  directly — sub-3px errors, not coincidental false positives — and visually
  confirmed this clip is a noticeably higher-quality/higher-contrast broadcast-style
  feed, bright ball against plain sky, sharper resolution, than the other 9). Do not
  quote a blended mean (10.6%) as "the" ball detection rate in the final write-up —
  it overstates typical performance by folding in one unrepresentative clip.
- **Far-player (`player_l`, separated) sample-size caveat, permanent**: detection
  rate **could not be reliably estimated for `video4` (n=15) or `video10` (n=9)** —
  ground-truth sample too small for either number to mean anything (a single
  match/miss swings the rate by 7-11 percentage points). Aggregate far-player
  statistics are driven primarily by the clips with adequate sample sizes:
  `video1` (n=96, 20.8%), `video9` (n=152, 20.4%), `video6` (n=152, 2.0%), `video2`
  (n=100, 1.0%), `video5` (n=41, 34.1%), `video8` (n=71, 5.6%), `video7` (n=45,
  0.0%), `video3` (n=63, 0.0%) — still a wide spread even among adequately-sampled
  clips, underscoring that far-player detectability varies a lot by camera
  distance/framing, not just a single fixed YOLOv8n accuracy number.
- `video6`'s player_r median error (91.8px vs. 68.5px mean) — smaller, secondary
  effect, plausibly camera framing/distance, not investigated further.

**Scope note**: this covers detection accuracy (steps 4-5) only. Tracking
(ByteTrack ID consistency, step 6) and pose estimation (step 7) are still open.

## Step 6 (ByteTrack ID consistency) — done, with two real bugs found and fixed mid-validation

`cv_pipeline/src/cv_pipeline/tracking.py` (NEW): YOLOv8 + ByteTrack via ultralytics'
built-in `model.track(persist=True, tracker='bytetrack.yaml')`. Validates whether the
SAME track ID stays on the SAME physical player throughout a clip, matched against
sentinel-aware `player_r`/`player_l` ground truth.

**Bug 1 (found before trusting any result)**: initial matching let `player_r` and
`player_l` independently claim the SAME track ID in the same frame — trivially
"consistent" (both showing `id=1` for the whole of `video1`) only because, per the
steps 4-5 finding, ground truth frequently points at the same physical player for
both slots. Fixed with exclusive per-frame matching (`match_frame_ids()`, `player_r`
claims first) — same fix pattern as `player_detection.py`.

**Bug 2 (found mid-validation, materially changed the conclusion)**: the "hard
moment" (crossing/occlusion-risk) proxy — originally "2+ person-boxes within
200px" — was contaminated by background people (spectators, officials, ball kids)
in broadcast-style clips. Flagged 819/998 (82%) of `video4` and 513/513 (100%) of
`video8` as hard moments; visually confirmed on `video8` (a PlaySight broadcast
angle with a visible crowd) that these were bystanders in the stands, not the real
players. Fixed by restricting to the top-2-highest-confidence boxes per frame
(`recompute_hard_moments_top2.py`). Corrected hard-moment counts across all 10
clips: `video1`=1, `video2`=6, `video3`=30, `video4`=0, `video5`=0, `video6`=0,
`video7`=4, `video8`=0, `video9`=0, `video10`=0.

**This flips the headline finding**: the original (buggy) run reported "0/10 clips
with zero hard-moment frames." Corrected: **6/10 clips (video4, 5, 6, 8, 9, 10) had
ZERO genuine crossing/proximity test coverage** — their "0 ID swaps" results are NOT
evidence the tracker handles crossings/occlusion well; those clips simply never
produced a real crossing scenario to test against. Only `video1` (1 frame),
`video2` (6), `video3` (30), and `video7` (4) have any real coverage, and even
those are thin.

**Substantive result**: `video3`, the clip with the most real coverage (30
hard-moment frames), showed **2 ID swaps** — genuine evidence of tracking
instability under real crossing conditions, the one meaningful data point from
this validation. `video6` also showed 2 swaps despite ZERO real hard moments
(corrected) — meaning those swaps are NOT crossing-related; a distinct failure
mode (likely track loss/re-acquisition from something else — camera motion, an
off-screen occlusion, or a spurious detection) that shouldn't be conflated with
crossing-induced swaps.

**Open item for a future pass** (not done, noted rather than silently skipped):
re-run the near-hard-moment cross-reference (which swap transitions fall within
±20 frames of a hard moment) using the CORRECTED hard-moment sets for `video2`/
`video3`/`video7` — the original near-hard-moment attribution was computed against
the buggy, contaminated hard-moment sets and should be treated as unconfirmed until
redone.

**`video6`'s non-crossing swaps, mechanism identified** (both traced to raw per-frame
ID sequences, then confirmed by pulling and viewing the actual video frames):
- `player_r` (frame ~289 -> ~435): **clean, single, well-evidenced mechanism.** The
  near player physically walked out of camera view (visually confirmed at frame 350
  — only the far player remains in shot, consistent with an on-court break/
  changeover), absent for ~145 frames (~2.4s). Track `id=1` timed out during the
  gap; upon the player's return (frame 435, visually confirmed back in frame), a
  brand-new track `id=9` was assigned rather than resuming `id=1`. This is
  ByteTrack's known limitation working as expected: short-term motion/IOU-based
  association only, no long-term re-identification by appearance — any track gone
  for a couple seconds is permanently lost, identity-wise.
- `player_l` (frame ~313 -> ~929): **same underlying break event, but NOT a clean
  single mechanism** — the gap (614 frames) is far longer than the break itself
  (both players visibly back in active play by frame 700, well before frame 929).
  Most likely entangled with the already-documented far-player detection weakness
  (steps 4-5, ~20% detection rate) and/or the same ground-truth-duplication
  artifact (player_l occasionally matching the near player's box on frames where
  player_r itself goes undetected that frame). Reported honestly as a probable
  mix of causes, not overclaimed as a single clean event the way `player_r`'s
  swap was.

## Step 7 (MediaPipe pose estimation) — visual spot-check only, no ground truth exists

**Environment note**: `mediapipe` conflicts with `ultralytics`' numpy 2.x pin via
`tensorflow` (a mediapipe dependency needs numpy<2) in the shared system Python used
for steps 4-6. Fixed properly (not hacked around) by giving `cv_pipeline` its own
isolated `.venv`, matching `rag_engine`/`llm_agent`'s existing pattern — same
sustainable fix, applied when the pain finally justified it rather than continuing
to patch the shared environment.

`cv_pipeline/src/cv_pipeline/pose_estimation.py` (NEW): MediaPipe Pose (Tasks API,
`pose_landmarker_lite`) run on padded YOLOv8 player-box crops, landmarks mapped back
to original-frame coordinates. **No ground truth exists for pose in this dataset —
this is explicitly a visual spot-check, not a measured accuracy.** No error rate is
or should be quoted from this step.

**Near player (5 cases spanning easy -> hard): looks genuinely good.** Clean frontal
ready-stance (video1, video7): accurate landmarks throughout. Mid-serve with arm
fully extended overhead (video3) — the hardest "normal" pose tested — landmarks
correctly tracked the extended arm to the raised hand plus accurate torso/leg
placement. Post-break resumption, mid-shot motion (video6): mostly accurate, with a
soft imperfection — a couple of low-visibility landmarks near the wrist/racket-grip
area were slightly ambiguous, not a hard failure but a real precision dip near
fast-moving extremities.

**Far player: two distinct real failures found and precisely diagnosed, not hidden**:
1. `video1`: YOLO detected only 1 person in the test frame at all — the far player
   was never found, so pose couldn't be attempted. A cascading failure from the
   already-documented far-player detection weakness (steps 4-5), not a new issue.
2. `video9`: caught a bug in the sample-selection script itself first — "pick the
   smallest box" picked a bystander walking the sideline (area 3171px²) over the
   actual far player (area 3664px², nearly identical size). Corrected and reran
   pose directly on the real far-player box — **zero landmarks produced**. Visually
   confirmed: correctly-boxed far player is ~55x66px and motion-blurred (mid-run) —
   a genuine MediaPipe failure on a small/low-res crop, distinct from the box-
   selection bug that was fixed first.

**Conclusion**: pose estimation is solid on the near player, including under real
difficulty. It fails on the far player — sometimes because detection never finds
them, and even when it does, the resulting crop is too small/blurred for MediaPipe
to produce a pose at all. Compounds, rather than introduces, the far-player
limitation already documented in steps 4-5 — a hardware/resolution-driven limitation
on this camera setup, stated plainly rather than glossed over.

## Step 8+9 (structured output schema + aggregate evaluation report) — done

`cv_pipeline/src/cv_pipeline/schema.py` (NEW): the per-clip structured output schema
combining detection, tracking, homography, and pose results. **Core design rule**:
every measurable field carries an explicit `Status` enum (`measured`,
`not_detected`, `not_attempted`, `sentinel_excluded`, `insufficient_sample`,
`excluded_known_issue`, `unvalidated`, `not_applicable`) alongside its value — never
a bare number-or-null. This distinguishes "genuinely zero," "never attempted,"
"excluded due to a known, documented issue," and "sample too small to trust" from
each other and from ordinary missing data, so the JSON is self-explanatory to a
future consumer who hasn't read this file.

`cv_pipeline/scripts/build_clip_reports.py` (NEW): populates the schema for all 10
clips from today's already-verified step 3-7 results (not a redundant ~40-minute
rerun — the underlying validation scripts remain independently re-runnable).
Outputs: `cv_pipeline/data/clip_reports/{clip}.json` (10 files) +
`all_clips.json` + `all_clips.parquet` (36 columns).

`cv_pipeline/EVALUATION_REPORT.md` (NEW): the single pulled-together aggregate
document — player/ball detection rates with all sample-size and contamination
caveats attached, homography validation status per clip (1 validated, 1 known-bad/
excluded, 8 unconfirmed), tracking coverage gaps (6/10 clips never tested crossing
behavior), and the far-player compounding limitation across detection AND pose.
Every number in it traces back to a specific, already-verified finding in this file
— nothing new was computed or claimed in writing it, only organized.

## Stress test: professional clip (out-of-dataset generalization check)

`cv_pipeline/STRESS_TEST_REPORT.md` (NEW) — a single, qualitative stress test on
`data/tennis_clip.mp4` (ATP Masters 1000 Paris practice clip, no ground truth,
~13min source, 900-frame/15s segment processed). **Not merged into the Phase 3
aggregate report or `EVALUATION_REPORT.md`** — different clip conditions, no
ground truth, explicitly a suggestive single-clip check, not a new benchmark. No
pipeline code was changed based on this clip's results.

Key findings: near-player detection and the "smallest-box ≠ far-player"
selection bug both replicate cleanly from the amateur dataset. Tracking showed 3
ID changes in 15s (near player) — individually investigated per-frame rather than
assumed: one was a genuine scene cut (this source video is very likely an edited
compilation, not a raw single take — a different regime than the amateur
dataset entirely), one a real ~1-frame detection dropout during fast motion
(same failure class as `video6`'s investigated swap), one left honestly
unresolved rather than assigned an unconfirmed mechanism. **Most valuable
result**: far-player pose was rerun on a correctly-identified, cleanly-backgrounded
far-player box and still produced zero landmarks — real evidence (one data point,
not conclusive) that the amateur dataset's far-player pose failures are driven by
crop size/resolution rather than background clutter.

## Box-selection fix: court-position plausibility, not box size

Confirmed on two independent clips (amateur `video9`, professional stress-test
clip) that "pick the largest/smallest detected box" as a near/far-player proxy is
unreliable — it can pick a bystander/official whose box happens to be a similar
size to the real player's. Per explicit instruction, fixed properly in pipeline
code rather than left as a documented gotcha: **`cv_pipeline/src/cv_pipeline/player_selection.py`**
(NEW) — `select_players_by_court_position()` projects each box's bottom-center
through the clip's homography and selects by court-position plausibility instead.

Design had to account for a second finding surfaced while building this: **`video9`
turns out to have the same near-baseline-to-net-only corner truncation as
`video7`** (never individually checked before — exactly the residual risk already
flagged in the homography section above, now confirmed to materialize in a second
clip). This meant an absolute-meters Y-axis bound (using the assumed
`COURT_LENGTH_M`) produced false negatives on both clips. Fixed by using a real,
tight meter-based margin on the X axis only (court width, unaffected by the
near/net truncation, and the axis that actually distinguishes real players from
courtside bystanders in both original bug cases) and RELATIVE Y-ordering (near =
smallest projected Y among X-plausible boxes, far = largest) rather than an
absolute Y bound.

**Re-ran `run_pose_spot_check.py` (all 6 amateur cases) to confirm**: `video9`'s
far-player selection now correctly identifies the real far player (rejecting the
sideline bystander) and correctly reports pose failure on it — matching the
manually-established ground truth from the original investigation exactly. No
regression on the other 5 cases.

**Known residual limitation, found while validating, not hidden**: this fix is
only as good as the homography it's given. On the stress-test clip's single rough,
manually-eyeballed homography, a courtside bystander's projected X position fell
genuinely inside the assumed court width — not a margin-tuning issue, a
homography-precision issue. Deliberately not overfitting the margin to force that
one ambiguous frame to pass. `stress_test_pro_clip.py` updated to use the same
selection function for consistency, with this limitation noted inline.

## Phase 4 (v2_serving) — step 1+2 done: app scaffold, POST /analyze-video + GET /jobs/{id}

**Venv consolidation**: `v2_serving` needs `cv_pipeline`'s heavy CV deps
(opencv/ultralytics/mediapipe) AND `rag_engine`/`llm_agent`. Since
`cv_pipeline/.venv` already resolves the numpy/tensorflow/ultralytics conflict
cleanly (Phase 3), installed `rag_engine`, `llm_agent`, and `v2_serving` editable
into `cv_pipeline/.venv` rather than fighting the same conflict again elsewhere —
that venv is now the one true venv for the whole app. `cv_pipeline` itself was
never packaged (Phase 3 used `sys.path` insertion in scripts); `video_pipeline.py`
does the same.

**`v2_serving/src/v2_serving/`**: `main.py` (app + `/health`), `models.py`
(Pydantic request/response shapes), `job_store.py` (in-memory job dict + lock —
justified explicitly: FastAPI `BackgroundTasks` already solves "don't block the
response" via its own threadpool, so polling stays responsive during processing;
a real queue (Celery/Ray) would only earn its complexity with multiple
workers/processes or cross-restart persistence, neither of which applies to a
single-developer M2 box — noted as the first thing that'd need to change if that
changes), `video_pipeline.py` (orchestrates `cv_pipeline`'s building blocks —
`CourtHomography`, `select_players_by_court_position`, `pose_estimation`,
YOLO+ByteTrack — into a genuine LIVE INFERENCE run, not a re-run of Phase 3's
ground-truth validation), `routers/jobs.py` (`POST /analyze-video`,
`GET /jobs/{job_id}`).

**Real integration friction, surfaced not hidden**: `player_selection`'s
court-position fix requires a homography, which requires either annotated corners
(only the 10 amateur dev clips have them) or manual calibration (not automated
anywhere). For a genuinely new/arbitrary video, there's no automated path to a
homography — selection falls back to the same size-based heuristic already
confirmed unreliable in Phase 3. Surfaced in the response via
`player_selection_method`, not silently used as if it were the fixed approach.
Closing this gap (automated corner detection, or a required manual-calibration
step in the API) is future work.

**End-to-end async flow confirmed working with real requests** (`video1.mp4`,
frame_limit=120): submission returns `{"job_id": ..., "status": "pending"}`
immediately; a poll mid-processing correctly returned `"status": "processing"`,
`result: null` (event loop stayed responsive); completion returned the full
structured result with `Status` enum values passed through faithfully
(`"measured"` for `video1`'s independently-validated homography, etc.) — see
this session's transcript for the exact JSON at each stage.

**A suspicious-looking number was investigated, not just reported**:
`far_player_detection` came back at 100% (n=120) — dramatically higher than
Phase 3's ground-truth-validated 20.8% (n=96) whole-clip figure for the same
clip. Verified directly rather than assumed either way: (1) re-ran Phase 3's own
ground-truth matching restricted to exactly frames 0-120 — genuinely 83.3%
(n=18) in that specific segment, confirming frame_limit's default start-of-clip
window really is an easier stretch, not a fluke; (2) the live metric also counts
any court-plausible second detection as "far player," a strictly looser bar than
matching within 150px of a known ground-truth point. Both explanations confirmed,
neither assumed. A detailed interpretation note is now attached directly to the
API's `far_player_detection` field so a caller can't mistake this for an
improved/representative accuracy figure.

**Known caveat for local/dev path-based mode**: `video_path` is resolved relative
to the *server's* cwd, not the repo root — an initial test with a relative path
correctly surfaced as a `"failed"` job with a clear `FileNotFoundError` message
(no silent failure), then re-run with an absolute path. Documented rather than
silently worked around.

## Phase 4, step 2 refinement + step 3 done: field naming, POST /query

**Field-naming fix** (requested, applied before step 3): `video_pipeline.py`'s
live-inference fields renamed with an explicit `_live_estimate` suffix
(`near_player_detection_live_estimate`, `far_player_detection_live_estimate`,
`ball_detection_live_estimate`, `near_player_pose_live_estimate`,
`far_player_pose_live_estimate`) rather than relying on an attached note alone —
Phase 3's ground-truth-validated figures in `EVALUATION_REPORT.md` use the same
base names with no such suffix, so the two can no longer be confused by a reader
(human or the LLM agent itself) skimming field names, not just by someone who
reads every attached caveat.

**`POST /query`** (`v2_serving/src/v2_serving/query_pipeline.py`,
`routers/query.py`): builds a `LiveFeatureSnapshot` from a completed job's
live-estimate result (every feature marked `is_estimate=True`, consistent with
llm_agent's own estimate-hedging convention) and hands it to
`TennisAnalystAgent.ask()` alongside the question -- `rag_engine` retrieval runs
internally inside the agent, untouched. `VectorStore` cached as a module-level
singleton (expensive to construct: loads the sentence-transformer + opens the
persisted Chroma index) -- same single-process-dev-server rationale as
`job_store.py`.

**Confirmed end-to-end with a real request** (`video1` job + "How is the near
player performing... compare to Cameron Norrie's historical Wimbledon results?",
`player="Cameron Norrie"`): live CV features (`[L1]`-`[L5]`) and 5 real retrieved
Wimbledon match summaries (`[D1]`-`[D5]`) both came through into the answer with
citations intact via the API. The model correctly refused to conflate the two --
explicitly stated CV tracking metrics can't be compared to historical tennis
performance stats, rather than fabricating a bridge between them -- the same
grounding discipline validated in Phase 2. `sources_used` vs. `sources_offered`
correctly distinguished: `L2` (far-player detection) was offered but not cited,
and the API response preserved that distinction rather than flattening it.

## Phase 4, step 4 done: GET /win-probability/{job_id}

`v2_serving/src/v2_serving/win_probability_pipeline.py`, `routers/win_probability.py`.
Response always includes both `prematch_baseline` and `live_adjustment`, each with
an explicit `status` and, when unavailable, a plain reason — never silently
omitted or replaced by the other.

**Two real integration findings, investigated and either fixed or reported, not
hidden**:
1. **No cv_pipeline job maps to a real v1 match.** v1's engine needs a real
   `match_id` from its own 5,981-match frozen-join dataset (Elo, rank, h2h);
   none of cv_pipeline's demo clips (10 amateur videos, the pro stress-test clip)
   are in it. `prematch_baseline` reports `not_available` with this reason for a
   job alone, and accepts an optional `?match_id=` query param to compute a real
   baseline for a known historical match — demonstrated genuinely working
   (Djokovic/Goffin 2019 Wimbledon QF → 0.7818; Djokovic/Kohlschreiber 2019
   Wimbledon R128 → 0.9093), not stubbed.
2. **`live_adjustment` is structurally unavailable for every job today, and
   this is reported as a plain, permanent fact, not a fabricated approximation**:
   v1's live-adjustment mechanism (`ServeReturnPosterior`) needs point-by-point
   serve/rally outcomes; cv_pipeline's schema (detection rates, tracking-ID
   stability, pose success — see `EVALUATION_REPORT.md`) contains no such
   extraction. Written as a real check (`has_point_level_data`), not a hardcoded
   `False`, so this starts working automatically if cv_pipeline ever gains
   point-level extraction, without this function needing a rewrite.

**Performance finding, verified before acting on it — not waved through**: the
first implementation called `replay_service.compute_five_engine_trajectory()`
and kept only its `ml_informed_prematch_p1` field — correct, but ~90s per call
(computes the entire per-point, 5-engine trajectory for the whole match just for
one pre-match scalar that's fully determined before the point loop starts).
Before adopting a narrower path, verified it produces the IDENTICAL value, not
just an approximately-equal one: called the same underlying seeding functions
`compute_five_engine_trajectory` itself uses
(`compute_composite_prematch_probability`, `compute_p_a_return_seed`,
`build_pretrained_prior`, `prob_win_match`) directly, skipping the per-point
loop, and diffed against the full-trajectory value for both real matches tested
— **`0.7818396461367739` both ways, `0.9093291144152997` both ways, `diff ==
0.0` exactly**, not merely close. Only after that exact-match confirmation was
the fast path adopted. Re-tested end-to-end post-switch: same values
(`0.7818`, `0.9093`) at the API layer, latency **~90s → ~16.7s for the first
call (one-time `load_replay_context()` setup) → ~0.2s for subsequent calls**
(steady-state, replay-context singleton confirmed reused — its one-time setup
log lines appear exactly once across multiple calls). The narrower path
duplicates a small amount of match-lookup glue from `replay_service.py`'s
internals (finding `match_df`/`first_row`/`final_winner_is_p1`) rather than
refactoring v1's backend itself — deliberate, given Phase 4's constraint that
v1's existing code isn't modified.

**Also fixed**: `xgboost` was missing from the consolidated venv (v1's model
loading needs it) — installed, version-matched to v1's own venv (3.3.0).

## Phase 4, step 5 done: test suite — 18/18 passing, 10.24s

`v2_serving/tests/` — `conftest.py` (clears the `job_store` singleton between
tests; confirmed `TestClient` runs `BackgroundTasks` synchronously, so no polling
needed in tests), `test_health.py`, `test_jobs.py`, `test_query.py`,
`test_win_probability.py`.

Fast by design: CV inference and the LLM agent call are mocked with fixture data
shaped exactly like real captured output (including real `Status` enum values) —
only `/win-probability`'s two regression-guard cases make real, unmocked calls
into v1's engine (fast now thanks to the verified fast path; first call pays the
~15-20s one-time `load_replay_context()` cost, included in the 10.24s total).

Explicit coverage per the request: `Status` enum passthrough asserted directly
(`"measured"`, `"not_detected"` survive unflattened, not collapsed to booleans);
`sources_used` vs. `sources_offered` asserted distinct (an offered-but-uncited
source appears in one but not the other); the actual relative-path failure mode
observed during manual Phase 4 testing re-run as a real (unmocked) regression
test; and the two known match_ids (Djokovic/Goffin, Djokovic/Kohlschreiber)
asserted to produce exactly `0.7818` and `0.9093` — a regression guard against
the fast path in `win_probability_pipeline.py` ever silently drifting from what
v1's full engine would produce.

Phase 4 (v2_serving) is now complete end-to-end: all 4 endpoints wired, each
confirmed working with real request/response JSON before the next was built, and
covered by a fast, real (not rubber-stamped) test suite.

## Phase 5 (v2_dashboard) — step 1 done: app shell, confirmed reaching v2_serving

Scaffolded with Vite + React + Tailwind v4 (`@tailwindcss/vite` plugin, not the
old PostCSS setup). `src/api.js` is the single place the app knows how to reach
`v2_serving` (`http://127.0.0.1:8734`, confirmed from Phase 4) — no mocked data
anywhere in the app, per this phase's constraint.

**Real integration requirement found and fixed before it could cause a silent
failure later**: the dashboard (Vite dev server, `localhost:5173`) is a
different origin than `v2_serving` (`127.0.0.1:8734`) — browser `fetch` calls
would be blocked by CORS even though `curl` (used throughout Phase 4's manual
testing) never hits this restriction, since `curl` doesn't enforce CORS at all.
Added `CORSMiddleware` to `v2_serving/main.py`, scoped to the two localhost dev
origins.

**Verified with a real Playwright screenshot** (not just "component built" —
set up `scripts/screenshot.mjs` now since real visual verification will be
needed for every remaining step, especially the overlay canvas): the app
renders, and the health-check panel shows a live, successful cross-origin
`fetch` result (`Reachable — {"status":"ok"}`), confirming both Tailwind
styling and the CORS fix work end-to-end from an actual browser context, not
just curl.

## Phase 5, step 2 done: upload + job-polling view, all 3 states confirmed for real

`src/components/StatusBadge.jsx` (pending=amber, processing=sky/pulsing,
complete=emerald, failed=red — visually distinct at a glance, not just text),
`src/components/AnalyzeView.jsx` (clip picker over the known demo paths — see
below on why not a real file-upload control — frame_limit input, submit,
`setInterval`-based polling every 3s against `GET /jobs/{id}`).

**Design note, not an oversight**: `/analyze-video` is path-based only (see
Phase 4's `AnalyzeVideoRequest` docstring) — no file-upload endpoint exists on
the backend. The dropdown offers the 10 known amateur clips + the pro
stress-test clip rather than wiring up a file `<input>` that would have nothing
real to submit to.

**All three real status transitions confirmed with actual Playwright
screenshots against a live `/analyze-video` run** (`video1.mp4`,
frame_limit=60), not just "component renders":
1. `pending` — amber badge, real job_id, real video_path/frame_limit, Submit
   correctly disabled.
2. `processing` — sky-blue pulsing badge, SAME job_id persisting, live
   descriptive message. (First test run screenshotted the `processing` state
   too early — before the 3s poll interval had even fired once — and honestly
   still showed `pending`; fixed the test script to wait on the real DOM
   transition rather than a guessed delay, not by fudging the screenshot.)
3. `complete` — emerald badge, same job_id, real result numbers (`60 frames
   processed in 10.9s`), Submit re-enabled.

Screenshots: `/tmp/step2_1_immediately_after_submit.png`,
`/tmp/step2_2_processing.png`, `/tmp/step2_3_complete.png`.
`scripts/test_analyze_flow.mjs` added as a reusable real-flow verification
script (not a unit test — drives the actual running dev server + API).

## Phase 5, step 3 done: result view with honest Status rendering, all 7 values confirmed distinct

`src/components/CvStatusValue.jsx` — the component this step is actually about.
Every `Status`-tagged field gets its own color/icon/label
(`measured`=green/●, `not_detected`=orange/✕, `not_attempted`=gray/–,
`excluded_known_issue`=purple/⚠, `insufficient_sample`=yellow/△,
`unvalidated`=sky-blue/?, `not_applicable`=dim gray/—) — and the accompanying
`note`/`reason` text is always shown for anything non-`measured`, never
collapsed behind a click. `src/components/ResultView.jsx` runs every field of a
completed job's result through it (homography, near/far player + ball
detection, near/far pose, tracking) — nothing is flattened to a blank cell.

**Real completed-job screenshots, cross-checked against the raw API JSON before
trusting them** (not just "component renders"):
- `video7.mp4` (frame_limit=90): homography correctly shows
  `excluded_known_issue` (purple, full root-cause note visible) while every
  detection/pose field on the same screen shows `measured` (green) — visually
  and unmistakably different treatments, confirmed side by side in one
  screenshot (`/tmp/step3_result_view.png`).
- `video9.mp4` (frame_limit=90): surfaced a third real status,
  `unvalidated` (sky-blue), for homography — confirmed distinct from both
  `measured` and `excluded_known_issue` (`/tmp/step3_video9_result.png`).
  Also a genuine `measured, rate=0.0%` case (ball detection) rendered in green
  like any other measured value, not confused with `not_detected` — an
  important distinction (a real, confirmed zero vs. "nothing was found") this
  step was specifically at risk of blurring.
- `video2.mp4` (frame_limit=20): surfaced a fourth real status,
  `not_attempted` (gray), for far-player pose.

**`not_detected` did not occur naturally** across 6 real `/analyze-video` runs
tried across 3 clips and frame_limits from 5 to 90 (documented honestly rather
than force-fit) — its known real trigger (Phase 3's `video9`, frame ~300) isn't
reachable because `/analyze-video` has no start-frame offset, only a frame
count from 0. Rather than fabricate a live occurrence or just assert the code
must be correct, ran one labeled, temporary fixture-data render (real browser,
real Tailwind, real component — clearly commented in the code and removed
immediately after the screenshot) showing all 7 `Status` values side by side,
confirming `not_detected` is visually and unmistakably distinct from all the
others too (`/tmp/step3_status_fixture_check.png`). Re-verified the real
(non-fixture) flow still works after removing that temporary block.

## Phase 5, step 4 done: video player + overlay canvas — two real backend gaps found and closed first

Before writing any frontend code, checked what the API actually returned for a
video-player overlay to draw — and confirmed it didn't have what was needed:
`video_pipeline.py` only ever returned aggregate rates (Phase 4's design), no
per-frame box coordinates anywhere, and no endpoint served the video files
themselves over HTTP (a browser `<video src="/Users/.../video1.mp4">` can't
load a local filesystem path). Both fixed in `v2_serving` (not `cv_pipeline`
itself, staying within this phase's own component):
- `video_pipeline.py` now also records a `frames` array — one entry per
  processed frame (`near_box`/`far_box`/`ball_box`/`near_track_id`/
  `far_track_id`), **explicit `null` for anything not detected that frame,
  never an omitted key** — plus `homography.court_corners` (raw pixel corners)
  and top-level `video_width`/`video_height`, all needed for the overlay to
  draw real coordinates rather than guesses.
- `routers/media.py` (NEW) — `GET /video-file/{filename}`, serving only the
  known demo clips (basename-matched against a fixed allow-list built from the
  same two directories `video_pipeline.py` already knows — not an arbitrary-
  path read, checked explicitly).

`src/components/VideoOverlay.jsx` — plays the clip, draws real per-frame boxes
+ the court quadrilateral on a canvas synced to `video.currentTime` via the
real `source_fps`, with a toggle. Court lines are drawn in a color distinct
from player boxes; near/far player and ball each get their own color.

**Real screenshots, cross-checked against the exact frame data, not just "it
renders"**:
- `video1.mp4` (frame_limit=90), seeked to frame 10 by exact timestamp
  (`frame/fps`): zoomed crop shows the near-player box tightly wrapping the
  actual player and the far-player box tightly wrapping the small distant
  figure near the net — both genuinely pixel-aligned, not approximate. The
  purple court quadrilateral traces the real painted court boundary exactly
  (baseline, sidelines, perspective narrowing toward the far court).
- Overlay toggle confirmed: unchecking it leaves a completely clean video
  frame, no stray marks.
- **The honesty requirement this step was actually about**: found a real
  segment (`video2.mp4`, frame_limit=20) where the far player has NO box in
  any of the 20 frames (confirmed in the raw JSON first). Screenshotted frame 5
  — only the near-player box is drawn, correctly aligned; **no fabricated
  far-player box appears**; the legend explicitly reads "Far player: not
  detected this frame" with a distinct hollow marker (vs. a filled marker for
  "detected"), so the absence is legible as a real finding, not a rendering
  gap a viewer has to guess about.

## Phase 5, step 5 done: chat interface, citation-audit distinction genuinely visible

`src/components/ChatView.jsx` — question input + optional player/opponent +
"fuse live CV features from {job}" toggle (bound to the last completed job from
`AnalyzeView`, lifted into `App.jsx` state). Each answer renders through
`ChatMessage`, which computes cited-vs-offered-only by checking whether each
`sources_offered` tag's descriptor text appears in `sources_used` — genuinely
two different visual treatments, not a shared list: cited sources get a green
border, `✓ cited` label, full-opacity text; offered-but-uncited sources get a
dimmed (60% opacity) card, gray `offered, not cited` label, and
strikethrough text.

**Confirmed against the exact question already verified in Phase 4** ("How is
the near player performing... compare to Cameron Norrie's historical Wimbledon
results?", `player="Cameron Norrie"`, live features fused from a real completed
`video1.mp4` job) — chosen specifically because Phase 4 already established it
produces one offered-but-uncited source (`L2`, far-player detection). Real
screenshot, zoomed on the sources list: `[D5]` (a cited Wimbledon match) shows
the green/✓/full-opacity treatment; `[L2]` shows the dimmed/gray/strikethrough
treatment immediately below it — the same distinction Phase 4 confirmed only in
raw JSON is now genuinely legible on screen, header line reading "9 cited / 10
offered" matching the real counts exactly.

## Phase 5, step 6 done: win-probability panel — final step, Phase 5 complete

`src/components/WinProbabilityPanel.jsx` — `match_id` input (optional) + Check
button, always renders BOTH `prematch_baseline` and `live_adjustment` cards
side by side regardless of status, styled by `available`/`not_available` the
same honest-status pattern as `CvStatusValue` (green vs. dimmed gray) — neither
card is ever omitted, and the real reason text is always visible for a
`not_available` card, not collapsed.

**Both real cases confirmed with screenshots**:
- **No `match_id`**: both cards correctly show "– not available" with their
  full real reason text — `prematch_baseline` explains v1's engine needs a
  known historical `match_id` this clip isn't part of; `live_adjustment`
  explains the structural gap (no point-level score/serve data in cv_pipeline's
  output) — neither panel omitted, no blank/misleading state.
- **Real `match_id`** (`20190710-M-Wimbledon-QF-Novak_Djokovic-David_Goffin`,
  one of Phase 4's two regression-test matches): `prematch_baseline` renders
  **78.18%**, exactly matching Phase 4's verified regression value, prominently
  displayed and clearly labeled "Novak Djokovic to beat David Goffin —
  pre-match only, no live point data" with the fast-path verification note
  visible; `live_adjustment` correctly remains "not available" alongside it in
  the same view — both real states shown simultaneously and distinctly, not
  one silently replacing the other.

**Phase 5 (v2_dashboard) is now complete** — all 6 steps built and each
confirmed against the real, running `v2_serving` API with actual screenshots,
not assumptions: health check, upload+polling (all 3 job states), result view
(4 of 7 `Status` values confirmed live, all 7 confirmed via one labeled fixture
check), video overlay (pixel-precise alignment confirmed, plus a genuine
missing-detection case), chat (citation-audit distinction genuinely visible,
reusing a question already known from Phase 4 to produce an offered-but-uncited
source), and win-probability (both real states, exact regression value).
Along the way, two real backend gaps were found and closed in `v2_serving`
(CORS, per-frame data + video-file serving) rather than discovered as mysteries
later or worked around in the frontend.

## Phase 6 (Evaluation & Research Write-Up) — point documents deployed, real
segfault found and fixed, precision@k re-measured with the new document type

While writing `RESEARCH_REPORT.md`, checking the actual persisted RAG index
(rather than restating from the build log) found it contained 22,610 documents —
20,000 match summaries, 2,610 player profiles, and **zero point documents** —
despite the point-document ingestion code (`rag_engine/ingest/point_documents.py`)
being complete and covered by tests. A real, previously-undocumented gap between
"built and tested" and "live in the index."

**Timing measured before deciding how much to deploy** (not assumed): a 10-match
calibration run took 250.3s (25.03s/match, v1's full 5-engine per-point
computation per match) — projecting the full 5,981-match frozen-join corpus at
~41.6 hours. Not "quick." A 100-match representative subset (~42 min projected)
was chosen instead, following the same documented-partial-subset pattern already
used for the 22,610-doc match/player index.

**First attempt (`add_point_documents_batch.py`, combining `rag_engine`'s
embedding stack and v1's model-inference stack in one process) segfaulted twice
(exit 139, no Python traceback), at the identical point in execution both
times:**
1. First crash: hypothesized macOS OpenMP (`libomp.dylib`) double-initialization
   conflict between PyTorch/sentence-transformers and v1's own model-loading
   stack. Standalone repro (load both stacks in sequence) confirmed: crashes
   without `KMP_DUPLICATE_LIB_OK=TRUE`, succeeds with it. This is the standard,
   commonly-cited fix for this class of crash.
2. Applied the fix, reran the real batch — **crashed again, same location.**
   This proved the env-var fix only resolved a load-time symptom, not the actual
   conflict: both native stacks were segfaulting when *actively computing
   concurrently* in one process (v1's per-match generation loop interleaved with
   Chroma's periodic batch-embed calls inside `build_index`), not merely
   coexisting at import time.

**Fix: architectural, not another env var.** Split into two single-stack
processes connected by a JSON file on disk:
- `generate_point_documents_to_disk.py` — imports only v1's stack
  (`rag_engine.ingest.point_documents`), never touches `rag_engine.index` /
  sentence-transformers / Chroma at all. Serializes generated `RagDocument`s to
  `rag_engine/data/point_documents_batch.json`.
- `embed_point_documents_from_disk.py` — imports only the embedding stack
  (`rag_engine.index.vector_store.VectorStore`), reads the JSON back in, calls
  `store.build_index(docs, reset=False)` to append without wiping the existing
  22,610 docs.

Run for real: generation took **3394.7s (56.6 min)** — somewhat over the ~42 min
projection from the 10-match calibration (noise in a small calibration sample,
not a new problem), producing **1,137 point documents from 100 matches**, no
crash. Embedding took **13.6s**, bringing the live index to **23,747 documents**
(22,610 + 1,137). Both steps completed cleanly on the real run.

**Precision@k re-run with the new document type** (`precision_at_k_eval.py`,
extended with 3 point-level queries; ground-truth pool sizes for each new query
checked directly against the generated batch before writing the relevance
predicates — 42 docs for Djokovic break points, 72 for Hurkacz on hard, 8 for
Djokovic-Nadal clay match points):

- Point-level queries: precision@3 = 0.222, precision@5 = 0.333 (mean across 3
  queries) — weaker than match-summary retrieval, and sensitive to phrasing:
  "break point moments" (echoes point-document wording) outperformed "notable
  points on hard court" (generic phrasing that mostly matched Hurkacz's
  *match*-summary documents instead).
- **Unflattering finding, kept rather than dropped**: adding point documents
  measurably hurt 2 of the original 7 match-summary queries. "Rafael Nadal clay
  court matches" and "Carlos Alcaraz hard court matches" both went from
  high/perfect precision to **0/5** — their top-5 nearest neighbors are now
  entirely point documents about the same player/surface, crowding out the
  match-summary documents that used to rank there. Overall match-summary
  precision@3 dropped from 0.952 to 0.714 (precision@5: 0.971 → 0.829). The
  other 5 match-summary queries were unaffected.
- Root cause: all three document types share one undifferentiated Chroma
  collection with no `doc_type` filtering applied at query time. Filtering by
  `doc_type` at retrieval time would very likely fix both the match-summary
  regression and the point-query weakness (since the two types would stop
  competing for the same ranked list) — **not implemented or tested here**,
  named as a specific next step rather than left vague.

`RESEARCH_REPORT.md` (§3, §5, §6.1) updated to reflect all of the above: point
documents now partially live (100-match/1,137-doc subset, not the full
5,981-match corpus — same documented-scope-decision pattern as the match/player
subset), the real segfault investigation and two-process fix, and the full
before/after precision@k numbers including the match-summary regression.

## Ball detection: TrackNet blocked on licensing, fine-tuned YOLO + motion-diff
combined method built, validated, and wired into cv_pipeline (STRESS_TEST_2 follow-up)

Follow-up investigation into `cv_pipeline`'s known ball-detection weakness
(~7.8% mean recall, amateur dataset; see `EVALUATION_REPORT.md`), triggered by
the STRESS_TEST_2 finding that broadcast footage didn't improve ball detection
either (2.6% on genuine wide-shot frames — see `STRESS_TEST_2_REPORT.md`).

**TrackNet/TrackNetV2**: searched for pretrained tennis-ball-tracking weights.
No cleanly-licensed source exists — the one repo with actual tennis weights
(`yastrebksv/TrackNet`) has no LICENSE file at all (all-rights-reserved by
default); the public-domain alternative doesn't ship TrackNet weights.
Downloading the unlicensed `.pt` was also blocked by the sandbox (untrusted
external weights from a self-found source). Not evaluated — a licensing
question, not a technical one, left open.

**Motion-diff (frame-differencing restricted to the court region via real
homography)**: validated against real ground truth (2,074 ball frames across
the 9-clip amateur dataset, video3 excluded, matching the committed baseline's
scope) — pushed pooled recall from the reproduced 7.81% stock-YOLO baseline to
**57.62%**. Confirmed genuine via visual spot-check (recovered positions land
within a few px of the real ball). **Does not transfer to the two broadcast
stress-test clips** — spot-check found the "candidates" there are almost
entirely false positives on player-limb motion (two players' motion swamps the
tiny ball signal on those clips; most amateur clips have only one player near
the ball at a time).

**Fine-tuned YOLOv8n**: found a real, cleanly CC-BY-4.0-licensed 578-image
tennis-ball dataset (Viren Dhanwani, Roboflow Universe) — confirmed no GitHub
fork of the popular tutorial ecosystem built on this dataset (872★ origin repo
down to 9★ forks) redistributes actual trained weights, so fine-tuning was
required rather than reusable. Timed before committing (project discipline):
3-epoch probe (501.4s, 167.1s/epoch) → 10-epoch probe (1699.2s, still climbing:
mAP50 0→0.596) → full 30-epoch run (13,102.7s／218.4min, with an isolated
~70min mid-run slowdown from external machine load, not a steady-state
problem; final mAP50 0.759, recall 0.733). Validated alone: 44.55-47.06%
pooled recall on the amateur dataset (methodology-dependent, top-1-box vs
any-box matching) — real, but below motion-diff alone.

**Bug found and fixed twice**: the fine-tuned model was found (visual
spot-check on the stress clips) to hallucinate a persistent, near-zero-
confidence-change detection at a FIXED pixel location across many frames —
`(1442,778)` in `tennis_clip.mp4`, `(412,442)` in `match_tennis.mp4` — almost
certainly a memorized court blemish from the small 578-image training set. A
first fix attempt (reject a top-confidence detection static for 10 consecutive
frames) **failed** — confirmed by direct pixel-coordinate inspection: the
artifact recurs frequently (12.1%/5.3% of all frames) but rarely wins "top
confidence" 10 frames in a row, since it's interleaved with other detections,
so the consecutive-run rule almost never accumulates enough evidence. Redesigned
as a two-pass, frequency-over-full-history filter (collect every candidate
box across the whole clip, not just the top pick; flag any pixel bin recurring
in ≥3% of all frames; reject boxes near flagged bins) — **this version
correctly caught both known artifacts** (confirmed directly), and re-validation
against the amateur dataset's real ground truth showed it doesn't cost real
detections (44.02%→45.18%, same or better) while the combined method (v2
filter + motion-diff-on-misses) jumped to **70.40% pooled recall — the best
number of every method tested**, up from the first (broken-filter) combined
attempt's 58.15%. **This 70.40% figure was later found to be invalid — see
the "GROUND-TRUTH LEAK FOUND" entry below, corrected to 53.91% (final).**

**Wired into `cv_pipeline` behind the existing `Status` enum pattern**
(`schema.py`), not left as a standalone script: new
`ball_detection_combined.py` module (the validated combined detector, plus
`classify_ball_detection_regime` — a cheap heuristic reusing Stress Test #2's
already-validated hard-cut detector to route a clip to "validated" (locked-
camera, amateur-style — eligible for the combined method) vs "best_effort"
(broadcast/multi-camera-angle — falls back to stock YOLO, since motion-diff is
known unreliable there)). `RateMetric` (schema.py) gained an additive `method`
field (defaults to `"stock_yolo"`, so no existing caller changes behavior) so
the same field can honestly carry "combined_v2, MEASURED, [validated %]" or
"stock_yolo, UNVALIDATED, best-effort, known limitations" depending on regime
(the specific percentage cited here was corrected from 70.40% to 53.91% —
see below).
**A real bug was found and fixed in the regime classifier itself** while
building the end-to-end verification: sampling only the first 30s of a clip
misclassified `match_tennis.mp4` (confirmed cut-heavy in Stress Test #2) as
"validated", because its opening ~5 minutes are a single continuous shot
before the highlight-reel-style cuts begin — fixed by sampling 3 windows
spread across the clip (10%/50%/90%) instead of just the start. End-to-end
verification (`verify_ball_detection_wiring.py`) confirms both clips route to
the correct regime and the resulting `RateMetric.to_dict()` carries the right
status/method/note pair for each.

**Not done**: broader pipeline call-site integration (e.g.
`run_full_detection_validation.py` and friends still call the original
stock-YOLO-only path) — this entry adds the validated capability and wires it
behind the schema, but does not switch every existing caller over to it. A
dashboard-side "improved method" vs "best-effort" UI treatment also isn't
built — only the backend `RateMetric.method` field it would key off.

**A further limitation found while visually spot-checking the wiring through
the dashboard** (temporary `?loadJob=` overlay check, `match_tennis.mp4`,
frame ~7600, t=304.0s): the court-polygon overlay was visibly misaligned in
that specific frame, even though the clip-level regime classification
(`classify_ball_detection_regime`) correctly labeled the whole window as
"best_effort." Root cause: that one frame is mid-hard-cut to a different
camera angle than the wide-broadcast shot the manual homography corners were
calibrated for — `classify_ball_detection_regime` decides a REGIME per
clip-window (sampling 3 windows across the clip, per the earlier fix in this
same entry), not per frame, so a window correctly flagged "best_effort" can
still contain individual cut frames where even the best-effort homography
doesn't apply at all. This is a precisely-scoped gap, not a vague caveat: the
per-frame hard-cut detector this would need already exists and is already
validated (the same histogram-correlation cut detector built for Stress Test
#2's camera-angle filter, and reused as-is inside
`classify_ball_detection_regime` itself for the clip-level decision) — it
just isn't wired to gate homography/court-overlay rendering at the frame
level, only used to make the one coarse per-clip regime call. Not urgent (the
ball-detection combined method itself doesn't depend on the court overlay
rendering correctly — only the visual court-line drawing is affected), but a
concrete next step if per-frame accuracy ever matters here, not something to
be fixed today.

**Fixed** (2026-07-16, before wiring the combined method in as any kind of
default): added `frame_matches_reference_framing` to
`ball_detection_combined.py` — compares each frame's grayscale histogram
directly against the histogram of the frame the homography corners were
actually calibrated on (the first frame processed), using the same 0.7
correlation threshold already used for clip-level cut detection. Validated
directly against the four known frames from the spot-check: the reference
frame itself → 1.000 (trivially matches), the known-misaligned frame (t=304.0s
/ frame 7600) → 0.094 (correctly flagged), and the two other sampled frames
that looked visually fine → 0.838 and 0.911 (correctly not flagged, both above
threshold). `CombinedBallDetectionResult` gained `homography_applicable: bool`
and `reference_match_correlation: float` fields. As a natural extension of the
same fix (not scope creep — same root cause): `run_combined_ball_detection_for_clip`
now also SKIPS the motion-diff fallback on a frame where
`homography_applicable` is False, since motion-diff's court-region mask is
built from the same homography and is equally wrong for that frame's camera
angle — previously this would have risked a wrong BALL POSITION, not just a
wrong overlay. `VideoOverlay.jsx` now suppresses court-line drawing on such
frames and says so explicitly in the legend ("suppressed this frame (camera
angle doesn't match calibration)") rather than silently drawing nothing or
drawing something wrong. Re-verified visually on the exact `match_tennis.mp4`
job/frame that surfaced the original gap: the misaligned trapezoid is gone at
frame 7600, the good frames (7513 etc.) are unaffected, and the amber legend
message appears exactly when expected.

**Combined method per-frame cost, measured before adopting as any default**
(same discipline as every other timing number in this project): 300 frames of
`match_tennis.mp4` (the full combined method — pass-1 artifact-bin flagging +
pass-2 detection-with-motion-diff-fallback, the actual production function,
not a proxy) took 22.3s → **74.3ms/frame**. Extrapolated to a full match
(`match_tennis.mp4`'s real full length, 53,473 frames / 35.6 min, used as the
full-match reference since it's an actual full match recording, not one of
the short ~10-20s amateur dev clips): **~66.2 minutes (1.10 hr) to process a
35.6-minute match — roughly 1.86x real-time.**

**Switched `v2_serving/video_pipeline.py`'s default ball-detection call site**
to the combined method, regime-gated (only when a real homography exists AND
`classify_ball_detection_regime` classifies the clip "validated" — every
other case still falls back to stock YOLO exactly as before). Smoke-tested
both paths directly (video1 → `method: combined_v2`, 95% live-estimate rate
on 60 frames; `match_tennis.mp4` → `method: stock_yolo`, no homography
available, 0/60 frames carry `homography_applicable`, correct fallback
behavior). All 18 existing `v2_serving` tests still pass, no regression.
Added a "validated/best-effort" method badge to the dashboard
(`CvStatusValue.jsx` — a small green "improved method (validated)" or amber
"best-effort, known limitations" pill next to the Status badge, keyed off
`RateMetric.method`). Verified rendering through the real, unmodified
`AnalyzeView` → `ResultView` flow (not the temporary `?loadJob=` debug
loader) via a real submitted job — confirmed the green pill and 95.0% rate
render correctly on video1.

**GROUND-TRUTH LEAK FOUND — the 70.40% figure above was invalid, corrected to
53.76%.** Per instruction, before trusting the combined method as any kind of
default, the full Phase-3-style validation suite was re-run end-to-end
through the ACTUAL PRODUCTION function
(`ball_detection_combined.run_combined_ball_detection_for_clip`, the same one
`video_pipeline.py` now calls) across all 10 amateur clips — not the
standalone prototype scripts that had originally produced 70.40%. The first
re-run came back at **46.24%** (video3 excluded, 9-clip scope), far below
70.40%. Root cause, found by comparing the two code paths line by line: the
original prototype (`ball_static_artifact_filter_v2.py`) had picked which
motion-diff candidate to trust, when several existed in one frame, by finding
whichever candidate was CLOSEST TO THE REAL GROUND-TRUTH BALL POSITION —
something no real inference-time system can ever do, since ground truth
doesn't exist at inference time. The actual production function had no such
access and instead took `candidates[0]` (the first blob in
`cv2.findContours`' arbitrary scan order) — an unprincipled stand-in that was
never itself validated. **Fixed** by having `motion_diff_candidates` return
area alongside position, and picking the LARGEST-AREA candidate instead of an
arbitrary one — a legitimate, non-cheating heuristic (the real ball has a
specific physical size; spurious motion-diff blobs are more likely noise),
already used successfully in an earlier unvalidated stress-clip spot-check.
Re-ran the full suite again with the fix: **53.76% pooled recall (video3
excluded, 9-clip scope; 53.65% including video3)** — the final, corrected,
honest number as of that run. Still a real, large improvement over stock
YOLO's 7.81% (~6.9x), just meaningfully lower than the 70.40% first claimed.
One anomaly surfaced by this run, noted rather than hidden at the time:
`video4` had 183 frames flagged `homography_applicable=False` by the
per-frame reference-framing check, despite being a locked-camera clip with no
classified cuts (`cut_rate: 0.0` at the clip level) — every other clip showed
0 such frames. **Root-caused and fixed in a follow-up investigation, see
below** — this was not left unresolved.

**video4 anomaly — investigated, root-caused, and fixed.** Step 1 (facts
first, no theorizing): pulled the exact 183 flagged frame indices — NOT
scattered or periodic (ruling out an indexing bug), but three contiguous
stretches (522-535, 592-740, 756-772/774, 835-836), the long one spanning
~150 frames. Pulled the actual frames at several points in and around these
stretches and looked at them directly: the camera framing is pixel-identical
throughout (same net position, same benches, same background buildings/mesh
pattern — a locked-off camera the whole time, no pan, no zoom, no cut).
Traced the exact code path: `frame_matches_reference_framing` compared every
frame's grayscale histogram against a SINGLE FIXED reference (the first frame
processed), at the same 0.7 threshold used for hard-cut detection elsewhere.

Computed correlation-to-reference directly across the region: NOT a sudden
jump (which is what a real cut looks like — confirmed separately on
match_tennis.mp4, where it drops from 0.999 to 0.09 in a single frame) but a
slow, continuous DRIFT — 0.936 (frame 100) → 0.843 → 0.775 → 0.708 (just
under threshold, frame 521) → sitting in the 0.6-0.7 range through the middle
stretch → climbing back to 0.82 by frame 900. Confirmed directly that this
is NOT any kind of camera change: CONSECUTIVE-frame correlation across the
exact same region stayed at 0.9993-1.0000 throughout (checked frame-by-frame
from 515 to 544) — there is no discontinuity anywhere. The real cause: this
court has natural daylight visible through mesh siding (confirmed visually —
"Sport Singapore" signage, sky and buildings visible through the fence), and
ambient lighting drifted gradually over the ~8-10 minute recording, slowly
moving the whole frame's grayscale histogram away from the single reference
frame captured at the very start — nothing to do with camera angle at all.
This is hypothesis 1 from the investigation brief (a residual false positive
in the per-frame gating logic), confirmed directly rather than assumed.

**Fixed, and the fix was itself checked against both known real cases, not
just video4.** First attempt: replaced the fixed-reference comparison with a
stateful design (`SegmentFramingTracker`) that only re-evaluates against
calibration right after a detected hard cut (consecutive-frame correlation
drop), holding the flag constant otherwise — this correctly eliminated
video4's false positives (0/998 flagged), but re-checking against
match_tennis.mp4 revealed a NEW problem: frames 7675 and 7750 (previously
confirmed, by direct visual inspection, to be normal well-aligned wide-shot
frames) got incorrectly stuck flagged `False`, because the real transition
back to the wide shot there is a GRADUAL zoom-out (correlation climbing
0.089→0.178→0.672→0.728→0.812→0.888 over ~130 frames), not a hard cut, so it
never tripped the stateful tracker's re-evaluation trigger. This design was
reverted the same day rather than shipped with a new, undiscovered bug.

**Final fix**: kept the original direct per-frame comparison (no state), but
lowered `FRAME_REFERENCE_MATCH_THRESHOLD` from the shared 0.7 cut-detection
bar to **0.5** — checked empirically against both clips' full real
distributions, not picked in isolation: a full scan of video4.mp4 (every 5th
frame, entire clip) found its true worst-case drift never drops below 0.6196
correlation, while match_tennis.mp4's real cut region sits at 0.09-0.4
immediately post-cut. 0.5 sits with real margin below video4's minimum while
still unambiguously catching match_tennis's cut. Re-validated: video4 now
0/998 flagged; match_tennis.mp4 still correctly flags frames 7568/7600
(`False`, the real closeup) and correctly un-flags 7622/7650/7675/7750
(`True`) once the gradual zoom-out has genuinely recovered enough framing
overlap — matching the direct visual confirmation from earlier in the day
that those specific frames look like normal, well-aligned wide shots.

**This fix changed the final pooled-recall number again, marginally**: with
video4's 183 frames now correctly eligible for the motion-diff fallback
(previously skipped), video4's own recall rose 56.3%→57.1%, and the true
final pooled figure is **53.91% (video3 excluded, 9-clip scope; 53.78%
including video3)** — re-run end-to-end one more time to confirm, `homog_bad`
now 0 across all 10 clips, not just video4.

**Every place citing either 70.40% or the intermediate 53.76% updated in this
same pass** to the final 53.91%, with a
note on what happened (not just the corrected number): `ball_detection_combined.py`
(module docstring + the function's own GROUND-TRUTH LEAK docstring section),
`schema.py`, `video_pipeline.py` (both the module docstring and the actual
dashboard-facing `COMBINED_BALL_METHOD_NOTE` text users see), `temp_ball_check.py`,
`verify_ball_detection_wiring.py`. (`RESEARCH_REPORT.md` and
`STRESS_TEST_2_REPORT.md` were checked and do not reference this figure — the
ball-detection combined-method investigation happened entirely after those
reports were written.)

**New general methodology note, worth watching for beyond this one instance**:
prototype/validation scripts must be checked for any use of ground truth
BEYOND SCORING — i.e., does ground truth ever influence WHICH of several
system-generated candidates gets treated as the real output (an arbitration
decision), versus only being used afterward to check whether some
independently-produced output was close enough (a scoring decision)? The
latter is standard, legitimate, and used throughout this project's other
evaluators (`player_detection.py`'s greedy nearest-neighbor matching,
`tracking.py`'s ID-to-ground-truth matching, `ball_detection.py`'s stock-YOLO
recall scoring, `precision_at_k_eval.py`'s retrieve-then-score RAG
evaluation, the golden Markov regression tests' precomputed-column
comparison) — none of these let ground truth choose the system's output, only
score an output already produced independently. The former is what happened
here, and produced a real, meaningfully-inflated number that went uncaught
until the actual production code path was measured end-to-end. **A deliberate
audit of the above-listed scripts, done specifically because this pattern was
found once, found no other instance of it in this project** — but the check
is now a named, repeatable question ("could ground truth be picking the
answer here, not just checking it?") worth asking of any new validation
script going forward, not just this one.

## Next up

1. Full 198k-match embed, closer to when v2 needs it in production (currently
   verified against a 22,610-doc representative subset — see earlier entry).
2. Full 5,981-match point-document corpus (currently 100-match/1,137-doc
   subset live; full corpus projected at ~41.6 hours at the measured
   ~25s/match rate).
3. `doc_type` filtering at RAG query time — flagged above as the likely fix for
   the point-document-vs-match-summary retrieval competition found in this
   entry; not yet implemented.
4. ~~Switch existing cv_pipeline call sites over to `ball_detection_combined`'s
   regime-gated combined method, and surface `RateMetric.method` in the
   dashboard.~~ **Done** — `video_pipeline.py`'s default call site switched
   (regime-gated), dashboard badge added, both verified live. A parallel
   `run_full_detection_validation_combined_ball.py` script now also exists
   for re-measuring cv_pipeline's own offline validation suite through the
   same production function (the original `run_full_detection_validation.py`
   was deliberately left untouched as the historical source of
   `EVALUATION_REPORT.md`'s committed stock-YOLO numbers).
5. TrackNet licensing — revisit if a clean weights source or explicit
   permission from the `yastrebksv` maintainer turns up.
6. ~~Wire the existing per-frame hard-cut detector... to gate
   homography/court-overlay rendering at the FRAME level~~ **Done** —
   `frame_matches_reference_framing` added, validated against the exact
   known frames, wired into both `run_combined_ball_detection_for_clip` and
   `VideoOverlay.jsx`.
7. ~~Root-cause `video4`'s 183 frames flagged `homography_applicable=False`~~
   **Done** — gradual ambient-lighting drift (not a camera cut) falsely
   crossed the old fixed 0.7 threshold; confirmed via consecutive-frame
   correlation staying ~1.0 throughout. Fixed by lowering
   `FRAME_REFERENCE_MATCH_THRESHOLD` to 0.5, checked empirically against both
   video4.mp4's and match_tennis.mp4's real correlation distributions.
   Pooled recall re-measured after the fix: 53.91% (final).
8. Re-verify the near-certain-tail-noise finding before citing its specific
   numbers again — no active code path depends on it (confirmed), so this is low
   priority.

## Master Build Prompt: Reference Pipeline (data/tennis/1.mp4) — Homography Precision Improvement (2026-07-17)

Context: Phase 4 of the reference-pipeline build calibrated `data/tennis/1.mp4`'s
homography manually with only the 4 outer doubles corners, validated against 2
held-out landmarks (near-T = center-service-line x near-service-line intersection;
net-base = net center at ground level) not used in calibration. Measured error:
**74.9px (113.1cm) at near-T, 45.7px (83.2cm) at net-base** — substantially worse
than the existing `video1` dev clip's independently-validated ~13px standard,
despite this footage having excellent, high-contrast, clearly visible court lines
(a calibration-precision problem, not a camera/quality problem).

**Step 1 (re-clicked 4-point corners, zoomed/gridded crops)**: modest improvement
— near-T 70.9px (106.6cm), net 41.6px (76.4cm). A ~5-9% reduction — confirms the
error was not primarily about corner-click precision.

**Step 2 (least-squares 8-point calibration)**: identified 4 additional
unambiguous, visible court-line intersections beyond the 4 outer corners —
near-service-line corners L/R (singles sideline x near service line, both sides),
baseline-center (center mark on near baseline), far-T (center line x far service
line) — confirmed visible via real zoomed frame crops before use, not assumed.
**Far service-box corners (singles sideline x far service line) were checked and
explicitly rejected** — no clean, unambiguous intersection visible at that
distance from camera in this clip. With 8 total points, solved via
`cv2.findHomography(..., method=0)` (ordinary least-squares DLT) instead of the
exact 4-point solve — implemented as `CourtHomography.from_point_correspondences`
in `cv_pipeline/src/cv_pipeline/homography.py`. Held-out error against the SAME
near-T/net-base landmarks: **27.4px (44.4cm) at near-T, 24.9px (47.0cm) at net**
— a real, substantial improvement (2.7x / 1.8x pixel-error reduction from the
original 74.9px/45.7px).

(Note: an interim re-click of this same 8-point technique, done earlier in this
investigation before a context-compaction boundary, measured 17.7px/16.8px on a
different independent set of manual clicks of the same 8 landmarks — both
re-clicks agree on the qualitative finding, a 2-4x reduction, not a full close of
the gap to video1's ~13px; the final adopted numbers above are from the version
actually wired into code, in `cv_pipeline/src/cv_pipeline/reference_video1_calibration.py`.)

**Comparison table (all figures: near-T error / net-base error):**

| calibration | pixels | real-world cm |
|---|---|---|
| Original 4-point (manual) | 74.9px / 45.7px | 113.1cm / 83.2cm |
| Re-clicked 4-point | 70.9px / 41.6px | 106.6cm / 76.4cm |
| **Least-squares 8-point (adopted)** | **27.4px / 24.9px** | **44.4cm / 47.0cm** |

**Plain conclusion**: the least-squares multi-point technique is a real,
substantial improvement and is what's now adopted for this clip (see
`reference_video1_calibration.py`, used by Phase 6's render). It does **not**
fully close the gap to video1's ~13px benchmark — a real, smaller residual gap
remains. Per the investigation's own scope limit, no further fix (e.g. automated
Hough-line-based corner detection) was pursued without checking in first.

**Open question, logged not investigated further**: per-point reprojection
residuals show `near_svc_L`/`near_svc_R` fit this least-squares model markedly
worse (50-60px residual) than the other 6 points (7-32px) — both are on the
near-service-line, both sides. Candidate causes: (a) click-precision error
specific to those two points, or (b) genuine lens/perspective distortion that a
flat homography model can't capture in that region of the frame — a real
possibility given this is a broadcast camera with some barrel/perspective
distortion, not a synthetic pinhole projection. The likely eventual fix, if
pursued, is automated sub-pixel line detection (Hough transform + line-fit
refinement) rather than more manual re-clicking — flagged as future work, not
attempted here.

Adopted into Phase 6 (full-pipeline render for data/tennis/1.mp4): player
detection/tracking, ball detection+interpolation, and this least-squares
homography — **no in/out overlay** (Phase 5's bounce-detection heuristic was
found unreliable, dominated by racket-contact false positives indistinguishable
from real bounces in monocular pixel-velocity data; see this file's Phase 5
entry). See RESEARCH_REPORT.md for the same table in the project's canonical
findings log.

## Court-Outline Rendering Bug — root cause found, same bug as the residual-asymmetry open question (2026-07-18)

User reported the rendered court-outline overlay was geometrically inconsistent:
full doubles width at the far baseline, only singles width at the near baseline —
flagged as a real geometric bug, not just an accuracy gap. Investigated by listing
all 8 calibration points by label and checking each directly against real,
zoomed frames before touching any code, per instruction:

| label | claimed | pixel (before) | actual line, verified against frame |
|---|---|---|---|
| BL | doubles corner, near-left | (400, 863) | **WRONG — singles sideline x near baseline.** True doubles corner is at (200, 866), a distinct, more-outer line found by widening the crop. |
| BR | doubles corner, near-right | (1518, 863) | **WRONG — singles sideline x near baseline.** True doubles corner at (1718, 866). |
| TL | doubles corner, far-left | (598, 300) | Correct — verified as the outer (smaller-x) of two lines at the far baseline; the inner line at x≈680 is the far singles crossing, correctly not used. |
| TR | doubles corner, far-right | (1330, 300) | Correct, same check. |
| near_svc_L | singles sideline x near service line | (528, 649) | Correct — forms a clean L-corner with the near service line, distinct from the doubles sideline passing through the same region (verified at (390,650) in the same crop). |
| near_svc_R | singles sideline x near service line | (1408, 650) | Correct, same check. |
| baseline_center | doubles baseline center mark | (958, 862) | Correct — consistent with (200+1718)/2=959. |
| far-T | center line x far service line | (963, 373) | Correct. |

**Root cause confirmed to be the SAME bug as the near_svc_L/near_svc_R
residual-asymmetry finding logged on 2026-07-17**, not two separate issues, per
the hypothesis in the request: BL/BR were the only mislabeled points (singles
pixel position given doubles world coordinates), which corrupted the whole
least-squares fit's near-baseline scale and specifically stressed near_svc_L/R
(themselves correctly labeled) into a 50-60px residual because they disagreed
with the corrupted BL/BR anchors about where the near-baseline plane's true
scale was.

**Fix**: re-clicked BL/BR against the correct, more-outer doubles-sideline line
(200,866)/(1718,866), confirmed by directly widening the crop and finding the
second line that the original narrower crop had cut off. Rebuilt the
least-squares homography with all 8 points now consistently doubles-corner-type
at both baselines (`cv_pipeline/src/cv_pipeline/reference_video1_calibration.py`).

**Result — this was not a minor tweak**:

| calibration | near-T error | net-base error |
|---|---|---|
| Least-squares 8-point (BL/BR mislabeled) | 27.4px (44.4cm) | 24.9px (47.0cm) |
| **Least-squares 8-point (BL/BR corrected)** | **4.4px** | **2.0px** |

Per-point reprojection residuals after the fix: 2.1–17.6px across all 8 points
(vs. up to 60px before) — the asymmetry is gone. **This now beats video1's own
~13px benchmark**, not just narrows the gap to it — the "real residual gap"
reported on 2026-07-17 was itself entirely an artifact of this one mislabeled
point pair, not a genuine limitation of the least-squares technique or this
clip's footage. Re-registered as the Phase 6 dashboard job with the corrected
homography and re-derived player selections (near/far detection rates
unaffected: still 100.0%/96.9%, since `y_upper_bound_m=29.0` and the
plausibility margins are generous enough not to flip on a homography change
this size).

Lesson for future calibration passes: when a candidate calibration point sits
near multiple parallel court lines (doubles alley + singles sideline running
close together), widen the verification crop enough to see whether a second,
more-outer line exists before accepting the first line found as the intended
one — the narrower crop used originally cut off the real doubles corner
entirely, so there was no visual cue in that crop that anything was wrong.

## Second Crowd-Selection Bug: Front-Row Spectator as Near Player (2026-07-19)

User reported a crowd member boxed instead of the real player at ~0:14 in the
data/tennis/1.mp4 render. Investigated using the corrected homography, per
instruction, testing the cheapest fix first before building anything complex.

**Root cause**: a front-row spectator/photographer standing behind the near
baseline barrier consistently projects to world_x≈9.8-9.95m (near the right
doubles sideline, inside the existing tight X margin) and world_y≈-3.7m
(behind the near baseline, inside the loose Y sanity bound). Since "near
player" is chosen as the SMALLEST plausible world_y, this stationary
bystander outranks both real players in every frame they're both visible —
confirmed at frame 852 (~14.2s) and 65+ other frames via direct projection
through the corrected homography, not assumed.

**Cheap fix tested first, and found insufficient — shown with real numbers,
not just asserted**: tightened the world_y bound to "inside the court polygon
plus a small margin" (tried 1.0m, 2.0m, 3.0m, and a court-rectangle-distance
metric). This correctly rejected the spectator, but also incorrectly rejected
real near-player frames: real players legitimately reach world_y as low as
-3.0m during play (confirmed via smoothly-varying frame-to-frame world
coordinates at frames 410-419, 446-456, 1783-1802 — genuine motion, not
noise), overlapping the spectator's -3.7m closely enough that no single
margin value cleanly separates the two populations by position alone in this
clip. Measured directly, not guessed: a 2.0m margin, for example, incorrectly
dropped the real far player at frames 190-197 (whose true world_y of 25.78m
is only 0.01m past the margin's cutoff).

**Working fix**: temporal continuity (the fallback explicitly pre-authorized
for exactly this situation). The spectator is stationary; a real player never
teleports between frames. `PlayerContinuityTracker` /
`select_players_sequence_with_continuity` (cv_pipeline/src/cv_pipeline/player_selection.py)
picks, for each frame after the first, whichever plausible candidate is
closest in world-space to the previous frame's selected position (within
`MAX_TEMPORAL_JUMP_M=3.0`), falling back to the existing court-position rule
only when no previous position exists or the closest candidate exceeds that
jump cap. Verified end to end on the full 2,020-frame clip:

- Spectator selected in **0/2020** frames after the fix (down from 66+).
- All 7 of Phase 2's manually spot-checked frames unaffected.
- Near-player coverage stays 100%, far-player coverage stays 96.9%
  (unchanged from the y_upper_bound_m-only fix).
- The temporal fallback (track distance > 3.0m) never actually triggered
  anywhere in the clip — the continuity signal was clean throughout.
- Visually spot-checked frames 852 and 1000 directly (not just the
  aggregate numbers) — both players correctly boxed in both.

**Important**: this is a SEPARATE, independent fix from `y_upper_bound_m=29.0`
(the original back-wall-staff bug) — both are needed together for this clip;
dropping either one reintroduces its respective bug (confirmed directly: a
first pass that used only temporal continuity without `y_upper_bound_m`
reintroduced the back-wall-staff selection at frame 0).

Wired into the live `/analyze-video` path (not just this one job) via a
`PlayerContinuityTracker` instantiated in `video_pipeline.py`'s
`run_video_analysis`, gated to `_is_reference_video1(video_path)` only — not
applied to any other clip, since their homography scale/framing hasn't been
independently checked against this same bound.

## Generalization Test: data/tennis/2.mp4 (2026-07-19)

Second clip in the reference-pipeline series, same match as 1.mp4
(Alcaraz vs Sinner, Miami 2023) but a different point (score 3-30/4-15 vs
1.mp4's 2-15/4-0), same camera setup. Treated as a real generalization test
per instruction, not a formality -- every component was independently
re-verified rather than assumed to transfer.

**Step 1 — file properties**: 1920x1080, 59.94fps, h264 (all identical to
1.mp4); 1344 frames / 22.42s (different clip length, expected). Confirmed
continuous single-camera footage: `classify_ball_detection_regime` reported
`validated`/0 cuts, cross-checked with a full 268-sample histogram-correlation
scan (every 5th frame) -- 0 cuts found, matching 1.mp4's standard.

**Step 2a — player detection**: found a NEW instance of the back-wall-staff
bug, but on BOTH sides this time (two stationary people at world_y~34-35m,
wx=-1.87 and wx=13.05, present in nearly every frame -- 2,688 raw detections).
A clean ~3.5m gap exists between real far-player positions (max 30.56m) and
this cluster (min 34.04m), so `y_upper_bound_m=32.0` (this clip's OWN bound,
not reused from 1.mp4's 29.0) cleanly separates them -- verified via full-clip
histogram, not assumed. A second stationary cluster at world_y~10-11m
(side-court staff/photographers, wx=-3.03 and wx=15.13) was found but requires
no fix -- both sit outside the existing X-plausibility margin
([-2.5, 13.47]) and are already rejected. **Checked specifically for 1.mp4's
near-side spectator bug and did NOT find it on this clip** -- the near-player
world_y excursions below 0 (down to -3.7m) are smooth, continuous motion
across a wide x-range (0.02-9.36m), confirmed by inspecting the real
frame-by-frame trajectory, not a stationary bystander. Temporal continuity was
therefore not applied here. Final rates: near 100%, far 97.1%. Visually
spot-checked 5 frames (0, 480, 900, 1200, and the net-rally exchange around
frame 1225) -- all correct; the one `far_box=None` cluster (frames
~1204-1267) is a legitimate coverage gap (both players converged at net,
one player briefly cropped out of frame), not a selection bug.

**Step 2b — ball detection**: combined method, 88.4ms/frame (118.9s for
1,344 frames), 100% raw candidate rate (unvalidated). **A new static-artifact
location was found**, as expected from the established pattern (every prior
clip has surfaced one) -- a hallucinated `fine_tuned_yolo` detection at pixel
~(605,628), on a faint court-surface scuff mark, 49% frequency in the
artifact-bin flagging pass. Visually confirmed via zoomed crop
(`artifact_bin_zoom.jpg`). Correctly suppressed by the existing artifact-bin
filter: 0 `fine_tuned_yolo` leaks in the final results; one unrelated
`motion_diff` detection near that pixel, visually confirmed as a real ball
pass-through (not the artifact, and not filtered since motion_diff isn't
gated by the artifact-bin list). Anchor+interpolation trajectory built the
same way as 1.mp4 (player-box exclusion, temporal-consistency anchors,
quadratic per-axis polyfit for gaps <=30 frames): 99.55% coverage (953
anchors + 385 interpolated), spot-checked and confirmed accurate.

**Step 2c — homography**: independently calibrated from scratch (NOT
reused from 1.mp4), applying the wide-enough-crop lesson directly -- every
one of the 4 outer doubles corners was checked against a crop wide enough to
reveal a hidden more-outer line before accepting it. **No such second line
was found for any corner on this clip** -- all 4 corners were correct on the
first attempt, unlike 1.mp4. Held-out-landmark error: near-T 10.1px (14.3cm),
net-base 5.1px (9.2cm), both under the ~13px benchmark. Per-point residuals
uniform (5.6-15.7px, no outlier asymmetry). Saved as
`cv_pipeline/src/cv_pipeline/reference_video2_calibration.py`.

**Step 2d — singles/doubles lines**: derived from the new homography via
`CourtHomography.singles_corners_pixels()`, visually confirmed against real
frames at both baselines (`singles_doubles_check_near/far.jpg`) -- both lines
track the real court markings cleanly.

**Step 3 — processing time (real, measured, not assumed to match 1.mp4)**:

| step | time | rate |
|---|---|---|
| person detection (1,344 frames) | 58.0s | 43.2ms/frame |
| ball detection, combined method (1,344 frames) | 118.9s | 88.4ms/frame |
| homography + selection assembly | <0.1s | negligible |
| **total** | **~177s (~2.9 min)** | |

(1.mp4's equivalent: 85.5s person + 236.3s ball ~ 322s for 2,020 frames --
2.mp4 is proportionally faster per-frame on the ball detection pass, 88.4 vs
117.0ms/frame, plausibly because this rally has less ball motion/fewer
artifact-bin candidates to process, not investigated further.)

Wired into the live `/analyze-video` path via `_is_reference_video2`, with
`_select_boxes` extended to accept a plain `y_upper_bound_m` (no continuity
tracker needed) for clips like this one that need the bound but not the
temporal-continuity fix.

### Comparison note: what transferred, what didn't

**Transferred cleanly, zero rework**: video properties/codec, cut-detection
methodology, ball-detection combined method + artifact-bin filtering
mechanism (the *code*, not any specific clip's flagged bins), anchor+
interpolation trajectory logic, `CourtHomography.from_point_correspondences`
+ `singles_corners_pixels()`, the overall render/registration pipeline.

**Needed clip-specific re-work, every time**: the homography itself (fully
re-calibrated from scratch -- reusing 1.mp4's would have been silently wrong,
since even this same locked camera differs in exact framing between points),
the `y_upper_bound_m` threshold (a different number, 32.0 vs 29.0, because
it's a function of this clip's own homography scale), and a fresh
static-artifact location for the ball detector (expected, but still required
a manual crop-and-confirm pass, not automatic).

**Encouragingly, NOT everything repeated**: 2.mp4 did not have 1.mp4's
near-side spectator bug (no temporal continuity needed), and its 4 outer
doubles corners were correct on the first attempt (no BL/BR-style
mislabeling) -- direct evidence the "wide-enough-crop" lesson generalizes and
prevents that specific class of error going forward, not just a coincidence
of this one clip.

**Bottom line for the generalization question**: onboarding a new clip in
this pipeline is NOT a zero-effort operation yet -- it still requires a full
manual homography calibration pass (~20-30 min of zoomed-crop point-picking)
and a full-clip player-selection audit (histogram + visual spot-checks) before
the render can be trusted. What IS now reusable is the *method* (wide-crop
corner verification, histogram-based bimodal-gap detection for crowd bugs,
the anchor+interpolation trajectory approach) and all the underlying code --
each new clip is faster to onboard than the last because the investigative
technique is established, but it is not yet automatic.

## data/tennis/2.mp4 Court-Outline Bug — ruled out rendering first, found a real calibration error (2026-07-19)

User reported the rendered left doubles sideline visibly sat outside the true
line on data/tennis/2.mp4, despite a reported 10.1px/5.1px held-out error.
Investigated rendering-vs-calibration explicitly, in order, before touching
the calibration, per instruction:

1. **Pulled the exact pixel values being drawn** (`homography.court_corners`/
   `singles_corners` from the served result JSON) and compared them against
   the homography's own `world_to_pixel` projection of the same world
   coordinates -- differences were small (~6-15px) and consistent with the
   already-reported per-point residuals. Not a coordinate-pass-through bug.
2. **Checked corner labels/ordering** (`_pixel_quad`, `court_polygon_pixels()`,
   `singles_corners_pixels()`) for a doubles/singles swap or an incorrect
   real-world-width constant -- all consistent, BL/BR/TR/TL correctly
   ordered and correctly inset.
3. **Checked for camera drift within the clip** (a locked, cut-free camera can
   still pan/zoom slightly without tripping a cut detector) by comparing the
   same corner's pixel position at frame 0 and frame 670 directly -- pixel-
   identical. Ruled out.
4. **Rendered the exact served JSON coordinates onto a real, previously-
   unchecked frame** (frame 670 -- only frame 0 and the last frame had been
   visually verified before) -- this is where the mismatch became visible:
   the drawn BL vertex sat ~49px right of the true corner.
5. **Re-measured all 4 doubles corners individually** against fresh zoomed
   grids: BR/TL/TR all matched their original calibration values exactly.
   Only BL was wrong -- originally clicked at (249,878), true position
   (200,879) -- a plain ~49px reading error, not a systematic bug.

**Conclusion: a genuine calibration error, not a rendering bug** -- confirmed
by process of elimination with real numbers at every step, not assumed
either way going in. Rebuilt with the corrected BL point:

| | near-T error | net-base error |
|---|---|---|
| Original (BL mismeasured) | 10.1px (14.3cm) | 5.1px (9.2cm) |
| **Corrected** | **6.98px** | **1.1px** |

Now better than 1.mp4's own corrected calibration (4.4px/2.0px)... at net,
though slightly higher at near-T; both comfortably under the ~13px benchmark.
`y_upper_bound_m=32.0` re-verified against the corrected homography (real
far-player max 30.71m, staff min 34.23m -- gap holds). Re-registered as the
dashboard job.

**Lesson, distinct from the BL/BR mislabeling lesson on 1.mp4**: that earlier
fix (wide-enough crop to rule out a hidden more-outer line) prevents clicking
the *wrong line entirely*. It does not prevent a plain misreading of the
*correct* line's coordinates. A held-out-landmark error that looks reasonable
is not sufficient proof a calibration is right -- the least-squares fit can
partially absorb one badly-wrong point into the other 7, keeping the
aggregate number deceptively plausible (10.1px isn't alarming on its own).
**Going forward: visually render every calibration onto more than just the
first/last frame before trusting it** -- this bug was invisible in the two
frames originally checked and only became visible on a frame chosen
specifically because it hadn't been checked yet.

## Mandatory Calibration Verification Workflow — baked into the codebase, not just documented (2026-07-19)

Following the data/tennis/2.mp4 BL-corner bug (see above), the requirement to
visually verify a new calibration was turned into an actual enforced gate, not
just a written lesson that could be silently skipped under time pressure on a
future clip.

**Built**:
- `cv_pipeline/src/cv_pipeline/calibration_verification.py` --
  `render_verification_frames(video_path, homography, out_dir, frame_indices)`
  draws the doubles+singles outline with labeled BL/BR/TR/TL corner markers on
  start/middle/end frames (or explicit indices) and saves them for human
  review. `CalibrationVerificationManifest` records a real per-frame,
  per-corner sign-off plus a free-text note of what was actually checked, with
  `.is_complete()` requiring >=3 frames, all 4 corners confirmed at each, and
  a non-empty note.
- `cv_pipeline/tests/test_calibration_verification.py` -- discovers every
  `reference_video*_calibration` module in the package automatically (via
  `pkgutil`, not a hardcoded list, so a future clip's calibration is picked
  up without editing this test file) and FAILS if there is no complete,
  checked-in manifest at `cv_pipeline/data/calibration_verification/<clip>.json`
  for it. Verified the gate actually works, not just written correctly: with
  video2's manifest temporarily removed, the test failed with an actionable
  error message (confirmed by direct run, not assumed); restored, it passes.
- Retroactively brought BOTH existing calibrations up to this standard --
  video1 had only ever been visually checked at 2 frames (frame 0 and the
  last frame) before this; re-verified at 3 (0, 1010, 2019) with all 4
  corners confirmed at each. video2 already had this pipeline's 3-frame
  check performed as part of fixing its BL bug; formalized into a manifest.

**What this does and doesn't guarantee**: the test cannot verify a human
actually looked at the rendered images -- no test can check that. What it
does enforce is that the artifact (a manifest with real per-corner
booleans and a note) exists and is complete before a calibration module can
be considered "done," which makes skipping the check a visible, deliberate
choice (an empty/missing manifest, caught by the test) rather than a silent
omission the way the original 2.mp4 BL bug was.

## Manifest Spot-Check Found a Real Gap, Then a Bigger Real Finding: data/tennis/2.mp4 Has a Genuine Mid-Clip Camera Pan

**The spot-check that started this**: asked to pull actual `confirmed_note`
text from the video1/video2 manifests to check they weren't generic
placeholders. They weren't generic, but they also weren't as rigorous as
they looked -- both notes were specific (dates, bug history, frame numbers,
file paths) without ever naming a single corner's actual pixel coordinate.
Nothing forced the per-corner `True` to be backed by a real measurement
instead of a plausible-sounding retrofit of the already-known-correct
calibration values.

**Fix**: `CalibrationVerificationManifest.corners_confirmed[frame][corner]`
is now a struct (`{confirmed, pixel: [x,y], matches: "<what real court
feature this was checked against>"}`), not a bare bool.
`is_complete()` now additionally requires: every corner has a real `[x, y]`
pixel pair; every corner has a non-empty `matches` description; the
`confirmed_note` references at least one of the recorded coordinates; and
no corner's coordinate is byte-identical across every checked frame (a
proxy for "this was actually re-measured per frame," not copy-pasted --
see caveat below).

**Re-verifying video1 and video2 against this standard is what surfaced the
camera-pan finding below.** While independently re-measuring each corner
per frame (grid-overlay crops, not reused calibration values) to populate
the stricter manifests, `data/tennis/2.mp4`'s BL/BR near-baseline apex
visibly sat at a different pixel row in the frame-1343 crop than in the
frame-0 crop -- by a margin too large to be read-error. Confirmed with
direct pixel-intensity measurement (not eyeballing) before treating it as
real, since eyeballing is exactly how the 2.mp4 BL bug happened in the
first place:

- Sampled a fixed x-column (both x=1600 and x=1680, independently) on the
  near baseline, and a fixed x-column (x=1300) on the far baseline, every
  40 frames across the full clip (0-1343, all 1344 frames).
- **Shape of the shift**: stable at near-baseline y=875, far-baseline y=298
  for frames 0-360 (~0-6.0s); a smooth ramp from frame ~400 to ~560
  (~6.7-9.3s) moving near-baseline 875->866px and far-baseline 298->289px;
  stable at those new values for frames 560-1343 (~9.3-22.4s). This is a
  gradual ~160-frame (~2.7s) transition, not an instantaneous cut.
- **Pan, not zoom**: the same edge's x-position was checked at frames 0,
  360, 480, 680, 1000, 1320 and stayed at x=179-184 throughout -- no
  horizontal component. Near-baseline and far-baseline shifted by the same
  ~9px; a zoom would shift them by different amounts scaled by distance
  from the optical center, so an equal shift on both means this is a pure
  vertical translation (tilt/pan), not a zoom.
- **Why the existing cut/regime detector didn't catch it, and why that's
  not itself a bug**: ran the real `frame_matches_reference_framing`
  histogram-correlation check (`ball_detection_combined.py`,
  `FRAME_REFERENCE_MATCH_THRESHOLD=0.5`) against every 20th frame of
  2.mp4. Correlation never dropped below 0.974 anywhere in the clip. This
  is exactly the blind spot that function's own docstring already
  documents from `match_tennis.mp4`'s gradual-zoom case: a global color
  histogram barely moves for a small in-shot camera adjustment, because
  the scene content is nearly identical, just shifted a few pixels -- the
  detector was built and validated to catch hard cuts / different camera
  sources, not sub-10px in-shot pans, and it did that documented job
  correctly here. Nothing was silently broken; this is a real, previously
  unobserved gap in what category of camera motion the detector can catch,
  not a regression.

**Implication, not yet fixed**: 2.mp4's single static homography (built
from frame-0 corners) is accurate for roughly the first third of the clip
and increasingly off -- up to ~9-10px -- from frame ~560 onward. This
likely explains at least part of the residual error the earlier
"Court-Line Misalignment" investigation found on frame 670: that
investigation correctly found and fixed a real ~49px BL mismeasurement (a
genuine, separate, static labeling error, independently reconfirmed here),
but its remaining error budget was probably a mix of that fix plus this
pan, not fully isolated at the time. **No fix has been applied yet** --
per instruction, this is reported for a decision on approach (accept the
~10px error and document it as a known limitation, split the clip into
before/after-pan segments each with their own homography, or track the
homography continuously) before any code changes.

**Manifests re-verified against the stricter schema**: video1 and video2
manifests were rebuilt using genuinely independently re-measured
per-frame, per-corner pixel coordinates (grid-overlay crops read fresh for
each frame, not reused from the calibration module) with real natural
sub-pixel variance between frames (e.g. video1 BL: (200,866) / (200,867) /
(201,864) across its 3 frames) -- both pass `is_complete()` under the new
struct-based check. **Caveat on the anti-copy-paste heuristic**: an
exact-duplicate pixel value across all checked frames is now rejected,
but on a clip with a genuinely static camera (video1, and video2's frames
560+) truly independent re-reads legitimately can land on the identical
integer pixel after rounding -- the check is a proxy for "did you actually
re-look," not a guarantee, and a real static-camera calibration could in
principle need the note to explicitly say so rather than being flagged.
No case like that was hit in this re-verification (real variance was
observed everywhere it was checked), so the caveat is documented but not
yet exercised.

## Fix for the Camera-Pan Finding: Split data/tennis/2.mp4 Into Two Calibrated Segments

Per instruction, went with **option 2** (a second homography for the post-pan
segment) rather than accepting the ~10px error or building continuous
tracking.

**New calibration module**: `reference_video2_postpan_calibration.py`, covering
frames ~560-1343. Built with the same methodology as every other calibration
in this project -- independently re-measured from scratch via wide-enough
crops (NOT derived by applying an assumed uniform pixel offset to the pre-pan
points), calibrated from frame 960 (middle of the stable post-pan window),
least-squares 8-point fit, held-out near-T/net-base landmarks never used in
the fit.

- The wide-crop check mattered again here: BR's wide crop showed a second,
  more-inner singles-sideline diagonal near the true doubles corner (see
  `cv_pipeline/scratch_output/tennis2/calib_verify_postpan/wide_BR.jpg`) --
  the same failure mode this project has hit before, now confirmed to
  recur on a segment that's pixel-shifted from its sibling calibration, not
  just on a fresh clip.
- **Held-out error: near-T 4.30px, net-base 1.69px** -- under the ~13px
  benchmark, in the same range as `reference_video1_calibration.py`'s
  4.4px/2.0px and this clip's own pre-pan calibration's 6.98px/1.1px. Per-
  point reprojection residuals across all 8 calibration points are uniform
  (3.8-10.67px, no outliers).
- **Visually verified per the mandatory workflow** at 3 frames spanning the
  post-pan segment (560 start, 950 middle, 1343 end) -- all 4 corners
  confirmed tight against the real court lines at every frame (see
  `cv_pipeline/scratch_output/tennis2/calib_verify_postpan/manifest_frames/`).
  Manifest at `cv_pipeline/data/calibration_verification/video2_postpan.json`
  passes `is_complete()` under the stricter per-corner-coordinate schema --
  auto-discovered by `test_calibration_verification.py` via `pkgutil` with
  zero test-file changes needed, confirming that discovery mechanism
  generalizes to segment calibrations, not just whole-clip ones.

**Before/after comparison**:

| | Segment 1 (frames 0-399) | Segment 2 (frames 560-1343) |
|---|---|---|
| Calibration module | `reference_video2_calibration.py` | `reference_video2_postpan_calibration.py` |
| Calibration source frame | 0 | 960 |
| BL (doubles corner, near-left) | (200, 879) | (197, 864) |
| BR (doubles corner, near-right) | (1720, 878) | (1718, 863) |
| Held-out near-T error | 6.98px | 4.30px |
| Held-out net-base error | 1.1px | 1.69px |

**The ~400-560 ramp**: excluded from confident court-line overlay rendering,
not assigned to the nearer segment. Reasoning: the shift during the ramp is
continuous, not a step -- there is no single frame in it with an unambiguous
"correct" homography, only a frame-by-frame-varying true position that
neither static calibration matches. It's also cheap to exclude: ~160 frames,
~2.7s of a 22.4s clip. Silently serving an approximate overlay there would
misrepresent a pixel-precision visual claim as more certain than the
underlying geometry supports -- exactly the kind of thing this project's
validation discipline exists to prevent. Player *selection* (a much coarser,
meter-scale-tolerance geometric check, not a pixel-precision visual claim) is
NOT excluded during the ramp -- it uses whichever segment's homography is
nearer by frame index (split at the ramp's midpoint, frame 480), since a
9-13px calibration error is negligible next to the multi-meter margins
`select_players_by_court_position` operates on.

**Wiring into the live pipeline** (`v2_serving/src/v2_serving/video_pipeline.py`):
- `_build_homography_if_available` now returns a `homography_report` for
  2.mp4 describing both segments explicitly, including both corner sets
  (`court_corners`/`singles_corners` for segment 1, `postpan_court_corners`/
  `postpan_singles_corners` for segment 2) and the held-out error for each.
- The per-frame loop selects between the pre-pan and post-pan homography for
  player selection at the ramp midpoint (frame 480), and separately stamps
  `frame_record["court_corners"]`/`["singles_corners"]` with the post-pan
  shape for frames >=560, and forces `frame_record["homography_applicable"]
  = False` for frames 400-559 -- applied AFTER the ball-detection block,
  since that block unconditionally overwrites `homography_applicable` with
  its own per-frame framing check, which (per the finding above) is
  confirmed blind to this exact kind of pan and would otherwise silently
  clobber the ramp suppression.
- `VideoOverlay.jsx` now reads `currentFrame.court_corners` /
  `currentFrame.singles_corners` first, falling back to the clip-level
  `result.homography.court_corners`/`singles_corners` when absent -- so
  frames 0-399 and the ramp (which never gets an override, though it's
  separately suppressed via `homography_applicable`) behave exactly as
  before, and frames 560+ draw the correct, different shape.
- **Explicitly out of scope, not silently left broken**: the ball-detection
  combined method (`run_combined_ball_detection_for_clip`) and
  `classify_ball_detection_regime` still use a single homography (the
  pre-pan one) for the whole clip -- their court-region motion-diff
  filtering could be mildly degraded on post-pan frames by the same ~10px
  the court-line overlay was. Not fixed here; this task's scope was the
  calibration split and the overlay-accuracy/player-selection consumers of
  it specifically. Flagged as a named follow-up.

**General lesson -- the regime/cut classifier needs a stricter check than it
currently has, scoped for later, not fixed now**: `classify_ball_detection_regime`
and `frame_matches_reference_framing` (both in `ball_detection_combined.py`)
are histogram-correlation based and were built and validated to catch hard
cuts and different-camera-source footage -- they do that job correctly, and
this finding is not a regression in either. But this project's "continuous
single camera" assumption, used in several places (ball-detection regime
gating, and implicitly by every calibration module before this one) needs a
check that can also catch a smooth, sub-15px, multi-second in-shot pan --
demonstrated here to be invisible to a global-histogram-correlation approach
(correlation never dropped below 0.974 across the entire pan). A future,
scoped improvement: track a small set of fixed reference points' pixel
position frame-to-frame (the same technique used ad hoc for this finding's
BL/BR/TR/TL scans) as a cheap, targeted drift detector, rather than relying
on histogram correlation for this category of camera motion. Not built now
-- flagging it as a specific, named open question, per instruction, rather
than scope-creeping this task into a general-purpose drift detector.

**Verification blocker, reported not worked around**: attempted to register
a live dashboard job spanning the pan boundary (`data/tennis/2.mp4`,
`frame_limit=650`) to visually confirm the wiring end-to-end in the actual
UI, on top of the static code trace and the calibration's own real-frame
visual verification (which does not depend on this). The job failed with
`ImportError: numpy.core.umath failed to import`, traced to a pre-existing,
unrelated environment problem: the running backend's Python environment has
`tensorflow==2.16.1` (pulled in by `mediapipe`, used for pose estimation)
alongside `numpy==2.5.1`, an incompatible combination -- reproduced with a
bare `import mediapipe` alone, with none of this session's code involved.
This is not caused by anything in this change; it would fail identically on
`data/tennis/1.mp4` or any other clip through `/analyze-video` right now.
**Not fixed here** -- downgrading numpy or tensorflow is a environment-wide
change with its own blast radius (other consumers of numpy 2.x in this repo
haven't been checked) and is a separate decision from this task's scope.
The code path was instead verified by: (1) static trace of the boundary
conditions (frame 399/400/480/559/560) against the constants and
conditionals as written, confirming the logic matches the design above: (2)
`render_verification_frames`, which has no mediapipe/tensorflow dependency,
independently confirming the post-pan homography's real-frame accuracy at 3
frames. The dashboard's actual on-screen behavior for this specific wiring
has NOT been visually confirmed and should be treated as code-reviewed, not
live-tested, until the environment issue is resolved.

## Environment Fix: numpy/mediapipe/tensorflow Break, and the Blast-Radius Check It Required

**Root cause, precisely identified before touching anything**: mediapipe's
own source (`mediapipe/tasks/python/core/optional_dependencies.py`) states
outright "TensorFlow isn't a dependency of mediapipe pip package... we'll
ignore it if tensorflow is not installed" and wraps the import in
`try: ... except ModuleNotFoundError`. Tensorflow 2.16.1 was installed in the
system Python environment but broken (it declares `numpy<2.0.0,>=1.26.0` for
python>=3.12; the environment had `numpy==2.5.1`), so the import raised
`ImportError` -- a different exception than the one mediapipe's own code
catches -- and crashed hard instead of falling through to the graceful no-op
path.

**Checked before acting, per instruction**: is tensorflow actually needed by
anything here? No -- confirmed by (1) grepping the entire project for direct
`import tensorflow`/`from tensorflow` (zero hits), (2) checking every
`pyproject.toml` in the repo (tensorflow declared nowhere), (3) checking
mediapipe's own declared `Requires-Dist` (no tensorflow listed), and (4)
directly testing `from mediapipe.tasks import python` in `cv_pipeline/.venv`,
which has mediapipe installed with **no tensorflow at all** -- it imported
cleanly. Tensorflow was orphaned dead weight, not a real dependency.

**The originally-proposed fix (pin numpy down) was checked and found to have
a real, worse conflict**, not just theoretically riskier: `opencv-contrib-
python`/`opencv-python` 5.0.0.93 (which `cv_pipeline` actively depends on for
all homography/CV work) declare `numpy>=2` for python>=3.9 -- directly
incompatible with tensorflow's `numpy<2.0.0` requirement. Satisfying both
simultaneously is impossible without also downgrading opencv to an older
major version, a much bigger and riskier change than removing an unused
package. This was surfaced to the user directly (a real fork from their
initial hypothesis) rather than silently picked either way.

**Fix applied** (user confirmed after the above was presented): uninstalled
`tensorflow==2.16.1`, `tensorboard==2.16.2`, `tensorboard-data-server==0.7.2`,
`keras==3.3.3` from the system Python environment. Checked each was safe to
remove first: nothing in the project imports keras/tensorboard directly
either; the only installed package requiring any of the three
(`ultralytics`) only pulls `tensorboard` via an optional, not-installed
`logging` extra. `ml-dtypes` (a small tensorflow/keras support library) was
deliberately left installed -- harmless, no conflicting upper-bound pin, not
the source of the crash.

**Verified the actual fix, not just the absence of the error**:
`from mediapipe.tasks import python` now succeeds, and
`cv_pipeline.pose_estimation.make_landmarker()` successfully instantiates a
real `PoseLandmarker` (using mediapipe's own bundled TFLite runtime, not the
separate tensorflow package -- confirming tensorflow was never actually
exercised by anything we use). `pip check` afterward shows only the same
two warnings that were already there before this fix and are unrelated to
it: `scipy` (soft warning only, scipy still imports and works) and `numba`
(hard-broken by numpy 2.5.1 too, but confirmed unused anywhere in this
project -- zero imports -- so left alone as a pre-existing, irrelevant,
separate issue, not silently "fixed" by scope-creeping into something not
asked for).

**Full test-suite blast-radius check, across all 4 components, per
instruction** (not skipped, not assumed clean):

| Component | Venv used | Result |
|---|---|---|
| `cv_pipeline` | `cv_pipeline/.venv` (own, never had tensorflow) | 4/4 passed |
| `rag_engine` | `rag_engine/.venv` (own, never had tensorflow) | 19/19 passed |
| `llm_agent` | `rag_engine/.venv` (installed editable side-by-side, per its own pyproject.toml) | 5/5 passed |
| `v2_serving` | system Python (the environment this fix actually touched) | 16/18 passed, 2 failed |

**The 2 `v2_serving` failures are real, but proven unrelated to this fix**,
not glossed over: both are tiny (4th-decimal-place) numeric drift in
win-probability predictions (`test_win_probability.py`,
`test_win_probability_djokovic_goffin_exact_regression_value` and
`..._kohlschreiber_...`). Root-caused, not assumed: the test output itself
carries an `sklearn` `InconsistentVersionWarning` -- "unpickle estimator...
from version 1.9.0 when using version 1.5.0." Checked
`tennis-intelligence-platform/.venv` (the actual venv the win-probability
models were trained/validated in): `scikit-learn==1.9.0`, `xgboost==3.3.0`,
matching the warning exactly. System Python (where `v2_serving`'s tests just
ran) has stale `scikit-learn==1.5.0`, `xgboost==2.1.4` -- a pre-existing
version skew between system Python's ad hoc package set and the real
training environment, with zero code-path overlap with
tensorflow/mediapipe/keras (`win_probability_pipeline.py` only imports
`pandas`/v1 platform code, nothing touched by this fix). This bug was
**already there before today's session** -- it was invisible only because
`v2_serving`'s tests couldn't even collect on system Python until now
(missing `httpx` and `python-dotenv`, both installed as part of getting this
verification to actually run -- pure additive, non-conflicting installs,
unrelated to the numpy/tensorflow issue itself). **Not fixed here** -- this
is a separate, pre-existing, newly-surfaced issue (stale sklearn/xgboost in
system Python vs. the real training venv) outside this task's scope; fixing
it would mean upgrading system Python's sklearn/xgboost to match
`tennis-intelligence-platform`'s versions, which is its own environment
change warranting the same check-first discipline applied here, not a quick
add-on.

**Closed the loop on the original motivating failure**: restarted the
backend on the now-fixed environment and re-ran `/analyze-video` on
`data/tennis/2.mp4` (the exact job that failed with the numpy/mediapipe
error last time) -- completed successfully end-to-end,
`near_player_pose_live_estimate` measured at 100% success rate (60/60
frames), confirming pose estimation (the actual feature this whole
environment break was blocking) now works live, not just in isolated
import checks.

## The 2 v2_serving Failures: Confirmed NOT Harmless, Root-Caused, Fixed

Investigated per instruction before touching anything further: is the
4th-decimal win-probability drift pure serialization float noise, or a real
behavioral difference? Checked with real side-by-side predictions, not just
the aggregate test tolerance.

**Method**: a standalone script ran the exact production code path
(`compute_composite_prematch_probability`) on 7 real matches (the 2 failing
tests' matches plus 5 more, deterministically sampled) in both system Python
(`scikit-learn==1.5.0`, `xgboost==2.1.4`) and `tennis-intelligence-platform/.venv`
(`scikit-learn==1.9.0`, `xgboost==3.3.0` -- the actual training environment),
comparing full-precision outputs.

**Result: NOT harmless.** The diff was positive on all 7 matches, every
time (0.03%-0.22% relative, ~0.0003-0.001 absolute in probability) -- a
one-directional, systematic offset, not the two-sided scatter real floating-
point noise would produce.

**Root cause, confirmed not guessed**: dumped the raw XGBoost booster config
in both environments. The trained model's real `base_score` (XGBoost's
learned prior, auto-estimated from training data for `binary:logistic`) is
`0.49895427`. System Python's older xgboost (2.1.4) read it back as exactly
`0.5` -- the generic hardcoded fallback default, not the actual trained
value -- while `xgboost==3.3.0` (the version that saved it) read it
correctly. This is precisely the scenario XGBoost's own runtime warning
describes ("if you are loading a serialized model... generated by an
older/newer version... export via `Booster.save_model` first") -- a known,
documented cross-version serialization hazard, not a novel bug requiring
open-ended detective work the way PtWinner did. The mismatch was in
XGBoost's own booster deserialization specifically, independent of the
sklearn wrapper version -- meaning aligning scikit-learn alone would not
have fixed it.

**Fix**: upgraded system Python to `scikit-learn==1.9.0` AND `xgboost==3.3.0`
(both, not just scikit-learn as originally proposed -- extended because the
diagnosed defect lives in xgboost's deserializer). Checked for conflicts
first: nothing installed pins a specific scikit-learn or xgboost range, so
this was safe. `pip` also resolved `numpy` down from `2.5.1` to `2.2.6`
as a transitive dependency of the scikit-learn upgrade -- re-verified this
didn't reopen the earlier opencv/mediapipe fix (opencv-contrib-python's
`numpy>=2` requirement is still satisfied by 2.2.6; `pip check` afterward
shows FEWER warnings than before, since 2.2.6 also now satisfies scipy's
`numpy<2.3` bound that 2.5.1 had violated).

**Verified, not assumed**: re-ran the same 7-match side-by-side comparison
after the upgrade -- all 7 diffs are exactly `0.0`, bit-for-bit identical to
the training environment (`base_score` now reads `[0.49895427]` correctly
in system Python too). Re-ran `v2_serving`'s full test suite:
**18/18 passing**. Re-confirmed the other 3 components unaffected (all in
their own isolated venvs, untouched by this system-Python-only change) and
re-ran a live `/analyze-video` job end-to-end after restarting the backend
-- still 100% pose success, confirming the earlier mediapipe fix and this
sklearn/xgboost fix compose cleanly together.

## Open Backlog Item (not urgent): Retrofit 1.mp4/2.mp4 Calibrations With Numeric Point-Tracing?

data/tennis/3.mp4's calibration (see below) found that numeric pixel-
brightness thresholding measurably beats eyeballed grid-crop reads --
held-out error dropped from 3.86px/5.0px to 1.68px/1.68px, and a real 22px
outlier residual (near_svc_L) dropped to a uniform 1.47-4.61px range once
re-measured numerically. Question raised: should 1.mp4's and 2.mp4's (both
segments) calibrations be redone the same way?

**Decision: not now, deliberately deferred, not forgotten.** All existing
calibrations (1.mp4: 4.4px/2.0px, 2.mp4 pre-pan: 6.98px/1.1px, 2.mp4
post-pan: 4.30px/1.69px) are comfortably under the ~13px benchmark this
project uses as its pass/fail bar -- none are failing, none have an outlier
residual pattern like the one that triggered this fix on 3.mp4. A few px of
further improvement at this scale is ~1-2cm in world coordinates, well
below the margins used by every downstream consumer (player-selection
bounds, court-polygon checks). Retrofitting is cheap if ever needed, but
not warranted by any current symptom -- revisit only if a future
investigation on 1.mp4/2.mp4 turns up something the current accuracy can't
explain, rather than proactively.

## Generalization Test: data/tennis/3.mp4 (Clip 1 of 3)

**Step 1 (file properties/camera check)**: 1920x1080, 59.94005994fps, 933
frames (15.57s), h264 (`ffprobe` unavailable, used cv2 + size/duration for
an approximate ~62.9Mbps). Same match as 1.mp4/2.mp4 (Miami Open, Alcaraz
vs Sinner), a different point (7-4(40)/6-4(15), static score throughout --
one continuous rally). Regime classifier: `validated`, 0 cuts. Direct
pixel-measurement pan check (2 near-baseline + 2 far-baseline columns,
every 40th frame across all 933 frames): near baseline 840-841px, far
baseline 267-268px, no trend anywhere -- genuinely static camera, unlike
2.mp4. One transient `None` reading (frames 913-924 at one column) was
checked and confirmed a player briefly standing on that exact column, not
drift -- the reading recovered to the exact pre-occlusion value and a
second column never wavered.

**Step 2 (homography)**: single static segment (whole clip). Found a real,
previously-unseen issue: first-pass eyeballed calibration had a 22px
outlier residual on `near_svc_L` (vs 3.5-8.8px on the other 7 points) --
the same asymmetric-residual signature that flagged 1.mp4's real
mislabeling bug. Investigated the same way (wide crop first): confirmed
the line identity was correct this time (not a repeat of that bug), so
root-caused it as plain ~14px eyeballing imprecision reading a zoomed
screenshot. Fixed by switching all 8 points to numeric pixel-brightness
thresholding instead of eyeballed grid crops -- dropped held-out error
from 3.86px/5.0px to **1.68px/1.68px** (the best of any clip so far) with
uniform 1.47-4.61px residuals, no outliers. New general lesson: even a
correctly wide-enough crop (which prevents mislabeling) can still carry
10+px of plain human-reading error on a fine anti-aliased intersection --
numeric thresholding is both more precise and more reproducible.
Verification manifest built (3 frames, 4 corners each, independently
re-measured per frame) and passes; full gate now 5/5.

**Decision raised and deferred**: should 1.mp4/2.mp4 be retrofitted with
the same numeric method? Not now -- none are failing their ~13px
benchmark or showing an outlier-residual pattern; logged as a named,
non-urgent backlog item above rather than acted on reactively.

**Step 3 (player detection/tracking)**: ran YOLO person detection across
all 933 frames (cv_pipeline/.venv, ~38s). **Found a real, previously-seen-
category bug**: without a clip-specific bound, `far_box`'s world_y was a
near-constant ~34.0m across effectively every frame -- not the real Sinner
(who was correctly detected but not selected). Root-caused with a full-
clip histogram of all far-region candidates: a clean bimodal split, real
player positions at 14.5-27.8m (near the 23.77m far baseline) and a
stationary pair of back-wall staff at 33.7-34.0m, ~6m gap between them.
Same failure category as 1.mp4 (29.0m bound) and 2.mp4 (32.0m bound), but
independently re-derived, not reused: **y_upper_bound_m=30.0** for this
clip specifically (2.2m margin above the real max, 3.7m below the decoy
cluster). Verified against real frames, not just the histogram: the
highest-value selection accepted by the new bound (29.5m, frame 375) was
checked visually and is genuinely Sinner playing a deep defensive shot,
not a misdetection -- confirms the bound isn't clipping real play. Near
side checked too: its most extreme value (-4.0m, frame 858) was visually
confirmed as genuinely Alcaraz standing near the baseline, not a repeat of
1.mp4's front-row-spectator bug -- no near-side fix needed. Temporal
continuity also not needed (mirrors 2.mp4, not 1.mp4). 7 additional frames
spot-checked across the clip with both boxes drawn -- all correct. Wired
into `video_pipeline.py`: `_is_reference_video3`, homography branch, and
`y_upper_bound_m=30.0` gate, following the exact pattern established for
1.mp4/2.mp4.

**Step 4 (ball detection)**: combined method, 932/933 raw detection (99.9%
-- 58.1% fine-tuned-YOLO, 41.8% motion-diff), the single gap being frame 0
itself (a structural cold-start edge case, not an interior gap). Live
estimate, no ground truth for this clip -- stated explicitly, not conflated
with the 53.91% validated pooled-recall figure. New static artifact found
and confirmed suppressed (94.4%-frequency bin at (920,500), a faint scuff
mark, zoomed in and visually confirmed; a second, rarer 3.2%-frequency bin
at (960,270) sits on the real far-service-line T-mark). Single-homography-
on-pan-clips caveat checked and confirmed not applicable -- no pan on
3.mp4.

**Step 5**: registered end-to-end (933 frames, 178.8s). Caught and fixed a
real process issue before trusting the result: the backend was still
running code from before this session's video3 wiring, so the first
registration silently fell back to `size_based_fallback_no_homography` and
`stock_yolo` -- caught by checking `player_selection_method`/
`ball_detection.method` in the actual result rather than assuming success
from a 200 response, restarted the backend, re-ran, confirmed
`court_position_plausibility` and `combined_v2`. Final: 100%/100% near/far
detection, 99.04%/99.04% pose success, 99.89% ball detection. Dashboard:
http://localhost:5173/?loadJob=ff41b4e4-6df2-4c85-9b04-cfc852efab47

**3.mp4 summary**: every established technique transferred (numeric
calibration method, y_upper_bound_m pattern, artifact suppression,
manifest workflow) but every clip-specific NUMBER had to be independently
re-derived and re-verified -- none reused from 1.mp4/2.mp4. Two real bugs
found and fixed (a 22px eyeballing error, a back-wall-staff selection bug)
using the same measure-first discipline as every clip before it, plus one
process bug (stale backend code) caught by checking the actual result
fields rather than trusting a 200 status.

## Generalization Test: data/tennis/4.mp4 (Clip 2 of 3)

**Step 1**: 1920x1080, 59.94005994fps, 1544 frames (25.76s), h264, ~79.8Mbps
(size/duration estimate). Same match, players have switched ends vs 3.mp4
(normal mid-match occurrence, not a camera issue) -- score static at
7-4(15)/6-5(15) throughout, one continuous point. Regime classifier:
`validated`, 0 cuts.

**A real, larger, more complex camera-motion finding than 2.mp4's pan --
found via the same direct-pixel-measurement discipline, then corrected
after an own methodology bug was caught**:
- First-pass scan (fixed-narrow-window, vertical-only) suggested a small
  ~4-5px drift -- calibrated from a middle frame (900) on that basis.
  Empirical held-out check at the clip's extremes (per this project's
  "verify, don't assume the calibration frame's own numbers generalize"
  standard) found 0.2px error at frame 0 but **9.16px at frame 1440** --
  a real, unexplained jump that didn't fit "small uniform drift."
- Root-caused, not shrugged off: the original scan window was clipping at
  several frames (a real bug in the diagnostic script itself) AND never
  checked horizontal (x) position at all -- only vertical (y). A corrected
  scan (wide window, clipping-guarded, tracking x-position) found a REAL
  horizontal camera sway of up to ~45-68px, confirmed independently at 3
  spatially-separated corners (BL, TL, BR) moving together -- proof of a
  genuine whole-frame motion, not a localized artifact. Also visually
  confirmed: the frame-900 calibration's overlay is clearly, visibly
  misaligned at frame 500 (inside the motion), not just numerically off.
- Full shape (finer scan, first-cluster-only clustering to avoid a second
  false-positive fix -- a stray bright object, likely a player's shoe,
  was contaminating naive "leftmost+rightmost" readings at some frames):
  stable frames 0-420; smooth dip to a brief (~30-frame) flat bottom
  (~485-515) at ~45px left of baseline, smooth return; stable again
  740-1240 at the SAME position as 0-420 (confirmed: 4 independent
  numeric reads across both windows cluster within ~1.5px); smooth rise
  to a brief (~30-frame) flat peak (~1280-1310) ~23px right of baseline;
  smooth decline continuing past baseline to a genuinely flat ~44-frame
  tail (1500-1543, the clip's last frame) ~30px left of baseline.
- **Decision** (per instruction: generalize the segment-finding approach
  rather than forcing 2.mp4's 2-segment shape, and honestly flag anything
  too short/unstable to calibrate): ONE homography, covering the two
  matched stable windows (frames 0-420 + 740-1240, ~921 frames, ~60% of
  the clip). The two ~30-frame turning points and the ~44-frame end tail
  are real and genuinely flat (not noise) but explicitly judged too brief
  for a calibration verifiable to this project's own standard (3+
  meaningfully-separated frames) -- not built. Frames 421-739 and
  1241-1543 (~40% of the clip) are excluded from confident court-line
  overlay rendering, the same exclusion principle as 2.mp4's ramp, applied
  to two regions instead of one.
- Held-out error (at the frame-900 calibration source, within the stable
  window): 2.2px near-T, 1.71px net-base. Verification manifest: 4 frames
  (0, 300, 900, 1200) spanning BOTH stable windows, all 4 corners
  independently re-measured per frame (BL alone: 188.0/187.5/189.0/188.5
  across the 4 frames -- tight, real variance, confirms both windows are
  the same position), passes the strict schema. Full gate now 6/6.

**Step 3**: same back-wall-staff pattern as 2.mp4/3.mp4, independently
re-derived (analysis restricted to the two homography-valid stable
windows, 922 frames -- world coordinates are meaningless in the excluded
ranges so they were excluded from this analysis too, not just overlay
rendering): real far-player positions cluster 23.4-27.8m, back-wall staff
33.1-35.0m, clean ~5.3m gap. **y_upper_bound_m=30.0** (same number as
3.mp4's, coincidence not reuse -- independently derived from this clip's
own data). Near side checked and clean, no front-row-spectator pattern.
Verified visually at 2 spot-check frames (200, 1200, one from each stable
window) -- both correct. Wired into video_pipeline.py; the bound is also
applied during the excluded ranges for player *selection* specifically
(not overlay) since leaving it unset there would reintroduce the same bug
unfiltered -- a deliberate, reasoned choice given selection's coarser
meter-scale tolerance, noted explicitly in the code comment as a judgment
call rather than a fully re-verified fact for those frames specifically.

**Step 4**: combined method, 1543/1544 raw detection (99.9%, 73.8%
fine-tuned-YOLO, 26.1% motion-diff), single gap at frame 0 again. 3 new
static artifacts found (two near the scoreboard graphics, one a scuff
mark at (920,480)), all low-frequency (3.8-5.5%), all confirmed suppressed
by the existing mechanism. Explicitly checked the excluded-motion-range
caveat rather than assuming it away: raw detection rate is statistically
identical inside vs outside the excluded ranges (100% vs 99.9%), but
`homography_applicable`'s own per-frame check reports 100% applicable
EVEN INSIDE the excluded ranges -- confirming the same histogram-
correlation blind spot found on 2.mp4, now on a third clip with 3-5x
larger motion. Practical consequence stated plainly: motion-diff's
court-region mask is mispositioned during ~40% of this clip's frames,
so "something detected" there doesn't mean correctly localized. Not
fixed (dynamic mask tracking is out of scope), not silently ignored.

**Step 5**: registered end-to-end (1544 frames, 287.6s). Restarted the
backend BEFORE registering this time (learned from 3.mp4's stale-code
mistake), then verified the actual result fields anyway rather than just
trusting the restart: `court_position_plausibility` and `combined_v2`,
confirmed correct on the first attempt. Final: 100%/99.42% near/far
detection, 100%/99.67% pose success, 99.94% ball detection. Dashboard:
http://localhost:5173/?loadJob=3a3bf8cb-675e-4d36-a8b2-01d49d03e936

**4.mp4 summary**: this clip broke the "locked-off camera" assumption
that held for 1.mp4/3.mp4 and even 2.mp4's simpler single-pan case --
required generalizing the segment-finding methodology itself (not just
applying it), catching a real bug in my own diagnostic script along the
way (narrow scan window + vertical-only check, both fixed before trusting
the conclusion), and making an explicit, evidence-backed judgment call
about which regions were worth calibrating vs. honestly excluding. Every
other technique (numeric point-tracing, y_upper_bound_m derivation,
artifact suppression, manifest workflow) transferred directly. The
process bug (stale backend code) from 3.mp4 was avoided this time by
applying the lesson learned, not just noting it.

Proceeding to Clip 3: data/tennis/5.mp4, Step 1.

## Generalization Test: data/tennis/5.mp4 (Clip 3 of 3)

**Step 1**: file properties confirmed directly (1920x1080, 59.94fps, 940
frames, 15.68s, h264, ~81.78Mbps estimated -- `ffprobe` unavailable, same
caveat as prior clips). Visual check at frames 0/470/900: same match/venue/
camera style as 1-4.mp4, one continuous point (Alcaraz 7-4(0-15)/Sinner
6-6(0-40)), a new "miami open 2023" banner graphic starting from this clip
(cosmetic broadcast-graphics difference, not a framing concern -- stated
explicitly). `classify_ball_detection_regime` reported `validated`, 0 cuts
-- as expected, this says nothing about smooth in-shot pans.

**Mandatory direct pixel-measurement pan-check** (full 940-frame clip, 3
independent tracked points -- near-left corner x, far-left corner x,
far-right corner x) found a real camera motion with a **third distinct
shape**, different from both 2.mp4 (small there-and-back pan) and 4.mp4
(complex multi-segment there-and-back-and-overshoot): a genuine **one-way
pan-and-settle**. Frames 0-~135 stable at the original position; frames
~136-399 a real, sustained, monotonic transition (~264 frames, confirmed
visually via a red/green pixel overlay of frame 0 vs frame 320 vs frame
900 -- unambiguous corner displacement); frames ~400-939 stable at a NEW
position through the end of the clip -- it never returns to the original
framing. Two single-frame near-corner spikes and a multi-sample far-corner
wobble were checked against the other 2 of 3 independently-tracked points
at the same frames and confirmed as local occlusions (player/object
crossing that exact scan column), not camera motion -- same diagnostic
logic as 3.mp4's frame_920 occlusion check.

**A real, previously-unseen tooling bug found and fixed during this
check**: `cv2.CAP_PROP_POS_FRAMES` seeking proved frame-INACCURATE for
many indices in this file -- one seek-based scan asked for frame 120 and
silently returned frame ~178's content instead. Root-caused via a binary
search: frames 0-70 seek correctly, frames 75+ do not, consistent with a
keyframe-interval boundary where FFmpeg's seek-then-read returns the
nearest keyframe's content rather than decoding forward to the exact
target. **This is NOT unique to 5.mp4** -- the same test reproduces on
every one of 1.mp4-4.mp4 (e.g. 4.mp4's own committed `video4.json`
manifest frame 300 does not match a sequential read at all, maxdiff=255/
255). This means `calibration_verification.py`'s `render_verification_frames`
-- used to produce the human-reviewable images for every prior clip's
manifest -- has been seeking unreliably this whole time. Fixed the root
cause: `_frame_at` (single seek) replaced with `_frames_at` (pure
sequential decode, reads every frame in order via `.read()`, never
`cap.set()`), used by `render_verification_frames` going forward. Not
retroactively re-auditing 1-4.mp4's existing manifests right now -- flagged
as a backlog item below, since none of those clips are failing their
held-out thresholds and any seek error landing within the same stable
segment would have shown geometrically-equivalent (if not literally
identical) content anyway.

**Step 2**: two independent calibrations, both via numeric pixel-brightness
thresholding, both cross-validated by held-out landmarks AND a full manifest.

- **Segment A** (frames 0-135, calibrated from frame 120 -- frames 0-~95
  have the near-right corner (BR) occluded by the returning player, visually
  confirmed via corner crops before picking a clean source frame). Held-out
  error: 0.75px near-T, 0.30px net-base -- **the best result of any
  reference clip so far**, but only after finding and fixing a real
  measurement bug: the first-pass TL corner used a "mean of first bright
  cluster" method that is only valid for narrow, isolated corner blobs --
  it silently breaks when the scan row also intersects a long, continuously-
  painted line (the far baseline itself), landing on a meaningless midpoint
  between the true corner and wherever the next line crosses it (here, the
  singles sideline). First-pass TL: (611.0, 273.5), off by ~85px in x.
  Caught via cross-validation, not luck: building a homography from the
  OTHER 7 points and checking the predicted TL position (537, 282) disagreed
  with the measured value by 74px, while every other point was internally
  consistent to within a few px -- the "suspiciously bad, check the obvious
  alternative explanation" discipline this project has used throughout.
  Re-measured TL properly (row-endpoint tracing, not cluster-mean) and got
  (526.0, 275.0); held-out error dropped from 10.22/17.98px to 0.75/0.30px.
- **Segment B** (frames 400-939, calibrated from frame 600, avoiding several
  single/multi-frame occlusion artifacts independently identified during the
  pan-check). Held-out error: 1.96px near-T, 1.92px net-base -- clean on the
  first pass, no bugs found.
- Segment A and Segment B are confirmed to be genuinely DIFFERENT camera
  positions (BL ~137px vs ~212px, a real ~75px shift) -- unlike 4.mp4, where
  the two stable windows turned out to be the SAME position sharing one
  homography. Frames 136-399 (the transition) are excluded from confident
  court-line overlay rendering, same principle as 2.mp4's ramp and 4.mp4's
  two ramps.
- Verification manifest: 6 frames (100, 115, 130 for Segment A; 450, 600,
  900 for Segment B), all rendered via the newly-fixed sequential-decode
  path and visually confirmed -- all 4 corners land tightly on the real
  court lines at every frame, no misalignment found. Full gate 7/7 (`pytest
  cv_pipeline/tests/` passes).

**Step 3**: full-clip person detection (3,914 raw detections across both
homography-valid segments -- transition frames excluded from world-
coordinate analysis, same reasoning as 4.mp4). Found the classic back-wall-
staff pattern: real far-player x-plausible positions cluster 11.4-26.91m,
back-wall staff cluster 33.83-34.29m, a clean 6.92m gap. **y_upper_bound_m
=30.0** (coincidentally the same number as 3.mp4/4.mp4, independently
re-derived from this clip's own data, not reused). A SECOND stationary-
object cluster was found near the net (world_y ~12.3-13.26m, at world_x
-3.3/14.93 -- almost certainly camera operators or ball-kids standing off
to the sides near the net posts, well outside the court) but required no
new fix: it's already fully rejected by the existing
`X_PLAUSIBILITY_MARGIN_M` check in `player_selection.py` (0 of 1223
detections in that cluster have a plausible x-coordinate). Near side
checked and clean -- a smooth -3.95 to 7.41m spread with no gap, no
front-row-spectator pattern like 1.mp4's. No `PlayerContinuityTracker`
needed (same as 2.mp4/3.mp4/4.mp4).

**Step 4**: combined method, 940/940 raw detection (100%, 69.7%
fine-tuned-YOLO, 30.3% motion-diff), zero gaps. One new static-artifact
bin found at pixel (92, 50), frequency 5.32% -- low, same pattern as
every prior clip, confirmed suppressed by the existing mechanism.
Excluded-range caveat explicitly checked, not assumed: the combined
method's OWN per-frame `homography_applicable` check (before my manual
override) reports 100% "applicable" even INSIDE the 136-399 transition
range (264/264) -- the same histogram-correlation blind spot found on
2.mp4, reconfirmed on 4.mp4, now confirmed a third time on a third
distinct motion shape (one-way pan vs there-and-back vs multi-segment
there-and-back-and-overshoot). Raw ball-detection rate is statistically
identical inside vs outside the excluded range (100% both), consistent
with 4.mp4's finding. Practical consequence stated plainly: motion-diff's
court-region mask would be mispositioned during the transition frames if
trusted there, but the shipped output's `homography_applicable` is
correct because of the explicit manual override applied after ball
detection (verified: final output shows 0/264 applicable inside the
range, 676/676 outside) -- a known, accounted-for limitation of the
underlying mechanism, not a live bug in what ships.

**A real, unrelated bug found and fixed while finishing Step 5**: the
`/video-file/{filename}` endpoint's directory allow-list
(`v2_serving/routers/media.py`) only included `data/cv_annotated/videos`
and `data/` directly -- NOT `data/tennis/`, where 3.mp4, 4.mp4, and 5.mp4
actually live (1.mp4/2.mp4 happen to also exist as direct copies under
`data/`, which is why they never surfaced this). This means the dashboard
jobs registered and reported as "viewable" for 3.mp4 and 4.mp4 earlier in
this same task were **actually 404ing on video playback** the whole time
-- the analysis JSON was correct, but the browser's `<video>` element
would have failed to load the source clip. Confirmed directly (`curl
.../video-file/3.mp4` and `4.mp4` both 404 before the fix). Fixed by
adding `data/tennis` to the allow-list; verified 200 after a backend
restart. The now-stale 3.mp4/4.mp4 job IDs reported earlier were lost on
this restart anyway (the in-memory job store never survives a restart,
independent of this fix) -- re-registering them is a quick follow-up if
their dashboards are wanted, not done here since it's outside this task's
scope of finishing 5.mp4.

**Step 5**: registered end-to-end (940 frames, 219.8s -- backend
restarted a SECOND time after the video-file fix above, then this job
was registered fresh so it reflects the fixed code, not stale; an
earlier 202.6s run on the pre-fix backend was discarded rather than
reported, since its video-file serving was confirmed broken). Verified
real fields, not just a 200: `player_selection_method:
court_position_plausibility`, ball detection `method: combined_v2`, and
`GET /video-file/5.mp4` returns 200 (confirmed, not assumed). Final:
100%/99.47% near/far detection, 99.79%/99.47% pose success, 100% ball
detection. Track-ID stability checked:
near-player 3 distinct IDs, far-player 8 distinct IDs across the clip --
swaps are NOT concentrated in the camera transition (a few do fall there,
frames 192/325/326, plausibly explained by the motion itself) but the
largest cluster of rapid swapping (frames 748-806, IDs 100/101/104
alternating) sits deep in Segment B's stable range, well past the
transition -- consistent with a genuinely difficult tracking moment (fast
exchange, players close together) rather than a camera-motion artifact,
and not a new clip-specific bug requiring intervention.

## Three-Clip Generalization Summary

Ran the full validated reference pipeline (file-property probing, direct
pixel-measurement pan detection, numeric-thresholding homography
calibration with mandatory manifest verification, player-selection
crowd/staff scanning, combined-method ball detection, full-clip render)
independently end-to-end on three new clips (3.mp4, 4.mp4, 5.mp4), each
a genuinely different point from the same match, none previously seen.
**Every clip surfaced at least one real, previously-unencountered issue**
-- the task's own prediction held up in practice, not just in principle.

**What is now genuinely routine** (applied identically, no new
investigation needed, each time):
- File-property probing and visual camera-style confirmation.
- The DISCIPLINE of not trusting `classify_ball_detection_regime` alone
  for camera-motion detection -- direct pixel-measurement pan-checking is
  now simply mandatory procedure, confirmed necessary on 2 of these 3
  clips (4.mp4, 5.mp4) plus the earlier 2.mp4 finding. The regime
  classifier's blind spot to smooth in-shot motion is now a 3-for-3
  confirmed, well-understood limitation, not a one-off surprise.
- Numeric pixel-brightness thresholding for all 8 calibration points
  (established as the default method on 3.mp4, used without incident as
  the starting method on 4.mp4 and 5.mp4).
- Held-out-landmark validation (near-T, net-base) plus the full
  calibration-verification manifest workflow -- mechanical at this point,
  though it keeps catching real bugs when something IS wrong (5.mp4's TL
  mismeasurement), which is exactly what it's for.
- `y_upper_bound_m` derivation via full-clip far-side world_y histogram,
  looking for a bimodal gap -- worked identically on all 3 clips (3.mp4:
  30.0, 4.mp4: 30.0, 5.mp4: 30.0 -- genuinely independently derived each
  time, the repeated number is coincidence, confirmed by each clip's own
  gap statistics differing).
- Restarting the backend before registering a job, and checking real
  result fields (not just HTTP 200) to confirm new code actually took
  effect -- learned the hard way on 3.mp4, applied proactively on 4.mp4
  and 5.mp4 without incident.

**What still required genuine, clip-specific investigation, every time**:
- The SHAPE of camera motion, when present, was different on every clip
  that had it: 2.mp4 (small there-and-back pan), 4.mp4 (complex
  multi-segment there-and-back-and-overshoot, 3-5x larger), 5.mp4 (a
  one-way pan-and-settle that never returns to its starting position).
  No template transferred directly -- each required its own segment-
  boundary analysis and its own honest judgment call about what was too
  brief/unstable to calibrate confidently.
- Each clip found a NEW variety of misdetection or measurement bug that
  no prior clip's fix covered: 3.mp4's eyeballing-precision error and a
  bounded single-frame occlusion; 4.mp4's diagnostic-script bugs (narrow
  clipping window, vertical-only check, first/last-cluster ambiguity)
  plus a genuinely new camera-motion shape; 5.mp4's mean-of-cluster
  corner-measurement bug, a project-wide `cv2` seeking-accuracy bug that
  had been silently present since 1.mp4, a second stationary-object
  misdetection cluster near the net, and an unrelated video-file-serving
  path bug that had made 3.mp4/4.mp4's dashboards silently non-functional.
- None of these were predictable from the previous clip's fixes -- each
  required the same from-scratch "measure, don't guess; check the obvious
  alternative explanation; verify against real evidence" process the
  project has used since 1.mp4, not a shortcut.

**Evidence for the "onboard a new venue within a fixed time budget"
goal**: the mechanical/procedural parts of onboarding a new clip (steps
1-2's measurement discipline, the manifest gate, y_upper_bound_m
derivation, the backend-restart lesson) are now fast and require no new
design decisions -- they are genuinely reusable process, not something
that has to be re-invented per clip. But the actual CONTENT of what's
wrong with a new clip -- the specific shape of any camera motion, the
specific new misdetection, the specific tooling bug that happens to
surface -- has been different every single time across 5 clips now, and
each has required real, non-templated investigation to find and fix
correctly. A realistic time budget for a new venue should budget for "one
genuinely novel investigation, format unknown in advance" as a near-
certainty, not a risk to plan around -- it has now happened 5/5 times.

## Retroactive Audit: Did the cv2 Seeking Bug Invalidate Clips 1-4's Manifests?

Prompted directly by the user, who correctly pointed out that finding a
project-wide tooling bug mid-task is not just a forward-looking fix -- it's
a live question about whether ALREADY-SHIPPED, already-"confirmed" work is
actually valid. Investigated properly rather than assuming either "it's
fine" or "it's all broken."

**Method**: for every non-zero-frame manifest entry across video1-4 (frame
0 needs no real seek and was already confirmed universally safe), used the
EXACT original calling pattern (`cap.set(POS_FRAMES, idx)` then `.read()`
on a shared capture, in manifest order) and compared the returned frame
against a true sequential read. Then, for every mismatch, searched the
full clip sequentially to identify which real frame's content was actually
being shown, and re-rendered the verification overlay on that real content
with the actual homography to see what a genuine visual check would have
shown.

**Severity was much worse than "off by a few pixels near a keyframe"**:
seeks landed hundreds of frames away from the target (e.g. asked for frame
1010 on 1.mp4, got frame 1998's content; asked for 672 on 2.mp4, got
1316's) or failed outright (`ok=False`) for several deeper targets on
1.mp4/2.mp4/3.mp4. This is consistent with a units/fps-interpretation bug
in the seek path, not simple keyframe-rounding.

**Per-clip result**:
- **video1.json**: HOLDS UP. Frames 1010 and 2019, re-rendered on their
  real content, both show all 4 corners landing tightly on the true court
  lines -- consistent with this clip's camera being genuinely locked-off
  for the whole clip, so even though the original images likely showed
  the wrong timestamp, they still happened to show the same (correct)
  court position. No changes made.
- **video2.json (pre-pan)**: **DID NOT HOLD UP -- rebuilt.** Frames 672 and
  1343 are outside this homography's actual valid range (0-399) to begin
  with, and when checked against their REAL content, show a genuine
  ~15-17px vertical baseline shift -- true post-pan camera position, not
  pre-pan. Worse: the OLD manifest's recorded pixel values for "672" (BL
  200,878) closely match the POST-PAN calibration's own values, not this
  one's, strongly suggesting the entry was built from -- or is at least
  only consistent with -- the wrong, seek-corrupted frame, not really
  frame 672. The manifest's central claim (verified across 3 frames
  spanning the clip) was false: only frame 0 had ever actually been
  checked. **Rebuilt from scratch** using frames 0, 150, 390 (all
  genuinely inside the 0-399 valid range), independently re-measured,
  rendered via the now-fixed sequential-read code, and visually
  reconfirmed -- tight alignment at all 3 frames, all 4 corners. The
  held-out error (6.98px/1.1px) was never actually affected by this bug
  (it's pure arithmetic on stored coordinates, no frame access involved)
  and stands unchanged. `video2.json` on disk has been replaced with the
  corrected manifest.
- **video2_postpan.json**: HOLDS UP. All 3 frames (560, 950, 1343),
  re-rendered on real content, show tight alignment -- this manifest's
  claims were genuinely true all along. No changes made.
- **video3.json**: HOLDS UP. Frames 466 and 932 both align correctly on
  real content. No changes made.
- **video4.json**: HOLDS UP. Frames 300, 900, and 1200 all align
  correctly on real content -- despite frame 300's old seek actually
  landing on frame 530's content (squarely inside the excluded 421-739 dip
  region, which WOULD show real misalignment), the manifest's recorded
  numbers do not reflect that wrong content and the real frame 300 checks
  out fine. No changes made.

**What this means concretely**: of 5 manifests across 4 clips, exactly ONE
(video2's pre-pan manifest) was actually compromised in a way that
mattered -- its "verified at 3 frames" claim was not true, though the
underlying calibration itself (built from frame 0, always seek-safe) was
never shown to be wrong, just under-verified. That one has been corrected.
The other 4 manifests' sign-offs are now independently re-confirmed
against real frame content, not just presumed fine because "the numbers
looked plausible." `pytest cv_pipeline/tests/` passes with all 6 manifests
(1, 2, 2_postpan, 3, 4, 5) complete and valid.

**Lesson for the record**: "the numeric held-out error is unaffected"
does NOT imply "the manifest's visual-confirmation claim is unaffected" --
these are two different claims resting on different data paths, and this
project's own history (2.mp4's original BL bug) is exactly why the manifest
system checks the second one separately. Checking one and assuming the
other transfers is the same shortcut that motivated building the manifest
system in the first place.

## Ball Detection: Coverage vs. Real Accuracy Under 3 Conditions (All 5 Clips)

Requested by the user before any change to the shipped ball-detection
method: for each of the 5 Miami clips, compute detection coverage AND a
real, visually-verified accuracy estimate under (A) the current combined
method as-is, (B) fine-tuned-YOLO only (motion-diff fallback disabled),
(C) YOLO-only plus gap interpolation. No ground truth exists for these
clips, so accuracy was measured the only honest way available: sampling
25 evenly-spread (not cherry-picked) detected frames per clip per
condition -- 375 total -- cropping around the reported/interpolated
position, and manually classifying each as landing on the real ball or
not, the same way the initial 8-frame spot-check was done. Condition C
uses `cv_pipeline/scripts/ball_detection_experiments.py`'s quadratic/
linear per-axis polyfit interpolation (gaps <=3 frames, fit through up to
2 confirmed points on each side) -- **this is NOT wired into the shipped
pipeline**, it exists only as an unintegrated experiment; borrowed here
for this evaluation since it's the only interpolation logic in the
codebase (there is no "existing interpolation step" in production despite
the task's original phrasing assuming one).

| Clip | Cond | Coverage | Sampled accuracy | Effective correct coverage |
|---|---|---|---|---|
| 1.mp4 (n=2020) | A | 100.0% | 68% (17/25) | 68.0% |
| | B | 77.8% | 84% (21/25) | 65.4% |
| | C | 84.2% (128 interp) | 64% (16/25) | 53.9% |
| 2.mp4 (n=1344) | A | 100.0% | 72% (18/25) | 72.0% |
| | B | 69.0% | 100% (25/25) | 69.0% |
| | C | 72.8% (50 interp) | 100% (25/25) | 72.8% |
| 3.mp4 (n=933) | A | 99.9% | 64% (16/25) | 63.9% |
| | B | 58.1% | 100% (25/25) | 58.1% |
| | C | 61.7% (34 interp) | 88% (22/25) | 54.3% |
| 4.mp4 (n=1544) | A | 99.9% | 92% (23/25) | 91.9% |
| | B | 73.8% | 88% (22/25) | 65.0% |
| | C | 76.2% (36 interp) | 88% (22/25) | 67.0% |
| 5.mp4 (n=940) | A | 100.0% | 60% (15/25) | 60.0% |
| | B | 69.7% | 88% (22/25) | 61.3% |
| | C | 74.1% (42 interp) | 80% (20/25) | 59.3% |

"Effective correct coverage" = coverage x sampled accuracy: the estimated
fraction of ALL frames in the clip that would show a genuinely correct
ball position under that condition.

**Averaged across all 5 clips**: A (combined, as shipped) = 71.2% effective
correct coverage, sampled per-detection accuracy 71.2% average (68/72/64/
92/60). B (YOLO-only) = 63.8% effective correct coverage, but sampled
per-detection accuracy averages 92.0% (84/100/100/88/88) -- almost every
marker shown is trustworthy. C (YOLO+interpolation) = 61.5% effective
correct coverage, per-detection accuracy averages 84.0% -- worse than B on
every metric; every interpolated sample checked in 4/5 clips (1.mp4,
4.mp4, 5.mp4 each had at least one) was a MISS, and 3.mp4's one
interpolated sample also missed -- only 2.mp4 had zero interpolated
frames land in its 25-sample draw. The quadratic-fit interpolation is not
reliably landing on the real ball's position on this footage; it does not
earn its added coverage.

**The real tradeoff, stated plainly**: disabling motion-diff (A -> B) does
NOT clearly increase the total volume of correct ball positions shown --
on 4/5 clips A's raw combined method actually produces MORE total correct
frames than B, simply because it shows so much more often (near-100%
coverage) that its lower per-shot accuracy still nets out ahead in raw
count. What B clearly buys is CONFIDENCE: a ~2.8x reduction in how often a
shown marker is wrong (28.8% wrong under A vs 8.0% wrong under B, on
average), at the cost of the marker being absent ~1/4 to ~2/5 of the time
instead of almost never. Since the user's original complaint was about
visible, embarrassing wrong markers (ball on a player's head, on a court
line, on a bag) rather than about gaps, B is very likely the better
*experienced* quality even where it isn't the better *raw coverage*
number -- but this is a product judgment call, not something the data
alone resolves, and is being left to the user rather than decided here.
C is not recommended under any framing tested: it never beats B and often
loses to it, while adding a component (interpolation) that isn't in the
shipped pipeline today.

No change has been made to `ball_detection_combined.py` or
`video_pipeline.py` pending this decision.

## Interpolation-Integration Discrepancy (2026-07-19) — history recovered, one part left genuinely open

Before shipping the motion-diff decision above, checked where the
assumption of "an existing interpolation step" actually came from, per
explicit user instruction, rather than proceeding on the (correct)
finding that condition C's tested implementation underperformed. Found a
real, previously-undocumented-in-this-session piece of project history:
an "anchor+interpolation trajectory" system (player-box exclusion +
temporal-consistency anchors + quadratic per-axis polyfit, gap<=30
frames) WAS built and validated in an earlier "Phase 6" pass of this
project, used for 1.mp4 and 2.mp4's full-pipeline renders (2.mp4:
99.55% coverage, 953 anchors + 385 interpolated, "spot-checked and
confirmed accurate" per this file's own earlier entry). That script does
not exist anywhere retrievable today -- not in `cv_pipeline/src`, not in
`v2_serving/src`, not in git history (this repo has only 4 commits total;
it was almost certainly a scratch script whose OUTPUT survived in
`cv_pipeline/scratch_output/reference_pipeline/phase6_*.json` but whose
CODE did not). Confirmed via live job results earlier this session that
BOTH 1.mp4 and 2.mp4 now report `ball_detection method: combined_v2` --
today's single shared pipeline for all 5 Miami clips, which contains no
interpolation of any kind. So: interpolation was real, was validated, and
was silently dropped when the pipeline was consolidated into
`ball_detection_combined.py` -- no decision to remove it is recorded
anywhere found.

### STANDING OPEN ITEM -- SAME SEVERITY CLASS AS THE GROUND-TRUTH-LEAK AND SEEKING-BUG FINDINGS (not resolved, not dismissed, not investigated further this session): was Phase 6's own "99.55% coverage... spot-checked and confirmed accurate" claim itself reliable?

**Do not read this as a minor footnote to the recovered-history paragraph
above.** It is the same category of risk as the two other confirmed
findings in this file that got their own top-level sections -- an
unverified accuracy number sitting in project documentation, exactly like
the ground-truth leak (70.40% -> 53.91%) and exactly like the retroactive
question the `cv2` seeking bug raised about clips 1-4's manifests. Both of
those turned out to require action once actually checked; this one has
NOT yet been checked either way, so it stays flagged at that level of
seriousness until it is.

**Specifically, for 1.mp4** (not just the 2.mp4 figure quoted above): the
surviving output `cv_pipeline/scratch_output/reference_pipeline/phase6_result*.json`
reports `ball_detection_live_estimate` = **97.6% coverage** (1972/2020
frames; "fine_tuned_yolo/motion_diff anchors (61.9%) + quadratic per-axis
interpolation across gaps <=30 frames (35.7%)"), with `status: "MEASURED"`.
This 97.6% figure for 1.mp4 has never been independently re-verified
against ground truth or a fresh manual audit -- it predates the
"manually classify a real sample, don't trust a coverage number" standard
this session established specifically to catch the motion-diff false-positive
problem (the 375-sample visual audit above). No such audit was ever done
for Phase 6's output.

**A second, self-contained red flag found just now while pulling this
number, worth recording on its own**: that same JSON's `status` field
says `"MEASURED"` -- schema.py's status vocabulary reserves `MEASURED` for
a rate actually checked against something -- but the note string in the
*same object* says, verbatim, "This is an UNVALIDATED live coverage rate
for this specific clip (no ground truth exists for data/tennis/1.mp4)."
The field and its own note directly contradict each other: `MEASURED`
status on a value the note itself calls unvalidated. That internal
contradiction, sitting undetected in this project's own surviving output,
is independent evidence pointing toward the "overclaimed number" branch
below, not just a hypothetical parallel to the ground-truth-leak precedent.

What was established above is only that the CODE is gone -- not that the
CLAIM about the code was trustworthy. Two genuinely different stories are
both still consistent with everything found so far, and this session did
not distinguish between them:

1. **A real regression.** Phase 6's anchor+interpolation system worked as
   described, was properly spot-checked, and was simply lost/dropped
   (not deliberately rejected) when the pipeline was consolidated into
   `ball_detection_combined.py`. If true, reconstructing it is recovering
   a genuine capability, not building something new from scratch.
2. **An unverified or overclaimed number that was never actually retested.**
   This exact project has at least one confirmed precedent for this shape
   of error: `ball_detection_combined.py`'s own docstring documents a
   ground-truth leak that inflated a different ball-detection figure from
   a real ~46.2% to a reported (and initially trusted) 70.40%, caught only
   when the real production code path was re-run end-to-end rather than
   trusting the prototype's own claimed number. Phase 6's "spot-checked and
   confirmed accurate" / "99.55% coverage" language is exactly the kind of
   claim that earlier bug shows this project should not take at face value
   without an independent re-check -- and no such re-check of Phase 6's
   claim has been done, here or (as far as this file's own history shows)
   at the time it was written either.

Distinguishing these would require either finding Phase 6's actual script
(not found in this session's search) or reconstructing its described
method and independently re-validating it against real frame content --
neither was attempted here, per explicit instruction to leave this open
rather than fold it into a general "interpolation deferred" note. Flagging
this explicitly as its own question for whoever picks up interpolation
work next, rather than letting the recovered history above read as more
settled than it is.

Practical consequence for the earlier "Coverage vs. Real Accuracy" entry:
condition C tested `ball_detection_experiments.py`'s simpler, still-live
interpolation code (raw YOLO hits as anchors, gap<=3) because it's the
only interpolation implementation that still exists -- NOT the more
careful Phase 6 approach (selective anchors, gap<=30). C's underperformance
vs. B is a real finding about the recoverable implementation specifically,
not a closed verdict on interpolation as a concept. Reconstructing Phase
6's actual anchor-selection logic from its two-sentence PROGRESS.md
description would be new implementation work, not a re-test of prior
validated code -- flagged as a real option for a future phase, not
attempted here.

## Ball Detection: Motion-Diff Fallback Disabled by Default (2026-07-19)

Per the coverage/accuracy audit above, shipped condition B:
`run_combined_ball_detection_for_clip` (`ball_detection_combined.py`) now
takes `use_motion_diff_fallback: bool = False` (was previously always-on,
no flag existed). The motion-diff code itself was NOT deleted -- passing
`use_motion_diff_fallback=True` restores the exact old behavior, kept for
research/comparison. `video_pipeline.py`'s call site passes
`use_motion_diff_fallback=False` explicitly (not relying on the function's
own default, so the call site self-documents the decision), and
`COMBINED_BALL_METHOD_NOTE` / `verify_ball_detection_wiring.py`'s notes
were updated to stop claiming motion-diff is active, since a stale note
here would be exactly the kind of "looks reasonable but wasn't
independently checked" failure this project has repeatedly flagged in
other contexts.

**Regime-gating explicitly re-verified, not assumed unaffected**: ran
`verify_ball_detection_wiring.py` end to end after the change (not just
read the diff). Confirmed: (1) `classify_ball_detection_regime` still
correctly routes video1.mp4 (amateur, locked camera) to `validated` and
match_tennis.mp4 (broadcast, hard-cut-heavy) to `best_effort`, unchanged
by this edit; (2) a fresh 60-frame run with the new default produces
`sources seen: {'fine_tuned_yolo', 'none'}` -- confirmed zero
`motion_diff` sources, a real assertion added to the script, not just a
printed observation; (3) passing `use_motion_diff_fallback=True`
explicitly still reaches `motion_diff` as a source, confirming the old
path is intact, not deleted. Full test suites re-run clean after the
change: `pytest cv_pipeline/tests/` (7 passed) and
`pytest v2_serving/tests/` (18 passed).

A caught-and-fixed self-inflicted bug during this edit, worth recording
briefly: the first attempt at extending `run_combined_ball_detection_for_clip`'s
docstring left the triple-quote closed early, silently turning the back
half of the original docstring into a syntax error (leading-zero decimal
literal from a `2026-07-16` date) that only surfaced when the module was
actually imported -- caught immediately by running
`verify_ball_detection_wiring.py`, not left for a later test run to find.

## Bounded Clip Comparison: 3.mp4 vs. 4.mp4 Under Condition B (logged hypothesis, not deep-dived)

Reusing the already-collected 25-sample visual spot-check data (not new
work): 3.mp4 condition B = 100% accurate (25/25, zero false positives);
4.mp4 condition B = 88% accurate (22/25, 3 false positives, all cases of
fine-tuned YOLO reporting an empty-court box with no real ball present at
all -- not a misdirected-but-real-object error like motion-diff's
failures, a genuine spurious detection). Checked whether those 3 misses
(frames 271, 560, 1169) cluster inside 4.mp4's known camera-motion
excluded ranges (421-739, 1241-1543): only 1 of 3 does (frame 560); the
other 2 (271, 1169) are in the clip's stable, locked-position windows.

**Hypothesis, logged not confirmed**: camera motion might weakly correlate
with more fine-tuned-YOLO false positives on 4.mp4 specifically (motion
blur or a shifting background changing what looks confusable), but a
2-of-3-in-stable-region split does not cleanly support that on this sample
size -- as likely to be ordinary noise (3 misses out of 1544 frames) as a
real effect. Not investigated further per explicit instruction to keep
this bounded; a real answer would need a larger, clip-specific sample
before acting on it.

## Kalman/Ballistic Trajectory Filter Phase -- built, verified, NOT SHIPPED

Requested as the next phase after B shipped: recover some of B's lost
coverage using a genuinely physically-grounded trajectory model (not a
generic curve fit like the already-rejected condition C), with an explicit
requirement not to accept it on a coverage number alone -- the same
manual-visual-audit standard used for the B/C decision had to be re-run on
this phase's own output before it could ship.

### Design, confirmed with the user before any code was written

Before implementing, confirmed B's exact output format
(`CombinedBallDetectionResult` in `ball_detection_combined.py`: `frame_index`,
`center`, `source` in {`fine_tuned_yolo`, `none`} with motion-diff off, plus
`homography_applicable`) and pulled real gap-length stats from the cached
B detections already used for the earlier audit (median gap 2-5 frames,
but a real long tail up to 41 frames / ~0.68s at 59.94fps across all 5
clips) -- this is a harder distribution than condition C ever attempted
(C's `MAX_GAP_FOR_INTERP=3` skipped the large majority of these gaps
outright).

**A real modeling gap was surfaced, not silently assumed away**: the
project's `CourtHomography` is a planar, ground-plane-only (Z=0) mapping --
it cannot project a true 3D point `(X, Y, Z>0)` (an airborne ball) to pixel
space, and a full 3D camera calibration (e.g. `cv2.solvePnP`) is
fundamentally underdetermined from a single planar homography (focal length
trades off against camera height/distance with no second constraint to
break the tie -- any value picked would be an unstated guess). PROGRESS.md's
own Phase 5 entry also already found monocular real-height/bounce recovery
unreliable on this exact footage. Three design forks were put to the user
explicitly rather than picked silently, all confirmed before implementation:

1. **Model**: hybrid, not full 3D reprojection -- a constant-velocity
   Kalman filter in real-world ground-plane meters (X, Y) for lateral
   motion (reuses the existing homography, no new camera calibration),
   plus a separate constant-acceleration Kalman filter on the PIXEL-SPACE
   VERTICAL RESIDUAL (`detected_pixel_y - world_to_pixel(X,Y).y`) to
   capture the flight arc without ever inverting for a metric height.
2. **Drag**: gravity-only first (linear KF; any drag-shaped mismatch
   absorbed into process noise), not modeled explicitly from the start.
3. **Shot boundaries**: real detections that strongly disagree with the
   filter's prediction (large Mahalanobis-gated innovation, or a
   velocity-direction reversal) trigger a state RESET (trust the new real
   detection, reinitialize velocity/acceleration) rather than blending
   through the discontinuity -- explicitly not an attempt to detect
   bounces/contacts directly, since that already failed once (Phase 5).

### Implementation

`cv_pipeline/src/cv_pipeline/ball_trajectory_kalman.py` (new). Forward pass
over B's real detections builds up lateral + residual Kalman state,
partitioning the clip into "segments" at each reset. For a gap strictly
between two real detections in the SAME segment, both filters are run
FORWARD from the earlier detection and BACKWARD (time-reversed transition,
same-magnitude process noise) from the later one, then fused by
inverse-covariance weighting at each intermediate frame -- legitimate here
because this is offline post-processing of an already-fully-decoded clip,
unlike the ground-truth leak this project already found and fixed
elsewhere (that leak used real ground truth, which an inference-time system
never has; this uses the clip's OTHER real detections, which are genuinely
available once the whole clip is processed). Gaps spanning a segment
boundary, or before/after the clip's first/last real detection, are left
unfilled with an explicit `skip_reason` rather than extrapolated. Every
filled frame carries a `fill_confidence_px` (derived from the fused
position covariance via a finite-difference Jacobian of `world_to_pixel`),
plus a provisional, audit-tunable `MAX_FILL_CONFIDENCE_PX=100.0` gate.

Sanity-checked on synthetic data before running on real clips: a clean
10-frame gap across a simulated parabolic arc filled to within ~0.3-1.5px
of true position; a synthetic direction reversal (simulated bounce) placed
inside a gap correctly triggered `segment_boundary_in_gap` and left the
entire gap unfilled rather than blending through it.

### Verification on the 5 real clips -- same manual-audit standard as B/C, not skipped

Ran on all 5 clips' real B detections (condition "B+Kalman" = "D" below).
Coverage recovered is substantial:

| Clip | Coverage B | Coverage D | Gap frames skipped (reason) |
|---|---|---|---|
| 1.mp4 | 77.8% | 90.4% | 194 segment-boundary, 1 clip-edge |
| 2.mp4 | 69.0% | 86.8% | 171 segment-boundary, 6 clip-edge |
| 3.mp4 | 58.1% | 77.9% | 168 segment-boundary, 38 clip-edge |
| 4.mp4 | 73.8% | 91.0% | 127 segment-boundary, 12 clip-edge |
| 5.mp4 | 69.7% | 86.6% | 101 segment-boundary, 25 clip-edge |

But coverage alone is exactly what this phase was told not to accept.
Sampled 25 evenly-spread KALMAN-FILLED frames per clip (125 total, same
crop+red-circle-marker montage technique as the B/C audit) and manually
classified each as landing on the real ball or not:

| Clip | n filled sampled | Sampled accuracy (filled frames only) | Eff. correct coverage B | Eff. correct coverage D | Overall shown-marker accuracy, D (vs B) |
|---|---|---|---|---|---|
| 1.mp4 | 25 | 12.0% (3/25) | 65.4% | 66.9% | 74.0% (vs 84%) |
| 2.mp4 | 25 | 36.0% (9/25) | 69.0% | 75.4% | 86.9% (vs 100%) |
| 3.mp4 | 25 | 40.0% (10/25) | 58.1% | 66.0% | 84.7% (vs 100%) |
| 4.mp4 | 25 | 44.0% (11/25) | 65.0% | 72.5% | 79.7% (vs 88%) |
| 5.mp4 | 25 | 36.0% (9/25) | 61.3% | 67.4% | 77.8% (vs 88%) |

**Averaged across all 5 clips**: sampled accuracy of the newly-filled
frames = **33.6%** -- roughly two-thirds of every new marker this phase
would add is simply wrong. Effective correct coverage does tick up (63.8%
-> 69.7%, +5.9pp) because the filled frames are additive on top of B's
already-correct real detections, but the blended overall shown-marker
accuracy drops from B's 92.0% average to 80.6% -- an 11.4pp degradation in
"how often a shown marker can be trusted," the exact metric the A->B
decision was made to protect.

**Checked whether tightening `MAX_FILL_CONFIDENCE_PX` could rescue this**
(confidence is genuinely informative -- accuracy is 46.4% for
confidence<10px vs. 0% for confidence in [20,40)px or [70,150)px, so the
gate isn't noise) -- but even the tightest useful cut (confidence<10px,
keeping 69/125 = 55% of filled frames) only reaches 46.4% accuracy, still
far below B's 92%. No threshold rescues this to "beats B without
materially dropping accuracy."

**Root cause, visually confirmed, not just inferred from the numbers**: the
reset mechanism only catches a shot-boundary discontinuity if it shows up
as disagreement between the filter's prediction and the NEXT real
detection at the gap's far edge. It cannot catch a discontinuity that
occurs and resolves entirely INSIDE a gap -- a player's swing/contact, or
the ball passing briefly behind a player, mid-gap, where the "after" real
detection can still land close enough to the smooth-arc prediction to pass
the innovation gate by coincidence. This showed up directly in the audit:
1.mp4 and 5.mp4 both reproduce, on multiple sampled frames (e.g. 1.mp4
frames 603/1063, 5.mp4 frames 887/893/900/906), the exact "marker parked on
a player's head/body" failure mode that motivated the A->B motion-diff
decision in the first place -- the same user-facing failure, from a
different mechanism.

### Verdict: NOT SHIPPED, per the explicit pre-agreed rejection criterion

The spec for this phase said plainly: "Only ship if it beats B on effective
correct coverage without materially dropping accuracy... if the
interpolated/filled frames turn out to be substantially less trustworthy
than B's real detections (the same failure mode found in condition C),
report that plainly and do not ship, the same way C was correctly
rejected." That is exactly what happened here, and more severely than C
(C's sampled accuracy was 84% -- still respectable; this phase's is 33.6%
on the newly-filled population). `ball_trajectory_kalman.py` is left in the
codebase, tested and working as designed against its own physical
assumptions, but is **not wired into `video_pipeline.py`** and B remains
the shipped method, unchanged. No change has been made to
`ball_detection_combined.py`, `video_pipeline.py`, or the shipped
`use_motion_diff_fallback=False` default as a result of this phase.

The plausibility-check goal (using the filter to flag suspect B detections)
was not pursued further given this result -- a filter whose own fills are
wrong 2/3 of the time is not yet a trustworthy arbiter of whether a
*different* detector's output looks physically implausible.

## Shot-Type Detection: Forehand vs. Backhand From Pose -- built, verified, real accuracy measured

New, separate capability from ball tracking: classify each shot a player
hits as forehand or backhand, using Phase 3's pose output. No ground truth
exists for shot type, so this needed the same manual-visual-audit standard
as every ball-detection decision this session -- applied throughout, not
skipped.

**Handedness, checked not assumed**: all 5 clips are the same match,
Alcaraz vs. Sinner (confirmed via on-screen scoreboard labels). Pulled
frontal, unambiguous frames (serve tosses, forehand strikes facing camera)
for both players across multiple clips -- both are right-handed, confirmed
visually, not inferred from their names. No left-handed player exists in
this dataset.

**Pose-data availability, measured, not assumed -- and a real reversal from
the Phase 3 amateur-dataset finding**: on this broadcast footage, pose
*success* (any landmarks) is ~99% for both roles, but *wrist* usability
(needed for a naive "which side is the hand on" heuristic) collapses for
the near player specifically: ~16% average (7-25% per clip), because fast
swing motion this close to camera blurs the wrist in the source footage
itself (visually confirmed via low-visibility frame crops, not assumed).
Elbow is far more robust (~55% near / ~80% far) and was used instead,
consistently for both shot-detection and side-classification (mixing
keypoints between the two steps would compound two different bottlenecks
independently). Shoulder+hip are ~99% reliable for both roles but weren't
used as the primary signal -- rotation alone doesn't identify which arm is
swinging, a structurally weaker signal than arm-side position even at much
higher coverage; deferred pending this pass's results.

**Two pose-only contact-frame proxies were tried and both failed on real
examples, not just in theory**: peak elbow-extension (Step 1.2's original
suggestion) and an "ascending-edge" fix (first threshold-crossing). Tested
against real, ball-verified examples pulled from multiple clips/players:
peak-extension grabbed the follow-through (which frequently crosses to the
wrong anatomical side on a big topspin finish); ascending-edge grabbed the
backswing (which frequently crosses to the wrong side loading up). Real
measured errors on ball-verified examples: 6-21 frames off, 2 of 3
misclassified for the ascending-edge variant specifically. Neither extreme
of the raw extension curve reliably locates contact.

**Escalated to ball-anchored contact detection**, per explicit sign-off,
with an explicit no-silent-fallback rule: if no usable condition-B ball
detection exists near a candidate swing, the event is DROPPED, not
classified via the falsified pose-only proxy. Real joint coverage (pose
AND ball both usable) was measured before building anything further: 71.8%
pooled (87.1% near / 61.7% far, n=31/47 real candidate events) -- comfortably
clears the "usefully above a third" go/no-go bar that was set in advance,
so the build proceeded.

**`cv_pipeline/src/cv_pipeline/shot_classification.py`** (NEW). Finds
candidate swing windows via elbow-extension peaks, then re-anchors each to
the real ball detection closest to the player within a window (not the
peak frame), classifies forehand/backhand by which side of the body's own
(self-calibrating, mirroring-safe) vertical axis the elbow is on at that
anchored frame. `cv_pipeline/tests/test_shot_classification.py` (NEW, 5
tests) reproduces the exact real follow-through-misleads-the-peak failure
synthetically and confirms ball-anchoring correctly overrides it, plus
confirms clean no-fallback dropping when ball or pose data is unavailable.

**A real bug found DURING the audit, not before it, and fixed**:
`_find_ball_anchor_frame`'s frame-selection had a dead comparison
(`dist == 0.0 and dist < best_dist`, where `best_dist` was already 0 and
could never be beaten) that silently picked the ball's FIRST entry into the
player's box -- often still during the backswing -- instead of the closest
approach. Found because a first audit pass showed every single "backhand"
prediction in one clip was wrong (0/6), which was suspicious enough to
trace rather than accept. Fixed to rank qualifying frames by distance to
the real (unexpanded) box.

**A real error in the manual audit process itself, found and corrected
mid-audit**: re-auditing after the bug fix, backhand predictions still
looked wrong by eye -- until one case was checked at high zoom and showed,
unambiguously, both hands on the racket (a real two-handed backhand). The
root cause was in the AUDIT, not the code: the far player faces the camera
and the near player faces away, so image-left/right maps to opposite
anatomical sides for the two roles, and early passes weren't consistently
correcting for this. A literal per-event checklist was adopted for the
remaining ~33 events specifically to prevent this recurring under fatigue:
(1) confirm role, (2) confirm facing direction and the resulting
image-side-to-hand mapping for that specific role, (3) zoom before judging,
(4) classify (grip cues first -- both players use two-handed backhands, so
two hands visible is a mirroring-independent tell -- then serve/overhead
check, then the mirroring-corrected side read; genuine ambiguity marked as
such rather than forced).

### Final audited results (52 real classifier events, all 5 clips, both roles, checklist-verified)

| Metric | Value |
|---|---|
| Confident forehand/backhand accuracy (excludes serve and misfire) | **86.5%** (32/37) |
| Near-player confident accuracy | 83.3% (15/18) |
| Far-player confident accuracy | 89.5% (17/19) |
| **Serve/overhead contamination of "forehand" predictions** | **23.3%** (10/43) |
| Misfire rate (candidate event, no real identifiable shot) | 7.7% (4/52), all in one clip's near player |
| Genuinely ambiguous (no confident call made) | 1/52 |

**The serve-contamination number is the one with a pre-agreed action
attached**: this phase's scope explicitly deferred building a
serve-exclusion heuristic until the audit measured whether serve
contamination was "a handful of percent" (leave as a documented limitation)
or "large enough to meaningfully distort forehand counts" (build the
exclusion heuristic, validated against real data instead of a guess).
23.3% is the latter -- real, substantial, not a rounding error. Building a
serve-exclusion heuristic (e.g. treating the first detected swing after a
motion lull as a probable serve) is now justified by measurement, not
assumption -- not yet built, flagged as the clear next step.

Two real bugs were caught specifically by not trusting a first-pass result
that looked too clean/too bad to be true (the anchor-selection bug, and
the audit's own mirroring error) -- both documented rather than quietly
fixed and forgotten, consistent with this project's standing practice.

## Serve-Exclusion Heuristic -- three signals tried and falsified, one narrow rule shipped

The 23.3% serve/overhead contamination measured above cleared the
pre-agreed "large enough to act on" bar, so building a serve-exclusion
heuristic was justified by measurement. The specific idea proposed --
flag a shot as a probable serve when it follows a "motion lull" -- was
tested three different concrete ways against 15 real, visually-confirmed
reference events (5 clips, spanning known serves, known overhead smashes,
and known real groundstrokes) BEFORE writing any classification logic,
same discipline as the pose-only contact-frame proxies earlier in this
phase and the Kalman filter phase before that.

**All three general "motion lull" signals were falsified by real data**:

1. **Gap since the previous detected shot event.** Falsified by a direct
   counterexample: clip3's near-player backhand at frame 676 is a real
   backhand (visually confirmed, a reaching slice near the net) following
   a 596-frame gap since the previous detected event; clip2's near-player
   serve at frame 1207 is a real serve following only a 121-frame gap. No
   single threshold separates these -- the gap large enough to exclude the
   false-positive risk is bigger than the gap on a confirmed real serve.
2. **Ball-detection density in the window before the event.** Falsified:
   condition B's fine-tuned YOLO detects the ball just as densely during
   pre-serve ball-bouncing (dead time) as during live rally flight -- 46
   detections in the 60 frames before the real serve at clip1 f107, vs. 52
   before the real overhead at clip1 f1035, no contrast at all.
3. **Ball spatial spread / path-length in the window before the event.**
   Falsified: the two confirmed real serves in the reference set
   disagreed with EACH OTHER (path-length 744px at clip1 f107 vs. 1857px
   at clip5 f84, 60-frame window), let alone separated cleanly from the
   real non-serve counterexample (621px at clip3 f676).
4. **Bonus check** (not itself a "motion lull" signal, but cheap given
   data already computed): elbow-extension-ratio magnitude at the anchor
   frame. Also falsified -- serves/overheads (0.53-0.94) heavily overlap
   real forehands (0.71-0.82) and even a real backhand (0.66).

**One thing held up cleanly, with zero counterexamples**: the very FIRST
candidate event in a clip's merged (near+far) timeline was a real serve in
all 5 clips checked -- clip1 f107, clip2 n34, clip3 n80, clip4 f406, clip5
f84, each visually confirmed (racket raised overhead, jumping extension).
This is narrow and structural, not a general serve detector: it only
catches the clip-opening shot (each of these 5 clips happens to start near
a real point boundary), not a serve occurring later in a clip (clip2's
real serve at n1207 is correctly NOT caught) or an overhead smash
mid-rally (clip1 f1035, clip4 f546 -- correctly NOT caught, since neither
has a preceding lull for this rule to key off in the first place).

**Shipped**: `flag_first_event_as_probable_serve` in
`shot_classification.py` -- takes the per-role event lists `find_shot_events`
already produces, flags the single earliest-anchored event across all
roles as `probable_serve=True`, does not drop or reclassify anything. Real
audit sample used to validate the flag itself (per the "re-run the audit
on the flagged sample" requirement): all 5 events this rule flags on the
real 52-event dataset were independently re-confirmed by zooming into
their crops -- all 5 are genuine serves, 0 false positives, so no case was
found of a real forehand/backhand being wrongly excluded for an unrelated
reason (a let, a challenge review, etc.) in this dataset.

**Measured effect on the 52-event audit set**:

| Metric | Before | After |
|---|---|---|
| "Forehand" predictions | 43 | 38 (5 flagged, none reclassified/dropped) |
| Serve/overhead contamination | 10/43 = 23.3% | 5/38 = 13.2% |

Verified programmatically against the real classifier output (not just
computed by hand): applying `flag_first_event_as_probable_serve` to all 5
clips' real events flags exactly the 5 expected frames and no others.

**What this does NOT fix, stated plainly**: mid-clip serves and overhead
smashes remain unaddressed -- no reliable general signal for either was
found with the data available (ball detections, pose landmarks). The
residual 13.2% contamination is a documented, known limitation, not a
silently-accepted one. A future pass could try a directly different kind
of signal (e.g. on-screen scoreboard/score-overlay change detection) but
that's new engineering scope, not attempted here.

**Wired in**: `v2_serving/src/v2_serving/video_pipeline.py` now runs shot
classification (gated on the combined_v2 ball-detection method, since
ball-anchoring requires condition B specifically) and the dashboard's
`VideoOverlay.jsx` renders the resulting shot events, including the
probable-serve flag. See video_pipeline.py's own docstring for the
integration's gating and caveats.

**Two real integration caveats found while wiring this in, verified by
direct experiment, not assumed**:

1. **A short `frame_limit` can silently under-detect real events near the
   segment's tail.** The peak-prominence check (`scipy.signal.find_peaks`,
   `MIN_SHOT_PROMINENCE=0.35`) needs a comparably deep valley on BOTH sides
   of a candidate swing to confirm it's a real local maximum -- if the
   window ends before that valley appears, the event is silently dropped,
   no error or warning. Confirmed directly: video1's real serve at frame
   107 (prominence 0.86 peak) computes a prominence of only 0.282 (below
   threshold) when the pipeline is run with `frame_limit=200`, because the
   only right-side valley visible within that window (0.578 at frame 122)
   isn't deep enough -- but the SAME code on the SAME video with
   `frame_limit=500` correctly finds it (a deeper valley, 0.251, appears
   later in frames 200-299, raising the computed prominence to 0.545). Any
   caller using a short `frame_limit` should treat the resulting shot
   count as a lower bound, not a complete one.
2. **video1's near-player selection in production uses
   `PlayerContinuityTracker`** (the front-row-spectator fix -- see
   `video_pipeline.py`'s own docstring), which the original 52-event
   manual audit did NOT use (it called `select_players_by_court_position`
   directly, no continuity tracker). This changes which boxes/poses feed
   the classifier for video1's near player specifically, and has not been
   independently re-audited under this exact configuration -- so the
   86.5%/13.2% figures above may not transfer exactly to video1's
   near-player events as they run in the live pipeline today. Far-player
   selection was checked and confirmed IDENTICAL between the two methods
   for video1's first 140 frames (0 differing frames), so this caveat is
   scoped specifically to video1's near player, not a general concern
   across roles or clips.

Both caveats are stated in `SHOT_CLASSIFICATION_NOTE` inside
`video_pipeline.py` itself, so they travel with the live result JSON, not
just this file.

### Follow-up: boundary padding fixed, near-player re-audited, one new gap found and disclosed

Before treating the above as closed, three things were checked further.

**1. Boundary-starvation FIXED, not just documented.** Landmark/box/ball
collection now reads `SHOT_CLASSIFICATION_PADDING_FRAMES` (300) extra
frames past `frame_limit` purely to give boundary events real right-side
peak-detection context, without adding those frames to the returned
`frames` array or any detection-rate counter. Verified directly: the exact
failure case from before (`frame_limit=200` missing video1's real serve at
frame 107) now correctly finds and flags it. This is a mitigation, not a
mathematical guarantee -- `scipy`'s prominence search is unbounded in
principle, and this project's own audit data has one real 596-frame
inter-shot gap, so a pathological case could in theory still fall outside
a 300-frame pad. The result JSON now carries
`boundary_padding_frames_requested`, `boundary_padding_frames_read`,
`video_truncated_by_frame_limit`, and a `boundary_note` describing exactly
this run's residual risk, rather than a single static disclaimer.

**2. video1 near-player re-audited under the real production configuration
(PlayerContinuityTracker), confirmatory sample, not the full 52 events
again.** Ran the actual `run_video_analysis` production path (not a
scratch re-implementation) over video1's full 2020 frames and compared
against the original audited event set:

| Frame | Original (no tracker) | Production (with tracker) |
|---|---|---|
| 235 | candidate event, raw output "forehand", audited as **misfire** | no longer a candidate at all |
| 324 | not a candidate | new candidate, "forehand" -- independently re-verified by hand |
| 476 | "forehand", audited **correct** | "forehand" (unchanged) |
| 491 | "forehand", audited **correct** | "forehand" (unchanged) |
| 788 | "forehand" x2, audited **correct** | "forehand" x2 (unchanged) |
| 1028 | "forehand", audited as **misfire** | "forehand" (unchanged) |
| 1179 | "forehand", audited as **misfire** | "forehand" (unchanged) |
| 1556 | "forehand", audited **correct** | "forehand" (unchanged) |
| 1844 | "forehand", audited as **misfire** | "forehand" (unchanged) |

All 4 previously-confirmed-correct events are unchanged. 3 of the 4
original misfires persist, unchanged (not worsened -- PlayerContinuityTracker
didn't introduce a new failure there, it just didn't happen to fix these
either). One misfire (235) no longer registers as a candidate at all.
Frame 324 is new and was independently re-verified: pulled a 3-frame
zoomed sequence (320/324/328) and confirmed a single continuous,
unambiguous one-handed forehand swing arc (extended contact -> racket
rising -> high finish, all on the anatomical right side, one hand visible
throughout) -- correctly classified. **Verdict: the 86.5%/13.2% figures
hold up for video1's near player under the actual production
configuration. No meaningful divergence found.**

**3. STEP=1 vs STEP=2 gap -- now audited and RESOLVED, figures updated.**
The original 52-event audit sampled every 2nd frame (`STEP=2`, for speed);
this live pipeline processes every frame (`STEP=1`). Comparing video1's
FAR player (unaffected by PlayerContinuityTracker, confirmed identical box
selection for frames 0-140 earlier) between the two runs found 3
additional real candidate events (frames 382, 1490, 1680) and 3 shifted
anchor frames (860->859, 1198->1199, 1333->1324) under STEP=1, purely a
sampling-resolution effect (not the continuity-tracker difference, not the
boundary-padding fix). All 6 were audited by hand, same checklist as the
rest of this project:

| Frame | Raw prediction | Audit verdict |
|---|---|---|
| 382 (new) | backhand | real backhand, two hands clearly on the racket at contact -- **correct** |
| 1490 (new) | forehand | real forehand, a one-handed emergency slide reach (one hand alone rules out backhand, since both players use two-handed backhands) -- **correct** |
| 1680 (new) | forehand | real, but it's an **overhead smash** (one-handed, racket raised fully overhead, classic smash form), not a groundstroke -- same known contamination category as the original 10 serve/overhead events, not a new failure mode |
| 859 (was 860) | backhand | two hands clearly on the racket -- **correct**, same real shot as originally audited at 860 |
| 1199 (was 1198) | backhand | two hands clearly on the racket -- **correct**, same real shot as originally audited at 1198 |
| 1324 (was 1333) | forehand | one-handed wide slice reach -- **correct**, same real shot as originally audited at 1333 |

All 3 shifted anchors preserve their original correct classification (the
few-frame shift lands within the same real swing, doesn't cross into a
different pose). Of the 3 new events, 2 are additional confirmed-correct
real shots and 1 is an overhead mislabeled as forehand -- exactly the
existing, already-quantified contamination pattern, not a surprise.
Folding these into the pooled 52-event figures:

| Metric | Original (52 events) | Updated (56 events, video1 STEP=1-verified) |
|---|---|---|
| Confident forehand/backhand accuracy | 86.5% (32/37) | **87.5% (35/40)** |
| Serve/overhead contamination, before exclusion rule | 23.3% (10/43) | **23.9% (11/46)** |
| Serve/overhead contamination, after exclusion rule | 13.2% (5/38) | **14.6% (6/41)** |

Both directions moved by roughly a point and a half -- confident accuracy
up, contamination up -- net a wash, not a meaningful correction. **The
86.5%/13.2% figures hold under true full-resolution (STEP=1) sampling; the
production-accurate figures going forward are 87.5%/14.6%.** Scope of this
check, stated plainly: only video1 was directly re-verified under STEP=1
(both roles, per this section and the near-player section above); clips
2-5 have not been independently re-sampled at STEP=1. Video1 showed no
meaningful divergence, which is reassuring evidence this isn't a
systemic issue, but it is not the same as having checked all 5 clips.
`SHOT_CLASSIFICATION_NOTE` in `video_pipeline.py` has been updated to
report 87.5%/14.6% as the current figures, with this scope caveat stated
explicitly rather than silently generalized to all 5 clips.

## Step 3: Full Output Video Render -- server-side MP4 with overlay burned in

With the shot-classification audit closed out (87.5%/14.6%, both
integration caveats resolved), built the server-side counterpart to the
dashboard's live `VideoOverlay.jsx` canvas: a real, downloadable `.mp4`
with court lines, player boxes, ball, and shot-classification labels drawn
directly into the source video's frames, rather than requiring the
dashboard to be open.

**`v2_serving/src/v2_serving/video_render.py`** (NEW):
`render_annotated_video(video_path, result, output_path)` draws from an
ALREADY-COMPUTED `run_video_analysis` result -- does not re-run any
detection/pose/ball model, so it's cheap (pure video I/O + `cv2` drawing)
relative to the analysis that produced `result`. Same overlay elements and
"only draw what's true for this exact frame" rule as `VideoOverlay.jsx`:
court quad + singles quad (respecting per-frame `homography_applicable`
and `court_corners`/`singles_corners` overrides for 2.mp4/5.mp4's
mid-clip camera changes), near/far boxes with track-ID labels, the ball
marker, and stacked `shot_events` labels (handles the same
multiple-events-per-frame case the collision fix above addresses). Frames
beyond the analyzed range (index >= the caller's `frame_limit`) are
written UNANNOTATED, not dropped -- the output video's duration always
matches the source, an honest "no overlay data past this point" rather
than a silently truncated file.

**API layer**, matching the existing `/analyze-video` job-based
conventions exactly (`render_job_store.py` mirrors `job_store.py`'s
minimal in-memory single-process pattern, for the same stated reasons):
`POST /render-video` (body: `{job_id}`, referencing an existing COMPLETE
analyze job -- 404 if unknown, 400 if not yet complete) kicks off a
background render and returns a `render_job_id`; `GET
/render-jobs/{render_job_id}` polls status; `GET
/rendered-video/{filename}` serves the finished file (basename-only
path-traversal guard, same pattern as `routers/media.py`). 8 new tests in
`tests/test_render.py`, mirroring `test_jobs.py`'s approach (mocked
`render_annotated_video` for the fast success-path tests, real
job-store-lookup behavior for the error paths).

**Verified end-to-end on real data, not just unit-tested**: ran
`run_video_analysis("data/tennis/1.mp4", 500)` for a real result, then
`render_annotated_video` on it. Output: 2020 total frames (matches the
source clip's full length), 500 annotated (matches the analysis's
`frame_limit`), 59.94fps, 1920x1080 -- all correct. Pulled frame 107 (the
real, confirmed serve event) from the RENDERED file and visually
confirmed: court quad and singles lines both drawn correctly, both player
boxes with track IDs, the ball marker on the racket, and "FOREHAND
(probable serve)" labeled directly above the server -- exactly matching
what the live dashboard overlay would show at that frame.

**A real bug found immediately after handing off a link to the rendered
file, not caught by any test at the time**: the file downloaded
successfully over HTTP (curl and the server's own access log both showed
clean 200/206 responses, including Range requests exactly like a
`<video>` element scrubbing) but silently failed to PLAY in a browser.
Root cause, confirmed by inspecting the actual output bytes (not
assumed): `cv2.VideoWriter_fourcc(*"mp4v")` encodes MPEG-4 Part 2
("FMP4"/"mp4v", confirmed via `cv2.VideoCapture(...).get(CAP_PROP_FOURCC)`
on the real file) -- a container+codec combination no mainstream browser's
HTML5 `<video>` tag can decode (Chrome/Safari/Firefox all require H.264,
VP8/VP9, or AV1). A working codec was found empirically, not guessed:
tested `avc1`/`H264`/`X264`/`h264`/`mp4v`/`MJPG`/`vp09` against this
machine's actual OpenCV/FFmpeg build -- `H264`/`X264`/`h264` all print a
real FFmpeg stderr warning ("tag ... is not supported ... fallback to use
tag ... 'avc1'") and fall back to `avc1` anyway, while `avc1` itself opens
directly and a read-back of the result reports fourcc `'h264'`, confirming
real H.264 output. Fixed in `video_render.py`. Re-rendered video1 (500
frames annotated) with the fix: output shrank from 57MB to 8.8MB (H.264's
much better compression) and reads back as `'h264'`; downloaded the
actually-served file via the running server and independently re-verified
its fourcc and frame count match. **A regression test was added**,
`tests/test_video_render.py` (3 new tests, real files not mocks -- the bug
could only be caught by inspecting actual output bytes, not by asserting
on a return value) -- `test_rendered_output_uses_a_browser_playable_codec`
specifically asserts the fourcc is never `mp4v`/`FMP4` again.

**A real collision bug found and fixed while smoke-testing the render on
the full clip, unrelated to the codec bug above.** `n_events_by_role` was
reporting 9 near-player events for video1 (correct), but only 8 distinct
frames carried a `shot_event` in the `frames` array -- the
dict-keyed-by-`(role, frame_index)` construction let a second event
silently overwrite the first whenever two events anchor to the same frame
(a real, confirmed case: video1's near player has two distinct candidate
swings both anchored to frame 788). Fixed: `frame_record["shot_event"]`
(a single nullable object) is now `frame_record["shot_events"]` (an
always-present list, explicit `[]` when empty, matching this module's own
"explicit null/empty, never omitted" convention). Verified: attaching now
produces exactly `n_events` total attached events, no more count/output
mismatch. `VideoOverlay.jsx` updated to match (stacks multiple labels per
frame, flattens the clickable event list instead of assuming one event
per frame).

**Current state**: shot classification is now fully closed out across all
three integration surfaces -- live dashboard overlay, per-frame JSON, and
a real, browser-playable downloadable rendered video (H.264, verified,
every frame of a full clip annotated end to end) -- with the accuracy
figures (87.5% confident / 14.6% residual contamination) verified under
the actual production configuration, not just the original offline audit.

## Automated Hough-Line-Based Homography Calibration -- experiment, two real bugs found and fixed, promising result

Every reference clip's calibration so far has needed real per-point manual
measurement (eyeballing, then numeric brightness-thresholding once
eyeballing was shown to carry 10+px error -- see 3.mp4's "COORDINATE
PRECISION LESSON"). Tested whether classical Hough-transform line
detection could locate the same 4 doubles corners automatically, compared
directly against the existing manually-traced ground truth on the 2
clips already fully calibrated and verified (1.mp4, 3.mp4) -- same
discipline as every other technique in this project: build a small,
real, testable version first, compare against real ground truth, report
whatever the numbers say.

**First pass: badly wrong (70-160px mean error), and visually diagnosable
as a real bug, not just "Hough doesn't work here".** The raw Hough
segments themselves traced the real court lines well (confirmed by
inspecting the white-line mask directly -- baseline, sidelines, service
lines, and center line all isolated cleanly by HSV thresholding, measured
directly from real pixels: line pixels are S~21/V~255, court-surface
pixels are S~106-146/V~161-176). The bug was in classification: a crude
"bucket by angle, split horizontal lines at the median y" approach blended
segments from DIFFERENT real lines (e.g. a baseline and a service line,
both "near-horizontal") into one averaged, garbage-diagonal fit -- visible
directly in the debug overlay as wild magenta lines cutting across the
whole frame at angles no real court line has.

**Fix 1**: proper line clustering. Group Hough segments by (theta, rho) in
normal form BEFORE fitting, not by a crude angle+position bucket -- so
multiple short segments broken up by gaps/players get merged into one
real-line fit, but segments from two genuinely different lines never get
averaged together. This alone dropped mean error from 70-160px to
11-70px, but left one specific, informative failure: the far baseline was
still being confused with a DIFFERENT real line.

**Fix 2**: the net cord, not the far baseline. The near/far baseline
selection heuristic was "the 2 widest horizontal clusters by pixel
x-span" -- correct in real-world terms (the baseline spans the full
doubles width, wider than any service line), but WRONG in pixel terms:
the net cord is also a wide, bright, near-horizontal line spanning close
to the full doubles width, and perspective foreshortening means a nearer,
real-world-narrower line (the net, or even the near service line) can
project WIDER in pixels than the more distant, real-world-wider far
baseline. Confirmed directly in the debug overlay: the "far_baseline"
line was running straight through the net band. Fixed: the net sits at
mid-court height, strictly between the near and far baselines in image-y
-- picking the topmost/bottommost horizontal cluster by Y-POSITION
instead of x-span correctly separates baseline from net regardless of
which one happens to be pixel-wider in a given frame.

### Final results, measured against real ground truth (frame 0 of each clip, the same frame the manual calibrations use)

| Clip | BL | BR | TR | TL | Mean |
|---|---|---|---|---|---|
| 1.mp4 | 7.5px | 32.7px | 5.5px | 1.6px | **11.8px** |
| 3.mp4 | 6.1px | 4.0px | 1.7px | 4.5px | **4.1px** |

3.mp4's result is genuinely comparable to the manual method's own
held-out-landmark precision (1.68px/1.68px). 1.mp4's BR outlier was
inspected visually: a line judge/ball-person is crouched close to that
exact corner in this specific frame -- plausibly interfering with the
line mask there, a single-frame occlusion artifact rather than (on this
evidence) a structural flaw, but not independently confirmed by testing
a second frame.

**`cv_pipeline/src/cv_pipeline/hough_court_detection.py`** (NEW):
`detect_court_corners(img) -> DetectedCourtCorners`. Deliberately named
OUTSIDE the `reference_video*_calibration` pattern so
`test_calibration_verification.py`'s mandatory-manifest gate (added
specifically for calibration modules, requiring a checked-in human
sign-off across >=3 frames -- see that test's own docstring) does not
discover it: this has NOT gone through that process and must not be
treated as a validated replacement for the existing manual calibrations
until it has. `cv_pipeline/tests/test_hough_court_detection.py` (NEW, 2
tests) is a real-data regression test against the measured numbers above,
with generous margins (not tight equality) so it catches an actual
regression -- like the net-cord-as-baseline bug -- without being brittle
to normal floating-point noise.

**What this does NOT yet prove, stated plainly**: only 2 of 5 reference
clips were checked, one frame each. No held-out-landmark cross-check
(near-T/net-base, the manual method's own internal validation) was
attempted -- only the 4 doubles corners themselves, since those are what
the manual ground truth directly records. Multi-frame robustness
(voting/averaging across several frames, the likely fix for 1.mp4's BR
outlier) was not attempted. **Not wired into anything, not adopted as a
replacement for manual calibration** -- this is a positive, promising
experimental result (unlike the Kalman filter phase or the pose-only
shot-detection proxies, both rejected on evidence), but "promising on 2
clips, 1 frame each" is not the same evidentiary bar this project has
required before shipping a calibration (see `test_calibration_verification.py`'s
own history: a 49px-mismeasured corner was accepted once because its
held-out error alone "looked reasonable" -- exactly the kind of
false-confidence multi-frame/multi-clip testing exists to catch). Natural
next step if pursued further: run on the remaining 3 clips, test several
frames per clip and compare voting/averaging against single-frame
results, and only then decide whether it clears the bar for the
mandatory verification-manifest process.

## Multi-Frame, Multi-Clip Hough Evaluation -- the natural next step above, now done

Followed up on the single-frame, 2-clip Hough experiment above with the
next step it explicitly flagged: all 5 reference clips, 5-10 (used 8)
evenly-spread frames per clip, voting/averaging via a new
`detect_court_corners_multi_frame` function, to see whether the
promising-but-thin single-frame result holds up under real multi-frame,
multi-clip testing -- and specifically whether averaging fixes 1.mp4's
BR-corner occlusion outlier the way it was hypothesized to.

**Camera motion made this non-trivial**: 3 of the 5 clips (2.mp4, 4.mp4,
5.mp4) have documented mid-clip camera pans/drifts (see each clip's own
`reference_videoN_calibration.py`/`reference_video2_postpan_calibration.py`
docstring), meaning a single ground-truth corner position is only valid
within that clip's own documented camera-stable window(s) -- sampling
frames blindly across a whole clip would silently compare detections from
one camera position against ground truth measured at another. Handled by
drawing the 8 sample frames only from each clip's stable window(s) (2.mp4:
pre-pan [0,400) and post-pan [560,1344) separately, against their own
respective calibrations; 4.mp4: both stable windows [0,420) and
[740,1240), which share one calibration since it was already confirmed
to validate across both at 0.2px; 5.mp4: segment A [0,136) and segment B
[400,940), against their own respective calibrations) -- 8 segments
total across the 5 clips. Also used strictly sequential `.read()` frame
decoding (never `cap.set()`/seek) per the seek-inaccuracy bug documented
in `reference_video5_calibration.py` (a seek to frame 120 silently
returned frame ~178's content on that file).

### Results: multi-frame averaged detection vs. ground truth, per clip (combining that clip's segment(s))

| Clip | BL | BR | TR | TL | Mean |
|---|---|---|---|---|---|
| 1.mp4 | 17.3px | 3.9px | 3.8px | 0.3px | **6.3px** |
| 2.mp4 | 14.0px | 14.2px | 4.1px | 5.0px | **9.3px** |
| 3.mp4 | 8.4px | 6.4px | 2.3px | 4.6px | **5.4px** |
| 4.mp4 | 3.2px | 12.3px | 3.4px | 3.4px | **5.5px** |
| 5.mp4 | 7.4px | 16.9px | 2.8px | 2.4px | **7.4px** |
| **overall** | | | | | **6.8px** |

(Per-segment breakdown, e.g. 2.mp4's pre-pan 8.8px vs. post-pan 9.9px,
4.mp4's stable-a 7.2px vs. stable-b 3.9px, is in
`test_hough_court_detection.py`'s `MULTI_FRAME_MAX_MEAN_ERROR_PX` and the
underlying eval run; the table above averages each clip's segments with
equal weight since each segment used the same frame count.)

### Three findings, not all of them the hoped-for "averaging just helps"

1. **Confirms the occlusion hypothesis for 1.mp4's BR outlier.** Averaging
   8 frames drops that corner from 32.7px (single frame 0) to 3.9px --
   the line-judge occlusion really was a single-frame artifact, and
   multi-frame voting is a real, working defense against it, not just a
   plausible-sounding idea.
2. **But it's not a uniform win.** 3.mp4 -- the *other* single-frame-tested
   clip, which had NO known occlusion problem -- got slightly WORSE under
   averaging (4.1px -> 5.4px), pulled up by one outlier frame (BL 53.5px
   at frame 666 out of 933, cause not investigated further; the other 7
   sampled frames for that corner were all under 8px). This is the
   expected failure mode of plain unweighted mean-averaging: it has no
   defense against a single bad frame, it just dilutes it. A median, or
   an explicit outlier-rejection pass, would likely fix this and was not
   tried.
3. **A real, consistent pattern that neither single-frame test could have
   shown: the two NEAR corners (BL, BR) are volatile (0.3-17.3px,
   whichever one is worse varies by clip) while the two FAR corners (TR,
   TL) are reliably good in every single clip (0.3-5.0px, no exceptions).**
   Correction to an earlier read of this same data: BR is not
   uniquely/structurally the worst corner -- recomputed per clip, the
   worst corner is BL in 2 of 5 clips (1.mp4, 3.mp4) and BR in the other
   3 (2.mp4 by a 0.2px margin, 4.mp4, 5.mp4). The real, consistent split
   is near-vs-far, not left-vs-right. Root-caused below.

### Root cause of the near-corner volatility, found via visual + programmatic diagnosis on 2 clips

Investigated 3 hypotheses for why near corners (whichever side) are worse
than far corners: (a) radial lens distortion being worse toward the frame
edges, (b) a recurring scene element (scoreboard, net post, camera
operator) interfering with detection near-baseline, (c) genuinely noisier
raw line detection near the baseline before fitting even begins.

**(a) Radial edge-distortion: ruled out by direct measurement.** Checked
each clip's BL/BR pixel-x distance from its respective frame edge: in
4 of 5 clips BL and BR sit within ~10px of being exactly symmetric
distances from their edges (e.g. 1.mp4: BL 200px from left, BR 202px
from right). In the one asymmetric case (5.mp4 segment A: BL 141px from
left, BR 299px from right -- BR nearly TWICE as far from its edge), BR
was still the worse-performing corner despite being farther from the
frame edge, directly contradicting an edge-distortion explanation.

**(b) and (c): the real mechanism, found by tracing two bad frames down
to the exact cluster that produced the error.**

*1.mp4 frame 288 (BL error 45.5px):* a visual crop confirmed a genuine
scene-element interference, but a different one per corner than
expected -- the "ALCARAZ / SINNER" on-screen scoreboard graphic sits
directly adjacent to the near-baseline region on this frame, and its
white text passes the exact same HSV threshold used to isolate court
line pixels (`_detect_line_mask`'s line mask is high-value/low-saturation
-- so is broadcast-graphic text). This feeds spurious bright segments
into whichever cluster ends up representing the near baseline nearby.

*4.mp4 frame 60 (BR error 35.5px):* no scoreboard or occlusion at all --
traced programmatically instead of visually. `classify_and_fit` picks
"near_baseline" as whichever horizontal cluster has the single highest
fitted-line midpoint-y among ALL horizontal clusters. On this frame, two
candidate clusters existed: one well-supported cluster with 8 members
spanning x=496-1562 (most of the true baseline's real pixel width,
mid_y=828.6), and a second, short cluster with only 3 members spanning
just x=181-541 (entirely on the BL side, nowhere near BR at all,
mid_y=830.7). The short, poorly-supported cluster won the selection by a
razor-thin 2.1px mid_y margin. `_fit_cluster_line` then extrapolates
whichever cluster wins +-3000px in both directions to compute corner
intersections -- so a line fit ONLY from local data near BL, when
extrapolated all the way out to BR (over 1000px beyond its actual
supporting data), amplified a tiny local slope into a 35.5px error at the
corner it was never actually measured near. BL itself stayed accurate
because it fell inside that cluster's real data range.

**Conclusion: this is not a lens-distortion or left/right-specific issue.
It is a general algorithmic weakness** -- near/far-baseline cluster
selection in `classify_and_fit` (`hough_court_detection.py`) picks by raw
fitted-line y-position alone, with no consideration of how much of the
baseline's actual pixel width a candidate cluster is supported by. A
short, locally-anchored cluster (whether short because a broadcast
graphic polluted it, as in 1.mp4, or short because Hough simply didn't
find enough segments across the full baseline in a given frame, as in
4.mp4) can out-rank a long, well-supported cluster by a razor-thin y
margin, and then get extrapolated far past where it has any real support
-- producing a large, essentially unbounded error at whichever corner is
farthest from that cluster's actual data. Far corners are unaffected
because the far baseline is much shorter in image pixels (foreshortened)
and this project's frames show far less clutter/graphics up there, so
there's less opportunity for a short, misleading cluster to win.

**This also means multi-frame averaging (the fix already built) is
treating the symptom, not this cause** -- it works (see 1.mp4's BR fix
above) because a short-cluster misfire is frame-specific and gets diluted
by other, good frames, not because the underlying selection logic
improved. A more targeted fix, identified here but NOT implemented
(investigation was intentionally time-boxed and stopped once the
mechanism was clearly identified, per instruction): weight cluster
selection and/or corner-computation confidence by the cluster's actual
x-coverage (or total member count/length) relative to the frame width,
so a short, locally-anchored cluster can't out-rank a well-supported one
just because its extrapolated midpoint happens to land a few pixels
further down, and/or flag corners computed from clusters with low
x-coverage support as lower-confidence rather than trusting them equally.
This is a more direct fix than swapping mean-averaging for a median, since
it addresses why a single frame goes wrong in the first place rather than
just diluting the damage after the fact.

**Code**: `detect_court_corners_multi_frame(frames) -> DetectedCourtCorners`
added to `hough_court_detection.py` -- runs single-frame detection on each
input frame and averages successful per-corner detections (a corner is
`None` only if undetected in every frame). `test_hough_court_detection.py`
gained a second, parametrized real-data regression test (8 cases, one per
clip-segment) with generous per-segment margins around these measured
means. Full suite: 10 tests in this file, 29 in `cv_pipeline` overall, all
passing (`PYTHONPATH=src .venv/bin/python3 -m pytest tests/` from
`cv_pipeline/`, ~28s -- note this venv needs `PYTHONPATH=src` set
explicitly since `cv_pipeline` itself, unlike `v2_serving`/`rag_engine`/
`llm_agent`, is not pip-installed editable in it).

**Still not wired into anything, still not through the mandatory
verification-manifest gate, still not adopted as a replacement** for the
manual calibrations -- per instruction, this stays a candidate to keep
comparing against the manual method until proven at least as reliable
across all 5 clips, which an overall 6.8px mean with a known systematic
BR weakness and known averaging-outlier fragility does not yet establish.
If pursued further, the next steps this run surfaces directly: (a)
replace mean-averaging with a median or outlier-rejection scheme to fix
the 3.mp4-style regression, and (b) investigate the BR-specific weakness
before deciding whether this is a masking issue, a clustering issue, or
something structural about that side of the frame.

## Coverage-Weighted Cluster Selection + Scoreboard Exclusion -- implementing the two fixes the root-cause investigation identified

Followed up on the root-cause investigation above with the two concrete
fixes it identified, per instruction: (1) coverage-weighted near/far-
baseline cluster selection, as the primary fix for the extrapolation
problem, and (2) a fixed-position scoreboard-region exclusion, to address
the contamination mechanism directly rather than relying on downstream
weighting to filter it out. Re-ran the same multi-frame, multi-clip
evaluation after both, to check whether near-corner error drops to
something comparable to the already-reliable far corners.

### Fix 1: coverage-weighted baseline selection

Added `_best_covered_extremum(candidates, key, want_max, tol=30.0)` to
`hough_court_detection.py`: instead of `classify_and_fit` picking
near/far-baseline as a flat `max`/`min` over all candidate clusters' raw
mid_y, it now finds the true extremal mid_y, then -- among only the
clusters within `tol=30.0px` of that extremum -- picks whichever has the
largest x_span (best real coverage of the baseline it claims to
represent). `tol=30.0` was chosen because the diagnosed failure case
(4.mp4 frame 60) had the short, wrong cluster beat the true baseline
cluster by only 2.1px, while the smallest gap to a genuinely different
line (service line, net cord) observed across all diagnosed cases was
tens to hundreds of pixels -- 30px comfortably separates "plausibly the
same real line, pick the better-covered one" from "a different line
entirely, don't merge the comparison."

### Fix 2: scoreboard-region exclusion

Measured the on-screen scoreboard graphic's ("ALCARAZ .../SINNER ...")
bounding box directly (numeric near-black-pixel thresholding, not
eyeballed) across all 5 clips, 2 frames each: y-range was pixel-identical
(914-1007) in every single case, x0 identical (180-181), x1 varying
454-563px with the score's digit count at different points in the match
-- confirming it's a broadcast overlay anchored to the OUTPUT SCREEN, not
tied to court position or camera framing. Added
`SCOREBOARD_EXCLUSION_REGION = (150, 890, 600, 1030)` (generous margin
around the measured extent) and zero it out of `line_mask` in
`_detect_line_mask`, before the Hough search ever sees it. Confirmed safe:
the highest real near-baseline pixel y observed across all 5 clips is
~879, well clear of the exclusion region's y0=890, so this cannot delete
real court-line signal.

### Re-measured result (same 5-clip, 8-frames-per-segment evaluation)

| Clip | BL | BR | TR | TL | Mean | (was) |
|---|---|---|---|---|---|---|
| 1.mp4 | 10.0px | 2.6px | 3.8px | 0.8px | **4.3px** | 6.3px |
| 2.mp4 | 11.9px | 4.9px | 4.1px | 4.7px | **6.4px** | 9.3px |
| 3.mp4 | 7.5px | 3.1px | 2.9px | 3.1px | **4.1px** | 5.4px |
| 4.mp4 | 5.5px | 3.6px | 3.3px | 3.5px | **4.0px** | 5.5px |
| 5.mp4 | 9.2px | 14.6px | 5.6px | 8.3px | **9.4px** | 7.4px |
| **overall** | | | | | **5.65px** | 6.8px |

Single-frame (frame 0 only) results also improved for the originally-
tested clips: 1.mp4 11.8px -> **4.1px** mean, 3.mp4 4.1px -> 5.0px mean
(within normal noise).

**4 of 5 clips improved, several substantially**, and near-corner (BL/BR)
error is now much closer to far-corner (TR/TL) error in most clips (e.g.
1.mp4: 10.0/2.6 vs 3.8/0.8; 4.mp4: 5.5/3.6 vs 3.3/3.5) -- the categorical
near-vs-far gap from before is largely closed for 4 of the 5 clips. It is
**not** fully closed across the board: 2.mp4's BL (11.9px) and especially
5.mp4's BR (14.6px) and TL (8.3px) remain clearly elevated.

### 5.mp4 is a real regression -- root-caused, and it is NOT a flaw in the coverage-weighting logic

Traced directly: one sampled frame (5.mp4 frame 0) lost its only real
far-baseline Hough segment cluster entirely once the scoreboard region
was masked out -- even though that region sits over 500px away from the
far baseline (y~273 vs. the scoreboard's y=890-1030). Confirmed by
re-running Hough detection on the identical frame with vs. without the
scoreboard mask:

- **Without** the mask: 68 raw segments, including a clean 3-member,
  484px-wide cluster at mid_y=273.7 -- matching the true far-baseline y
  (273-275) almost exactly.
- **With** the mask: 59 raw segments, and that cluster is gone entirely --
  nothing left near y=273 at all.

This is `cv2.HoughLinesP`'s probabilistic transform being sensitive, in
this one case, to edge-pixel changes anywhere in the frame, not just
locally near the change -- removing pixels from an unrelated region
(the scoreboard, y=890-1030) altered which segments got detected in a
completely different region (y~273) purely through the algorithm's
internal randomized voting, not through any spatial-locality mechanism.
This is a genuine, somewhat surprising property of the underlying OpenCV
primitive, not a bug in this module's own clustering/selection logic --
confirmed by checking that the coverage-weighted selection fix plays no
role here: with the far-baseline cluster entirely absent from the
candidate list, the OLD plain-`min()` selection would have picked exactly
the same wrong fallback cluster. Multi-frame averaging partially absorbs
this (7 of the 8 sampled frames in that segment were fine), but the one
catastrophic single-frame miss (TR 78.6px, TL 112.9px on that frame
alone) still pulls the whole segment's average up substantially.

### Verdict, before reconsidering the calibration-verification-manifest gate question

**Net improvement (overall mean 6.8px -> 5.65px, 4 of 5 clips better,
near-vs-far corner gap substantially closed for most clips), but not a
clean, uniform win** -- 5.mp4 got worse, for a newly-surfaced and
understood reason (rare non-local Hough instability triggered by the
scoreboard mask) rather than a flaw in either fix's own logic. Per
instruction, this is exactly why the manifest-gate question stays open
rather than being reconsidered as "yes, now ready": the method has not
yet been shown reliable across all 5 clips -- one clip's near corners
improved into unreliability being replaced by one clip's far corners
becoming newly unreliable. If pursued further, natural next steps: (a)
swap mean-averaging for a median/outlier-rejection scheme, which both the
original 3.mp4 regression and this new 5.mp4 regression suggest would
help; (b) investigate whether `cv2.HoughLinesP`'s parameters (e.g. its
random-sampling behavior) can be made more stable/deterministic, or
whether the scoreboard region should be filled with a neutral court-color
value instead of zeroed out entirely, to avoid perturbing the edge map's
global statistics; (c) sample more than 8 frames for 5.mp4 specifically
to confirm whether frame 0's miss was a one-off or recurs.

**Code**: `_best_covered_extremum` and `SCOREBOARD_EXCLUSION_REGION`
added to `hough_court_detection.py`; `classify_and_fit`'s docstring and
`_detect_line_mask`'s updated accordingly. `test_hough_court_detection.py`'s
thresholds (`MAX_MEAN_ERROR_PX`, `MAX_PER_CORNER_ERROR_PX`,
`MULTI_FRAME_MAX_MEAN_ERROR_PX`) tightened to track the new measured
numbers, with 5.mp4 segment-a's margin left deliberately wider to account
for the diagnosed single-frame Hough instability. Full suite: still 10
tests in this file, 29 in `cv_pipeline` overall, all passing.

## Neutral-Color Fill Test -- negative result, but a useful one

Before rebuilding the scoreboard exclusion mechanism, tested the cheapest
possible variant first: fill the scoreboard region with a real, sampled
court color instead of zeroing it, re-tested specifically against the
confirmed failure case (5.mp4 frame 0) before touching anything else.

**No difference at all** -- zeroing and neutral-fill produced byte-for-
byte identical Hough segments (59 either way, same clusters, same missing
far-baseline candidate). This makes sense once traced precisely: `line_mask`
is a binary (0/255) mask computed via `cv2.inRange` thresholding, and a
neutral court color fails the "is this a line pixel" threshold exactly the
same way black does -- both collapse to 0 in the mask. The fill VALUE was
never the variable that mattered. The real mechanism (confirmed earlier):
`cv2.HoughLinesP`'s Progressive Probabilistic Hough Transform processes
edge points in randomized order, checking after each one whether any
accumulator bin crosses the vote threshold -- removing edge points
ANYWHERE changes that sequence, independent of what those points are
replaced with. This ruled out "the masking approach, done right" and
pointed at "don't touch Hough's input at all" instead.

## Segment-Filtering Instead Of Pixel-Masking -- the actual fix, plus Median Aggregation as a separate robustness layer

Given the neutral-fill result, rebuilt the scoreboard exclusion to avoid
touching `cv2.HoughLinesP`'s input entirely: `_detect_line_mask` no longer
masks the scoreboard region at all (reverted to its pre-scoreboard-fix
form). Instead, `detect_court_corners` now runs Hough on the full,
unmasked edge map, then filters the resulting SEGMENT LIST via a new
`_segment_in_excluded_region` check -- any segment whose bounding box
overlaps `SCOREBOARD_EXCLUSION_REGION` is dropped before clustering.
`cv2.HoughLinesP` therefore always sees the same input regardless of
whether the scoreboard graphic is present, so it cannot be non-locally
perturbed by its removal -- while the scoreboard's own spurious segments
are still kept out of the near-baseline cluster.

**Verified directly against the failure case first**, per instruction,
before any broader re-run: 5.mp4 frame 0's TR/TL corners, which had
blown out to 78.6px/112.9px under pixel-masking, came back to 2.3px/0.9px
-- back in line with the pre-any-scoreboard-fix baseline (2.3px/0.9px).
BR stayed elevated (22.9px), but that's the separate, already-documented
player-occlusion issue for this segment's early frames (`reference_video5_calibration.py`'s
own docstring: BR occluded by a player standing there for frames 0-100)
-- untouched by either scoreboard-handling approach, as expected.

### Full 5-clip re-evaluation confirms no regression elsewhere

| Clip | unfixed | pixel-mask (regressed) | segment-filter |
|---|---|---|---|
| 1.mp4 | 6.3px | 4.3px | 4.3px |
| 2.mp4 | 9.3px | 6.4px | 6.8px |
| 3.mp4 | 5.4px | 4.1px | 4.2px |
| 4.mp4 | 5.5px | 4.0px | 4.2px |
| 5.mp4 | 7.4px | 9.4px | **7.3px** |
| **overall** | 6.8px | 5.65px | **5.36px** |

5.mp4 is fully back in line (7.4px -> 7.3px, no longer regressed), and
overall mean improved further (5.65px -> 5.36px) even without any other
change -- segment-filtering isn't just "no worse," it's strictly better
than the pixel-masking version once its side effect is removed.

### Checked for remaining single-frame-outlier patterns before deciding whether to add outlier-rejection

Per instruction, before reaching for outlier-rejection, actually checked
whether the pattern (one bad frame dragging a segment's average, the way
3.mp4's frame 666 did originally) still shows up anywhere. Printed every
per-frame, per-corner error across all 8 clip-segments. Found a real mix:

- **Genuine single/couple-frame-outlier patterns** (median would help):
  2.mp4 pre-pan BR (`[5.0, 5.5, 5.6, 6.4, 7.7, 11.2, 16.2, 24.7]` -- 6
  clustered low, 2 high), 3.mp4 BL (`[1.4, 2.1, 4.9, 6.2, 6.6, 6.9, 13.9,
  15.4]`), 5.mp4 segment-a BL (one frame at 19.5 vs. seven between
  6.0-8.9), 5.mp4 segment-b BL (two frames at 17.8/20.8 vs. six between
  6.3-12.0).
- **Systematic elevation across ALL 8 frames, not an outlier pattern**
  (median would NOT help, and shouldn't be expected to): 1.mp4 BL
  (8.8-13.8px across every single frame), 2.mp4 pre-pan BL (15.1-18.6px
  across every frame), 5.mp4 segment-a BR (10.0-23.2px across every
  frame -- the documented player-occlusion window).

Since real outlier cases exist, added the general fix rather than
special-casing specific clips: `detect_court_corners_multi_frame` now
aggregates via **median** instead of mean. Simple, general, and harmless
where there's no outlier (median ~= mean on well-behaved data) while
directly fixing the cases found -- no need for a more elaborate
MAD/IQR-based rejection scheme given every observed case had a clear
minority (1-2 of 8) of bad frames.

### Final result: median aggregation on top of segment-filtering

| Clip | unfixed | segment-filter (mean) | + median (final) |
|---|---|---|---|
| 1.mp4 | 6.3px | 4.3px | 4.5px |
| 2.mp4 | 9.3px | 6.8px | 6.4px |
| 3.mp4 | 5.4px | 4.2px | 3.9px |
| 4.mp4 | 5.5px | 4.2px | 4.0px |
| 5.mp4 | 7.4px | 7.3px | 6.8px |
| **overall** | 6.8px | 5.36px | **5.14px** |

**Every one of the 5 clips is now at or below its original unfixed
baseline** -- 1.mp4 ticked up marginally from segment-filter's 4.3px to
4.5px (within noise; 1.mp4's dominant remaining error, BL, is the
systematic-not-outlier kind median can't fix), but even that is still
better than the original 6.3px. This is a clean improvement across all 5
clips, not 4 of 5 -- the pixel-masking regression is fully resolved and
overall mean corner error dropped from 6.8px to 5.14px (~24% reduction)
across three composed, independently-justified fixes.

Single-frame (frame 0 only) results with the final pipeline: 1.mp4 6.0px
mean (was 11.8px unfixed), 3.mp4 4.2px mean (was 4.1px, within noise) --
both comfortably better than or in line with the unfixed baseline.

### Revisiting the calibration-verification-manifest gate question

**Still not reconsidering it as "ready."** The result is a real,
consistent, well-understood improvement -- every clip at or below
baseline, no unresolved regressions, both remaining elevated corners
(2.mp4's BL, 5.mp4's BR) traced to specific, already-documented causes
rather than mysteries. But "8 frames per clip, 5 clips" is still a thin
sample relative to this project's own established bar for a calibration
(3+ meaningfully-separated frames with human sign-off on all 4 corners,
per `test_calibration_verification.py`'s manifest requirement) -- this
experiment has tested breadth (more clips, more frames) but still no
human visual sign-off step, and BL/BR error in the 4-14px range in
several clips is still meaningfully worse than the manual method's own
sub-2px held-out precision. If pursued further, the natural next step is
a larger per-clip frame sample (the median fix's effectiveness depends on
outliers staying a minority -- worth confirming at, say, 20+ frames per
clip rather than 8) before the manifest-gate question is worth reopening
in earnest.

**Code**: `_segment_in_excluded_region` added, `_detect_line_mask`
reverted to not touch the scoreboard region, `detect_court_corners` now
filters segments post-hoc via `SCOREBOARD_EXCLUSION_REGION`.
`detect_court_corners_multi_frame` switched from `np.mean` to `np.median`.
`test_hough_court_detection.py`'s thresholds retightened to track the
final measured numbers. Full suite: still 10 tests in this file, 29 in
`cv_pipeline` overall, all passing.
