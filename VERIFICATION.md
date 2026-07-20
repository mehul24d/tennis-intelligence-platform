# Verification Guide

This is for a skeptical reader: for each major claim in `RESEARCH_REPORT.md` and
`PROGRESS.md`, what to run yourself and what you should see. Every command below was
actually run against this exact commit before this file was written — none of it is
copied from memory of an earlier pass.

**Honesty up front, per `README.md`'s "What's not included" section**: raw video
clips, v1's raw/processed match datasets, and the RAG engine's persisted vector index
are excluded from this repo (large, and in the video case, not freely redistributable
source footage). Claims whose *code* is fully tested but whose *headline number*
depends on that missing data are marked **(data not included)** below — the checklist
says exactly what that means for each one, rather than leaving it ambiguous.

All commands assume you're at the repo root. Component venvs referenced below are
each component's own `.venv` (see `README.md` for setup); `cv_pipeline/.venv` is
shared by `cv_pipeline`, `llm_agent`, and `v2_serving` (see README for why).

---

## 1. v1 win-probability engine — bug fixes and retrain

**Claim**: the `PtWinner` server-relative/literal convention bug (§4.2 of
`RESEARCH_REPORT.md`) was found, fixed everywhere it touched, and the retrained model
beats the pre-fix model in all 4 rolling-origin folds.

**Run**:
```
cd tennis-intelligence-platform
PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -q
```
**Expect**: `211 passed`. This is the full v1 regression suite, including
`tests/unit/test_ml_informed_markov.py` (the corrected server-relative point-winner
derivation) and `tests/unit/test_golden_markov_outputs.py` (a golden-value regression
guard — this one specifically **skips**, not fails, if
`data/processed/day11_head_to_head_v2_predictions.parquet` isn't present, since that
file is excluded per the data policy above; the other 210 tests don't depend on it).

The retrain comparison numbers themselves (log_loss/Brier across 4 folds, §6.3) are
a one-off analysis captured in `tennis-intelligence-platform/docs/
ptwinner_convention_correction.md`, not a re-runnable script — re-deriving them from
scratch would require the full ~198k-match dataset **(data not included)**.

---

## 2. RAG retrieval quality (precision@k)

**Claim**: mean precision@5 = 0.517 across 12 queries (7 match-summary, 2
player-profile, 3 point-level) against the live 23,747-document index, with a
measured, reported-not-hidden regression on 2 match-summary queries after point
documents were added (§6.1).

**Code-level check (always reproducible)**:
```
cd rag_engine
PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -q
```
**Expect**: `19 passed`. These tests build their own temporary Chroma index
(`tempfile.mkdtemp()`) — they verify the ingestion/embedding/retrieval *mechanism*
works correctly, not the specific precision@k numbers above.

**The actual precision@k numbers — (data not included)**: reproducing §6.1's table
requires the real, persisted 23,747-document index at `rag_engine/data/chroma/`
(430MB, gitignored) built from v1's full dataset. The evaluation script itself is
real and re-runnable once that index exists:
```
cd rag_engine
PYTHONPATH=src .venv/bin/python3 scripts/precision_at_k_eval.py
```
To build the index from scratch you'd need v1's full match dataset (not included) and
`rag_engine/src/rag_engine/build_index.py` — per `RESEARCH_REPORT.md` §3, the full
corpus embed takes multiple hours on CPU-only hardware, which is exactly why this
project scoped it to a documented subset rather than the full ~198k matches.

---

## 3. LLM agent grounding

**Claim**: `TennisAnalystAgent` cites every factual claim with `[L#]`/`[D#]` tags, and
3 numeric-heavy answers were independently re-verified against raw parquet rows,
bypassing the RAG layer entirely, with an exact match all 3 times (§3).

**Code-level check (always reproducible, mocked)**:
```
cd llm_agent
PYTHONPATH=src /path/to/cv_pipeline/.venv/bin/python3 -m pytest tests/test_agent.py -q
```
**Expect**: `5 passed`. Mocked — verifies the citation/grounding *scaffolding*
(prompt construction, source tracking, hedging enforcement), not a live model call.

**The grounding claim itself — (data not included / requires a live API key)**:
`llm_agent/tests/manual_eval.py` is the real, unmocked 10-question harness that calls
the actual Gemini API against actual retrieved context — requires `GEMINI_API_KEY` in
`rag_engine/.env` (see `rag_engine/.env.example`) and the same persisted RAG index as
§2 above. Not part of the pytest suite by design (it calls a paid, non-deterministic
external API) — run manually:
```
cd llm_agent
PYTHONPATH=src /path/to/cv_pipeline/.venv/bin/python3 tests/manual_eval.py
```
The 3-answer cross-check against raw `matches_with_elo.parquet` rows (§3) was a
one-off manual verification, not a script — the method is described in
`RESEARCH_REPORT.md` §3 precisely so it can be repeated by hand against the same
questions once you have the data.

---

## 4. CV pipeline detection accuracy (player/ball/pose/homography)

**Claim**: near-player detection 91.3–99.8% (mean 96.4%), far-player detection
0–34% (a confirmed hardware limit, not a bug), ball detection 53.91% pooled recall
via the combined YOLO+motion-diff method (up from 7.81% stock YOLO), 1 of 10
homographies independently validated (§6.2, full detail in
`cv_pipeline/EVALUATION_REPORT.md`).

**Code-level check (always reproducible)**:
```
cd cv_pipeline
PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -q
```
**Expect**: `29 passed`. Covers ball-trajectory Kalman gap-filling, shot
classification, the calibration-verification manifest gate, and the experimental
Hough-detection module (see §6/§7 below) — all with synthetic or checked-in fixture
data, no video required.

**The accuracy numbers themselves — partially (data not included)**: the ground-truth
annotation CSVs (`data/cv_annotated/annotations/*.csv` — player/ball/court positions
for 10 clips) **are** included in this repo, since they're small and are the actual
evidence the numbers above were computed from. What's **not** included is the source
video those annotations were measured against (`data/cv_annotated/videos/`, excluded
per the data policy) — re-running detection to reproduce the recall/precision numbers
requires supplying your own copy of those 10 clips at that path. The scripts that
compute these numbers (`cv_pipeline/scripts/run_full_detection_validation.py`,
`run_tracking_validation_all_clips.py`) are real and included; they just need the
video files present to run.

---

## 5. Shot classification (serve-exclusion rule)

**Claim**: 87.5% of flagged non-serve shot events are real shots (14.6%
contamination), confirmed manually under the actual production configuration
(Addendum, 2026-07-19).

**Code-level check (always reproducible)**:
```
cd cv_pipeline
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_shot_classification.py -q
```
**Expect**: `9 passed`. Covers `flag_first_event_as_probable_serve`'s logic
(earliest-event-across-roles, earliest-within-role, no-events, no-mutation) with
synthetic events — deterministic, no video required.

**The 87.5%/14.6% figures — not mechanically re-runnable, by nature**: this was a
manual, visual, per-event audit (facing direction, zoom, frame-by-frame confirmation
against real video) — the methodology is fully described in `PROGRESS.md`'s
"Serve-Exclusion Heuristic" entry so it can be repeated by a human against the same
clips, but there is no script that reproduces a human visual judgment call. What
*is* verifiable mechanically is that the shipped rule's code does exactly what
`PROGRESS.md` says it does (via the test suite above), and that it's wired into
`video_pipeline.py` only after — not before — that audit passed.

---

## 6. Ball detection: motion-diff recovery + the ground-truth-leak correction

**Claim**: pooled recall 7.81% (stock YOLO) → 53.91% (combined method), corrected
down from an invalid 70.40% after a ground-truth leak in candidate selection was
found and fixed (Addendum, 2026-07-19).

**Code-level check**: covered by the same `cv_pipeline` suite as §4 (`ball_detection`,
`ball_detection_combined`, `ball_trajectory_kalman` modules are exercised by
`tests/test_ball_trajectory_kalman.py` and indirectly by the pipeline-level tests).

**The recall numbers — (data not included)**: same constraint as §4 — reproducing
7.81%/53.91% requires the real video files at `data/cv_annotated/videos/`. The
scripts are real and included: `cv_pipeline/scripts/ball_detection_experiments.py`,
`ball_finetuned_combined_eval.py`. The ground-truth-leak story itself (§Addendum,
`PROGRESS.md`'s "GROUND-TRUTH LEAK FOUND" entry) is a one-off investigation with a
before/after number, not a re-runnable script — the fixed logic it resulted in
(largest-area candidate selection) is what's shipped in
`cv_pipeline/src/cv_pipeline/ball_detection_combined.py` today.

---

## 7. Automated Hough-transform court calibration (experimental)

**Claim**: overall mean corner error across all 5 clips = 5.14px (down from 6.8px
before two fixes + a switch to median aggregation), still below this project's own
bar for the mandatory calibration-verification-manifest gate (Addendum, 2026-07-19).

**Run**:
```
cd cv_pipeline
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_hough_court_detection.py -v
```
**Expect (with `data/tennis/*.mp4` present)**: `10 passed` (~30s — opens and decodes
real video frames). **Expect (without the video files, i.e. a fresh clone with no
data supplied)**: `10 skipped` — every test in this file explicitly checks whether
its required video file exists first and calls `pytest.skip(...)` if not, rather than
failing or silently reporting a false pass. This is the one CV test file written to
degrade gracefully without the video data; worth checking it actually does, not just
trusting the code comment.

---

## 8. Rendered output video (codec fix)

**Claim**: the first render used the `mp4v` FourCC (unplayable in any mainstream
browser despite downloading successfully); fixed by switching to `avc1` (real H.264).

**Run**:
```
cd v2_serving
PYTHONPATH=src /path/to/cv_pipeline/.venv/bin/python3 -m pytest tests/test_video_render.py -v
```
**Expect**: `3 passed`. Fully self-contained — these tests write their own tiny
synthetic source video via `cv2.VideoWriter` and inspect the real output bytes
(`cv2.VideoCapture(...).get(cv2.CAP_PROP_FOURCC)`), so no project video data is
needed. `test_rendered_output_uses_a_browser_playable_codec` is the actual regression
guard against the original bug — it asserts the output FourCC is never `mp4v`/`FMP4`.

---

## 9. Full-repo test suite (all 5 components, one pass)

**Claim**: `RESEARCH_REPORT.md` §6.5 reports 211 + 29 + 19 + 5 + 29 = 293 tests
passing across v1, `cv_pipeline`, `rag_engine`, `llm_agent`, and `v2_serving`.

**Run each of the 5 commands in §1–5 above in sequence** (or see `README.md`'s setup
section for the one-time venv install steps first). Expected: 293 passed total, 0
failed, plus 10 `skipped` from §7 if `data/tennis/*.mp4` isn't supplied. This exact
sequence was run against this exact commit before writing this file — see the commit
message / `PROGRESS.md`'s most recent entries for the real pass counts at the time
they were last verified.
