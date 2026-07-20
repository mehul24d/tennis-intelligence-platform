# Known issue: `ml_informed_p1[i]` reflects the score BEFORE point i, not after

**Status: FIXED (2026-07-14).** See "Resolution" below. Found 2026-07-14 while
grounding a RAG document example in real `get_point_timeline` output (v2 planning
work). Related, already-fixed issue: see `ml_informed_markov.py`'s
`a_won_this_point` bug-fix note (`PtWinner` is server-relative, not
player-relative) — that fix landed the same day, but did NOT resolve this
second, separate issue (verified: post-fix direction-correctness on a real
match was 62/119 = 52.1%, essentially unchanged from before that fix).

## The issue

`compute_five_engine_trajectory` (`src/tennis_intel/serving/replay_service.py`,
~line 191) builds one `MatchState` per point via `_row_to_match_state(row)` /
`row_to_match_state` (`src/tennis_intel/live/match_state_conversion.py`), then
calls `ml_informed_markov_p_player1(row, ...)` to get that point's probability.

`row_to_match_state` builds `MatchState` from `p1_points`/`p2_points`/`Set1`/
`Gm1`/etc. — and per `point_timeline_service.py`'s own documented convention
(`_server_perspective_score`'s docstring), these columns represent the score
**before** the point in that row is played, not after.

So `ml_informed_markov_predict(state, row, ...)` computes
`prob_a_wins_match_from_state` using the **pre-point-i** score, blended with
the classifier's context-aware prediction for the upcoming point. The real
outcome of point `i` (`row["PtWinner"]`) is only used afterward, to update the
`ServeReturnPosterior` for *future* points — it is never used to advance the
`MatchState` itself before computing this row's returned probability.

`compute_five_engine_trajectory` appends this value to `ml_informed_p1[i]`.
`point_timeline_service.get_point_timeline` (and every other consumer of the
same shared trajectory: `match_summary_service.py`, `model_agreement_service.py`,
`replay_match_by_id`) then treats `ml_informed_p1[i]` as `probability_after_p1`
for point `i` via `all_p1 = [prematch_p1] + smoothed_p1`. That's a semantic
off-by-one: the value is really "the pre-point-i prediction," not "the
probability once point i's real outcome is known."

## Evidence

On a real match (`20220725-M-Kitzbuhel-R32-Jurij_Rodionov-Hernan_Casanova`,
119 points), checking whether `probability_after_p1 > probability_before_p1`
whenever player 1 (the tracked winner) won a point, and the reverse whenever
player 2 won:

- Pre-`PtWinner`-fix: ~53% direction-correct (near chance).
- Post-`PtWinner`-fix: 62/119 = 52.1% direction-correct (essentially
  unchanged) — confirming the `PtWinner` fix, while correct in isolation
  (unit-tested in `tests/unit/test_ml_informed_markov.py`), is not the
  dominant cause of the observed direction-inconsistency.

## Proposed fix (not yet implemented)

Advance the `MatchState` to the **post**-point-i score (using `row["PtWinner"]`/
`row["Svr"]` to increment points/games/sets correctly, including game/set/
tiebreak transitions) before computing the probability that gets stored as
`ml_informed_p1[i]` — or, more conservatively, keep `ml_informed_p1[i]` as the
pre-point value it actually is and fix the *consumers* (`point_timeline_service.py`
et al.) to align indices correctly (`probability_before_p1` for point `i` =
`ml_informed_p1[i]`, `probability_after_p1` for point `i` = `ml_informed_p1[i+1]`
computed from point `i+1`'s pre-point state, which numerically equals the
post-point-`i` state only if no intervening game/set boundary logic differs —
needs care).

Whichever approach is chosen, it touches a function shared by four services,
so any fix needs: (1) a regression test replaying a short synthetic point
sequence with known scores and asserting the state used matches the intended
before/after semantics at each step, and (2) re-verification of
direction-correctness on the same real match used here, expecting close to
100% (not just "the new test passes").

## Resolution (2026-07-14)

Took the conservative option above: **`ml_informed_p1[i]` itself was left
unchanged** (it correctly represents "the prediction just before point `i`",
which is also the quantity every calibration/evaluation pipeline script
(`evaluate_full_match_calibration.py`, `sweep_prior_strength.py`, etc. — 15+
call sites of `ml_informed_markov_predict`) legitimately wants and already
uses correctly for pre-point-vs-outcome calibration). Changing
`ml_informed_markov_predict`'s semantics would have silently broken that much
larger surface for no benefit.

Instead, precisely traced where each of the four downstream consumers
*re-indexes* the shared `ml_informed_p1` trajectory, and found the bug was
narrower than originally scoped: only **two** of the four actually construct a
before/after pair per point (the other two use the trajectory directly,
point-by-point, with no before/after pairing — never bugged):

- `point_timeline_service.py::get_point_timeline` — was pairing point `i` with
  `(all_p1[i], all_p1[i+1])` = `(smoothed_p1[i-1], smoothed_p1[i])` for `i>0`,
  i.e. **the previous point's** before/after pair, mislabeled as point `i`'s.
- `match_summary_service.py::get_match_summary`'s `largest_probability_swing` —
  same `all_p1 = [prematch_p1] + smoothed_p1` construction, same off-by-one in
  `swings[i] = |all_p1[i] - all_p1[i-1]|`, attributing the largest swing to the
  point *after* the one that actually caused it.
- `model_agreement_service.py::get_model_agreement` — **not bugged.** Uses
  `engine_matrix[i]` (= each engine's pre-point-`i` value) directly per point,
  with no before/after pairing claim.
- `replay_service.py::replay_match_by_id` — **not bugged**, same reason: plots
  `computed[engine][i]` directly per point, no pairing.

**Fix**: both bugged services now compute `prob_before = smoothed_p1[i]` and
`prob_after = smoothed_p1[i+1]` if `i+1 < n_points` else the actual terminal
match outcome (`1.0`/`0.0`) for the match's final point — correct because
`row_to_match_state`'s pre-point convention means row `i+1`'s pre-point state
*is*, by construction of consecutive point-dataset rows, exactly the
post-point-`i` state.

**Verification performed** (see `tests/unit/test_point_probability_indexing.py`
and `tests/unit/test_ml_informed_markov.py`):
1. Hand-computed synthetic 4-point sequence (mocked `compute_five_engine_trajectory`
   output) with a deliberate large swing on point 2 — confirmed the pre-fix code
   attributed it to point 3 (off-by-one), confirmed the post-fix code correctly
   attributes it to point 2, for both `point_timeline_service` and
   `match_summary_service`.
2. Real match (`20220725-M-Kitzbuhel-R32-Jurij_Rodionov-Hernan_Casanova`, 119
   points) direction-correctness: **89.9% overall (107/119)**, and **100%
   (35/35) on every point with a non-negligible swing (≥0.01)** — the 12
   remaining "wrong-direction" points are all tiny-magnitude (≤0.007) noise at
   already near-certain match states (probability already ≥0.98), exactly the
   pre-agreed legitimate-exception category, not a residual bug.
3. All 209 existing unit tests still pass (no regressions).
4. All four downstream consumers exercised end-to-end against the real match
   above: `match_summary_service` and `point_timeline_service` produce
   corrected, sane output; `model_agreement_service` and `replay_match_by_id`
   run unchanged (confirmed never bugged, per the trace above) and their output
   is unaffected, as expected since neither was touched.

`rag_engine/ingest/point_documents.py` (deferred in `PROGRESS.md`) can now be
built against real, trustworthy `get_point_timeline` output.
