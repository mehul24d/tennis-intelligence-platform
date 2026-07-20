# Known issue: "probability after point i" partly reflects point i+1's own context, not just point i's outcome

**Status:** documented, not a bug — a real, inherent property of the ML-informed
engine's per-point context features, distinct from the (fixed) pre-point/post-point
indexing bug in `known_issue_ml_informed_markov_pre_point_state.md`. Found 2026-07-14
while sourcing real `notable_point` examples for `rag_engine/ingest/point_documents.py`.

**RE-VERIFIED 2026-07-14, post `PtWinner` convention fix** (see
`docs/ptwinner_convention_correction.md`): this finding was originally measured using
`point_timeline_service.py`'s `winner` field under the since-reverted server-relative
`PtWinner` convention. Re-checked directly under the corrected literal convention —
**the mechanism holds, and if anything more strongly than originally reported** (see
"Re-verification" section below). The original worked example (Wimbledon point 203)
no longer applies as an illustration — under the corrected convention it turns out to
be a perfectly ordinary, direction-consistent point, not a mismatch at all — and has
been replaced with a new, genuinely mismatched example (point 95) found fresh under
the corrected data.

## The finding

After fixing the pre-point/post-point indexing bug (see the sibling doc), a second
real match (`20230716-M-Wimbledon-F-Novak_Djokovic-Carlos_Alcaraz`) was checked the
same way the first was validated. Direction-correctness on points with a meaningful
swing (≥0.01) was **88.4% (237/268)**, not ~100% like the first match — with several
"wrong-direction" points at moderate probability and non-negligible swing (not the
near-certainty-tail noise pattern from the first match), e.g.:

- Point 203 (score before 0-40, Alcaraz serving, Alcaraz wins the point saving a break
  point): `probability_before_p1` (p1=Djokovic) = 0.7697, `probability_after_p1` =
  0.8378 — Djokovic's probability *rose* 6.8 points despite Alcaraz (not Djokovic)
  winning the point.

## Root-caused via controlled counterfactual (not guessed)

`probability_after_p1` for point `i` = `smoothed_p1[i+1]`, computed from row `i+1`'s
full feature vector via `ml_informed_point_probabilities`. Row `i+1` differs from row
`i` in exactly 9 of 55 feature columns. Diffing them for point 203 showed **every
outcome-derived feature moved consistently in Alcaraz's favor** (his return streak
reset from 7 to -1 was actually Djokovic's streak breaking, `p1_momentum_last10`
0.7→0.6, `p2_momentum_last10` 0.3→0.4) — yet the composed probability moved the
opposite way. One of the 9 differing features was `is_second_serve_point: False →
True` — a property of point `i+1` (204) itself, not a consequence of point `i`
(203)'s outcome.

Two controlled tests (`ml_informed_point_probabilities` called directly, isolating
one change at a time):

1. **Flip only the outcome-derived features** (streak/momentum, as if Djokovic had
   won point 203 instead of Alcaraz), holding `is_second_serve_point` and everything
   else fixed at row `i+1`'s real values: `p_a_serve` moved by only **-0.0058**,
   `p_a_return` by only **-0.0054** — tiny, and even pointed the "wrong" way (Alcaraz's
   predicted probabilities were marginally *higher* under the "Djokovic won"
   counterfactual). This tiny, wrong-signed residual is most plausibly model noise at
   that scale, not a systematic bug — its magnitude is negligible next to what follows.
2. **Flip only `is_second_serve_point`** (holding all point-203-outcome features fixed
   at their pre-point-203 values): `p_a_serve` moved by **-0.2254**, `p_a_return` by
   **+0.2092** — **93% and 96% of the entire real observed swing**, respectively.

This isolates the cause precisely: `is_second_serve_point` (whether point `i+1`
itself will be a second serve — legitimately predictive, since second serves are won
at substantially lower rates than first serves, real tennis signal) dominates the
"after point i" prediction, overwhelming and in this case reversing the much smaller,
correctly-signed-in-principle effect of point `i`'s actual outcome.

## Why this is (b) and not a residual indexing bug

- The state/score mechanics are independently verified correct: `state_after` (via
  `row_to_match_state(row_after)`) exactly matches the real post-point-203 score
  (`p2_points` 0→1, consistent with the server winning the point) — confirmed by
  direct inspection, not assumed.
- The swing's cause is fully attributable to a real, named, sensible feature
  (`is_second_serve_point`) with a well-understood, legitimate direction of effect
  (second serves are weaker) — not an unexplained residual or a sign-flip bug in the
  indexing/orientation logic.
- This is architecturally inherent to how `compute_five_engine_trajectory` was
  designed: `ml_informed_p1[i]` is *supposed* to be context-aware (that's the entire
  point of the "ML-informed" engine vs. the constant-rate pure Markov engine) — the
  context necessarily includes information about the point about to be played, which
  the pre-point/post-point fix's pairing (`smoothed_p1[i]`, `smoothed_p1[i+1]`)
  correctly surfaces as "the next known state," but that next state's own context
  is not purely a function of point `i`'s outcome.

## Re-verification (2026-07-14, post `PtWinner` fix)

Re-ran `get_point_timeline` on the same Wimbledon match with the corrected `winner`
field. Point 203's `PtWinner` value (raw `2` at that row) means **Djokovic (player 1)
won the point** under the literal convention — not Alcaraz as originally reported.
With that correction, `probability_before_p1` (Djokovic) = 0.213,
`probability_after_p1` = 0.232 — Djokovic's own probability rose *while Djokovic won*.
**No longer a mismatch at all**; `direction_matches_winner` would not flag this point
today, so it can no longer serve as an example of the phenomenon this doc describes.

Scanned the same match fresh under the corrected data for a genuine replacement:
**64 hedge-worthy mismatches (swing ≥0.01) still exist** out of 334 points — the
phenomenon is real and still common, just not at point 203 anymore. Picked point 95
(Alcaraz wins the point; Djokovic's probability rises) and re-ran the identical
controlled-counterfactual method:

```
REAL row_after (point 96's context):     p_a_serve=0.6737  p_a_return=0.2746
CF (flip ONLY is_second_serve_point):    p_a_serve=0.4558  p_a_return=0.4971
delta from is_second_serve_point alone:  p_a_serve=+0.2178  p_a_return=-0.2225

REAL total delta (row 95 -> row 96):     p_a_serve=+0.2006  p_a_return=-0.2022
```

`is_second_serve_point` alone explains **108.6%** of the real `p_a_serve` swing and
**110.0%** of the real `p_a_return` swing — it doesn't just dominate, it *overshoots*
the observed total (the other 12 differing features, mostly the same
streak/momentum family as before, net out to a small partial offset). **The
mechanism is confirmed under the corrected convention, at least as strongly as
originally found** — if anything the isolated effect is now even cleaner (>100% vs.
the original 93-96%).

**Conclusion**: the underlying phenomenon this doc describes was never an artifact of
the `PtWinner` bug — it's a real, independent property of `ml_informed_point_probabilities`
and unrelated to which point-winner convention is in use (confirmed by it holding
under both the old and new convention, just attached to different specific points).
The only thing that changed is *which point* serves as the example. `point_documents.py`'s
swing-neutral hedge phrasing remains justified on this finding.

## Implication for `rag_engine/ingest/point_documents.py`

A "notable point" swing at point `i` should **not** be narrated as if caused solely by
point `i`'s own outcome — text like "X's break-point save swung the probability by
6.8 points" would be misleading when a large share of that swing is actually
attributable to point `i+1` being a second serve, unrelated to point `i`'s outcome.
`point_documents.py` should either (a) phrase swing descriptions neutrally ("the win
probability moved by N points around this juncture") rather than asserting causal
attribution to the specific point's outcome, or (b) prefer points where the server/
return-side raw features are stable across the pairing (harder to guarantee without
extra bookkeeping), or (c) simply not claim causality in generated text and let the
retrieved `probability_before`/`probability_after`/`swing` fields speak for
themselves alongside the point's own score/context, leaving causal narration to the
downstream LLM agent (Phase 2), which can reason about it with more context than a
templated ingestion string can.
