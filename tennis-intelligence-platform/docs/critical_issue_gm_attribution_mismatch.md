# CRITICAL: Gm1/Gm2 game-count attribution is wrong whenever player 2 serves

**Status: RESOLVED — root cause was NOT Gm1/Gm2 (they were correct all along). See
`docs/ptwinner_convention_correction.md` for the actual fix and full record.** This
doc is kept for its investigation history (the hypothesis-testing sequence that led
to the real answer), but its own title and framing are now known to be wrong: the
"Gm1/Gm2 attribution" appeared broken only because `PtWinner` was being read under
the wrong (server-relative) convention at the time this doc was written. Once
`PtWinner` was corrected back to its original, literal, fixed-player-relative
convention, `Gm1`/`Gm2` matched it at 99.91% (the residual consistent with ordinary
charting error) — `Gm1`/`Gm2` needed no fix at all. Do not act on this doc's original
"Gm1/Gm2 is broken" framing; read `ptwinner_convention_correction.md` instead.

---

*(Original doc follows, preserved for the investigation record.)*

Found 2026-07-14
while verifying `rag_engine/ingest/point_documents.py`'s swing-neutral phrasing on an
automatically-flagged example. Escalated because it is a **v1 foundational-correctness
issue**, not a v2/RAG-layer concern — `Gm1`/`Gm2` are read directly by
`row_to_match_state` (`src/tennis_intel/live/match_state_conversion.py`) into every
engine's `MatchState`, used throughout the entire live win-probability system (Markov,
ML+MC, ML-informed, hybrid), `point_timeline_service`, `match_summary_service`,
`model_agreement_service`, `replay_match_by_id`, and all downstream evaluation/
calibration pipeline scripts.

**BLOCKS ALL FURTHER V2 WORK** — `point_documents.py`, the LLM agent, everything —
until resolved. Do not resume v2 work before this is fixed and re-verified.

## How this was found

While picking an automatically-flagged (not hand-picked) `notable_point` example for
`point_documents.py`, scanned the corpus and got
`19921031-M-Stockholm_Masters-SF-Stefan_Edberg-Goran_Ivanisevic` point 36 — Edberg wins
the point at 0-40 (receiver, triple break point) — this should unambiguously end the
game in Edberg's favor. Instead, the following game-boundary row showed `Gm2`
(Ivanisevic's, supposedly) incrementing, not `Gm1` (Edberg's). This didn't fit the
"next-point-context" pattern already documented in
`known_issue_after_point_swing_includes_next_point_context.md` — it's a disagreement
about the raw score state itself, not the model's interpretation of it.

## Confirmation #1: hand-traced, two independent methods, zero shared code

Match: `19690703-M-Wimbledon-SF-Rod_Laver-Arthur_Ashe` (mcp_Player 1 = Laver,
mcp_Player 2 = Ashe; day6 confirms Laver won the match; real historical score
`2-6 6-2 9-7 6-0`).

**Method A** — the already-shipped, already-audited `_point_winner_is_p1` helper
(`match_summary_service.py`): traced every point from 1 to 21 by hand. At point 20
(entering it: Laver at Advantage, Ashe at 40, `Svr=2`, `PtWinner=2`), the formula gives
`point_winner_is_p1 = True` — Laver wins the point, ending the game in his favor.

**Method B** — pure first-principles score-delta reasoning using *only*
`p1_points`/`p2_points` (ordinal 0,1,2,3=40,4=AD), with **no reference to `PtWinner`,
`Svr`, or any existing helper function**: traced the same 21 points via score deltas
alone (whoever's ordinal increased won; AD reverting to deuce means the advantage
holder lost; deuce advancing to AD means that player won). At point 19→20, Laver
(`p1_points`) advances 3→4 (deuce→AD) — Laver won point 19. Entering point 20, Laver is
at AD. A new game demonstrably starts at point 21 (`Gm1+Gm2` changes). From an
advantage score, a game can *only* end if the advantage-holder wins the point (losing
it reverts to deuce, extending the game) — so point 20 **must** have been won by
Laver, purely by elimination, independent of any interpretation of `PtWinner`.

**Both independent methods agree: Laver should win the game.** Actual data:
`Gm2` (Ashe) increments at the point-21 boundary row, not `Gm1` (Laver). Confirmed
contradiction, not a false alarm from either method.

Aggregate cross-check (to rule out `Gm1`/`Gm2` being simply globally swapped): the
final `Gm1`/`Gm2` values at the end of set 1 for this match trend toward 2-6
(`Gm1→2, Gm2→5` observed late in set 1), matching the real historical set-1 score of
2-6 in Ashe's favor. **The aggregate/global `Gm1`=player1, `Gm2`=player2 convention is
correct** — the error is specific to individual game-boundary attribution, not a
wholesale column swap.

## Confirmation #2: corpus-wide scale (vectorized, all 5,981 matches)

For every non-tiebreak game boundary in the full frozen-join corpus: does
`Gm1` incrementing (vs. `Gm2`) match the winner of the immediately preceding
(game-deciding) point, per the already-validated `PtWinner`/`Svr` convention?

- **147,290 game boundaries checked, 71,815 mismatches = 48.76%.**
- Present in **100% of the 5,981 matches checked** (not scattered/rare) — ruling out
  "isolated old-match charting noise" as the explanation on its own (a real per-match
  charting error rate this uniform and this high across every single match, old and
  modern alike, would be implausible).

## Shift test (ruling out a simple one-row indexing lag)

Hypothesis: `Gm1`/`Gm2` are attached to the wrong row by a constant one-row offset
during dataset construction. Tested by comparing the deciding point's outcome against
the `Gm1` increment measured at shifts -1, 0, +1 rows from the original boundary:

| shift | checked | mismatches | rate |
|-------|---------|------------|------|
| -1 | 147,290 | 104,215 | 70.76% |
| **0** (original) | 147,290 | 71,815 | **48.76%** |
| +1 | 147,287 | 94,975 | 64.48% |

The rate does **not** collapse toward 0% at either neighboring shift — it gets *worse*
in both directions, and shift 0 is already the best fit of the three. **A simple
systematic one-row lag/lead is ruled out.**

## Server/returner and p1/p2 split (the decisive finding)

Split the shift-0 mismatches by (a) whether the deciding point was won by the server
(hold) or returner (break), and (b) whether the "expected" game winner (per
`PtWinner`/`Svr`) was player 1 or player 2:

```
                                                 count    sum      mean
decisive_winner_is_server expected_winner_is_p1
False (break)              False (p2 wins)      14353     24    0.0017
                            True  (p1 wins)      57513  57454    0.9990
True  (hold)                False (p2 wins)      14291  14277    0.9990
                            True  (p1 wins)      61133     60    0.0010
```

A hold/break-only split alone is moderate and not clean (server-won mismatch rate
19.0%, returner-won 80.0% — suggestive but murky). The cross-tab resolves it: **the
true driver is not hold vs. break at all** — both "correct" cells and both "wrong"
cells span both hold and break. Isolating purely by **which player served the
deciding point**:

```
decisive_point_svr   count    sum      mean
1 (player 1 serves)  75486     84    0.0011
2 (player 2 serves)  71804  71731    0.9990
```

**This is the clean signature.** When player 1 serves the game-deciding point,
`Gm1`/`Gm2` attribution is correct 99.89% of the time. When player 2 serves the
game-deciding point, attribution is wrong 99.90% of the time — regardless of whether
player 2 holds or is broken, regardless of who the actual game winner is. The bug
tracks server identity at construction time, not point outcome, not hold/break, not a
simple row-shift.

## What this rules in / rules out

- **Ruled out**: a global `Gm1`↔`Gm2` column swap (aggregate set totals check out
  against real history).
- **Ruled out**: a simple one-row indexing lag or lead (shift test).
- **Ruled out**: scattered, low-rate historical charting errors (100% of matches
  affected, ~49% of all boundaries).
- **Ruled out** (by the cross-tab): a hold/break-specific bug (both holds and breaks
  appear on both sides of the clean split).
- **Strongly implicated**: something in the point-level dataset construction pipeline
  (most likely `src/tennis_intel/features/point_level_features.py` and/or
  `src/tennis_intel/data/join_tml_mcp.py` / `build_point_dataset.py`) that correctly
  attaches `Gm1`/`Gm2` to rows when player 1 is serving, but incorrectly does so when
  player 2 is serving — e.g. a join, copy, or forward-fill step keyed off server
  identity, or a "server's games"/"returner's games" intermediate representation that
  only happens to align with the `Gm1`=player1/`Gm2`=player2 convention when player 1
  is the server.

## Next steps (not yet started)

1. Read `point_level_features.py` and the raw MCP → point-dataset join pipeline
   end-to-end, specifically any step touching `Gm1`/`Gm2`, looking for logic
   conditioned on or interacting with `Svr`.
2. Once a candidate mechanism is found, construct a synthetic minimal reproduction
   (a few rows, hand-computed expected `Gm1`/`Gm2`) to confirm before fixing, matching
   this project's established discipline for every prior fix this session.
3. After fixing: re-run the exact same corpus-wide check in this doc (all four
   sub-checks: aggregate cross-check, corpus-wide rate, shift test, server split) and
   confirm the mismatch rate collapses to ~0% (allowing for genuine, rare historical
   charting errors, not the current systematic ~49%).
4. Re-verify every consumer of `Gm1`/`Gm2` downstream: `row_to_match_state` and
   therefore every live-probability engine, `point_timeline_service`,
   `match_summary_service`, `model_agreement_service`, `replay_match_by_id`, and
   flag whether any already-published calibration/evaluation results
   (`docs/day*_freeze.md`, `evaluate_*.py` outputs) need to be treated as suspect and
   potentially re-run, since they may have been computed on the same corrupted
   game-state attribution.
5. Only after 3 and 4 are clean: resume `point_documents.py` and any other paused v2
   work.
