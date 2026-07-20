# Day 7 — Point-Sequence Parsing & In-Match State Features (Frozen)

Status: **frozen**. Scope: parsing and leakage-safe in-match state features only — pre-match
feature attachment, the Markov baseline, and the ML+simulation engine are Days 8-9 (see
`live_win_probability_extension_analysis.md`), deliberately not built in this pass.

## What real-data inspection revealed (grounded, not assumed)

Confirmed by direct inspection of `charting-m-points-2020s.csv` before writing any parsing
code (547,478 points in the file), consistent with this project's established discipline of
verifying real schema before coding:

- **`Pt` is clean** — 1..N per match, zero gaps, verified on a real 141-point match.
- **The raw file is NOT sorted chronologically by point within a match** — the first rows
  returned for a match started at `Pt=92`, not `Pt=1`. Explicit sorting by `(match_id, Pt)`
  is mandatory before any processing; skipping it would silently corrupt every downstream
  leakage-safety guarantee.
- **`TbSet` does NOT mean "this point is in a tiebreak"** — confirmed `True` for an entire
  141-point match including ordinary non-tiebreak games late in the match (game 21, score
  progressing 0-0 -> 15-40, standard deuce-set scoring). Initially assumed otherwise; caught
  by direct inspection before it became a bug. Tiebreak status is derived independently
  from `Gm1==6 and Gm2==6`, which is unambiguous.
- **`1st`/`2nd` shot-notation columns were NOT parsed** — deliberately out of v1 scope (see
  `live_win_probability_extension_analysis.md` Section 3). The only signal extracted is
  whether `2nd` is populated (second-serve point), which needs no notation decoding.
  `PtWinner` is already a clean target requiring no derivation from shot notation at all.

## What was built

- `point_score_parser.py`: parses `Pts` notation (regular game scores and tiebreak scores),
  and implements break-point / set-point / match-point detection from first principles
  against standard tennis scoring rules (ad-scoring, best-of-3 and best-of-5, tiebreak
  win-by-2).
- `point_level_features.py`: loads and correctly sorts raw point files, applies score
  parsing and situational flags to every point, and computes leakage-safe **in-match**
  momentum (a player's rolling point-win rate within the current match only — explicitly
  distinct from Day 5's across-match rolling form, which this must never be conflated with).

## Synthetic correctness — every rule hand-verified against real tennis scoring logic

- Score parsing: `"0-15" -> (0,1)`, `"40-40" -> (3,3)`, `"AD-40" -> (4,3)`, tiebreak
  `"6-5" -> (6,5)`
- Break point: 40-40 correctly NOT a break point; 30-40 and 40-AD correctly ARE; AD-40
  (server has advantage) correctly NOT a break point
- Set point: 40-15 while leading 5-3 in games correctly IS a set point; the same score
  while only leading 3-2 correctly is NOT (winning the game wouldn't win the set)
- Match point: correctly requires the right number of sets for the format — up 1 set in a
  best-of-3 correctly counts, but sets-even in a best-of-3 correctly does not; up 2-0 in a
  best-of-5 correctly counts (2+1=3 sets needed)
- Tiebreak win condition: 6-5 -> 7-5 correctly a win; 6-6 -> 7-6 correctly NOT (needs 2-clear)
- Chronological sorting: proven the pipeline correctly reorders an out-of-order input file
- In-match momentum: exact match to hand-calculated value (2 prior points, 1 win -> momentum
  = 0.5 before the 3rd point); first point of a match correctly has no history -> `NaN`, not
  fabricated
- **Leakage test:** removing a future point from the input does not change any earlier
  point's momentum value — verified by direct comparison, same discipline as every prior
  stage's leakage proof

## Explicitly deferred (not silently dropped)

- Attaching pre-match Elo/rolling/serve-return features as static per-point context (Day 8+,
  low-risk — reuses the exact `mcp_match_id`-anchored merge pattern already proven safe in
  the Day 6 bug-fix)
- Full shot-notation parsing (ace/winner/error detail) — a substantial separate undertaking,
  scoped out per `live_win_probability_extension_analysis.md`
- The analytical Markov-chain baseline (Day 8) and point-level ML classifier + Monte Carlo
  simulation engine (Day 9)

## Next step

Day 8: the analytical Markov-chain baseline, validated against known closed-form reference
values from the tennis-analytics literature — the fast, interpretable baseline the ML
approach in Day 9 will be compared against, exactly as Milestone 5 compared tree models
against logistic regression rather than assuming complexity wins.