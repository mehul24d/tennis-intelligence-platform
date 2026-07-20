# Day 8 — Analytical Markov-Chain Baseline (Frozen)

Status: **frozen**. This is the interpretable, theory-grounded baseline that the Day 9
point-level ML + Monte Carlo approach will be measured against — the same "does added
complexity actually beat a principled simple model" framing used in Milestone 5 (trees vs.
logistic regression).

## What was built

- `markov_baseline.py` — closed-form pre-match win probability: `prob_win_game`,
  `prob_win_tiebreak`, `prob_win_set`, `prob_win_match`. Given each player's per-serve
  point-win probability, composes point -> game -> set -> match analytically.
- `live_win_probability.py` — `prob_a_wins_match_from_state(MatchState, ...)`: win
  probability from an ARBITRARY in-match score state (sets/games/points/server), which is
  what makes this a live engine rather than only a pre-match predictor.

## Foundational assumption (stated plainly, per the literature)

Points are i.i.d. GIVEN who is serving (Klaassen & Magnus). Each player has a constant
point-win probability on their own serve. This is a known simplification — real points show
momentum, fatigue, and pressure effects — and that gap is precisely the motivation for the
richer Day 9 model. The baseline's value is being fast, exact, interpretable, and a
principled yardstick, not being the final model.

## Validation

**Against known literature reference values:**
- `prob_win_game(0.60) = 0.7357` — matches the canonical textbook figure exactly
- `prob_win_game(0.70) = 0.9008`, `prob_win_game(0.50) = 0.5000` — all match to <1e-3

**Symmetry (to machine precision):** game, tiebreak, set, and match all return exactly 0.5
when both players are equally skilled (p=0.5), for both best-of-3 and best-of-5.

**Monotonicity:** win probability rises strictly with serve skill at every level.

**Best-of-5 amplifies the favorite** vs best-of-3 — a real, known tennis phenomenon (more
sets = less variance = the stronger player's edge compounds), confirmed by the model.

**Live-engine consistency (the key test):** at 0-0-0 with the first server serving, the
live in-match engine reproduces the pre-match analytical model to a difference of exactly
0.0 (both 0.736502 for p_serve=0.65, p_return=0.40). This proves the two independent code
paths are mutually consistent.

**Live-engine intuition:** triple match point on serve -> 0.999; facing triple match point
-> 0.007; win probability increases monotonically as the point score improves within a game.

## A real bug caught and fixed during development

The first implementation of the in-match game and tiebreak calculators recursed through the
deuce/advantage states (deuce -> advantage -> deuce -> ...), which is an infinite cycle with
no base case — it raised `RecursionError` and would have hung the tiebreak deuce phase.
Fixed by solving the deuce subgame with its closed-form fixed point (server wins from deuce
with probability p^2 / (p^2 + q^2); tiebreak deuce phase via the analogous two-point-cycle
formula) instead of recursing. This is the same closed form `prob_win_game` already used —
the live path now matches it exactly, which is why the consistency test passes to 0.0.

## Explicitly deferred to Day 9 (not silently dropped)

- Estimating each player's actual per-serve point-win probabilities from the frozen Day 6
  serve/return features (this baseline takes them as inputs; Day 9 wires real values in)
- The point-level ML classifier and Monte Carlo simulation engine that will be compared
  against this baseline on real charted matches
- Evaluating both on the same held-out point-level data using the frozen `evaluation/`
  framework (temporal CV, log loss, calibration)

## Next step

Day 9: point-level ML classifier + Monte Carlo simulation engine, evaluated head-to-head
against this Markov baseline on real charted matches — the empirical core of the live
win-probability contribution.