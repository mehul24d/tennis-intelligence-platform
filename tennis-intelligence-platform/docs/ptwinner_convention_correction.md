# PtWinner convention correction: the full investigation, fix, and decision record

**Status: FULLY RESOLVED (2026-07-15).** Code fixed (literal player-relative
convention restored everywhere it was found), impact measured, model retrained on
corrected features, retrain result reviewed and approved, deployed. This doc is the
permanent, closed record of the entire investigation: from the first RAG-example
review that surfaced the anomaly through the final retrain decision.

Read this before touching `PtWinner`, `Gm1`/`Gm2`, or the point-level classifier
again.

---

## TL;DR

- `PtWinner` is **literal, fixed-player-relative**: `PtWinner==1` means player 1 won
  the point, full stop, independent of who served. This was the ORIGINAL, correct
  convention this project used.
- A same-day "fix" (2026-07) briefly changed it to **server-relative**
  (`PtWinner==1` means the server won), across `ml_informed_markov.py`,
  `point_timeline_service.py`, `match_summary_service.py`, and five functions in
  `point_level_features.py`. That fix was wrong. It has been traced and reverted in
  every one of those files.
- The wrong fix was built on `check_ptwinner_disagreement_at_scale.py`'s "0.00%
  disagreement" claim, which only tests internal self-consistency between `PtWinner`
  and fixed-player `Pts` on interior (non-game-boundary) points — a check that cannot
  distinguish server-relative from literal `PtWinner` (they're mirror images that
  only diverge at `Svr==2`, exactly the case that script never examines).
- Checked directly against `Gm1`/`Gm2` (the one independently-recorded, external
  signal) at real game boundaries: **literal `PtWinner` matches at 99.91%
  corpus-wide, symmetric across `Svr==1`/`2`; server-relative matches only ~51%**
  (chance level).
- `day9_point_classifiers.joblib` (the trained point-level classifier) was retrained
  on features computed under the corrected convention and **deployed 2026-07-15**.
  Rolling-origin evaluation across four independent years showed a consistent, real
  improvement (log_loss 0.6281 → 0.6247, Brier 0.2187 → 0.2172, every single fold).
  See "Retrain results" below for the full comparison, including a follow-up
  correlation check that explains (rather than just observes) the one surprising
  SHAP result.
- v1's flagship match-level XGBoost/LightGBM/CatBoost win-probability models are
  **confirmed unaffected** — verified by direct search, not assumption (see below).

---

## How this was found

While grounding a `rag_engine/ingest/point_documents.py` example in real
`get_point_timeline` output, an automatically-flagged (not hand-picked) point in
`19921031-M-Stockholm_Masters-SF-Stefan_Edberg-Goran_Ivanisevic` looked wrong: the
point winner and the win-probability direction disagreed. Investigating that
disagreement led to checking `Gm1`/`Gm2` (game-count columns) against the point-level
data, which surfaced a ~49%, corpus-wide, chance-level mismatch rate — present in all
5,981-7,532 matches checked, not scattered charting error.

Multiple hypotheses for this were proposed and tested directly against real data, in
order, each with actual numbers before moving to the next (not accepted or rejected
by discussion):

1. **"`Gm1`/`Gm2` are simply swapped"** — refuted: aggregate set-final scores (e.g.
   Laver/Ashe's real historical set-1 score of 2-6) match the standard
   `Gm1`=player1/`Gm2`=player2 convention.
2. **"A one-row indexing lag/lead in dataset construction"** — refuted: mismatch
   rate at shift -1 (70.76%) and +1 (64.48%) is *worse* than shift 0 (48.76%), not
   better.
3. **"Hold vs. break specific bug"** — refuted: cross-tab of (server-won vs.
   receiver-won) x (expected winner is p1 vs. p2) showed the split wasn't
   hold/break at all.
4. **"Server identity at the deciding point"** (`Svr==1`: 0.11% mismatch, `Svr==2`:
   99.90% mismatch) — this was the closest clean signature, but subsequent direct
   counter-evidence (three real games in one match, hand-traced, all reportedly
   correct at `Svr==2`) forced a deeper look, since a script-level re-check of that
   *same* match actually reproduced 10/19 (52.6%) mismatches, contradicting the
   informal hand-trace.
5. **"`Pts` is server-first, not fixed-player, for regular games"** (mirroring the
   already-fixed tiebreak notation bug) — tested directly at corpus scale using the
   same methodology the tiebreak fix used (match rate against `PtWinner`-implied
   transitions): fixed-player parsing scored 99.9991%, server-first scored 50.48%
   (chance). **Refuted** — this was a wrong analogy to the tiebreak fix; regular-game
   `Pts` parsing was already correct.
6. **The real answer, found by testing the missing combination**: server-first `Pts`
   (translated via `Svr`) checked against **literal** `PtWinner` (not server-relative)
   matched at 99.9990%, symmetric across `Svr==1`/`Svr==2` — the first combination
   that was both internally self-consistent *and* symmetric. Checked against `Gm1`
   at real game boundaries: 99.91% match (181,258 boundaries, 167 mismatches,
   corpus-wide). This settled it.

### Why the existing "0.00% disagreement" claim didn't catch this

`check_ptwinner_disagreement_at_scale.py`'s `compute_disagreement_rate` explicitly
skips every game-boundary transition:
```python
if row["Gm1"] != next_row["Gm1"] or row["Gm2"] != next_row["Gm2"]:
    continue
```
It only ever tests interior, within-game point-to-point consistency between
`PtWinner` and fixed-player `Pts`. There are exactly two internally self-consistent
labeling systems possible here — (A) server-relative `PtWinner` + fixed-player `Pts`,
and (B) literal `PtWinner` + server-first `Pts` — which coincide whenever `Svr==1`
and are exact mirror opposites whenever `Svr==2`. A same-row self-consistency check
can rule out *mixed* pairings (which is all the script's "FIXED vs. RELATIVE"
comparison actually demonstrates) but can never distinguish between two
self-consistent systems, because neither ever contradicts itself internally. Only an
independently-recorded external signal — `Gm1`/`Gm2` — can do that, and the script
never looks at it.

### Hand-verification, in plain tennis terms

Two real matches, hand-traced using literal `PtWinner`:
- **Laver/Ashe, 1969 Wimbledon SF, game 2** (Ashe serving throughout): the score
  oscillates through deuce and advantage twice before resolving. Literal `PtWinner`
  says: **Ashe survives an extended deuce battle and holds serve from his own
  advantage.** `Gm2` increments. Consistent.
- **Nadal/Shapovalov, 2019 Davis Cup Final, game 2** (Shapovalov serving throughout):
  score climbs from love-30 down through deuce/advantage. Literal `PtWinner` says:
  **Shapovalov claws back and holds serve from his own advantage.** `Gm2`
  increments. Consistent.

Both are the ordinary, ordinary-frequency outcome (a server grinding out a deuce
game and holding) — the old (server-relative) reading required the *less common*
outcome (a break) in both hand-checked cases, and didn't match the recorded `Gm`
column in either.

---

## Files fixed

All changed to literal, fixed-player-relative `PtWinner` (`PtWinner==1` means player
1 won, no `Svr` cross-reference for the winner determination itself):

| file | what changed |
|---|---|
| `src/tennis_intel/live/ml_informed_markov.py` | `a_won_this_point` in `ml_informed_markov_predict` reverted to literal; comment rewritten with full investigation history and cross-references. |
| `src/tennis_intel/serving/point_timeline_service.py` | `point_winner_is_p1` simplified to `row["PtWinner"] == 1` directly. `_server_perspective_score` was confirmed **never affected** (it reorders already-correct fixed-player `p1_points`/`p2_points` via `Svr` for display, never touches `PtWinner`). |
| `src/tennis_intel/serving/match_summary_service.py` | Three spots: `_point_winner_is_p1` (literal now), `longest_server_run`'s `server_won` (restored the `Svr` cross-reference this needs — an example of the reverse mistake: the old "fix" had *removed* a needed cross-reference here, since "did the server win" genuinely does need `PtWinner` combined with `Svr`, unlike "did player 1 win"), `p1_bp_converted` (was inverted, fixed; `p2_bp_converted` turned out correct-by-coincidence and needed no change). |
| `src/tennis_intel/features/point_level_features.py` | Five functions: `compute_in_match_momentum`, `compute_consecutive_points_streak`, `compute_split_points_streak`, `compute_in_match_serve_return_rate`, `compute_in_match_serve_return_rate_rolling` — all reverted to `p1_won_point = (df["PtWinner"] == 1)`, no `Svr` cross-reference. |
| `pipelines/check_game_counter_consistency_at_scale.py` | Ground-truth logic fixed to literal; docstring flags prior output as built on the refuted convention and should be re-derived, not cited (including the referenced "Athens match" finding). |
| `pipelines/check_ptwinner_vs_points_progression.py` | Fixed a subtler bug: it was comparing literal `PtWinner` against *fixed-player* `Pts` progression — a self-inconsistent pairing that would show ~50% "disagreement" even under the correct convention. Now derives server-first `p1`/`p2` (via `Svr`) to correctly pair with literal `PtWinner`. |
| `pipelines/check_ptwinner_disagreement_at_scale.py` | Docstring updated to explain its own blind spot (see above) rather than restating the refuted "0.00%" conclusion as settled. |
| `tests/unit/test_ml_informed_markov.py` | Rewritten for literal convention (6 parametrized cases + 1 end-to-end), 7/7 passing. |
| `tests/unit/test_point_streak.py` | `test_ptwinner_is_relative_to_server_not_fixed_player` replaced with `test_ptwinner_is_literal_player_relative_not_server_relative`, asserting the corrected direction. |
| `tests/unit/test_split_points_streak.py` | `test_return_streak_hand_traced` re-derived (second time) for the literal convention: `[0, 0, 0, 1, -1, -1, -1, -2]`. |
| `tests/unit/test_in_match_serve_return_rates.py` | `test_return_rate_hand_traced` re-derived: `[None, None, None, 1.0, 0.5, 0.5, 0.5, 1/3]`. |

**Not affected, confirmed by inspection, no change needed:**
- The model's training **target**, `server_wins_point = (PtWinner == Svr)` in
  `build_point_dataset.py` — a direct identity comparison between two
  player-identity-coded values (`PtWinner` and `Svr` both take values 1/2 denoting
  which named player), only logically valid under the literal convention, and
  already correct before this investigation.
- `points["player1_is_winner"]` in `replay_service.py` (`(Svr==1) == server_is_winner`)
  — `server_is_winner` there is match-outcome-derived (who eventually wins the whole
  match), not `PtWinner`-derived; unrelated to this question.
- All match-level modeling code (see "Match-level models" section below).

All 211 unit tests pass after every fix.

---

## Impact analysis: does this matter for the deployed classifier?

Requested explicitly before any retrain decision, per this project's standing
discipline of measuring before acting.

**Feature importance** (`day9_point_classifiers.joblib`, GradientBoostingClassifier,
55 features):

| feature | rank | importance | Svr==2 rows changed | shift magnitude |
|---|---|---|---|---|
| `is_second_serve_point` (unaffected) | 1 | 0.7746 | — | — |
| `server_is_player1` (unaffected) | 2 | 0.0332 | — | — |
| **`p1_in_match_return_rate`** | **3** | **0.0297** | 96.8% | mean|diff|=0.32 (0-1 scale; max 1.0) |
| `p1_in_match_serve_rate` | 4 | 0.0242 | 97.9% | mean|diff|=0.013 (tiny — built from `Svr==1` points, forward-filled onto `Svr==2` rows, only residual drift) |
| **`points_streak`** | **5** | **0.0172** | 86.9% | mean|diff|=3.6 (max diff 78) |
| `p1_return_streak` | 17 | 0.0027 | — | same construction as `points_streak`, comparably large shift expected |
| `points_streak_x_break_point` | 21 | 0.0021 | — | interaction of `points_streak` |
| `p1_serve_streak` | 28 | 0.0017 | — | moderate |
| `p1_momentum_last20` | ~39 | 0.0006 | 91.0% | mean|diff|=0.17 |
| `pressure_index_x_momentum10` | 36 | 0.0009 | — | interaction term |
| `p1_momentum_last10` | 45 | 0.0003 | 78.6% | mean|diff|=0.18 |
| `p2_momentum_last10/20` | ~49 | ~0.0002 | — | mirror of p1, same magnitude |

**Conclusion**: this is not the "barely moved the needle" case. `p1_in_match_return_rate`
(the model's **#3** most important feature) shifts on 96.8% of `Svr==2` rows (half
the dataset) by an average of ~1/3 of its full 0-1 range. `points_streak` (**#5**)
shifts on 86.9% of `Svr==2` rows by an average of 3.6 streak-points. Together with
the lower-importance affected features, roughly **7-8% of total model importance**
sits on features that are materially different at inference time than at training
time.

**Decision**: retraining is likely warranted based on this evidence, but **retraining
is explicitly NOT happening as part of this fix**. It is scoped as its own dedicated
follow-up task (see "Retraining plan" below), not folded into today's correctness fix
— deliberately, so the decision is made with the numbers in hand rather than by
default in either direction.

---

## Known-stale model: RESOLVED, warnings removed

`day9_point_classifiers.joblib` was flagged wherever it was loaded from the point
the mismatch was found until the retrain landed:

- `src/tennis_intel/serving/replay_service.py::load_replay_context()` — the one
  production loader. Carried both a large docstring block explaining the mismatch and
  a `logger.warning(...)` firing on every load (verified firing in this session's own
  diagnostic runs). **Both removed 2026-07-15**, replaced with a short note pointing
  at this doc's retrain-results section.
- `pipelines/build_day9_point_model.py` — carried a loud warning at the top
  explaining not to re-run it without reading the impact analysis first. **Updated
  2026-07-15** to reflect that the mismatch is resolved; re-running it now is a
  normal refresh, not something requiring special caution.

(The classifier is also loaded directly, via `joblib.load`, in ~26 one-off
`pipelines/*.py` diagnostic/evaluation scripts, never individually annotated — if any
of those scripts' *conclusions* get reused or cited, they were generated against the
pre-retrain classifier and should be treated accordingly.)

---

## Retrain results (2026-07-15) — DEPLOYED

Executed the scoped retraining plan in full, via
`pipelines/retrain_day9_candidate_and_compare.py` (new script; does not touch the
deployed path until explicitly told to).

### Step 1 — regenerated the point-level dataset

Rebuilt via `build_point_dataset` with the now-corrected `point_level_features.py`
functions, saved to `data/processed/day10_point_dataset_RETRAIN_CANDIDATE.parquet`
(1,042,831 points, matches spanning 1968–2026).

### Step 2 — sanity-checked the fix actually took effect in the training set

Not assumed — measured directly on the fresh dataset: `p1_in_match_return_rate`
changed on **96.8% of `Svr==2` rows, mean|diff|=0.3185** — matching the earlier
isolated-function measurement exactly (96.8%, 0.32). Confirmed the correction was
genuinely present in the regenerated training data, not just in unit tests.

### Step 3 — retrained

Same methodology and hyperparameters as the original `build_day9_point_model.py`
(`GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
subsample=0.8, random_state=42)` in a `Pipeline` with median imputation; also trained
`LogisticRegression` alongside, same as the original script, for parity — not the
deployed model type). Saved to
`data/processed/day9_point_classifiers_RETRAIN_CANDIDATE.joblib` first, reviewed, and
only copied to the deployed path after approval (see Deployment below).

### Step 4 — old vs. new comparison, rolling-origin (not a single split)

**Point-level accuracy** — single split (test year ≥ 2022, matching the original
script's holdout), both models scored on the *same* corrected-feature test set (old
model re-scored on corrected features — the fairest apples-to-apples test, isolating
the model difference from any feature difference):

| | log_loss | Brier |
|---|---|---|
| OLD (deployed pre-retrain) | 0.6280 | 0.2186 |
| NEW (retrained) | **0.6248** | **0.2172** |

**Rolling-origin, four independent expanding-window folds** (train = strictly before
`test_year`, test = exactly that year — the same discipline
`src/tennis_intel/evaluation/temporal_cv.py::generate_temporal_folds` implements,
applied directly since the point dataset keys on an integer `match_year`, not a date
column):

| test_year | n_test | old_ll | new_ll | old_brier | new_brier |
|---|---|---|---|---|---|
| 2022 | 87,196 | 0.6275 | 0.6246 | 0.2184 | 0.2172 |
| 2023 | 90,587 | 0.6286 | 0.6250 | 0.2189 | 0.2173 |
| 2024 | 94,553 | 0.6280 | 0.6246 | 0.2187 | 0.2171 |
| 2025 | 81,054 | 0.6282 | 0.6248 | 0.2187 | 0.2172 |
| **mean** | | **0.6281** | **0.6247** | **0.2187** | **0.2172** |

**The new model wins in every single fold**, by almost exactly the same margin each
time (~0.003 log_loss, ~0.0015 Brier) — that consistency across four independent
years is the strongest evidence this is a real improvement, not noise from one split.

**Calibration** (reliability by predicted-probability decile, single-split test):
both models reasonably calibrated; OLD tends toward mild underconfidence in the
upper-middle deciles (e.g. predicted 0.699 vs. actual 0.721), NEW toward mild
overconfidence in a couple of adjacent bins (predicted 0.681 vs. actual 0.666).
Neither clearly better — not a deciding factor either way. Full table:
`data/processed/day9_retrain_calibration_comparison.csv`.

**SHAP feature importance** (mean |SHAP value|, 2000-row test sample):

| feature | old rank | new rank | old importance | new importance |
|---|---|---|---|---|
| `is_second_serve_point` | 1 | 1 | 0.4180 | 0.4305 |
| `server_is_player1` | 2 | 2 | 0.0648 | 0.0397 |
| `p1_in_match_return_rate` | 3 | 3 | 0.0564 | 0.0326 |
| `p1_in_match_serve_rate` | 4 | 4 | 0.0274 | 0.0301 |
| `loser_first_serve_win_pct_career` | 8 | 5 | 0.0229 | 0.0245 |

Top-4 ranks identical (stable model structure). `p1_serve_streak` moved rank 22 → 14.
Full table: `data/processed/day9_retrain_shap_comparison.csv`.

### The one surprising result — investigated, not accepted by default

`p1_in_match_return_rate` kept its #3 rank but its importance magnitude dropped ~42%
(0.0564 → 0.0326). Two competing explanations were possible: (a) the feature's real
predictive signal genuinely shrank, or (b) collinearity with `is_second_serve_point`
increased post-fix, and some of return-rate's apparent old importance was actually
`is_second_serve_point` "claiming" signal that used to show up as return-rate's own
(a five-minute correlation check, requested explicitly before accepting either
explanation by default):

```
Test 1 — collinearity with is_second_serve_point:
  OLD (server-relative, reconstructed): r = -0.00489
  NEW (corrected, literal):             r = +0.00462
  → negligible in both, negligible change. Refutes explanation (b).

Test 2 — correlation with the actual training target (p1_won_point):
  OLD (server-relative): r = -0.02896  (negative — backwards!)
  NEW (corrected):       r = +0.02283  (positive — correct sign)
```

**Test 2 is the real explanation, and it's cleaner than either hypothesis offered
initially.** Under the old (buggy) convention, a *higher* computed return rate was
weakly associated with *losing* the next point — backwards, a direct fingerprint of
the bug. The model had partially fit to that spurious, wrong-signed correlation,
inflating the feature's apparent importance. Under the corrected convention, the
correlation flips to the intuitively correct positive sign, smaller in magnitude but
genuine. The importance drop reflects the model losing access to noise it used to
exploit, not a real predictor becoming meaningless — if anything, this strengthens
confidence in the retrain rather than raising a concern.

### Deployment decision: APPROVED, deployed

Reviewed and approved. `day9_point_classifiers.joblib` now contains the retrained
model. The pre-retrain classifier is preserved at
`data/processed/day9_point_classifiers_PRE_PTWINNER_FIX.joblib` (not deleted,
matching this project's standing practice of backing up before overwriting a
deployed artifact — see the parallel `..._PRE_LEAKAGE_FIX.joblib` from an earlier,
unrelated fix). Load-time warnings removed from `replay_service.py` and
`build_day9_point_model.py` (see above).

---

## Match-level models: confirmed unaffected

Checked explicitly, not assumed, since v1's flagship win-probability system (the
XGBoost/LightGBM/CatBoost match-level models, distinct from the day9 point-level
classifier) is the core of this project:

```
grep -rln "point_level_features\|compute_in_match_momentum\|compute_consecutive_points_streak\|
            compute_split_points_streak\|compute_in_match_serve_return_rate" \
  src/tennis_intel/modeling/ src/tennis_intel/features/feature_engineering_day5.py \
  src/tennis_intel/features/serve_return_features.py \
  src/tennis_intel/features/surface_serve_return_features.py \
  src/tennis_intel/features/head_to_head_features.py \
  pipelines/build_xgboost_prematch_model.py pipelines/train_baseline_models.py \
  pipelines/build_joined_dataset.py pipelines/build_day5_features.py pipelines/build_day6_features.py
→ (no output — zero matches)

grep -rln "PtWinner|\bSvr\b" src/tennis_intel/modeling/ src/tennis_intel/features/*.py \
  src/tennis_intel/ratings/*.py
→ only point_level_features.py and point_score_parser.py themselves
```

`serve_return_features.py`/`surface_serve_return_features.py` (which back the
match-level serve/return career features) are built entirely from **match-summary
aggregate columns** (`w_ace`, `w_svpt`, `w_1stIn`, `w_1stWon`, etc. — the standard
per-match TML stat columns covering the full ~198k-match corpus), not from the
point-by-point charted data (which only covers the ~5,988-match frozen-join subset —
far too small to train a match-level model on the full corpus in the first place).
Elo (`ratings/elo.py`, `ratings/surface_elo.py`) is built from match win/loss
outcomes, independent of point-level data entirely.

**Conclusion: v1's core match-level win-probability models (the actual portfolio
centerpiece) do not consume `PtWinner`/`Svr`-derived features at all, directly or
indirectly. They are unaffected by this investigation and this fix.** Only the day9
point-level classifier (used for the *live*, in-match win-probability engine, a
separate and smaller component) and the serving-layer display/trajectory code fixed
above are in scope.

---

## Open items flagged for future re-verification (not re-run today)

Two earlier same-day findings were verified using `point_timeline_service.py`'s
`winner` field, which at the time still used the (since-reverted) server-relative
convention. Their **structural** fixes are independent of the `PtWinner` convention
and remain valid, but the **specific numbers** reported alongside them were measured
against a since-corrected ground truth and should be re-measured before being cited
again:

- `docs/known_issue_ml_informed_markov_pre_point_state.md` — the pre-point/post-point
  `MatchState` indexing fix itself (pairing `smoothed_p1[i]`/`smoothed_p1[i+1]`) does
  not depend on `PtWinner`'s convention and stands. The reported verification numbers
  (89.9% overall / 100% on meaningful-swing points, on the Rodionov/Casanova match)
  were computed using the `winner` field for direction-checking, which has since
  changed. Re-run before citing those specific percentages again.
- `docs/known_issue_after_point_swing_includes_next_point_context.md` — the
  `is_second_serve_point` mechanism finding (93-96% of a swing explained by the next
  point's serve context, independent of the current point's outcome) used
  `ml_informed_point_probabilities` and `PtWinner`-based counterfactuals directly; the
  narrative labeling ("Alcaraz won this point") used the `winner` field. Re-verify
  which named player actually won the traced points before reusing this doc's
  specific example.

Neither of these was re-run as part of today's fix — flagged here as explicit,
scoped follow-up rather than silently left stale.
