# Milestone 5 — Baseline Model Comparison Results (v1: Elo + Rolling Form)

Status: **frozen baseline**, pending Day 6 (MCP point-level serve/return integration) for a
direct before/after comparison under the identical evaluation harness below.

## Setup

- **Data:** 198,062 ATP matches (Days 1–5, frozen), converted to a symmetric player_1/
  player_2 dataset via `build_symmetric_dataset.py` (deterministic assignment via hash of
  `(tourney_id, match_num)`, never row order or a global seed).
- **Features (v1):** 11 numeric `*_diff` features — Elo, win% (5/10/20-match rolling),
  surface win%, game differential (overall + surface), opponent Elo strength, win/loss
  streak, rest days, straight-set rate. All leakage-safe by construction (Days 4–5, frozen).
- **Evaluation:** rolling-origin temporal cross-validation, 8 folds (test years 2018–2025),
  expanding training window (176,060 → 195,065 matches). Every fold's train set is strictly
  earlier in time than its test set (verified in `tests/unit/test_temporal_cv.py`).
- **Models:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost — each
  evaluated both raw and with isotonic calibration (3-fold internal CV on the train fold
  only, never touching test data).
- **Metrics:** log loss (primary), Brier score, both with bootstrap 95% CI (1,000 resamples
  — reduced to 200 for the full 40-run sweep here for runtime; report should use 1,000 for
  the final numbers going in the paper).

## Aggregate results (mean across 8 folds)

| Model | Mean Log Loss | Mean Brier | Mean Calibrated Log Loss | Mean Calibrated Brier |
|---|---|---|---|---|
| **CatBoost** | **0.6196** | **0.2159** | 0.6200 | 0.2161 |
| LightGBM | 0.6199 | 0.2160 | **0.6198** | **0.2159** |
| XGBoost | 0.6201 | 0.2160 | 0.6199 | 0.2160 |
| Logistic Regression | 0.6213 | 0.2166 | 0.6217 | 0.2168 |
| Random Forest | 0.6218 | 0.2168 | 0.6227 | 0.2172 |

All five models comfortably beat the no-skill baseline (constant p=0.5 prediction gives
log loss = ln(2) = 0.6931) — roughly a 10–11% relative improvement.

## Key finding: model complexity buys almost nothing on this feature set

The full model spread is under 0.003 log-loss (0.6196 to 0.6218). Gradient-boosted trees
(CatBoost/LightGBM/XGBoost) edge out linear and bagged-tree baselines, but only marginally
— logistic regression is within 0.002 of the best tree model.

**Interpretation:** this is consistent with the current feature set being largely linear in
its signal. Elo itself is already a well-engineered, nonlinearly-derived single-number
summary of relative player strength (it's the *output* of a rating system, not a raw stat),
so a tree model has comparatively little additional nonlinear structure to exploit once Elo
and the rolling-form diffs are already in the feature set. This is a legitimate empirical
result, not a modeling failure — it's exactly the kind of finding that motivates Day 6:
does richer, point-level information (serve/return stats, which are NOT already
Elo-summarized) give tree models more genuine nonlinear structure to work with, thereby
widening the gap between simple and complex models? That comparison is the actual point of
adding MCP data, not just "more features = better."

## Calibration observations

Calibration effects are small and occasionally slightly negative (e.g., Logistic Regression
2018: raw log loss 0.6254 → calibrated 0.6257). This is plausible, not a bug: these models'
raw outputs are already reasonably well-calibrated (logistic regression is calibrated by
construction when correctly specified; tree ensembles with enough estimators tend to
produce fairly smooth probability outputs too), so a 3-fold-internal isotonic calibration
step has limited genuine gap to close and can introduce small variance from its own
train/calibration split. Worth revisiting with a larger internal calibration CV (e.g. 5-fold)
or a held-out calibration set once the pipeline scales to the MCP-integrated feature set.

## Fold-level stability

Log loss is stable across all 8 years (range: 0.6093–0.6268 across all models/folds) with
no sudden jumps or degenerate folds — no fold shows a suspiciously low log loss that would
suggest leakage, and no fold collapses to near-random performance that would suggest a
broken feature or a data gap in a particular year.

## What this result does NOT yet tell us

- Whether point-level (serve/return) features add value beyond match-level Elo/form — this
  requires Day 6's MCP integration and an apples-to-apples rerun of this exact harness.
- Full calibration curves (reliability diagrams) — `calibration_table()` is built and tested
  but not yet run/plotted against the real predictions; worth adding to the next iteration
  of this report.
- Hyperparameter-tuned versions of each model — current results use reasonable fixed
  defaults (documented in `train_and_evaluate.py::get_model_registry`), not a tuned search;
  tuning is deferred until the feature set is finalized (post-Day 6), since tuning against
  a feature set that's about to change is wasted effort.

## Artifacts

- `data/processed/model_comparison_per_fold.csv` — full per-fold, per-model results
- `data/processed/model_comparison_aggregate.csv` — aggregate table above
- `docs/shap_summary_catboost.png` — SHAP feature-importance summary for the winning model
- MLflow run `milestone5_baseline_comparison` in experiment `tennis-baseline-model-comparison`

## Next step

Day 6: integrate MCP point-level serve/return statistics (first serve %, break points
saved/converted, etc.) as leakage-safe rolling features, using the same chronology
discipline as Days 4–5. Rerun this exact evaluation harness unchanged, and compare directly
against this frozen baseline to measure serve/return data's incremental value — the core
empirical contribution the paper is building toward.