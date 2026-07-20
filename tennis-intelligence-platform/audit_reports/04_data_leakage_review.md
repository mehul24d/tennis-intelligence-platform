# Phase 4 - Data Leakage Review

Priority: highest. Each required category is explicitly addressed.

## 1) Temporal leakage

### Checked
- Elo chronology/update order in `src/tennis_intel/ratings/processor.py`.
- Day5 rolling features in `src/tennis_intel/features/feature_engineering_day5.py` (shift then rolling).
- Day6 serve/return rolling in `src/tennis_intel/features/serve_return_features.py` (shift then rolling).
- H2H update order in `src/tennis_intel/features/head_to_head_features.py`.

### Result
- No direct temporal leakage found in those modules; update pattern is pre-read then post-update.

## 2) Split leakage (train/test split by row vs by match)

### Checked
- `pipelines/build_day9_point_model.py` split logic (`match_year` threshold).
- `pipelines/tune_day9_hyperparameters.py` split logic (`<2020`, `2020-2021`, `>=2022`).
- `src/tennis_intel/modeling/train_and_evaluate.py` temporal fold logic.

### Result
- For point dataset, split is by year and all points in a match share year, so match is not split across train/test by row in current implementation.
- No direct split leakage found here.

## 3) Group leakage across matches (duplicate identifiers / aliasing)

### Checked
- Join path uses `mcp_match_id` anchors in `build_point_dataset.py`.
- Duplicate handling in MCP stats in `serve_return_features.py` (`drop_duplicates(match_id, player)`).

### Result
- No direct evidence of train/test duplication leakage in current reviewed paths.
- Residual risk remains from upstream duplicate match identities not normalized globally (not proven in this audit).

## 4) Target leakage

### Critical finding
- File: `src/tennis_intel/live/build_point_dataset.py:142`
- Severity: Critical
- Mechanism: `server_is_winner` is constructed from true match winner identity and used as model feature (`pipelines/build_day9_point_model.py:63`).
- Why leakage: eventual winner is unknown at live prediction time; model receives post-outcome information proxy during training/evaluation.
- Inflation direction: optimistic point-model performance and downstream engine metrics.
- Rough magnitude: likely material (binary strong signal directly tied to outcome strength).
- Fix:
  - remove `server_is_winner` from features,
  - replace with non-leaky orientation signal (`server_is_player1` / `server_is_tracked_player`),
  - retrain and recompute all reported metrics.

## 5) Preprocessing leakage

### Checked
- `SimpleImputer` usage in `pipelines/build_day9_point_model.py`, `pipelines/tune_day9_hyperparameters.py`, `pipelines/build_xgboost_prematch_model.py`.
- Median statistics usage in `src/tennis_intel/modeling/train_and_evaluate.py`.

### Result
- Imputation appears fit only on train partitions in reviewed scripts.
- No direct preprocessing leakage found.

## 6) Calibration leakage

### Checked
- `CalibratedClassifierCV(..., cv=3)` in `src/tennis_intel/modeling/train_and_evaluate.py:167` fitted on train fold then tested on holdout fold.

### Result
- No direct calibration leakage found in this path.

## 7) Backtest leakage on Elo/H2H prior

### Checked
- Elo pre/post extraction ordering in `src/tennis_intel/ratings/processor.py`.
- H2H pre-match read then state increment in `src/tennis_intel/features/head_to_head_features.py`.

### Result
- No evidence that match result leaks into its own pre-match Elo/H2H features.

## 8) Duplicate leakage

### Checked
- Duplicate `(match_id, player)` in MCP stats explicitly deduplicated in `serve_return_features.py`.
- No global near-duplicate detector found for full point/match training tables.

### Result
- Local duplicate guard exists for MCP stats table.
- Repository lacks a dedicated near-duplicate leakage guard at model training stage.

## Leakage verdict

- Confirmed Critical: outcome-derived feature leakage (`server_is_winner`).
- Other required leakage categories: no direct confirmed leakage from code paths inspected, with residual risks documented where explicit guards are absent.