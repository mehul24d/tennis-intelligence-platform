# Phase 0 - Repository Mapping

Date: 2026-07-06
Scope: `/Users/mehuldahiya/Desktop/tennis-intelligence-platform`

## 0.1 Logic File Tree (annotated)

### Root standalone scripts
- `check_matchpoint_spread.py`: distribution diagnostics at match-point states.
- `diagnose_markov_inputs_at_scale.py`: large-scale Markov input sanity diagnostics.
- `diagnose_ml_informed_markov.py`: compares ML-informed Markov behavior to alternatives.
- `diagnose_point1_jump.py`: investigates early-match jump behavior in trajectories.
- `diagnose_prematch_ml.py`: pre-match ML diagnostics.
- `inspect_2008_wimbledon_state.py`: historical state inspection utility.
- `inspect_iw_anomaly.py`: investigates anomaly in Indian Wells-related output.
- `inspect_markov_inputs.py`: inspects per-row Markov probability inputs.
- `inspect_ml_prematch_features.py`: inspects pre-match feature values for ML engine.
- `inspect_raw_predictions.py`: quick inspection of saved prediction files.
- `isolate_second_serve_flag.py`: targeted flag debugging utility.
- `trace_markov_at_matchpoint.py`: traces recursion behavior around match point.

### `src/tennis_intel/live` (core engines)
- `markov_baseline.py`: closed-form game/set/match model from serve/return point rates.
- `live_win_probability.py`: from-state recursion for live match probability.
- `markov_inverse.py`: inverts pre-match match-win probability to point-level serve prior.
- `ml_informed_markov.py`: Beta-Binomial-updated point-rate + Markov recursion engine.
- `monte_carlo_engine.py`: simulation engine (`simulate_match_from_state`, batched rollouts).
- `hybrid_engine.py`: fixed-weight blend of Markov and ML+MC match probabilities.
- `build_point_dataset.py`: point-level dataset assembly and feature broadcasting.

### `src/tennis_intel/features`
- `point_score_parser.py`: point score parsing and break/set/match-point flags.
- `point_level_features.py`: per-point state + momentum derivation.
- `score_parser.py`: match score parsing at match-level.
- `feature_engineering_day5.py`: rolling match-level features (leakage-safe shift/roll).
- `serve_return_features.py`: rolling serve/return features from MCP stats.
- `surface_serve_return_features.py`: surface-conditioned serve/return feature extensions.
- `head_to_head_features.py`: overall and tournament-specific H2H pre-match features.

### `src/tennis_intel/ratings`
- `base.py`: rating interface abstraction.
- `elo.py`: Elo implementation.
- `surface_elo.py`: per-surface Elo ratings.
- `processor.py`: chronological rating pass with leakage-safe pre/post extraction.

### `src/tennis_intel/modeling`
- `build_symmetric_dataset.py`: winner/loser -> player_1/player_2 symmetric training rows.
- `train_and_evaluate.py`: baseline model registry, temporal CV, isotonic calibration.

### `src/tennis_intel/evaluation`
- `metrics.py`: log loss, Brier, ECE, calibration tables, paired bootstrap.
- `temporal_cv.py`: temporal fold generation and holdout split helper.

### `src/tennis_intel/data` and `src/tennis_intel/entities`
- `join_tml_mcp.py`: TML <-> MCP joining logic.
- `join_validation.py`: join-quality diagnostics.
- `canonical_player_names.py`: name normalization.
- `canonical_players.py`: canonical player table tools.
- `player_aliases.py`: alias normalization logic.
- `player_registry.py`: player registry creation/access.
- `validate_players.py`: player identity QA checks.

### `src/tennis_intel/viz`
- `trajectory_generation.py`: package-level trajectory assembly.
- `trajectory_plot.py`: plotting utilities for trajectories.
- `trajectory_events.py`: event annotations on trajectories.

### Pipeline entrypoints (`pipelines/*.py`)
- Build/train: `build_elo.py`, `build_day5_features.py`, `build_day6_features.py`, `build_day9_point_model.py`, `build_xgboost_prematch_model.py`, `build_joined_dataset.py`, `build_player_registry.py`, `train_baseline_models.py`, `train_day6_comparison.py`, `tune_day9_hyperparameters.py`.
- Evaluate/compare: `evaluate_live_engines.py`, `evaluate_live_engines_v2.py`, `evaluate_ml_informed_markov.py`, `evaluate_hybrid_engine.py`, `evaluate_all_engines_unified.py`.
- Validate/sanity: `validate_markov_engine.py`, `validate_markov_inputs.py`, `validate_ml_mc_engine.py`, `validate_tiebreak_notation.py`, `test_prematch_probability_sanity.py`, `test_hybrid_vs_ml_informed_prematch.py`.
- Diagnostics/profiling: `diagnose_*`, `profile_ml_informed_markov.py`, `sweep_prior_strength.py`, `audit_markov_call_sites.py`, `analyze_n0_fix_impact.py`.
- Replay/publication: `replay_match.py`, `generate_trajectories.py`, `generate_publication_trajectory.py`.

### Tests (`tests/unit/*.py`)
- Markov/live core: `test_markov_baseline.py`, `test_live_win_probability.py`, `test_markov_input_construction.py`, `test_markov_inverse_and_prior.py`.
- Features/ratings/modeling: `test_feature_engineering_day5.py`, `test_serve_return_features.py`, `test_point_level_features.py`, `test_point_score_parser.py`, `test_score_parser.py`, `test_elo.py`, `test_elo_v2_extensions.py`, `test_head_to_head_and_tournament_features.py`, `test_build_symmetric_dataset.py`.
- Identity/registry: `test_canonical_names.py`, `test_canonical_players.py`, `test_player_aliases.py`.

## 0.2 Required inventories

### Standalone run entry points
- All root-level `*.py` scripts listed above.
- All pipeline scripts in `pipelines/` are standalone `python pipelines/<script>.py` entry points.

### Modules defining model/probability computation
- `src/tennis_intel/live/markov_baseline.py`
- `src/tennis_intel/live/live_win_probability.py`
- `src/tennis_intel/live/monte_carlo_engine.py`
- `src/tennis_intel/live/ml_informed_markov.py`
- `src/tennis_intel/live/hybrid_engine.py`
- `src/tennis_intel/live/markov_inverse.py`
- `src/tennis_intel/evaluation/metrics.py`

### Modules touching loading/preprocessing
- `src/tennis_intel/live/build_point_dataset.py`
- `src/tennis_intel/features/*`
- `src/tennis_intel/ratings/processor.py`, `src/tennis_intel/ratings/surface_elo.py`
- `src/tennis_intel/data/*`, `src/tennis_intel/entities/*`
- Pipeline build scripts in `pipelines/build_*.py`

### Test files
- 18 unit test files under `tests/unit` (all discovered tests currently unit-level).

### Evaluation/metrics scripts
- `pipelines/evaluate_live_engines.py`
- `pipelines/evaluate_live_engines_v2.py`
- `pipelines/evaluate_ml_informed_markov.py`
- `pipelines/evaluate_hybrid_engine.py`
- `pipelines/evaluate_all_engines_unified.py`
- `src/tennis_intel/evaluation/metrics.py`

## 0.3 Data flow diagrams in words

### A) Pure Markov path
1. Point/match row -> `_row_to_match_state` in `pipelines/evaluate_live_engines_v2.py`.
2. Extract `winner_first_serve_win_pct_career` and opponent serve for return construction in `markov_p_winner`.
3. Call `prob_a_wins_match_from_state` in `src/tennis_intel/live/live_win_probability.py`.
4. Inside recursion, use `prob_win_game`, `prob_win_set`, tiebreak helpers from `src/tennis_intel/live/markov_baseline.py`.
5. Output point-time `P(A wins match)`.

Ambiguity noted: multiple wrappers (`evaluate_live_engines.py`, `replay_match.py`, `evaluate_live_engines_v2.py`) each construct state; conventions are similar but duplicated.

### B) ML + Monte Carlo path
1. Build point data (`build_point_dataset`) and load classifier (`day9_point_classifiers.joblib`).
2. For each point state, call `batch_simulate_dynamic` in `src/tennis_intel/live/monte_carlo_engine.py`.
3. At each simulated point:
   - regenerate situational flags (`is_break_point`, `is_set_point`, `is_match_point`),
   - build feature matrix,
   - call classifier `predict_proba`,
   - sample point outcome, advance state via `_advance_point`.
4. Return fraction of simulation paths where A wins.

Ambiguity noted: there are three simulation entrypoints (`simulate_match_from_state`, `simulate_match_with_classifier`, `batch_simulate_dynamic`) with different safeguards and behavior.

### C) ML-informed Markov hybridized-by-rate path
1. Build same row/state (`_row_to_match_state`).
2. At match start, compute pre-match `p0` (`compute_ml_pre_match_probability` in `generate_publication_trajectory.py` called from `evaluate_ml_informed_markov.py`).
3. Convert to point-rate prior with `build_pretrained_prior` -> `invert_prematch_probability` + `n0` scaling.
4. For each point:
   - create two synthetic rows (`server_is_winner` true/false), infer `p_a_serve_raw`, `p_a_return_raw`.
   - compute recursion sensitivity, blend classifier estimate with Beta posterior mean (`sensitivity_aware_blend`).
   - call `prob_a_wins_match_from_state`.
   - update posterior from actual point outcome (`PtWinner`) on serve/return channel.
5. Output point-time `P(A wins match)`.

Ambiguity noted: pre-match probability source is not centralized in one module and is imported from a pipeline script (`generate_publication_trajectory.py`), increasing fragility.

## 0.4 Testing infrastructure and coverage

### What exists
- Unit tests (`pytest`) under `tests/unit`.
- Pipeline validation scripts (`pipelines/validate_*.py`) for scenario checks.
- Saved-prediction comparison scripts (`evaluate_*`, `diagnose_*`).

### Executed in this audit
- `pytest -q tests/unit` -> `140 passed`.
- Focus tests: Markov/live/prior tests -> `32 passed`.
- Coverage run:
  - total across selected modules: 58%.
  - `markov_baseline.py`: 96%
  - `live_win_probability.py`: 80%
  - `markov_inverse.py`: 86%
  - `ml_informed_markov.py`: 44%
  - `monte_carlo_engine.py`: 0%
  - `hybrid_engine.py`: 0%
  - `build_point_dataset.py`: 0%

### Coverage assessment of core probability-computation code
- High coverage: closed-form Markov baseline and most score parsers.
- Medium coverage: live recursion from arbitrary state.
- Low/none coverage in high-risk modules: Monte Carlo engine and ML-informed Markov path.
- Practical conclusion: core deterministic recursion is tested; simulation and integration-heavy online update path are not sufficiently protected by unit tests.