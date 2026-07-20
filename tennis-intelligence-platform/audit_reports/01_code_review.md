# Phase 1 - Code Review Findings

Focus: probability computation, data loading/preprocessing, evaluation paths.

## Findings (ordered by severity)

### 1) `simulate_match_from_state` ignores already-terminal states
- File: `src/tennis_intel/live/monte_carlo_engine.py:132`
- Severity: Critical
- Description: `simulate_match_from_state` enters simulation loops even when input state already has match winner decided (`a_sets >= sets_needed` or `b_sets >= sets_needed`).
- Why it matters: deterministic-state requirement is violated. In this audit run:
  - `simulate_match_from_state(2,0,0,0,0,0,True,False,3,0.6,n=200) -> 0.8`
  - `simulate_match_from_state(0,2,0,0,0,0,True,False,3,0.6,n=200) -> 0.3`
  should be exactly `1.0` and `0.0`.
- Suggested fix: add immediate terminal-state guards before simulation loop.

### 2) Monte Carlo engine can hang on degenerate serve probabilities
- File: `src/tennis_intel/live/monte_carlo_engine.py:132` and `src/tennis_intel/live/monte_carlo_engine.py:450`
- Severity: Critical
- Description: no hard per-simulation step cap in `simulate_match_from_state` and `simulate_match_with_classifier`; at `p_server_wins_point=1.0`, execution hung in audit and had to be killed.
- Why it matters: violates degenerate-input robustness requirement; can freeze batch jobs or live serving path under edge inputs/model saturation.
- Suggested fix: add `max_points` safety cap (as already done in `batch_simulate_dynamic`) and return explicit undecided handling/logging.

### 3) Outcome-dependent feature in point model (`server_is_winner`) is target-leakage-prone by design
- File: `src/tennis_intel/live/build_point_dataset.py:142`, `pipelines/build_day9_point_model.py:63`
- Severity: Critical
- Description: feature is computed using match winner identity from historical truth and then used in model training/inference features.
- Why it matters: this is not available in true live prediction; it leaks post-match information into point predictions and distorts downstream engine metrics.
- Suggested fix: replace with outcome-free feature (`server_is_player1` or `server_is_tracked_player`) and retrain/evaluate.

### 4) Dead/contradictory code path in point dataset feature construction
- File: `src/tennis_intel/live/build_point_dataset.py:129-136`
- Severity: High
- Description: an initial `server_is_winner` assignment with a redundant/opaque map-lambda expression is immediately overwritten by a second simpler assignment.
- Why it matters: dead code in high-risk orientation logic increases regression risk and audit ambiguity.
- Suggested fix: remove first assignment block and keep single authoritative construction.

### 5) Duplicated score-state/orientation conversion logic across scripts
- File: `pipelines/evaluate_live_engines_v2.py:154`, `pipelines/evaluate_live_engines.py:84`, `pipelines/replay_match.py:100`
- Severity: High
- Description: multiple `_row_to_match_state` implementations exist in different pipeline scripts.
- Why it matters: orientation bugs already happened in this codebase; duplicated conversion logic is a known recurrence vector.
- Suggested fix: centralize state-conversion helper in one importable module and test it once.

### 6) Duplicated feature-list definitions across training/evaluation/replay scripts
- File: `pipelines/build_day9_point_model.py:50`, `pipelines/tune_day9_hyperparameters.py:60`, `pipelines/evaluate_live_engines_v2.py:93`, `pipelines/replay_match.py:79`, `pipelines/generate_publication_trajectory.py:89`
- Severity: Medium
- Description: feature columns are manually duplicated in multiple files.
- Why it matters: drift causes silent NaN inputs or shape mismatches; comments already acknowledge this risk.
- Suggested fix: move feature schema to single source (`src/tennis_intel/live/feature_schema.py`) and import everywhere.

### 7) Simulation-path modules have no unit-test coverage
- File: `src/tennis_intel/live/monte_carlo_engine.py` (coverage run result)
- Severity: High
- Description: coverage is 0% for Monte Carlo engine and 0% for `hybrid_engine.py`.
- Why it matters: critical runtime bugs (#1, #2) were not caught pre-audit.
- Suggested fix: add unit tests for terminal states, degenerate probabilities, and bounded termination.

## Checked and found acceptable
- `src/tennis_intel/live/markov_baseline.py`: boundary checks and monotonicity are solid.
- `src/tennis_intel/live/live_win_probability.py`: regular-game and tiebreak recursion logic is structurally consistent and backed by passing tests.
- `src/tennis_intel/live/markov_inverse.py`: clipping and bisection behavior are sane, with regression tests for target extremes.

## Executed evidence
- `pytest -q tests/unit/test_markov_baseline.py tests/unit/test_live_win_probability.py tests/unit/test_markov_input_construction.py tests/unit/test_markov_inverse_and_prior.py` -> `32 passed`.
- `pytest -q tests/unit` -> `140 passed`.
- Runtime checks (this audit): reproduced terminal-state MC failure and degenerate-input hang.