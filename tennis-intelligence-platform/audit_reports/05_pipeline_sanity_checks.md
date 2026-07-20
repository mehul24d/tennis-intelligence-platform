# Phase 5 - Pipeline and Sanity Checks

All checks below were executed in this audit session unless explicitly marked otherwise.

## 1) Probability coherence (`P(A)+P(B)=1`)

### Markov engine
- Method: mirrored state/orientation on sampled real points built from `build_point_dataset`.
- Output:
  - `n=120`
  - `max_abs_error=6.661338147750939e-16`
  - `mean_abs_error=9.62193288008469e-17`
- Result: PASS (numerically coherent).

### ML-informed and ML+MC
- Attempted mirrored checks reveal strong orientation sensitivity if mirror features are not perfectly swapped.
- Observed outputs in naive mirror harness:
  - ML-informed max error up to `0.0374`
  - ML+MC max error up to `0.9787`
- Interpretation: these numbers are not accepted as final coherence verdict because mirrored feature construction is not uniquely defined in current API.
- Residual risk: there is no canonical public function `p_player1` + `p_player2` from same state for these engines.

## 2) Deterministic terminal-state monotonicity

### Markov
- `prob_a_wins_match_from_state(MatchState(2,0,...), ...) = 1.0`
- `prob_a_wins_match_from_state(MatchState(0,2,...), ...) = 0.0`
- Result: PASS.

### ML-informed Markov
- With terminal states, output returned exactly `1.0` and `0.0` in audit check.
- Result: PASS.

### ML+MC (`simulate_match_from_state`)
- `simulate_match_from_state(2,0,0,0,0,0,True,False,3,0.6,n=200) = 0.8`
- `simulate_match_from_state(0,2,0,0,0,0,True,False,3,0.6,n=200) = 0.3`
- Result: FAIL (critical).

## 3) Degenerate input checks

### Markov
- `prob_win_match(1.0,1.0,3) = 1.0`
- `prob_win_match(0.0,0.0,3) = 0.0`
- Result: PASS.

### ML+MC
- At `p_server_wins_point=1.0`, simulation run hung and had to be killed.
- Result: FAIL (critical robustness issue).

## 4) Format-boundary checks (bo3/bo5, tiebreak/advantage-set)

### Existing automated tests found
- bo3 vs bo5 checks exist in `tests/unit/test_markov_baseline.py`.
- Tiebreak deuce-phase checks exist in `tests/unit/test_live_win_probability.py`.

### Gap
- No explicit unit test found for real advantage-set/no-final-set-tiebreak historical case (e.g., Wimbledon-era long final set) despite code comments discussing this logic.
- Result: PARTIAL coverage; add explicit regression for advantage-set path.

## 5) Golden-output regression tests

### Existing status before audit
- No dedicated golden file test for concrete real-match output rows was present.

### Added in this audit
- New test: `tests/unit/test_golden_markov_outputs.py`
- Uses 3 real rows from `data/processed/day11_head_to_head_v2_predictions.parquet`:
  - `20220120-M-Australian_Open-R64-Steve_Johnson-Jannik_Sinner`, Pt `1`, `markov_pred=0.363702`
  - same match, Pt `2`, `markov_pred=0.376799`
  - same match, Pt `3`, `markov_pred=0.404097`
- Validation run: `1 passed`.

## 6) Numerical stability

- Markov complement test errors were at machine precision (`~1e-16`) on sampled states.
- No visible float-drift issue observed in deterministic Markov path.
- ML+MC path has larger correctness issues before float drift becomes the binding concern.

## 7) Reproducibility

### ML+MC
- Same-run default seed reproducibility:
  - run1 = `0.529`
  - run2 = `0.529`
  - exact equality: `True`
- Different seeded runs (`n=1000`) produced:
  - `[0.515, 0.512, 0.485, 0.492, 0.499, 0.502, 0.502, 0.499, 0.510, 0.489]`
  - std dev = `0.0099247`

### Markov
- Deterministic by construction.

### ML-informed Markov
- Deterministic for fixed model/features/state (no sampling in recursion path).

## Phase 5 summary
- Markov path passes coherence and deterministic constraints.
- ML-informed path passes key deterministic checks in this audit.
- ML+MC has two critical failures: terminal-state handling and degenerate-input non-termination.