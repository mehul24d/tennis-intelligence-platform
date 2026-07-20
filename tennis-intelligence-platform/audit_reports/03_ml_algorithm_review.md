# Phase 3 - ML Algorithm Review

## 3.1 What each model predicts vs how it is used

- Point classifier (`pipelines/build_day9_point_model.py`): predicts `server_wins_point`.
- ML+MC (`src/tennis_intel/live/monte_carlo_engine.py`): uses point-level classifier outputs to simulate full continuation.
- ML-informed Markov (`src/tennis_intel/live/ml_informed_markov.py`): uses classifier-derived point-rate estimates (serve/return contexts) blended with Beta posterior, then feeds Markov recursion.

Consistency check: target label and usage are aligned at point level. However, see leakage risk in Phase 4 for `server_is_winner` feature.

## 3.2 Monte Carlo variance and reproducibility

Executed check:
- Command run in audit with `simulate_match_from_state(... p=0.6, n_simulations=1000)`.

Results:
- Same seed path is exactly reproducible (default internal RNG seed):
  - run1 = 0.529
  - run2 = 0.529
- Different seeds produce measurable stochastic spread:
  - values = `[0.515, 0.512, 0.485, 0.492, 0.499, 0.502, 0.502, 0.499, 0.510, 0.489]`
  - std dev = `0.0099247`

Interpretation:
- Engine is reproducible under fixed seed.
- Monte Carlo noise is non-trivial at n=1000 and must be accounted for in comparisons.

## 3.3 Calibration findings (recomputed from saved predictions)

Data used:
- `data/processed/day11_head_to_head_v2_predictions.parquet`
- `data/processed/ml_informed_markov_predictions.parquet`
- Inner merge count = `25881` points, target mismatch = `0`.

Recomputed metrics:
- Markov: `log_loss=0.6287`, `brier=0.1996`, `ECE10=0.0903`, `ECE20=0.0950`, `ECE30=0.0982`
- ML+MC: `log_loss=0.1976`, `brier=0.0510`, `ECE10=0.0914`, `ECE20=0.0914`, `ECE30=0.0914`
- ML-informed Markov: `log_loss=0.2652`, `brier=0.0801`, `ECE10=0.0479`, `ECE20=0.0479`, `ECE30=0.0482`

Conclusion on prior claim ("log loss/Brier improved but ECE didn't"):
- Not true for current saved outputs: ML-informed Markov materially improves ECE vs both Markov and ML+MC.
- The historical statement appears stale/outdated relative to current artifacts.

## 3.4 Required Beta-Binomial sanity checks (executed)

### Check 1: point index 0 posterior mean equals prior exactly
- Executed with `build_pretrained_prior` + `ServeReturnPosterior.from_pretrained_prior`.
- Output:
  - `p_serve0 = 0.6648008298873902`
  - `posterior_mean_serve = 0.6648008298873902`
  - exact equality: `True`
- Result: PASS.

### Check 2: N-point run matches analytic Beta-Binomial update
- Scenario: N=12 service points, wins=9, losses=3.
- Output:
  - code posterior mean = `0.6822279328649695`
  - analytic posterior mean = `0.6822279328649695`
  - abs error = `0.0`
- Result: PASS.

### Check 3: early-match volatility behavior
- Proxy measured at initialization (`n0_serve=46.666...`):
  - +1 win delta = `+0.00703`
  - +1 loss delta = `-0.01395`
- Interpretation:
  - Bayesian posterior itself is stable early (small moves).
  - Any observed large early jump is likely from blend mechanics and/or classifier term, not Beta update algebra.

## 3.5 Algorithmic risks

1. `server_is_winner` dependence contaminates practical validity (see Phase 4 critical).
2. MC engine has deterministic-state bugs (Phase 5 critical), which can distort algorithm comparisons.
3. No variance-reduction method (CRN/antithetic) implemented in MC rollout; comparison noise remains seed-sensitive.