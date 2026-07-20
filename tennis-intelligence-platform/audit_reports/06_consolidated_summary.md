# Phase 6 - Consolidated Findings and Action Plan

## Master findings table

| Phase | File:Line | Severity | Finding | Effort |
|---|---|---|---|---|
| 1,5 | `src/tennis_intel/live/monte_carlo_engine.py:132` | Critical | `simulate_match_from_state` does not short-circuit already-terminal states; returns non-1/0 values. | S |
| 1,5 | `src/tennis_intel/live/monte_carlo_engine.py:132`, `src/tennis_intel/live/monte_carlo_engine.py:450` | Critical | No hard step cap in non-batched simulators; hangs on degenerate inputs (`p_server_wins_point=1.0`). | M |
| 4 | `src/tennis_intel/live/build_point_dataset.py:142` | Critical | `server_is_winner` feature uses true match winner identity (target leakage for live use). | M |
| 1 | `src/tennis_intel/live/build_point_dataset.py:129-136` | High | Dead/overwritten orientation code block in `server_is_winner` construction path. | S |
| 2 | `pipelines/evaluate_live_engines_v2.py:154`, `pipelines/replay_match.py:100`, `pipelines/evaluate_live_engines.py:84` | High | Duplicated state/orientation mappers increase recurrence risk of orientation bugs. | M |
| 2 | `src/tennis_intel/live/hybrid_engine.py:25`, `pipelines/evaluate_hybrid_engine.py:1` | High | Fixed-weight hybrid path still active despite architectural direction toward prior->posterior engine. | S |
| 1,2 | `pipelines/build_day9_point_model.py:50`, `pipelines/tune_day9_hyperparameters.py:60`, `pipelines/evaluate_live_engines_v2.py:93` (+others) | Medium | Feature schema duplicated across scripts; drift risk acknowledged in comments and still present. | M |
| 2 | `src/tennis_intel/live/ml_informed_markov.py:200-251` | Medium | `n0` confidence uses Elo match-count proxy only, not full prior evidence uncertainty. | M |
| 5 | `tests/unit/*` coverage report | Medium | Coverage gaps on high-risk modules (`monte_carlo_engine.py`, `hybrid_engine.py`, `build_point_dataset.py`). | M |

## Critical (fix in this order)

1. Remove leakage feature path (`server_is_winner`) and retrain/evaluate all point and engine metrics.
2. Fix Monte Carlo terminal-state handling (`if a_sets>=needed: return 1.0`, etc.).
3. Add bounded termination (`max_points`) to all MC simulation APIs, with explicit undecided handling.
4. Recompute and republish engine metrics after fixes 1-3.

## High-impact

1. Centralize state/orientation conversion in one tested module.
2. Remove dead orientation code in `build_point_dataset.py`.
3. Decommission fixed-weight hybrid from default evaluation path.
4. Centralize feature schema constants across train/tune/eval/replay scripts.

## Nice-to-have

1. Upgrade `n0` from single-proxy scaling to composite uncertainty model.
2. Add variance-reduction options in MC simulation for lower comparison noise.
3. Add explicit advantage-set/no-tiebreak historical regression tests.

## Verified fine (checked explicitly)

- Elo and H2H pre-match extraction/update ordering is leakage-safe by code path.
- Markov closed-form baseline is stable and coherent (`P(A)+P(B)=1` at machine precision in sampled checks).
- Beta-Binomial initialization and update equations in ML-informed Markov match analytic values exactly.
- Calibration metric recomputation from saved prediction artifacts reproduces consistent values and target alignment.
- Unit test suite health: `140 passed` in current environment.

## Pre-Match-Prior -> In-Match-Posterior Transition: Verdict

The transition is implemented coherently only in the ML-informed Markov path (`build_pretrained_prior` -> `invert_prematch_probability` -> `ServeReturnPosterior` updates -> Markov recursion), but it is not correct end-to-end for the project as currently evaluated because the point-level model feeding that transition is trained with an outcome-derived leakage feature (`server_is_winner`). The first hard break in the chain is therefore at feature construction/training time (`build_point_dataset.py`), before prior inversion or Beta updates even execute. Separately, alternative active engine paths (pure Markov and ML+MC) do not implement this prior->posterior mechanism at all.

## Confidence in reported metrics

Current reported Log Loss/Brier/ECE numbers are **partially trustworthy at best**. Deterministic Markov recursion metrics are more credible, but any metrics involving the point classifier and MC rollout are compromised by (a) target leakage via `server_is_winner` and (b) critical MC simulator correctness bugs in terminal/degenerate states. Metrics should be treated as provisional until these are fixed and all evaluations are rerun.