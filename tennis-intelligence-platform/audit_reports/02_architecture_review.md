# Phase 2 - Architecture Review

Lens: correctness of pre-match-prior -> in-match-posterior transition.

## Findings

### A) Fixed-weight hybrid path is still live/reachable
- File: `src/tennis_intel/live/hybrid_engine.py:25`, `pipelines/evaluate_hybrid_engine.py:1`
- Severity: High
- Issue: architecture still contains and evaluates a fixed-weight match-probability blend (Markov + ML+MC), despite project direction to move to ML-informed Markov.
- Impact on central objective: this path bypasses principled prior/posterior updating and reintroduces ad hoc match-level blending.
- Recommendation: hard-deprecate or isolate behind explicit experimental flag.

### B) Prior inversion chain exists in ML-informed path, but only there
- File: `src/tennis_intel/live/ml_informed_markov.py:200`, `src/tennis_intel/live/markov_inverse.py:33`, `pipelines/evaluate_ml_informed_markov.py:78`
- Severity: Medium
- Issue: the intended chain (pre-match probability -> Markov inversion -> Beta posterior update -> recursion input) is implemented only in ML-informed Markov flow.
- Impact: pure Markov and ML+MC paths do not implement this central transition mechanism.

### C) `n0` confidence uses Elo match-count proxy only
- File: `src/tennis_intel/live/ml_informed_markov.py:200-251`
- Severity: Medium
- Issue: `n0` scales off `elo_matches_played_pre_*` only; it does not account for availability/strength of H2H/tournament-H2H evidence directly.
- Impact on objective: prior strength does not fully reflect pre-match evidence richness.
- Recommendation: move from single-proxy confidence to composite uncertainty signal.

### D) Serve/return posterior updates are correctly channel-specific
- File: `src/tennis_intel/live/ml_informed_markov.py:176-198`, `src/tennis_intel/live/ml_informed_markov.py:384-389`
- Severity: Verified fine
- Observation: when `state.server_is_a` is true, only serve posterior updates; else only return posterior updates.
- Impact: aligns with required update semantics.

### E) No single source of truth for score-state construction
- File: `pipelines/evaluate_live_engines_v2.py:154`, `pipelines/replay_match.py:100`, `pipelines/evaluate_live_engines.py:84`
- Severity: High
- Issue: multiple implementations of state construction/orientation mapping.
- Impact on objective: increases probability of reintroducing prior orientation bugs and breaking transition correctness at ingestion boundary.
- Recommendation: centralize in one module with strict tests.

### F) Pre-match probability source is imported from a pipeline script
- File: `pipelines/evaluate_ml_informed_markov.py:38` (imports `compute_ml_pre_match_probability` from `generate_publication_trajectory.py`)
- Severity: Medium
- Issue: core engine dependency is coupled to publication pipeline code.
- Impact: brittle dependency path for critical prior initialization logic.
- Recommendation: move pre-match estimator to `src/tennis_intel/live/prematch_prior.py`.

## Central architectural verdict (phase-level)
- The pre-match-prior -> in-match-posterior architecture exists and is mostly coherent inside ML-informed Markov only.
- It is not an end-to-end project-wide architecture yet because:
  1. alternative active engine paths bypass it,
  2. state/orientation ingestion is duplicated,
  3. prior confidence (`n0`) is only partially evidence-aware.