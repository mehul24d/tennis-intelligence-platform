PROJECT OBJECTIVE (read this first — it is the lens for the entire audit)

The objective of this project is: produce a live win-probability curve that starts at a historically-grounded pre-match baseline (Elo, surface Elo, head-to-head, tournament head-to-head, and other pre-match features) and then updates coherently, point by point, as real in-match evidence accumulates — rather than either ignoring the pre-match context entirely (pure in-match Markov using only static career rates) or ignoring the scoreboard/structure entirely (pure ML output driven only by noisy recent-point features).

Concretely, this means: before a single point is played, the system must output a probability derived from pre-match historical data (Elo/surface-Elo/H2H/tournament-H2H). From that exact starting point, in-match performance should progressively update it — with the rate of that update governed by a principled, tunable notion of how much weight the pre-match prior deserves versus how much weight new in-match evidence deserves (e.g., a Beta-Binomial prior with an effective-sample-size parameter, or an equivalent formal mechanism).

Treat the correctness of this prior-to-posterior transition as the central, unifying concern of the whole audit, not one finding among many. Nearly every specific bug or oddity already suspected in this codebase (career-rate staleness in the pure Markov engine, the ML-informed hybrid's smoothed estimate collapsing back onto pure Markov for long stretches, the smoothed estimate being more volatile than the unsmoothed one early in a match, the fixed-weight hybrid damping into an uninformative ~0.5 band) is plausibly a symptom of this same root issue: the mechanism that is supposed to carry a match from "pre-match prior" to "in-match-updated belief" is either missing, mis-specified, or incorrectly implemented somewhere in the pipeline.

Therefore, in every phase below — code review, architecture review, ML algorithm review, data leakage review, and pipeline/sanity checks — explicitly evaluate each finding against this question: "does this cause the system to depart from, or fail to properly implement, a principled pre-match-prior → in-match-posterior update?" Findings that bear directly on this question should be treated as higher priority than findings that don't, regardless of which phase they were found in. The final consolidated report (Phase 6) must include a dedicated section explicitly answering: is the pre-match-prior-to-in-match-posterior transition currently implemented correctly, anywhere in this codebase, end to end — and if not, exactly where does it break?


ROLE

You are acting as a Staff Machine Learning Engineer and code auditor doing a rigorous, adversarial review of this repository. Your job is not to describe what the code does — it is to find where it is wrong, inconsistent, statistically invalid, or silently leaking information it shouldn't have. Assume bugs exist until you've verified otherwise by reading the actual logic, not by reading variable names or comments. Do not give the benefit of the doubt anywhere. If something "looks fine" but you haven't traced the actual data flow, say so explicitly rather than passing it.

CONTEXT (read first, then verify against the actual code — do not assume this description is accurate)

This project is a live tennis win-probability system with (at least) these components:


A Markov/recursive engine that computes exact win probability from current score state given a serve-point-win probability and return-point-win probability per player.
An ML + Monte Carlo engine that predicts point-level probabilities from engineered features and simulates matches forward.
An ML-informed Markov hybrid that is supposed to feed ML-derived, in-match-updated serve/return probabilities into the Markov recursion instead of static career rates.
A pre-match prior step that is supposed to derive a starting win probability (and corresponding point-level serve/return rate priors) from Elo, surface Elo, head-to-head, tournament head-to-head, and other pre-match features, inverted through the Markov recursion into point-level rates, expressed as Beta distributions with a tunable effective-sample-size (n0) parameter, then updated online during the match via Beta-Binomial conjugate updating.
Historical bugs already found and reportedly fixed in a prior session: a _row_to_match_state orientation bug (server/returner or player-A/player-B mixed up somewhere), a p_return construction bug (return probability not correctly derived as complement/adjustment of opponent's serve probability), and an _advantage_set recursion gap (the exact recursion not correctly handling advantage-set / no-tiebreak-final-set scoring).
Evaluation code computing Log Loss, Brier Score, and Expected Calibration Error (ECE), reportedly with a finding that ECE didn't improve even when Log Loss/Brier did for the ML-informed Markov engine.


Do not trust this summary. Find and read the actual files. If any of the above doesn't match what's actually implemented, flag the mismatch explicitly as its own finding — a stale/inaccurate mental model of your own system is itself a risk.


PHASE 0 — Repository Mapping (do this first, output before anything else)


Produce a full file tree of the repository, annotated with a one-line purpose for every file that contains logic (skip pure config/lockfiles).
Identify and list: every entry point/script that can be run standalone, every module that defines a model or probability computation, every module that touches data loading/preprocessing, every test file, and every evaluation/metrics script.
Build a data flow diagram in words: for a single live match, trace exactly which functions are called in what order, from raw input (match/point data) to the final probability output, for each of the three engines (Markov, ML+MC, hybrid). Note any place this trace required guessing because the code path was unclear — that ambiguity is itself a finding.
Identify what testing infrastructure exists (unit tests, integration tests, backtests) and what fraction of the core probability-computation code is actually covered.


Write this to audit_reports/00_repo_map.md.


PHASE 1 — Code Review (general software quality)

For every file involved in probability computation, data loading, or evaluation:


Read the actual logic line by line, not just signatures/docstrings.
Check for: off-by-one errors (especially in score-state indexing — 0-indexed vs 1-indexed games/points, deuce/advantage boundary conditions, tiebreak point-target boundaries e.g. 7 vs 10 points), incorrect boolean logic (especially any if server == player_A type branching — these are exactly where orientation bugs hide), silent type coercion issues, mutable default arguments, exception handling that swallows errors silently (except: pass or broad except Exception without logging), inconsistent handling of None/missing data, and any global/module-level mutable state that could leak between matches or between test runs.
Check for dead code and contradictory duplicate implementations — e.g., is there more than one function that computes "point win probability given score" and do they agree with each other on the same inputs? If there are two implementations of similar logic anywhere (this is common after "fixing" a bug in one place but not a duplicate), find and reconcile them.
Check naming/units consistency: is p_serve always "probability the server wins the point" everywhere it's used, or does it silently sometimes mean "probability player A wins the point" in some function signatures? This exact confusion is the class of bug already found once (orientation bug) — assume it can recur elsewhere until proven otherwise by reading every call site.


Write findings to audit_reports/01_code_review.md, each finding with: file:line, description, why it matters, suggested fix, severity (Critical/High/Medium/Low).


PHASE 2 — Architecture Review


Is the "Markov recursion as backbone, ML supplies point-level rate inputs" separation actually respected in the code, or does the ML component's output get used in ways that bypass/duplicate the recursion (e.g., does anything blend a P(match win) output from ML directly with a P(match win) output from Markov using a fixed weight — this was already flagged as an inferior fixed-weight-hybrid approach; confirm whether it still exists anywhere alongside the newer hybrid, and if so, whether it's still reachable/used by anything)?
Are configuration values (prior effective sample size n0, feature lists, model hyperparameters, surface definitions, tournament-format best-of-3-vs-5 logic) centralized, or duplicated/hardcoded in multiple places with risk of drift?
Is there a clean, single source of truth for "current match/score state," or do different engines maintain their own parallel copies of score state that could desynchronize?
Trace how the pre-match prior (Elo/H2H/surface/tournament-derived) is threaded into the online Beta-Binomial update. Specifically verify: (a) the prior is inverted through the actual Markov forward recursion to derive point-level rates, not approximated some other way; (b) n0 is not a single global constant applied identically regardless of how much Elo/H2H data actually exists for the two players (thin-history matchups should arguably get a different n0 than deep-history ones — check whether this distinction exists anywhere or is planned-but-missing).
Check whether the online update only updates a player's serve posterior on that player's own service points, and their return posterior on their own return points (not conflating the two, and not updating both players' rates on a single point when only one player's serve/return rate is actually being observed).


Write to audit_reports/02_architecture_review.md.


PHASE 3 — ML Algorithm Review


For every ML model in the pipeline: what is it actually predicting (next point, next game, latent state), what features does it consume, and does the training label match what the inference-time usage assumes it means?
Check the Monte Carlo simulation component specifically: how many paths are simulated per estimate, is a fixed random seed used anywhere that could artificially reduce/inflate apparent variance run-to-run, and is there any variance-reduction technique in place (antithetic variates, common random numbers) — if not, quantify (by running it) how much estimate-to-estimate noise exists purely from simulation variance at a fixed input.
Check the calibration layer (if implemented) — is isotonic/Platt regression fit on a held-out set, or does it risk being fit and evaluated on overlapping data (which would itself be a leakage issue, see Phase 4)?
Specifically investigate the reported "Log Loss/Brier improved but ECE didn't" finding for the ML-informed Markov engine: recompute ECE from scratch, plot/describe a reliability diagram, and determine whether this is a genuine sharpness-without-calibration problem or an artifact of the ECE binning/implementation (e.g., too few bins, bins computed on a tiny eval set, wrong binning scheme).
Check whether the Beta-Binomial online updater's n0 and update rule actually reproduce the following required sanity behaviors — implement and run these as executable checks, don't just reason about them:

At point index 0 (pre-match), the posterior mean must exactly equal the Elo/H2H-derived prior (no accidental smoothing/blending toward some other default at initialization).
Given a fixed n0, a run of N consecutive service points won should shift the posterior mean by an amount consistent with the Beta-Binomial update formula — compute this analytically and compare to what the code actually outputs.
Early in a match (few points observed), the estimate should not be more volatile than a raw unsmoothed per-point ML estimate — if it currently is (as suspected from prior chart review), find the specific cause (e.g., n0 too small, prior variance miscomputed, update applied per-point instead of per-service-point, or the prior mean itself being recomputed/refreshed mid-match incorrectly).





Write to audit_reports/03_ml_algorithm_review.md, including the actual numbers/outputs from the executable checks above, not just descriptions of what you expect them to show.


PHASE 4 — Data Leakage Review (highest priority — be exhaustive)

For every model training and evaluation path in the repository:


Temporal leakage: Does any feature used at "time t" (a given point in a match) incorporate information that would not have been known at that real-world moment? Specifically check: career serve/return rate features — are they computed using only matches prior to the match being predicted, or averaged over a player's full career including matches that happened after the match in question? Check H2H features the same way. Check any "current form" or "momentum" feature for the same issue.
Split leakage: Are train/test/validation splits performed by match (all points from a given match entirely in one split) or naively by row/point? If by row, points from the same match appearing in both train and test would let the model implicitly learn match-specific information. Check this explicitly in every place a split is performed, including inside any cross-validation loop.
Group leakage across matches: Could the same match, player, or tournament-edition appear under different identifiers/rows and end up split across train/test without being recognized as the same underlying entity?
Target leakage: Does any feature encode the outcome directly or near-directly — e.g., a "current win probability" or "momentum score" field that was itself computed using the final match result, or using future points within the same match?
Preprocessing leakage: Are any scalers, imputers, encoders, or feature-selection steps fit on the full dataset (train+test combined) before the split, rather than fit only on train and applied to test/validation?
Calibration leakage: Is the isotonic/Platt calibration layer (if present) fit and evaluated on the same data, or properly on a disjoint calibration set separate from both training and final test data?
Backtest leakage on the pre-match Elo/H2H prior specifically: Elo and H2H are themselves computed from historical results — verify the Elo update and H2H aggregation used as of the start of a given match do not include that match's own result or any later match.
Duplicate leakage: Check for exact or near-duplicate rows (e.g., a match logged twice under slightly different metadata) that could inflate apparent train or test performance.


For every issue found: cite file:line, explain the exact mechanism by which information leaks, estimate the direction and rough magnitude of metric inflation it would cause, and propose a concrete fix (e.g., "switch to GroupKFold on match_id", "recompute career rate as of match_date using only prior matches"). If you cannot find an issue after genuinely checking a category above, say so explicitly and show what you checked — do not silently omit a category.

Write to audit_reports/04_data_leakage_review.md.


PHASE 5 — Pipeline & Sanity Checks

Run (don't just describe) the following, using the actual codebase, and report the real outputs:


Probability coherence: at any given point in a match, do P(player A wins) and P(player B wins) sum to exactly 1.0 across all three engines? Check numerically across a sample of real matches, not just in theory.
Monotonicity at deterministic states: does win probability correctly go to exactly 1.0 the instant a player wins match point, and 0.0 for the opponent, in all three engines?
Degenerate-input checks: feed each engine p_serve = 1.0, p_return = 1.0 (a player who never loses a point) and confirm win probability is exactly 1.0; feed p_serve = 0.0 and confirm sensible (not NaN/crashing) behavior.
Format-boundary checks: verify best-of-3 vs best-of-5 logic, and advantage-set vs final-set-tiebreak logic, are each independently tested against known real match examples with hand-verifiable outcomes.
Regression/golden-output tests: is there a pinned set of known inputs → known expected outputs checked in CI/tests, so a future change that silently alters recursion behavior gets caught automatically? If not, create a minimal one from 2-3 real matches in this repo's data as part of this audit.
Numerical stability: check for any place floating point drift could accumulate over a long match (400+ point recursions) and whether probabilities ever visibly fail to sum to 1 due to this.
Reproducibility: running the same match through the same engine twice — does it produce identical output? If the ML+MC engine doesn't (due to unseeded randomness), quantify the run-to-run variance.


Write to audit_reports/05_pipeline_sanity_checks.md, including actual command output/numbers, not hypothetical expectations.


PHASE 6 — Consolidated Findings & Prioritized Action Plan

Read all five reports you just wrote and produce a single consolidated report at audit_reports/06_consolidated_summary.md containing:


A table of every finding across all phases, each with: Phase, File:Line, Severity (Critical/High/Medium/Low), one-line description, estimated effort to fix (S/M/L).
A Critical section: anything that would invalidate reported metrics or produce wrong probabilities in production, listed first, in the order you'd fix them.
A High-impact section: real problems, not currently invalidating results, but degrading quality or robustness.
A Nice-to-have section.
A "Verified fine, checked explicitly" section — list what you checked and confirmed was actually correct, so it's clear these weren't skipped, not just assumed safe.
A dedicated section titled "Pre-Match-Prior → In-Match-Posterior Transition: Verdict" that directly answers, end to end: is this mechanism implemented correctly anywhere in the codebase today? Trace it from the Elo/H2H/surface-Elo inputs, through any inversion into point-level rates, through the Beta-Binomial (or equivalent) online update, through to what actually gets fed into the Markov recursion at each point — and state plainly, in one paragraph, exactly where in that chain it first breaks down (if it does), rather than leaving this implicit across scattered findings.
An honest closing paragraph stating your overall confidence in the current metrics (Log Loss/Brier/ECE numbers reported anywhere in the repo) given what you found — explicitly say whether you believe those numbers are currently trustworthy, partially trustworthy pending fixes, or not trustworthy, and why.


RULES FOR THIS ENTIRE AUDIT


Prefer running actual code (unit tests, small scripts, REPL checks) over reasoning from reading code alone, wherever a claim can be verified by execution.
If something is ambiguous, state the ambiguity as a finding rather than silently picking the more favorable interpretation.
Do not soften severity ratings to be polite. A Critical issue that invalidates a reported metric should be called Critical even if it means work already presented as "done" and "validated" is not actually validated.
Cite exact file paths and line numbers for every finding — no finding should be vague enough that it can't be located and fixed directly from your report.