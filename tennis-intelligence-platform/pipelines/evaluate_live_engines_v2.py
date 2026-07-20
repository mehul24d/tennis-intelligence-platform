"""
evaluate_live_engines_v2.py — corrected head-to-head evaluation (handoff Tasks 1, 2, 4, 5, 6).

FIXES OVER v1 (see docs/day10_head_to_head_freeze.md for why each was needed):

1. VALID CALIBRATION TARGET (Task 1): v1 always tracked the eventual winner, making the
   target degenerate (always 1) and every calibration number a mathematical artifact
   (gap == 1 - mean_predicted in every bucket, verified exactly). v2 tracks a
   DETERMINISTICALLY RANDOM player per match — a hash of match_id decides whether the
   tracked player is the winner or the loser (same hash-based assignment pattern as the
   frozen build_symmetric_dataset.py). The target is therefore a genuine ~50/50 mix of
   1s and 0s and calibration/ECE/reliability tables are statistically meaningful.
   Implementation note: both engines' predictions are computed for P(winner wins) exactly
   as before; when the tracked player is the loser, prediction := 1 - p and target := 0.
   This is an information-preserving reframing, not a change to either engine.

2. DYNAMIC ROLLOUT (Task 7): uses batch_simulate_dynamic, which re-derives break/set/match
   point flags and momentum from each simulation's own evolving state at every tick —
   fixing the stale-context limitation identified as the likely dominant cause of v1's
   large Markov-vs-ML gap.

3. BEST-OF-SCALED SIMULATION CAP (Task 6): v1's flat max_points=400 was genuinely reachable
   by real-length best-of-5 continuations (long 5-setters exceed 400 points), so many of
   its non-termination warnings were expected behavior on Slam matches, not purely a bug.
   v2 uses 350 (bo3) / 700 (bo5) and logs non-termination COUNTS in aggregate rather than
   per-occurrence (v1's per-warning logging flooded the console). The v1 throughput
   degradation is consistent with date-sorted match_ids clustering best-of-5 Slam matches
   together (each such point costs several times more simulation work), compounded by
   plausible thermal throttling on a fanless laptop over a 4.5-hour run; run
   diagnose_day10_runtime.py against the saved v1 parquet to confirm empirically.

4. FULL METRICS + PAIRED SIGNIFICANCE + PROFILING (Tasks 2, 4, 5): log loss, Brier, ECE,
   sharpness, reliability tables, runtime/throughput per engine, paired bootstrap CIs on
   the log-loss and Brier differences, and a per-stage runtime/memory summary.

Reproducibility: all seeds fixed; match selection and player assignment are deterministic
functions of match_id, independent of row order.

Usage:
    python pipelines/evaluate_live_engines_v2.py > day11_output.txt 2>&1
"""

from __future__ import annotations

import logging
import os
import random
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.live_win_probability import MatchState, prob_a_wins_match_from_state
from tennis_intel.live.match_state_conversion import row_to_match_state
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.monte_carlo_engine import batch_simulate_dynamic
from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error,
    calibration_table, sharpness, paired_bootstrap_diff,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_MCP = PROJECT_ROOT / "data" / "raw" / "tennis_MatchChartingProject"
PROCESSED = PROJECT_ROOT / "data" / "processed"

POINT_FILES = [
    RAW_MCP / "charting-m-points-to-2009.csv",
    RAW_MCP / "charting-m-points-2010s.csv",
    RAW_MCP / "charting-m-points-2020s.csv",
]

HOLDOUT_YEAR = 2022
N_MATCHES = 150
N_SIMULATIONS = 200
ROLLOUT_MODEL_NAME = "gradient_boosting"  # switch to "logistic_regression" for ~6x faster runs
RANDOM_STATE = 42
N_WORKERS = None  # None -> os.cpu_count(); set to an int to cap worker processes

# Feature schema centralized (external audit, 2026-07, Code Review finding #6): this was
# an independently-maintained duplicate of the same 24-item pre-match feature list defined
# in three other files — verified identical before consolidating, see
# src/tennis_intel/live/feature_schema.py for the single source of truth (PREMATCH_FEATURE_NAMES
# there is derived programmatically from POINT_FEATURE_COLS, not separately copied).
from tennis_intel.live.feature_schema import PREMATCH_FEATURE_NAMES as STATIC_FEATURE_NAMES



class StageProfiler:
    """Task 5: per-stage wall time + peak memory. Prints an automatic summary."""

    def __init__(self):
        self.stages: list[tuple[str, float, float]] = []
        tracemalloc.start()

    def __call__(self, name: str):
        profiler = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, *a):
                elapsed = time.perf_counter() - self.t0
                _, peak = tracemalloc.get_traced_memory()
                profiler.stages.append((name, elapsed, peak / 1e6))
                logger.info("[profile] %s: %.2fs (peak traced mem %.0f MB)",
                            name, elapsed, peak / 1e6)

        return _Ctx()

    def summary(self) -> str:
        total = sum(s[1] for s in self.stages)
        lines = ["=== Runtime Profile ===",
                 f"{'Stage':<32} {'Time (s)':>10} {'% of total':>11} {'Peak MB':>9}"]
        for name, t, mem in self.stages:
            lines.append(f"{name:<32} {t:>10.2f} {100*t/total:>10.1f}% {mem:>9.0f}")
        lines.append(f"{'TOTAL':<32} {total:>10.2f}")
        return "\n".join(lines)


def tracked_player_is_winner(match_id: str) -> bool:
    """Task 1: deterministic per-match coin flip — same hash-based, order-independent
    assignment principle as the frozen build_symmetric_dataset.py.

    Uses md5, NOT Python's built-in hash(): the built-in is salted per process
    (PYTHONHASHSEED), which would silently change the winner/loser assignment between
    runs and break reproducibility — a real bug caught before the first real run."""
    import hashlib
    digest = hashlib.md5(match_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 2 == 0


def _row_to_match_state(row: dict) -> MatchState:
    """Thin alias to the single canonical implementation — kept under this file's
    original name so every existing call site below (markov_p_winner, ml_p_winner, etc.)
    continues to work unchanged. See match_state_conversion.py for the full
    implementation, docstring, and the two bug-fix histories this centralization
    consolidates (per external audit Architecture Review finding E / Code Review #5:
    three near-duplicate implementations existed independently, and a fix applied to one
    had already failed to reach another — see that module's docstring for specifics)."""
    return row_to_match_state(row)


def markov_p_winner(row: dict) -> float:
    """
    BUG FIX (found via generate_publication_trajectory.py's implausible 0.995 pre-match
    output for a real Sinner-Alcaraz final): p_return must be 1 - the OPPONENT's actual
    serve-win rate (per prob_a_wins_match_from_state's documented definition), not the
    winner's own generic return-stat in isolation. The original version used
    winner_return_pts_won_pct_career (the match winner's own career return average against
    a mix of past opponents) as p_return, silently ignoring the actual loser's real serve
    ability — this affected every point evaluated in the frozen Day 11 results and requires
    a re-run to produce corrected numbers.

    BUG FIX #2 (external review, 2026-07, found via the Sinner-Alcaraz "does the model
    neglect its pre-match prior" investigation): both ps AND pr here were using
    first_serve_win_pct_career directly as if it were each player's TRUE overall serve-
    win rate — but that column is, by construction, only the win rate on points where the
    first serve landed, ignoring second-serve points (won at a meaningfully lower rate)
    entirely. This systematically UNDERSTATED both the winner's own serve strength (ps)
    and the opponent's serve strength feeding pr's construction. Fixed together, in one
    pass, since both are symptoms of the same missing column — see build_point_dataset.py's
    combined_serve_win_pct_career (properly weights first+second serve by how often each
    actually occurs) and return_seed.py for the corrected, opponent-conditioned
    construction. See return_seed.py's own module docstring for a documented near-miss
    where an earlier attempted fix reintroduced the ALREADY-REJECTED "own return average"
    regression this function's BUG FIX #1 (above) specifically moved away from.
    """
    state = _row_to_match_state(row)
    ps = row.get("winner_combined_serve_win_pct_career")
    if ps is None or pd.isna(ps):
        ps = row.get("winner_first_serve_win_pct_career")  # known-inferior fallback
    ps = 0.65 if (ps is None or pd.isna(ps)) else float(ps)
    pr = compute_p_a_return_seed(row, track_winner=True)
    return prob_a_wins_match_from_state(state, ps, pr)


def ml_p_winner(row: dict, model, feature_cols: list, rng_seed: int) -> tuple[float, float]:
    state = _row_to_match_state(row)
    static = {c: row.get(c, np.nan) for c in STATIC_FEATURE_NAMES if c in feature_cols}
    seed_mom = {"p1_momentum_last10": row.get("p1_momentum_last10"),
                "p1_momentum_last20": row.get("p1_momentum_last20")}

    def predict_fn(fm):
        return model.predict_proba(fm)[:, 1]

    t0 = time.perf_counter()
    p = batch_simulate_dynamic(
        (state.a_sets, state.b_sets, state.a_games, state.b_games,
         state.a_points, state.b_points, state.server_is_a, state.is_tiebreak),
        static, feature_cols, predict_fn, best_of=state.best_of,
        player1_is_winner=bool(row["player1_is_winner"]),
        seed_momentum=seed_mom, n_simulations=N_SIMULATIONS,
        rng=random.Random(rng_seed),
    )
    return p, time.perf_counter() - t0


def point_seed(match_id: str, pt: int) -> int:
    """
    Deterministic seed for a single point's simulation, derived from (match_id, Pt) rather
    than loop position. NECESSARY for parallel execution: the original v2 draft seeded via
    a global loop counter, which has no stable meaning once matches are processed by
    multiple worker processes in a non-deterministic order. This function gives every
    point a fixed seed independent of execution order, process, or scheduling — the run is
    fully reproducible, but produces DIFFERENT specific random draws than a sequential run
    seeded by loop position would have. This is an intentional, documented tradeoff: the
    two are statistically equivalent (same estimator, same expected accuracy) but not
    byte-identical, which is unavoidable when moving from position-based to
    order-independent seeding.
    """
    import hashlib
    digest = hashlib.md5(f"{match_id}:{pt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# Populated once per worker process by _init_worker — NOT shared across processes (each
# process gets its own copy via the pool's initializer, avoiding repeated
# pickling/unpickling of the model on every task).
_WORKER_MODEL = None
_WORKER_FEATURE_COLS = None


def _init_worker(model_path: str, rollout_model_name: str) -> None:
    global _WORKER_MODEL, _WORKER_FEATURE_COLS
    payload = joblib.load(model_path)
    _WORKER_MODEL = payload[rollout_model_name]
    _WORKER_FEATURE_COLS = payload["feature_cols"]


def evaluate_one_match(match_id: str, rows: list[dict]) -> dict:
    """
    Runs both engines on every point of ONE match. This is the unit of parallel work —
    matches are fully independent given a loaded model and deterministic per-point seeds,
    so dispatching one of these per worker process requires no shared mutable state and no
    change to either engine's validated logic.
    """
    track_winner = tracked_player_is_winner(match_id)
    markov_preds, ml_preds, ml_times, pts = [], [], [], []

    for row in rows:
        p_markov_w = markov_p_winner(row)
        p_ml_w, ml_t = ml_p_winner(
            row, _WORKER_MODEL, _WORKER_FEATURE_COLS,
            rng_seed=point_seed(match_id, row["Pt"]),
        )
        markov_preds.append(p_markov_w if track_winner else 1.0 - p_markov_w)
        ml_preds.append(p_ml_w if track_winner else 1.0 - p_ml_w)
        ml_times.append(ml_t)
        pts.append(row["Pt"])

    return {
        "match_id": match_id,
        "pts": pts,
        "markov_preds": markov_preds,
        "ml_preds": ml_preds,
        "ml_times": ml_times,
        "target": 1.0 if track_winner else 0.0,
    }


def main() -> None:
    prof = StageProfiler()

    model_path = str(PROCESSED / "day9_point_classifiers.joblib")

    with prof("load_model"):
        # Load once in the main process too, purely to fail fast on a missing/corrupt
        # file before spinning up worker processes.
        payload = joblib.load(model_path)
        feature_cols = payload["feature_cols"]

    with prof("build_point_dataset"):
        frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
        day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
        points = build_point_dataset(POINT_FILES, frozen_join, day6)
        points["match_year"] = points["match_id"].str[:4].astype(int)
        test_points = points[points["match_year"] >= HOLDOUT_YEAR].copy()
        test_points["player1_is_winner"] = (test_points["Svr"] == 1) == test_points["server_is_winner"]

    with prof("select_matches"):
        match_ids = np.sort(test_points["match_id"].unique())
        n_use = min(N_MATCHES, len(match_ids))
        selected = np.random.RandomState(RANDOM_STATE).choice(match_ids, size=n_use, replace=False)
        eval_df = test_points[test_points["match_id"].isin(selected)].copy()
        eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
        logger.info("Evaluating %d matches, %d points, every point", n_use, len(eval_df))

    # Group into per-match record lists — one unit of work per match, dispatched to the
    # pool. Matches are fully independent given a loaded model and deterministic per-point
    # seeds (see point_seed's docstring), so this parallelizes with zero change to either
    # engine's already-validated simulation logic (Task 8's "do not change model behavior
    # for speed alone" constraint).
    match_groups: dict[str, list[dict]] = {}
    for rec in eval_df.to_dict("records"):
        match_groups.setdefault(rec["match_id"], []).append(rec)

    n_workers = min(N_WORKERS, len(match_groups)) if N_WORKERS else min(
        os.cpu_count() or 1, len(match_groups))
    logger.info("Dispatching %d matches across %d worker process(es)",
                len(match_groups), n_workers)

    results_by_match: dict[str, dict] = {}
    with prof("run_both_engines"):
        t_start = time.perf_counter()
        completed = 0
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker,
            initargs=(model_path, ROLLOUT_MODEL_NAME),
        ) as executor:
            futures = {
                executor.submit(evaluate_one_match, mid, rows): mid
                for mid, rows in match_groups.items()
            }
            for future in as_completed(futures):
                mid = futures[future]
                try:
                    results_by_match[mid] = future.result()
                except Exception:
                    logger.exception("Match %s failed in worker — excluded from results", mid)
                completed += 1
                if completed % 20 == 0 or completed == len(match_groups):
                    elapsed = time.perf_counter() - t_start
                    logger.info("Completed %d / %d matches (%.1fs elapsed)",
                                completed, len(match_groups), elapsed)

    # Reassemble in deterministic (match_id, Pt) order regardless of completion order —
    # aggregate metrics are order-invariant, but deterministic output ordering matters for
    # the saved parquet and for anyone diffing runs.
    markov_preds, ml_preds, ml_times, targets, ordered_match_ids, ordered_pts = [], [], [], [], [], []
    for mid in sorted(match_groups.keys()):
        r = results_by_match.get(mid)
        if r is None:
            continue  # failed match, already logged
        markov_preds.extend(r["markov_preds"])
        ml_preds.extend(r["ml_preds"])
        ml_times.extend(r["ml_times"])
        targets.extend([r["target"]] * len(r["pts"]))
        ordered_match_ids.extend([mid] * len(r["pts"]))
        ordered_pts.extend(r["pts"])

    n_failed = len(match_groups) - len(results_by_match)
    if n_failed:
        logger.warning("%d match(es) failed and were excluded from the results", n_failed)

    y = np.array(targets)
    mk = np.clip(np.array(markov_preds), 1e-6, 1 - 1e-6)
    ml = np.clip(np.array(ml_preds), 1e-6, 1 - 1e-6)

    print("\n=== Head-to-Head v2 (valid mixed target, dynamic rollout) ===")
    print(f"Matches: {n_use}, points: {len(y):,}, target balance: {y.mean():.3f} "
          f"(valid calibration requires this to be well away from 0 and 1)\n")

    print(f"{'Engine':<10} {'LogLoss':>9} {'Brier':>8} {'ECE':>8} {'Sharpness':>10}")
    for name, p in [("Markov", mk), ("ML+MC", ml)]:
        print(f"{name:<10} {compute_log_loss(y, p):>9.4f} {compute_brier_score(y, p):>8.4f} "
              f"{expected_calibration_error(y, p):>8.4f} {sharpness(p):>10.4f}")

    total_wall = sum(s[1] for s in prof.stages if s[0] == "run_both_engines")
    print(f"\nML+MC mean inference: {np.mean(ml_times)*1000:.1f}ms/point "
          f"(n_simulations={N_SIMULATIONS}, model={ROLLOUT_MODEL_NAME})")
    print(f"Overall throughput: {len(y)/total_wall:.1f} points/sec")

    print("\n=== Paired Bootstrap (Markov - ML+MC; negative favors Markov) ===")
    for metric_fn, name in [(compute_log_loss, "log_loss"), (compute_brier_score, "brier")]:
        r = paired_bootstrap_diff(y, mk, ml, metric_fn, name, n_bootstrap=1000,
                                  random_state=RANDOM_STATE)
        sig = "SIGNIFICANT" if not r.zero_in_ci else "not significant"
        print(f"{name:>9}: diff={r.point_estimate_diff:+.4f}  "
              f"95% CI=[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]  -> {sig}")

    print("\n=== Reliability: Markov ===")
    print(calibration_table(y, mk).to_string(index=False))
    print("\n=== Reliability: ML+MC ===")
    print(calibration_table(y, ml).to_string(index=False))

    results_df = pd.DataFrame({
        "match_id": ordered_match_ids,
        "Pt": ordered_pts,
        "target": targets,
        "markov_pred": markov_preds,
        "ml_pred": ml_preds,
        "ml_runtime_sec": ml_times,
    })
    # Merge explicitly on (match_id, Pt) rather than assuming positional row-order
    # alignment with eval_df — the reassembly above iterates matches in sorted order,
    # which is very likely (but not guaranteed) to coincide with eval_df's own sort order.
    # An explicit merge is correct regardless of that assumption.
    eval_df = eval_df.merge(results_df, on=["match_id", "Pt"], how="inner")
    if len(eval_df) != len(results_df):
        logger.warning("Merge row count mismatch: eval_df=%d, results=%d — some points "
                       "may be missing from the saved output.", len(eval_df), len(results_df))
    out = PROCESSED / "day11_head_to_head_v2_predictions.parquet"
    eval_df.to_parquet(out, index=False)
    print(f"\nSaved per-point predictions to {out}")

    print("\n" + prof.summary())


if __name__ == "__main__":
    main()