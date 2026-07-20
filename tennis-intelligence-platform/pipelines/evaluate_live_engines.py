"""
evaluate_live_engines.py — Day 10: the scientific evaluation. Runs BOTH the Day 8 analytical
Markov baseline and the Day 9 ML + Monte Carlo engine forward from EVERY point of a sample
of held-out (2022+) charted matches, and compares them head-to-head on:

  - Log loss, Brier score (overall probabilistic accuracy)
  - Expected Calibration Error (ECE) and reliability diagrams (is the model trustworthy?)
  - Sharpness (does confidence appropriately increase as the match progresses?)
  - Runtime per prediction (practical live-inference usability)
  - Full win-probability trajectories saved for a few illustrative matches

This is the result that turns "two approaches were built" into "two approaches were
rigorously compared" — see live_win_probability_extension_analysis.md and the Day 8/9
freeze docs for the approaches being compared here.

PERFORMANCE: the naive way to do this (call the classifier once per simulated point, once
per simulation, at every real point of every match) is computationally intractable at this
scale. Uses batch_simulate_with_classifier, which issues ONE classifier call per simulation
TICK across all active simulations simultaneously — a ~100x+ reduction in classifier calls,
validated empirically in monte_carlo_engine.py's development. If runtime is still excessive
on your machine, reduce N_MATCHES or N_SIMULATIONS below — the harness degrades gracefully
to a smaller, faster sample rather than needing a different design (per the user's own
fallback guidance).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.live_win_probability import MatchState, prob_a_wins_match_from_state
from tennis_intel.live.match_state_conversion import row_to_match_state
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.monte_carlo_engine import batch_simulate_with_classifier
from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error,
    calibration_table, sharpness,
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
N_MATCHES = 150          # per the user's guidance: 100-200 matches, every point
N_SIMULATIONS = 150      # per-point MC sample size for the ML engine
MAX_POINTS_PER_MATCH = 400  # safety cap; real matches rarely exceed ~250 points

# Which trained model to use for the Monte Carlo rollout. Logistic Regression is chosen
# deliberately, not as a downgrade: it is ~10x faster per predict_proba call than Gradient
# Boosting (benchmarked directly — 0.06ms vs 0.63ms per call on comparable data), and Day 9
# showed the two differ by only ~0.001 log-loss on this exact task. Since the MC rollout
# calls the classifier at every simulated tick, this speed difference compounds into the
# difference between a ~70-minute and a ~15-minute evaluation run — a well-justified
# tradeoff, not a silent accuracy compromise.
ROLLOUT_MODEL_NAME = "logistic_regression"

MOMENTUM_COLS = ["p1_momentum_last10", "p2_momentum_last10",
                  "p1_momentum_last20", "p2_momentum_last20"]
SITUATIONAL_COLS = ["is_tiebreak_game", "is_break_point", "is_set_point",
                     "is_match_point", "is_second_serve_point"]


def load_model():
    payload = joblib.load(PROCESSED / "day9_point_classifiers.joblib")
    feature_cols = payload["feature_cols"]
    model = payload[ROLLOUT_MODEL_NAME]
    return model, feature_cols


def _row_to_match_state(row: pd.Series) -> MatchState:
    """
    Thin alias to the single canonical implementation (see match_state_conversion.py).

    BUG FIX (found during centralization, external audit 2026-07, Code Review finding
    #5): this file's own local copy read row["p1_points"]/row["p2_points"] unconditionally,
    with no branch for tiebreak points — the same tb_points bug found and fixed in
    evaluate_live_engines_v2.py's copy, which this pre-v2 file never received. Since
    p1_points/p2_points are NaN during tiebreaks (point_level_features.py stores the real
    count in tb_p1_points/tb_p2_points instead), every tiebreak point evaluated by this
    script was silently treated as "the tiebreak just started, 0-0" regardless of the real
    score. Centralizing onto the shared, already-fixed implementation resolves this as a
    direct consequence, rather than requiring a second, separate bug-fix pass on this file.
    """
    return row_to_match_state(row)


def markov_prediction(row: pd.Series) -> float:
    """Markov baseline's live P(winner wins) from this point's state, using the winner's
    career serve rate and the OPPONENT's (loser's) real serve rate to derive p_return.

    BUG FIX: p_return must be 1 - the opponent's actual serve-win rate, not the winner's
    own generic return statistic — see the full explanation in evaluate_live_engines_v2.py's
    markov_p_winner, which had the identical bug (found via a real match's implausible
    0.995 pre-match probability)."""
    state = _row_to_match_state(row)

    # BUG FIX (external review, 2026-07): see return_seed.py's module docstring and
    # evaluate_live_engines_v2.py's markov_p_winner for the full derivation.
    p_a_serve = row.get("winner_combined_serve_win_pct_career")
    if p_a_serve is None or pd.isna(p_a_serve):
        p_a_serve = row.get("winner_first_serve_win_pct_career", 0.65)  # known-inferior fallback
    p_a_serve = 0.65 if pd.isna(p_a_serve) else float(p_a_serve)
    p_a_return = compute_p_a_return_seed(row, track_winner=True)

    return prob_a_wins_match_from_state(state, p_a_serve, p_a_return)


def ml_prediction(row: pd.Series, model, feature_cols: list, rng_seed: int) -> tuple[float, float]:
    """
    ML + batched Monte Carlo P(winner wins) from this point's state.

    DOCUMENTED SIMPLIFICATION: in-match situational flags (break/set/match point) and
    momentum are held at their value AT THE STARTING POINT throughout each simulated
    continuation, rather than being re-derived from the simulated point sequence at every
    simulated step. Re-deriving true rolling momentum inside the simulation would require
    tracking full point history per simulation path, which is a real additional engineering
    step beyond this evaluation's scope — flagged explicitly, not silently assumed away.
    This means the ML engine's forward simulation is somewhat less state-aware than its
    training conditions; if the head-to-head result still favors ML+MC despite this
    handicap, that is a stronger, not weaker, result.
    """
    import random as _random
    state = _row_to_match_state(row)

    static_values = {c: row.get(c, np.nan) for c in feature_cols}

    def feature_matrix_fn(states):
        rows = []
        for s in states:
            f = dict(static_values)
            f["is_tiebreak_game"] = s["is_tiebreak"]
            # server_is_winner must reflect the CURRENT simulated server, not the starting one
            if "server_is_winner" in f:
                f["server_is_winner"] = s["server_is_a"] if row["player1_is_winner"] else (not s["server_is_a"])
            rows.append([f.get(c, np.nan) for c in feature_cols])
        return np.array(rows, dtype=float)

    def predict_fn(feature_matrix):
        return model.predict_proba(feature_matrix)[:, 1]

    t0 = time.time()
    p = batch_simulate_with_classifier(
        (state.a_sets, state.b_sets, state.a_games, state.b_games,
         state.a_points, state.b_points, state.server_is_a, state.is_tiebreak),
        feature_matrix_fn, predict_fn, best_of=state.best_of,
        n_simulations=N_SIMULATIONS, rng=_random.Random(rng_seed),
        max_points=MAX_POINTS_PER_MATCH,
    )
    elapsed = time.time() - t0
    return p, elapsed


def main() -> None:
    logger.info("Loading trained model and building point dataset...")
    model, feature_cols = load_model()

    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)

    points["match_year"] = points["match_id"].str[:4].astype(int)
    test_points = points[points["match_year"] >= HOLDOUT_YEAR].copy()
    test_points["player1_is_winner"] = (test_points["Svr"] == 1) == test_points["server_is_winner"]

    match_ids = test_points["match_id"].unique()
    n_available = len(match_ids)
    n_use = min(N_MATCHES, n_available)
    logger.info("Evaluating %d of %d available held-out matches, every point", n_use, n_available)
    selected_matches = np.random.RandomState(42).choice(match_ids, size=n_use, replace=False)

    eval_df = test_points[test_points["match_id"].isin(selected_matches)].copy()
    eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
    logger.info("Total points to evaluate: %d", len(eval_df))

    markov_preds, ml_preds, ml_times = [], [], []
    t_start = time.time()

    # Convert to plain dicts up front — repeatedly calling .iloc[i] on a DataFrame inside a
    # tight loop creates a new pandas Series object on every call, which is real, avoidable
    # overhead at 25,000+ iterations. A plain dict has none of that construction cost.
    records = eval_df.to_dict("records")

    for i, row_dict in enumerate(records):
        markov_p = markov_prediction(row_dict)
        ml_p, ml_t = ml_prediction(row_dict, model, feature_cols, rng_seed=i)

        markov_preds.append(markov_p)
        ml_preds.append(ml_p)
        ml_times.append(ml_t)

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(eval_df) - i - 1) / rate
            logger.info("Processed %d / %d points (%.1f pts/sec, ETA %.0fs)",
                       i + 1, len(eval_df), rate, eta)

    eval_df["markov_pred"] = markov_preds
    eval_df["ml_pred"] = ml_preds
    eval_df["ml_runtime_sec"] = ml_times
    y_true = np.ones(len(eval_df))

    print("\n=== Head-to-Head: Markov Baseline vs ML + Monte Carlo ===")
    print(f"Matches evaluated: {n_use}, total points: {len(eval_df):,}\n")

    for name, preds in [("Markov", eval_df["markov_pred"]), ("ML+MC", eval_df["ml_pred"])]:
        preds_clipped = preds.clip(1e-6, 1 - 1e-6)
        ll = compute_log_loss(y_true, preds_clipped)
        bs = compute_brier_score(y_true, preds)
        ece = expected_calibration_error(y_true, preds)
        sh = sharpness(preds)
        print(f"{name:10s}  log_loss={ll:.4f}  brier={bs:.4f}  ECE={ece:.4f}  sharpness={sh:.4f}")

    print(f"\nML+MC mean runtime per prediction: {np.mean(ml_times)*1000:.1f}ms "
          f"(n_simulations={N_SIMULATIONS})")

    print("\n=== Reliability Table: Markov ===")
    print(calibration_table(y_true, eval_df["markov_pred"].values).to_string(index=False))
    print("\n=== Reliability Table: ML+MC ===")
    print(calibration_table(y_true, eval_df["ml_pred"].values).to_string(index=False))

    eval_df.to_parquet(PROCESSED / "day10_head_to_head_predictions.parquet", index=False)
    print(f"\nSaved full per-point predictions to "
          f"{PROCESSED / 'day10_head_to_head_predictions.parquet'}")


if __name__ == "__main__":
    main()