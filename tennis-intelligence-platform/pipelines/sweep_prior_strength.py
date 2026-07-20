"""
sweep_prior_strength.py — tests whether a weaker Beta-Binomial prior recovers most of the
accuracy lost to smoothing (evaluate_ml_informed_markov.py: log loss roughly doubled,
0.1815 -> 0.4178, when Bayesian + sensitivity-aware smoothing was added) while keeping
most of the calibration gain (ECE improved from 0.0917 to 0.0440) — the natural next
question raised by that result, rather than accepting either version as final.

Builds the point dataset and selects the SAME 150 matches ONCE, then re-runs the
point-by-point evaluation loop for each candidate prior_strength against that same data —
avoiding repeating the ~10s dataset-build cost per configuration.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_live_engines_v2 import (
    tracked_player_is_winner, _row_to_match_state, markov_p_winner,
    HOLDOUT_YEAR, N_MATCHES, RANDOM_STATE, POINT_FILES, PROCESSED,
)
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.ml_informed_markov import ml_informed_markov_predict, ServeReturnPosterior
from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CANDIDATE_PRIOR_STRENGTHS = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]

REFERENCE_RESULTS = {
    "Unsmoothed (no blending)": {"log_loss": 0.1815, "brier": 0.0454, "ece": 0.0917},
    "ML+MC (Day 11)": {"log_loss": 0.2652, "brier": 0.0781, "ece": 0.0468},
    "Pure Markov": {"log_loss": 0.6287, "brier": 0.1996, "ece": 0.0903},
}


def run_one_configuration(eval_df: pd.DataFrame, model, feature_cols: list,
                          prior_strength: float) -> dict:
    ml_informed_preds, targets = [], []
    posterior = None
    current_match_id = None

    for row in eval_df.to_dict("records"):
        if row["match_id"] != current_match_id:
            current_match_id = row["match_id"]
            winner_serve = row.get("winner_first_serve_win_pct_career")
            loser_serve = row.get("loser_first_serve_win_pct_career")
            winner_serve = 0.65 if winner_serve is None or pd.isna(winner_serve) else float(winner_serve)
            loser_serve = 0.65 if loser_serve is None or pd.isna(loser_serve) else float(loser_serve)
            posterior = ServeReturnPosterior.from_career_rate(
                winner_serve, 1.0 - loser_serve, prior_strength=prior_strength
            )

        track_winner = tracked_player_is_winner(row["match_id"])
        state = _row_to_match_state(row)
        p_winner, posterior = ml_informed_markov_predict(state, row, model, feature_cols, posterior)

        if track_winner:
            ml_informed_preds.append(p_winner)
            targets.append(1.0)
        else:
            ml_informed_preds.append(1.0 - p_winner)
            targets.append(0.0)

    y = np.array(targets)
    p = np.array(ml_informed_preds)
    return {
        "log_loss": compute_log_loss(y, p),
        "brier": compute_brier_score(y, p),
        "ece": expected_calibration_error(y, p),
    }


def main() -> None:
    logger.info("Loading trained classifier...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model = payload["gradient_boosting"]
    feature_cols = payload["feature_cols"]

    logger.info("Building point dataset ONCE, reused across all prior_strength values...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["match_year"] = points["match_id"].str[:4].astype(int)
    test_points = points[points["match_year"] >= HOLDOUT_YEAR].copy()
    test_points["player1_is_winner"] = (test_points["Svr"] == 1) == test_points["server_is_winner"]

    match_ids = np.sort(test_points["match_id"].unique())
    n_use = min(N_MATCHES, len(match_ids))
    selected = np.random.RandomState(RANDOM_STATE).choice(match_ids, size=n_use, replace=False)
    eval_df = test_points[test_points["match_id"].isin(selected)].copy()
    eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
    logger.info("Evaluating %d matches, %d points per configuration (SAME selection as Day 11)",
               n_use, len(eval_df))

    results = {}
    for prior_strength in CANDIDATE_PRIOR_STRENGTHS:
        t0 = time.perf_counter()
        results[f"prior_strength={prior_strength}"] = run_one_configuration(
            eval_df, model, feature_cols, prior_strength
        )
        elapsed = time.perf_counter() - t0
        logger.info("  prior_strength=%.1f done in %.1fs", prior_strength, elapsed)

    print(f"\n{'Configuration':<35} {'LogLoss':>10} {'Brier':>10} {'ECE':>10}")
    print("-" * 67)
    for name, r in REFERENCE_RESULTS.items():
        print(f"{name:<35} {r['log_loss']:>10.4f} {r['brier']:>10.4f} {r['ece']:>10.4f}")
    print("-" * 67)
    for name, r in results.items():
        print(f"{name:<35} {r['log_loss']:>10.4f} {r['brier']:>10.4f} {r['ece']:>10.4f}")

    print("\nLooking for: a prior_strength that recovers accuracy CLOSER to the unsmoothed")
    print("baseline (0.1815 log loss) while keeping ECE meaningfully better than the")
    print("unsmoothed baseline's 0.0917 — i.e. a genuine improvement on BOTH axes over at")
    print("least one of the two extremes already measured, not just a midpoint that's")
    print("worse than one extreme on accuracy and worse than the other on calibration.")


if __name__ == "__main__":
    main()