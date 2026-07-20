"""
export_model_comparison.py — computes LogLoss/Brier/ECE for ALL FIVE prediction
engines on the holdout set and saves the result to a JSON file the API serves
directly (see api/routers/model_comparison.py) — NOT recomputed on every HTTP
request, since a full-holdout run takes ~9 minutes (confirmed elsewhere in this
project's own logs) and involves per-point Monte Carlo simulation for ML+MC, making
it completely unsuitable for a live request/response cycle.

Reuses the SAME per-point prediction functions replay_match.py/replay_service.py
already use for all five engines (markov_p_player1, ml_p_player1,
ml_informed_markov_p_player1, ml_informed_markov_p_player1_unsmoothed, hybrid_predict)
— evaluate_full_match_calibration.py's own main() only computes THREE of the five
engines (missing ML+MC and Hybrid), so this script is NOT a thin wrapper around that
one; it independently runs the point-level loop across all five, following the same
per-point construction already validated in replay_service.py.

Usage:
    python pipelines/export_model_comparison.py            # smoke-test sample
    python pipelines/export_model_comparison.py --full      # full holdout (~9+ min,
                                                             # slower still with ML+MC's
                                                             # per-point Monte Carlo)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.ml_informed_markov import ServeReturnPosterior, build_pretrained_prior
from tennis_intel.live.markov_baseline import prob_win_match
from tennis_intel.live.hybrid_engine import hybrid_predict
from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error,
)

from replay_match import (
    markov_p_player1, ml_p_player1, ml_informed_markov_p_player1,
    ml_informed_markov_p_player1_unsmoothed, PROCESSED, POINT_FILES, ROLLOUT_MODEL_NAME,
)
from generate_publication_trajectory import compute_composite_prematch_probability

HOLDOUT_YEAR = 2022  # matches this project's own established train/test split
N_MATCHES_SAMPLE = 150
RANDOM_STATE = 42
OUTPUT_PATH = PROCESSED / "model_comparison_export.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    print("Loading trained classifier...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload[ROLLOUT_MODEL_NAME], payload["feature_cols"]

    print("Building point dataset...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["match_year"] = points["match_id"].str[:4].astype(int)
    points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]
    test_points = points[points["match_year"] >= HOLDOUT_YEAR].copy()

    match_ids = np.sort(test_points["match_id"].unique())
    if args.full:
        selected = match_ids
        print(f"Running FULL holdout set: {len(selected)} matches. This will take a while.")
    else:
        n_use = min(N_MATCHES_SAMPLE, len(match_ids))
        selected = np.random.RandomState(RANDOM_STATE).choice(match_ids, size=n_use, replace=False)
        print(f"Running smoke-test sample: {n_use} matches. Pass --full for the complete "
              f"holdout set once this looks sane.")

    eval_df = test_points[test_points["match_id"].isin(selected)].copy()
    eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
    print(f"Evaluating {len(selected)} matches, {len(eval_df)} points total")

    targets = {name: [] for name in ["markov", "ml_mc", "ml_informed_unsmoothed",
                                      "ml_informed_smoothed", "hybrid"]}
    preds = {name: [] for name in targets}

    current_match_id, posterior = None, None
    t0 = time.time()
    records = eval_df.to_dict("records")

    for idx, row in enumerate(records):
        if idx % 5000 == 0 and idx > 0:
            print(f"  {idx} / {len(records)} points ({100*idx/len(records):.1f}%, "
                  f"{time.time()-t0:.0f}s elapsed)")

        if row["match_id"] != current_match_id:
            current_match_id = row["match_id"]
            p0_a_wins = compute_composite_prematch_probability(row)
            p_a_return_seed = compute_p_a_return_seed(row, track_winner=True)
            elo_a, elo_b = row.get("elo_matches_played_pre_winner"), row.get("elo_matches_played_pre_loser")
            h2h = None
            if pd.notna(row.get("winner_h2h_wins_pre_match")) and pd.notna(row.get("loser_h2h_wins_pre_match")):
                h2h = float(row["winner_h2h_wins_pre_match"]) + float(row["loser_h2h_wins_pre_match"])
            best_of_val = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3
            p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
                p0_a_wins, p_a_return_seed, best_of_val,
                elo_matches_played_a=elo_a, elo_matches_played_b=elo_b, h2h_meetings=h2h,
            )
            posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)

        target = 1.0 if row["player1_is_winner"] else 0.0

        p_markov = markov_p_player1(row)
        p_ml_mc = ml_p_player1(row, model, feature_cols, rng_seed=idx)
        p_ml_informed, posterior = ml_informed_markov_p_player1(row, model, feature_cols, posterior)
        p_ml_informed_unsmoothed = ml_informed_markov_p_player1_unsmoothed(row, model, feature_cols)
        p_hybrid = hybrid_predict(markov_p=p_markov, ml_mc_p=p_ml_mc)

        for name, p in [("markov", p_markov), ("ml_mc", p_ml_mc),
                        ("ml_informed_unsmoothed", p_ml_informed_unsmoothed),
                        ("ml_informed_smoothed", p_ml_informed), ("hybrid", p_hybrid)]:
            targets[name].append(target)
            preds[name].append(p)

    print(f"Done in {time.time()-t0:.1f}s")

    engine_display_names = {
        "markov": "Analytical Markov",
        "ml_mc": "Machine Learning + Monte Carlo",
        "ml_informed_smoothed": "ML-Informed Markov (Smoothed)",
        "ml_informed_unsmoothed": "ML-Informed Markov (Unsmoothed)",
        "hybrid": "Hybrid Engine",
    }
    results = {}
    for name in targets:
        y = np.array(targets[name])
        p = np.clip(np.array(preds[name]), 1e-6, 1 - 1e-6)
        results[name] = {
            "display_name": engine_display_names[name],
            "n_points": len(y),
            "log_loss": round(compute_log_loss(y, p), 6),
            "brier": round(compute_brier_score(y, p), 6),
            "ece": round(expected_calibration_error(y, p), 6),
        }
        print(f"{engine_display_names[name]:<35} log_loss={results[name]['log_loss']:.4f} "
              f"brier={results[name]['brier']:.4f} ece={results[name]['ece']:.4f}")

    export = {
        "n_matches": len(selected), "n_points": len(eval_df),
        "holdout_year": HOLDOUT_YEAR, "is_full_holdout": args.full,
        "engines": results,
    }
    OUTPUT_PATH.write_text(json.dumps(export, indent=2))
    print(f"\nSaved model comparison export to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()