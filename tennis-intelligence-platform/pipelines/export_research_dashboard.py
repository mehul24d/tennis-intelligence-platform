"""
export_research_dashboard.py — computes the Research Dashboard data (reliability
diagram / calibration curve, bootstrap confidence intervals for LogLoss and Brier,
and per-engine prediction distributions) and saves to JSON, served by the API rather
than recomputed live — same reasoning as export_model_comparison.py.

REUSES the SAME per-point five-engine computation as export_model_comparison.py, and
metrics.py's ALREADY-BUILT calibration_table/bootstrap_metric/expected_calibration_
error/sharpness functions directly — no new metric implementations here.

Usage:
    python pipelines/export_research_dashboard.py            # smoke-test sample
    python pipelines/export_research_dashboard.py --full      # full holdout
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
from tennis_intel.live.hybrid_engine import hybrid_predict
from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error,
    calibration_table, bootstrap_metric, sharpness,
)

from replay_match import (
    markov_p_player1, ml_p_player1, ml_informed_markov_p_player1,
    ml_informed_markov_p_player1_unsmoothed, PROCESSED, POINT_FILES, ROLLOUT_MODEL_NAME,
)
from generate_publication_trajectory import compute_composite_prematch_probability

HOLDOUT_YEAR = 2022
N_MATCHES_SAMPLE = 150
RANDOM_STATE = 42
OUTPUT_PATH = PROCESSED / "research_dashboard_export.json"
N_CALIBRATION_BINS = 10
N_BOOTSTRAP = 1000

ENGINE_DISPLAY_NAMES = {
    "markov": "Analytical Markov",
    "ml_mc": "Machine Learning + Monte Carlo",
    "ml_informed_unsmoothed": "ML-Informed Markov (Unsmoothed)",
    "ml_informed_smoothed": "ML-Informed Markov (Smoothed)",
    "hybrid": "Hybrid Engine",
}


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
        print(f"Running smoke-test sample: {n_use} matches. Pass --full once this looks sane.")

    eval_df = test_points[test_points["match_id"].isin(selected)].copy()
    eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
    print(f"Evaluating {len(selected)} matches, {len(eval_df)} points total")

    targets = {name: [] for name in ENGINE_DISPLAY_NAMES}
    preds = {name: [] for name in ENGINE_DISPLAY_NAMES}

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

    results = {}
    for name in ENGINE_DISPLAY_NAMES:
        y = np.array(targets[name])
        p = np.clip(np.array(preds[name]), 1e-6, 1 - 1e-6)

        cal_table = calibration_table(y, p, n_bins=N_CALIBRATION_BINS)
        reliability_points = [
            {
                "bin_index": int(row["bucket"]), "n": int(row["n"]),
                "mean_predicted": round(float(row["mean_predicted"]), 6),
                "observed_win_rate": round(float(row["observed_win_rate"]), 6),
                "calibration_gap": round(float(row["calibration_gap"]), 6),
            }
            for _, row in cal_table.iterrows()
        ]

        print(f"Computing bootstrap CIs for {ENGINE_DISPLAY_NAMES[name]} "
              f"({N_BOOTSTRAP} resamples)...")
        ll_boot = bootstrap_metric(y, p, compute_log_loss, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE)
        brier_boot = bootstrap_metric(y, p, compute_brier_score, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE)

        hist_counts, hist_edges = np.histogram(p, bins=20, range=(0, 1))

        results[name] = {
            "display_name": ENGINE_DISPLAY_NAMES[name],
            "n_points": len(y),
            "log_loss": {
                "point_estimate": round(ll_boot.point_estimate, 6),
                "ci_lower": round(ll_boot.ci_lower, 6), "ci_upper": round(ll_boot.ci_upper, 6),
            },
            "brier": {
                "point_estimate": round(brier_boot.point_estimate, 6),
                "ci_lower": round(brier_boot.ci_lower, 6), "ci_upper": round(brier_boot.ci_upper, 6),
            },
            "ece": round(expected_calibration_error(y, p, n_bins=N_CALIBRATION_BINS), 6),
            "sharpness": round(sharpness(p), 6),
            "reliability_diagram": reliability_points,
            "prediction_histogram": {
                "bin_edges": [round(float(e), 3) for e in hist_edges],
                "counts": [int(c) for c in hist_counts],
            },
        }
        print(f"  {ENGINE_DISPLAY_NAMES[name]:<35} log_loss={results[name]['log_loss']['point_estimate']:.4f} "
              f"[{results[name]['log_loss']['ci_lower']:.4f}, {results[name]['log_loss']['ci_upper']:.4f}]")

    export = {
        "n_matches": len(selected), "n_points": len(eval_df),
        "holdout_year": HOLDOUT_YEAR, "is_full_holdout": args.full,
        "n_calibration_bins": N_CALIBRATION_BINS, "n_bootstrap": N_BOOTSTRAP,
        "engines": results,
    }
    OUTPUT_PATH.write_text(json.dumps(export, indent=2))
    print(f"\nSaved research dashboard export to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()