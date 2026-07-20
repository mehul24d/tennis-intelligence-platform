"""
check_mc_simulation_count_tradeoff.py — directly tests whether the AGGREGATE LogLoss/
Brier for ML+MC is stable when n_simulations is reduced, on the SAME 150-match sample
already run at n_simulations=300 (which produced log_loss=0.535854, brier=0.178754).

The hypothesis: since these are averages over ~26,000 independent points, per-point
Monte Carlo noise should largely cancel out in the aggregate metric even with far
fewer simulations per point — worth verifying DIRECTLY before reducing
export_model_comparison.py's simulation count for the full holdout, rather than
assuming this holds without checking (this project's own standing discipline).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.evaluation.metrics import compute_log_loss, compute_brier_score
from replay_match import ml_p_player1, PROCESSED, POINT_FILES, ROLLOUT_MODEL_NAME

HOLDOUT_YEAR = 2022
N_MATCHES_SAMPLE = 150
RANDOM_STATE = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-counts", type=int, nargs="+", default=[300, 100, 50, 25, 10])
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
    n_use = min(N_MATCHES_SAMPLE, len(match_ids))
    selected = np.random.RandomState(RANDOM_STATE).choice(match_ids, size=n_use, replace=False)

    eval_df = test_points[test_points["match_id"].isin(selected)].copy()
    eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
    records = eval_df.to_dict("records")
    print(f"Evaluating {n_use} matches, {len(records)} points, at each of {args.sim_counts} simulation counts\n")

    import tennis_intel.live.ml_informed_markov as mim_module

    results = {}
    for n_sim in args.sim_counts:
        # Monkey-patch N_SIMULATIONS for this run only — ml_p_player1 reads the
        # module-level constant from replay_match.py at call time via its own
        # closure, so patching replay_match's own N_SIMULATIONS is what actually
        # takes effect, not the ml_informed_markov module.
        import replay_match
        original_n_sim = replay_match.N_SIMULATIONS
        replay_match.N_SIMULATIONS = n_sim

        targets, preds = [], []
        t0 = time.time()
        for idx, row in enumerate(records):
            target = 1.0 if row["player1_is_winner"] else 0.0
            p = ml_p_player1(row, model, feature_cols, rng_seed=idx)
            targets.append(target)
            preds.append(p)
        elapsed = time.time() - t0

        replay_match.N_SIMULATIONS = original_n_sim  # restore

        y = np.array(targets)
        p = np.clip(np.array(preds), 1e-6, 1 - 1e-6)
        ll, br = compute_log_loss(y, p), compute_brier_score(y, p)
        results[n_sim] = {"log_loss": ll, "brier": br, "elapsed": elapsed}
        print(f"n_simulations={n_sim:<5} log_loss={ll:.6f}  brier={br:.6f}  "
              f"({elapsed:.1f}s for {len(records)} points)")

    print("\n=== Stability check: difference from n_simulations=300 (if included) ===")
    if 300 in results:
        baseline = results[300]
        for n_sim, r in results.items():
            if n_sim == 300:
                continue
            ll_diff = r["log_loss"] - baseline["log_loss"]
            br_diff = r["brier"] - baseline["brier"]
            speedup = baseline["elapsed"] / r["elapsed"] if r["elapsed"] > 0 else float("inf")
            print(f"n_simulations={n_sim:<5} log_loss_diff={ll_diff:+.6f}  "
                  f"brier_diff={br_diff:+.6f}  speedup={speedup:.1f}x")

    print("\nInterpretation:")
    print("- If log_loss_diff/brier_diff stay small (e.g. < 0.005) even at much lower")
    print("  n_simulations, the aggregate metric is genuinely stable and a reduced")
    print("  simulation count is safe to use for the full-holdout export -- the ~350K+")
    print("  point aggregate averages out per-point MC noise even with fewer trials")
    print("  per point.")
    print("- Pick the LOWEST n_simulations where the diff stays small, then multiply")
    print("  the speedup shown here by roughly (5981/150) to estimate the full-holdout")
    print("  runtime at that setting.")


if __name__ == "__main__":
    main()