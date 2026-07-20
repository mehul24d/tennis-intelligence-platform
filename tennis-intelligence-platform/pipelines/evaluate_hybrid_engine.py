"""
*** DEPRECATED ENGINE UNDER TEST — kept for historical reproducibility of the finding ***

Running this script reproduces the measurement that led to the hybrid engine's
deprecation: the fixed-weight blend underperforms BOTH pure Markov and pure ML+MC
individually on every metric (see hybrid_engine.py's module docstring for the full
explanation of why, and ml_informed_markov.py for the architecture that replaced this
approach). This script is retained so that finding remains independently reproducible,
not as an invitation to keep iterating on the hybrid design — per external audit
(Architecture Review, finding A), any further work on live win-probability blending
should build on ml_informed_markov.py's single-point-probability-into-one-recursion
architecture, not this one.

evaluate_hybrid_engine.py — evaluates the fixed-weight hybrid (Markov + ML+MC) against
each engine individually, reusing Day 11's ALREADY-SAVED per-point predictions
(day11_head_to_head_v2_predictions.parquet) rather than re-running the expensive rollout.

LEAKAGE NOTE: this is safe to evaluate directly on Day 11's existing test predictions
because hybrid_engine.py's weighting function is FIXED and hand-specified (motivated by
Day 11's OWN prior reliability findings, not fit/tuned on this specific data) — there is no
new parameter being chosen based on this evaluation set. A genuinely LEARNED meta-model
blend would require its own fresh held-out split and must not reuse this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tennis_intel.live.hybrid_engine import hybrid_predict
from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, expected_calibration_error, paired_bootstrap_diff,
)

PROCESSED = "data/processed"
PRED_FILE = f"{PROCESSED}/day11_head_to_head_v2_predictions.parquet"


def main() -> None:
    df = pd.read_parquet(PRED_FILE)
    print(f"Total points: {len(df):,}\n")

    df["hybrid_pred"] = [
        hybrid_predict(mk, ml) for mk, ml in zip(df["markov_pred"], df["ml_pred"])
    ]

    y = df["target"].values
    print(f"{'Engine':<12} {'LogLoss':>10} {'Brier':>10} {'ECE':>10}")
    for name, col in [("Markov", "markov_pred"), ("ML+MC", "ml_pred"), ("Hybrid", "hybrid_pred")]:
        p = df[col].values
        ll = compute_log_loss(y, p)
        brier = compute_brier_score(y, p)
        ece = expected_calibration_error(y, p)
        print(f"{name:<12} {ll:>10.4f} {brier:>10.4f} {ece:>10.4f}")

    print("\n=== Paired Bootstrap: Hybrid vs. each individual engine ===")
    for name, col in [("Markov", "markov_pred"), ("ML+MC", "ml_pred")]:
        result = paired_bootstrap_diff(y, df["hybrid_pred"].values, df[col].values,
                                       metric_fn=compute_log_loss, metric_name="log_loss")
        direction = "Hybrid better" if result.point_estimate_diff < 0 else "Hybrid worse"
        sig = "SIGNIFICANT" if not result.zero_in_ci else "not significant"
        print(f"Hybrid vs {name}: diff={result.point_estimate_diff:+.4f} "
              f"95% CI=[{result.ci_lower:+.4f}, {result.ci_upper:+.4f}] -> {sig} ({direction})")

    # Weight distribution: how often is the hybrid effectively "using" each engine?
    from tennis_intel.live.hybrid_engine import hybrid_weight_markov
    weights = df["ml_pred"].apply(hybrid_weight_markov)
    print(f"\nWeight given to Markov across all points: mean={weights.mean():.3f}, "
          f"median={weights.median():.3f}")
    print(f"Fraction of points where Markov gets >50% weight: {(weights > 0.5).mean():.1%}")
    print("(High Markov weight = ML+MC was making an extreme prediction on that point;")
    print(" low Markov weight = ML+MC was near a toss-up, so it's trusted more there.)")


if __name__ == "__main__":
    main()