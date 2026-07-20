"""
diagnose_feature_importance.py — checks whether the trained Day 9 classifiers are actually
USING the new surface Elo / matches-played-confidence columns, or effectively ignoring
them despite being available. Two independent measures, since either alone can mislead:

  1. GBM's built-in feature_importances_ (impurity-based) — fast, but known to be biased
     toward high-cardinality/continuous features regardless of true predictive value.
  2. Permutation importance on the HELD-OUT TEST SET (both models) — the harder, more
     convincing test: shuffle one feature's values and measure how much real held-out
     performance degrades. This directly answers "does the model actually rely on this,"
     not just "did the training algorithm happen to split on it somewhere."

Reuses the exact same train/test split and dataset-building logic as
build_day9_point_model.py, so results are directly comparable to that pipeline's own
numbers — no separate data pipeline to maintain or accidentally diverge from.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from tennis_intel.live.build_point_dataset import build_point_dataset

PROCESSED = "data/processed"
RAW_MCP = "data/raw/tennis_MatchChartingProject"
POINT_FILES = [
    f"{RAW_MCP}/charting-m-points-to-2009.csv",
    f"{RAW_MCP}/charting-m-points-2010s.csv",
    f"{RAW_MCP}/charting-m-points-2020s.csv",
]
HOLDOUT_YEAR = 2022
TARGET = "server_wins_point"

NEW_COLS = [
    "elo_surface_pre_match_winner", "elo_surface_pre_match_loser",
    "elo_matches_played_pre_winner", "elo_matches_played_pre_loser",
]
LIKELY_REDUNDANT_COMPARISON = [
    "winner_surface_win_pct_last10", "loser_surface_win_pct_last10",
    "elo_pre_match_winner", "elo_pre_match_loser",
]


def extract_match_year(match_id: str) -> int:
    try:
        return int(str(match_id)[:4])
    except (ValueError, TypeError):
        return 0


def main() -> None:
    print("Loading trained classifiers...")
    payload = joblib.load(f"{PROCESSED}/day9_point_classifiers.joblib")
    feature_cols = payload["feature_cols"]
    print(f"Model was trained on {len(feature_cols)} features.\n")

    print("Rebuilding the exact same point-level dataset used for training/eval...")
    frozen_join = pd.read_parquet(f"{PROCESSED}/joined_matches_m.parquet")
    day6 = pd.read_parquet(f"{PROCESSED}/matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["match_year"] = points["match_id"].apply(extract_match_year)
    test_pts = points[points["match_year"] >= HOLDOUT_YEAR].copy()

    missing_new = [c for c in NEW_COLS if c not in test_pts.columns]
    if missing_new:
        raise SystemExit(f"New columns not found in rebuilt dataset: {missing_new} — "
                          f"did build_point_dataset.py's PREMATCH_FEATURE_COLS get updated?")

    X_test = test_pts[feature_cols].values
    y_test = test_pts[TARGET].values
    print(f"Test set: {len(X_test):,} points\n")

    for model_name in ["gradient_boosting", "logistic_regression"]:
        model = payload[model_name]
        print("=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        baseline_pred = model.predict_proba(X_test)[:, 1]
        baseline_ll = log_loss(y_test, baseline_pred)
        print(f"Baseline log loss on test set: {baseline_ll:.4f}\n")

        # --- 1. GBM's built-in impurity-based importance (GBM only) ---
        if model_name == "gradient_boosting":
            clf = model.named_steps["clf"]
            importances = clf.feature_importances_
            imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False)
            print("--- GBM impurity-based feature importance (top 15) ---")
            print(imp_df.head(15).to_string(index=False))
            print("\n--- Where do the NEW Elo columns rank? ---")
            for c in NEW_COLS:
                rank = imp_df.reset_index(drop=True)
                pos = rank[rank["feature"] == c].index[0] + 1
                val = rank[rank["feature"] == c]["importance"].iloc[0]
                print(f"  {c}: rank {pos}/{len(feature_cols)}, importance={val:.5f}")
            print()

        # --- 2. Permutation importance on the real held-out test set (both models) ---
        print("--- Permutation importance on held-out test set (subsample for speed) ---")
        # Full test set permutation importance can be slow; subsample for a fast, still
        # statistically meaningful check (10k points, 5 repeats).
        rng = np.random.RandomState(42)
        n_sample = min(10000, len(X_test))
        sample_idx = rng.choice(len(X_test), size=n_sample, replace=False)
        X_sample, y_sample = X_test[sample_idx], y_test[sample_idx]

        result = permutation_importance(
            model, X_sample, y_sample, scoring="neg_log_loss",
            n_repeats=5, random_state=42, n_jobs=1,
        )
        perm_df = pd.DataFrame({
            "feature": feature_cols,
            "perm_importance_mean": result.importances_mean,
            "perm_importance_std": result.importances_std,
        }).sort_values("perm_importance_mean", ascending=False)

        print(perm_df.head(15).to_string(index=False))
        print("\n--- Focused comparison: new Elo columns vs. their likely-redundant counterparts ---")
        focus = perm_df[perm_df["feature"].isin(NEW_COLS + LIKELY_REDUNDANT_COMPARISON)]
        print(focus.to_string(index=False))
        print("\n(perm_importance_mean = how much log loss WORSENS when this feature's values")
        print(" are randomly shuffled — near-zero or negative means the model isn't relying")
        print(" on it meaningfully, even if it was included in training.)")
        print()


if __name__ == "__main__":
    main()