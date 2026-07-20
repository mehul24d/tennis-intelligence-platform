"""
train_baseline_models.py — pipeline entrypoint for the model comparison milestone.

Reads the FROZEN Day 5 output (matches_with_day5_features.parquet), builds the symmetric
player_1/player_2 modeling dataset, runs temporal cross-validated comparison across
Logistic Regression / Random Forest / XGBoost / LightGBM / CatBoost, logs everything to
MLflow, and produces a SHAP summary for the best-performing tree model.

Usage (from project root, with .venv activated, MLflow tracking to local ./mlruns):
    python pipelines/train_baseline_models.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from tennis_intel.modeling.build_symmetric_dataset import build_symmetric_dataset
from tennis_intel.modeling.train_and_evaluate import (
    run_model_comparison, get_model_registry, _select_feature_cols,
)
from tennis_intel.evaluation.metrics import calibration_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DAY5_PATH = PROCESSED_DIR / "matches_with_day5_features.parquet"
DOCS_DIR = PROJECT_ROOT / "docs"

# Broad walk-forward evaluation across the years with the richest data coverage. Adjust once
# you've seen how much training data exists per fold in the diagnostics output.
TEST_YEARS = list(range(2018, 2026))


def main() -> None:
    if not DAY5_PATH.exists():
        raise FileNotFoundError(f"{DAY5_PATH} not found — run pipelines/build_day5_features.py first.")

    matches = pd.read_parquet(DAY5_PATH)
    logger.info("Loaded %d matches", len(matches))

    symmetric_df = build_symmetric_dataset(matches)
    logger.info("Built symmetric dataset: %d rows, label balance=%.3f",
                len(symmetric_df), symmetric_df["label"].mean())

    mlflow.set_experiment("tennis-baseline-model-comparison")

    with mlflow.start_run(run_name="milestone5_baseline_comparison"):
        mlflow.log_param("test_years", TEST_YEARS)
        mlflow.log_param("n_matches", len(matches))
        mlflow.log_param("feature_cols", _select_feature_cols(symmetric_df))

        result = run_model_comparison(symmetric_df, test_years=TEST_YEARS)

        summary = result.summary_table()
        aggregate = result.aggregate_by_model()

        print("\n=== Per-fold results ===")
        print(summary.to_string(index=False))
        print("\n=== Aggregate by model (sorted by mean log loss, lower is better) ===")
        print(aggregate.to_string(index=False))

        summary_path = PROCESSED_DIR / "model_comparison_per_fold.csv"
        aggregate_path = PROCESSED_DIR / "model_comparison_aggregate.csv"
        summary.to_csv(summary_path, index=False)
        aggregate.to_csv(aggregate_path, index=False)
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(aggregate_path))

        for _, row in aggregate.iterrows():
            mlflow.log_metric(f"{row['model']}_mean_log_loss", row["mean_log_loss"])
            mlflow.log_metric(f"{row['model']}_mean_brier", row["mean_brier"])
            mlflow.log_metric(f"{row['model']}_mean_calibrated_log_loss", row["mean_calibrated_log_loss"])

        best_model_name = aggregate.iloc[0]["model"]
        print(f"\nBest model by mean log loss: {best_model_name}")
        mlflow.log_param("best_model", best_model_name)

        # --- SHAP analysis on the best tree-based model, trained on the full dataset's
        # most recent fold's training set (for a representative, reasonably current model) ---
        _run_shap_analysis(symmetric_df, best_model_name)


def _run_shap_analysis(symmetric_df: pd.DataFrame, model_name: str) -> None:
    if model_name == "logistic_regression":
        logger.info("Best model is logistic regression — SHAP is most informative for tree "
                    "models; skipping SHAP plot (coefficients are already directly "
                    "interpretable for linear models, see model.coef_).")
        return

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("shap or matplotlib not installed — skipping SHAP analysis.")
        return

    feature_cols = _select_feature_cols(symmetric_df)
    registry = get_model_registry()
    if model_name not in registry:
        logger.warning("Best model '%s' not available for SHAP (library not installed?) — skipping.", model_name)
        return

    train_medians = symmetric_df[feature_cols].median()
    X = symmetric_df[feature_cols].fillna(train_medians)
    y = symmetric_df["label"]

    model = registry[model_name]()
    model.fit(X, y)

    sample = X.sample(min(2000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, sample, show=False)
    output_path = DOCS_DIR / f"shap_summary_{model_name}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    mlflow.log_artifact(str(output_path))
    print(f"\nWrote SHAP summary plot to {output_path}")


if __name__ == "__main__":
    main()