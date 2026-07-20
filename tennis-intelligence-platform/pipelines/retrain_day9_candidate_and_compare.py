"""retrain_day9_candidate_and_compare.py — retrains day9_point_classifiers.joblib on
the corrected (literal PtWinner) point-level features, per
docs/ptwinner_convention_correction.md's scoped retraining plan.

Does NOT touch the deployed day9_point_classifiers.joblib. Saves the new candidate
model to day9_point_classifiers_RETRAIN_CANDIDATE.joblib and evaluates OLD vs NEW
side by side (point-level log loss/Brier, rolling-origin across multiple test years
via generate_temporal_folds, and SHAP feature importances) so the comparison can be
reviewed before any decision to deploy.

Usage:
    python pipelines/retrain_day9_candidate_and_compare.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import log_loss, brier_score_loss

from tennis_intel.live.feature_schema import POINT_FEATURE_COLS, TARGET

from build_day9_point_model import build_models, extract_match_year, HOLDOUT_YEAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

CANDIDATE_MODEL_PATH = PROCESSED / "day9_point_classifiers_RETRAIN_CANDIDATE.joblib"
DEPLOYED_MODEL_PATH = PROCESSED / "day9_point_classifiers.joblib"
FRESH_DATASET_PATH = PROCESSED / "day10_point_dataset_RETRAIN_CANDIDATE.parquet"

ROLLING_TEST_YEARS = [2022, 2023, 2024, 2025]


def main() -> None:
    logger.info("Loading freshly-built point dataset (corrected features)...")
    points = pd.read_parquet(FRESH_DATASET_PATH)
    points["match_year"] = points["match_id"].apply(extract_match_year)

    feature_cols = [c for c in POINT_FEATURE_COLS if c in points.columns]
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    # --- Load the DEPLOYED (old) model for comparison ---
    deployed_payload = joblib.load(DEPLOYED_MODEL_PATH)
    old_feature_cols = deployed_payload["feature_cols"]
    old_gb_model = deployed_payload["gradient_boosting"]
    assert old_feature_cols == feature_cols, (
        f"Feature column set differs between old and new — comparison would not be "
        f"apples-to-apples. old={old_feature_cols} new={feature_cols}"
    )

    # --- Single-split (same HOLDOUT_YEAR=2022 as the original script) train NEW model ---
    train_pts = points[points["match_year"] < HOLDOUT_YEAR].copy()
    test_pts = points[points["match_year"] >= HOLDOUT_YEAR].copy()
    logger.info(
        "Single-split: Train %d points (%d matches), Test %d points (%d matches)",
        len(train_pts), train_pts["match_id"].nunique(),
        len(test_pts), test_pts["match_id"].nunique(),
    )

    X_train = train_pts[feature_cols].values
    y_train = train_pts[TARGET].values
    X_test = test_pts[feature_cols].values
    y_test = test_pts[TARGET].values

    new_models = {}
    for name, model in build_models().items():
        logger.info("Training NEW %s on corrected features...", name)
        model.fit(X_train, y_train)
        new_models[name] = model

    new_gb_model = new_models["gradient_boosting"]

    # --- Save candidate (NOT the deployed path) ---
    candidate_payload = {name: model for name, model in new_models.items()}
    candidate_payload["feature_cols"] = feature_cols
    joblib.dump(candidate_payload, CANDIDATE_MODEL_PATH)
    logger.info("Saved retrain candidate to %s (deployed model untouched)", CANDIDATE_MODEL_PATH)

    # --- Single-split point-level comparison (old model on SAME test_pts, re-scored
    # with the CORRECTED features -- this is the fairest apples-to-apples test: same
    # held-out points, same feature values, only the model differs) ---
    print("\n=== Single-split (test year >= 2022) point-level comparison ===")
    print(f"{'model':<30} {'log_loss':>10} {'brier':>8}")
    old_prob = old_gb_model.predict_proba(X_test)[:, 1]
    old_ll, old_bs = log_loss(y_test, old_prob), brier_score_loss(y_test, old_prob)
    print(f"{'OLD (deployed, re-scored on corrected features)':<30} {old_ll:>10.4f} {old_bs:>8.4f}")
    new_prob = new_gb_model.predict_proba(X_test)[:, 1]
    new_ll, new_bs = log_loss(y_test, new_prob), brier_score_loss(y_test, new_prob)
    print(f"{'NEW (retrained on corrected features)':<30} {new_ll:>10.4f} {new_bs:>8.4f}")

    # --- Rolling-origin comparison across multiple years ---
    # NOTE: generate_temporal_folds (src/tennis_intel/evaluation/temporal_cv.py) expects
    # a real date column; the point-level dataset only carries match_year (an int derived
    # from the match_id prefix), so folds are built directly here using the identical
    # expanding-window logic that function implements (train = strictly before test_year,
    # test = exactly that year) -- same discipline, just inlined for this int-keyed column.
    print("\n=== Rolling-origin comparison (expanding-window folds, multiple test years) ===")
    print(f"{'test_year':>10} {'n_test':>8} {'old_ll':>8} {'new_ll':>8} {'old_brier':>10} {'new_brier':>10}")
    rolling_results = []
    for test_year in ROLLING_TEST_YEARS:
        fold_train = points[points["match_year"] < test_year]
        fold_test = points[points["match_year"] == test_year]
        if len(fold_test) == 0:
            continue
        Xf_train = fold_train[feature_cols].values
        yf_train = fold_train[TARGET].values
        Xf_test = fold_test[feature_cols].values
        yf_test = fold_test[TARGET].values

        old_fold_prob = old_gb_model.predict_proba(Xf_test)[:, 1]
        old_fold_ll = log_loss(yf_test, old_fold_prob)
        old_fold_bs = brier_score_loss(yf_test, old_fold_prob)

        fold_model = build_models()["gradient_boosting"]
        fold_model.fit(Xf_train, yf_train)
        new_fold_prob = fold_model.predict_proba(Xf_test)[:, 1]
        new_fold_ll = log_loss(yf_test, new_fold_prob)
        new_fold_bs = brier_score_loss(yf_test, new_fold_prob)

        print(f"{test_year:>10} {len(fold_test):>8} {old_fold_ll:>8.4f} {new_fold_ll:>8.4f} "
              f"{old_fold_bs:>10.4f} {new_fold_bs:>10.4f}")
        rolling_results.append({
            "test_year": test_year, "n_test": len(fold_test),
            "old_log_loss": old_fold_ll, "new_log_loss": new_fold_ll,
            "old_brier": old_fold_bs, "new_brier": new_fold_bs,
        })

    rolling_df = pd.DataFrame(rolling_results)
    rolling_df.to_csv(PROCESSED / "day9_retrain_rolling_origin_comparison.csv", index=False)
    print(f"\nSaved rolling-origin comparison to {PROCESSED / 'day9_retrain_rolling_origin_comparison.csv'}")
    print(f"\nMean across folds: old_log_loss={rolling_df['old_log_loss'].mean():.4f}  "
          f"new_log_loss={rolling_df['new_log_loss'].mean():.4f}")
    print(f"Mean across folds: old_brier={rolling_df['old_brier'].mean():.4f}  "
          f"new_brier={rolling_df['new_brier'].mean():.4f}")

    # --- Calibration: reliability by decile, on the single-split test set ---
    print("\n=== Calibration (reliability by predicted-probability decile, single-split test) ===")
    calib_rows = []
    for label, prob in [("OLD", old_prob), ("NEW", new_prob)]:
        df_c = pd.DataFrame({"prob": prob, "actual": y_test})
        df_c["decile"] = pd.qcut(df_c["prob"], 10, duplicates="drop")
        grp = df_c.groupby("decile", observed=True).agg(
            mean_predicted=("prob", "mean"), mean_actual=("actual", "mean"), n=("actual", "size")
        )
        grp["model"] = label
        calib_rows.append(grp.reset_index())
    calib_df = pd.concat(calib_rows, ignore_index=True)
    calib_df.to_csv(PROCESSED / "day9_retrain_calibration_comparison.csv", index=False)
    print(calib_df.to_string())
    print(f"\nSaved calibration comparison to {PROCESSED / 'day9_retrain_calibration_comparison.csv'}")

    # --- SHAP feature importances, old vs new, on a shared sample of the test set ---
    print("\n=== SHAP feature importance comparison (top 15, mean |SHAP value| on a 2000-row test sample) ===")
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]

    def shap_importance(pipeline_model):
        clf = pipeline_model.named_steps["clf"]
        imputer = pipeline_model.named_steps["imputer"]
        X_imputed = imputer.transform(X_sample)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_imputed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return np.abs(shap_values).mean(axis=0)

    old_shap = shap_importance(old_gb_model)
    new_shap = shap_importance(new_gb_model)

    shap_df = pd.DataFrame({
        "feature": feature_cols, "old_mean_abs_shap": old_shap, "new_mean_abs_shap": new_shap,
    })
    shap_df["old_rank"] = shap_df["old_mean_abs_shap"].rank(ascending=False).astype(int)
    shap_df["new_rank"] = shap_df["new_mean_abs_shap"].rank(ascending=False).astype(int)
    shap_df["rank_change"] = shap_df["old_rank"] - shap_df["new_rank"]
    shap_df = shap_df.sort_values("new_rank")
    shap_df.to_csv(PROCESSED / "day9_retrain_shap_comparison.csv", index=False)
    print(shap_df.head(15).to_string(index=False))
    print(f"\nSaved full SHAP comparison to {PROCESSED / 'day9_retrain_shap_comparison.csv'}")

    print("\n=== DONE. Deployed model NOT touched. Candidate saved separately for review. ===")


if __name__ == "__main__":
    main()
