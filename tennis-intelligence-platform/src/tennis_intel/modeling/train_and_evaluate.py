"""
train_and_evaluate.py — trains and temporally cross-validates the four baseline models
(Logistic Regression, Random Forest, XGBoost, CatBoost — LightGBM included if installed)
on the symmetric player_1/player_2 dataset, with calibration and MLflow tracking.

SCOPE (v1): uses only the numeric *_diff features from build_symmetric_dataset.py.
Categorical context (surface, tourney_level) is a natural v2 extension, deferred because
correct categorical encoding inside temporal CV requires per-fold-fit encoders (to avoid
leaking category statistics across folds) — real, but a distinct scope from "get baseline
models running temporally-correctly first."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from tennis_intel.evaluation.metrics import (
    compute_log_loss, compute_brier_score, bootstrap_metric, calibration_table,
)
from tennis_intel.evaluation.temporal_cv import generate_temporal_folds, TemporalFold

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("xgboost not installed — skipping XGBoost in model comparison.")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("lightgbm not installed — skipping LightGBM in model comparison.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("catboost not installed — skipping CatBoost in model comparison.")


def get_model_registry(random_state: int = 42) -> dict:
    """Returns {model_name: sklearn-compatible estimator factory}. Built as factories (not
    instances) so each fold gets a fresh, unfitted model — reusing a fitted instance across
    folds would itself be a leakage bug (the model would remember prior folds' data)."""
    registry = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
        ),
    }
    if HAS_XGBOOST:
        registry["xgboost"] = lambda: XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=random_state, n_jobs=-1,
        )
    if HAS_LIGHTGBM:
        registry["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=random_state, n_jobs=-1, verbosity=-1,
        )
    if HAS_CATBOOST:
        registry["catboost"] = lambda: CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.05,
            random_state=random_state, verbose=False,
        )
    return registry


@dataclass
class FoldResult:
    model_name: str
    test_year: int
    n_train: int
    n_test: int
    log_loss: float
    log_loss_ci: tuple[float, float]
    brier: float
    brier_ci: tuple[float, float]
    calibrated_log_loss: float
    calibrated_brier: float


@dataclass
class ModelComparisonResult:
    fold_results: list[FoldResult] = field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for r in self.fold_results:
            rows.append({
                "model": r.model_name, "test_year": r.test_year,
                "n_train": r.n_train, "n_test": r.n_test,
                "log_loss": r.log_loss, "log_loss_ci_lo": r.log_loss_ci[0], "log_loss_ci_hi": r.log_loss_ci[1],
                "brier": r.brier, "brier_ci_lo": r.brier_ci[0], "brier_ci_hi": r.brier_ci[1],
                "calibrated_log_loss": r.calibrated_log_loss, "calibrated_brier": r.calibrated_brier,
            })
        return pd.DataFrame(rows)

    def aggregate_by_model(self) -> pd.DataFrame:
        table = self.summary_table()
        return table.groupby("model").agg(
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            mean_calibrated_log_loss=("calibrated_log_loss", "mean"),
            mean_calibrated_brier=("calibrated_brier", "mean"),
            n_folds=("test_year", "count"),
        ).reset_index().sort_values("mean_log_loss")


def _select_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_diff")]


def run_model_comparison(
    symmetric_df: pd.DataFrame,
    test_years: list[int],
    date_col: str = "tourney_date",
    label_col: str = "label",
    random_state: int = 42,
) -> ModelComparisonResult:
    feature_cols = _select_feature_cols(symmetric_df)
    logger.info("Using %d numeric diff features: %s", len(feature_cols), feature_cols)

    folds = generate_temporal_folds(symmetric_df, date_col, test_years)
    registry = get_model_registry(random_state)
    result = ModelComparisonResult()

    for fold in folds:
        train_df = symmetric_df.loc[fold.train_idx]
        test_df = symmetric_df.loc[fold.test_idx]

        if train_df.empty or test_df.empty:
            logger.warning("Fold %d has empty train or test set — skipping.", fold.test_year)
            continue

        # Median-impute using ONLY the train fold's statistics — fitting on test data here
        # would itself be a (small but real) leakage bug.
        train_medians = train_df[feature_cols].median()
        X_train = train_df[feature_cols].fillna(train_medians).to_numpy()
        X_test = test_df[feature_cols].fillna(train_medians).to_numpy()
        y_train = train_df[label_col].to_numpy()
        y_test = test_df[label_col].to_numpy()

        for model_name, model_factory in registry.items():
            model = model_factory()
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            ll = compute_log_loss(y_test, y_prob)
            bs = compute_brier_score(y_test, y_prob)
            ll_ci = bootstrap_metric(y_test, y_prob, compute_log_loss, n_bootstrap=200, random_state=random_state)
            bs_ci = bootstrap_metric(y_test, y_prob, compute_brier_score, n_bootstrap=200, random_state=random_state)

            # Calibration: fit ONLY on train fold (via internal CV), never touching test data
            calibrated = CalibratedClassifierCV(model_factory(), method="isotonic", cv=3)
            calibrated.fit(X_train, y_train)
            y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
            ll_cal = compute_log_loss(y_test, y_prob_cal)
            bs_cal = compute_brier_score(y_test, y_prob_cal)

            result.fold_results.append(FoldResult(
                model_name=model_name, test_year=fold.test_year,
                n_train=len(train_df), n_test=len(test_df),
                log_loss=ll, log_loss_ci=(ll_ci.ci_lower, ll_ci.ci_upper),
                brier=bs, brier_ci=(bs_ci.ci_lower, bs_ci.ci_upper),
                calibrated_log_loss=ll_cal, calibrated_brier=bs_cal,
            ))
            logger.info("Fold %d, %s: log_loss=%.4f (calibrated=%.4f), brier=%.4f (calibrated=%.4f)",
                        fold.test_year, model_name, ll, ll_cal, bs, bs_cal)

    return result