"""
tune_day9_hyperparameters.py — hyperparameter search for the Day 9 GradientBoosting point
classifier, deferred at Milestone 5, Day 6, and Day 9 (see each stage's freeze doc) and
picked up now that the feature set is richer and validated (26+ features vs. the original
handful) and worth actually tuning against.

LEAKAGE DISCIPLINE: the existing train (<2022) / test (>=2022) split from
build_day9_point_model.py must NOT be used for tuning — repeatedly checking performance on
the test set to pick hyperparameters would leak test information into model selection,
silently invalidating the final reported test metric (a form of researcher-degrees-of-
freedom overfitting, distinct from but just as real as the leakage bugs found elsewhere in
this project). Instead, a THIRD split is carved out of the training period specifically for
tuning:

  TRAIN (tune):      matches before 2020
  VALIDATION (tune):  matches 2020-2021  <- hyperparameters are chosen using ONLY this
  TEST (untouched):   matches 2022+       <- never touched until the final, one-time check

Once a winning configuration is chosen here, build_day9_point_model.py should be updated
to use it and re-run normally (retraining on the FULL pre-2022 period, evaluating ONCE on
the untouched 2022+ test set) — this script does not modify that file itself.

This is a random search (not exhaustive grid search) over a deliberately bounded parameter
space — full grid search would be prohibitively slow given each GBM fit already takes
several minutes on the real dataset size (~350k+ points).
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from tennis_intel.live.build_point_dataset import build_point_dataset

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

# Feature schema centralized (external audit, 2026-07, Code Review finding #6): this file
# previously maintained its OWN duplicated copy of POINT_FEATURE_COLS, reasoning at the
# time that duplication was SAFER than a shared import ("so this script can be run/modified
# independently without risk of accidentally changing the frozen training pipeline's own
# feature list via a shared-mutable-import mistake"). That reasoning was disproven by the
# actual outcome: this copy still had "server_is_winner" — the confirmed leakage feature
# already removed everywhere else after the Phase 4 audit finding — meaning if this script
# had been run again, it would have silently retrained with the leaky feature
# reintroduced. A shared, read-only imported list carries no such risk (nothing here
# mutates it in place); the duplication was the actual vulnerability, not the fix for one.
from tennis_intel.live.feature_schema import POINT_FEATURE_COLS, TARGET

TUNE_TRAIN_CUTOFF = 2020   # train < this
TUNE_VAL_CUTOFF = 2022     # val is [TUNE_TRAIN_CUTOFF, TUNE_VAL_CUTOFF), test is >= this

# Deliberately bounded search space — chosen to bracket the CURRENT fixed configuration
# (n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8) rather than search
# blindly, so the search is centered on values already known to be reasonable.
PARAM_SPACE = {
    "n_estimators": [100, 150, 200, 300, 400],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_samples_leaf": [1, 5, 10, 20, 50],
}
N_RANDOM_SEARCH_ITERATIONS = 25


def extract_match_year(match_id: str) -> int:
    try:
        return int(str(match_id)[:4])
    except (ValueError, TypeError):
        return 0


def sample_params(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in PARAM_SPACE.items()}


def main() -> None:
    logger.info("Loading frozen join and Day 6 features...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")

    logger.info("Building point-level dataset...")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["match_year"] = points["match_id"].apply(extract_match_year)

    train_pts = points[points["match_year"] < TUNE_TRAIN_CUTOFF].copy()
    val_pts = points[(points["match_year"] >= TUNE_TRAIN_CUTOFF) &
                      (points["match_year"] < TUNE_VAL_CUTOFF)].copy()
    logger.info("Tuning split — Train: %d points (<%d), Validation: %d points (%d-%d). "
                "Test set (>=%d) is NOT touched by this script.",
                len(train_pts), TUNE_TRAIN_CUTOFF, len(val_pts),
                TUNE_TRAIN_CUTOFF, TUNE_VAL_CUTOFF - 1, TUNE_VAL_CUTOFF)

    if len(train_pts) == 0 or len(val_pts) == 0:
        raise SystemExit("Empty train or validation split — check TUNE_TRAIN_CUTOFF/"
                          "TUNE_VAL_CUTOFF against the actual match_year distribution.")

    X_train = train_pts[POINT_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y_train = train_pts[TARGET].values
    X_val = val_pts[POINT_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y_val = val_pts[TARGET].values

    rng = random.Random(42)
    results = []

    logger.info("Starting random search: %d configurations", N_RANDOM_SEARCH_ITERATIONS)
    for i in range(N_RANDOM_SEARCH_ITERATIONS):
        params = sample_params(rng)
        t0 = time.perf_counter()

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(random_state=42, **params)),
        ])
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)[:, 1]
        val_ll = log_loss(y_val, val_pred)
        val_brier = brier_score_loss(y_val, val_pred)
        elapsed = time.perf_counter() - t0

        results.append({**params, "val_log_loss": val_ll, "val_brier": val_brier,
                        "fit_seconds": elapsed})
        logger.info("[%d/%d] %s -> val_log_loss=%.4f, val_brier=%.4f (%.1fs)",
                    i + 1, N_RANDOM_SEARCH_ITERATIONS, params, val_ll, val_brier, elapsed)

    results_df = pd.DataFrame(results).sort_values("val_log_loss")
    print("\n=== Top 10 configurations by validation log loss ===")
    print(results_df.head(10).to_string(index=False))

    out_path = PROCESSED / "day9_hyperparameter_search_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    best = results_df.iloc[0]
    print(f"\n=== Best configuration ===")
    print(f"n_estimators={int(best['n_estimators'])}, max_depth={int(best['max_depth'])}, "
          f"learning_rate={best['learning_rate']}, subsample={best['subsample']}, "
          f"min_samples_leaf={int(best['min_samples_leaf'])}")
    print(f"Validation log_loss={best['val_log_loss']:.4f}, brier={best['val_brier']:.4f}")
    print(f"\nCurrent fixed config (n_estimators=200, max_depth=4, learning_rate=0.05, "
          f"subsample=0.8, min_samples_leaf=default) for comparison:")
    current = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, random_state=42)),
    ])
    current.fit(X_train, y_train)
    current_val_pred = current.predict_proba(X_val)[:, 1]
    print(f"Validation log_loss={log_loss(y_val, current_val_pred):.4f}, "
          f"brier={brier_score_loss(y_val, current_val_pred):.4f}")

    print("\nNEXT STEP: if the best configuration meaningfully beats the current one on "
          "this validation set, update build_day9_point_model.py's GradientBoostingClassifier "
          "call with these hyperparameters, then re-run it normally — that will retrain on "
          "the FULL pre-2022 period and report final metrics on the untouched 2022+ test "
          "set exactly once, preserving the leakage discipline this script was built to protect.")


if __name__ == "__main__":
    main()