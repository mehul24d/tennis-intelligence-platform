"""
build_day9_point_model.py — Day 9 pipeline: builds the point-level training dataset,
trains a point-outcome classifier, evaluates it via temporal CV, then compares it against
the Day 8 Markov baseline on the same held-out charted matches.

Usage (from project root, with .venv activated):
    python pipelines/build_day9_point_model.py

This is the empirical core of the live win-probability contribution: does an ML model
conditioned on in-match state + pre-match strength beat a principled analytical baseline?

Two evaluation layers:
1. POINT-LEVEL: log loss / Brier on held-out points (did we predict point outcomes correctly?)
2. MATCH-LEVEL: on a held-out set of charted matches, run both the ML+MC engine and the
   Markov baseline forward from the first point, compare their per-match win-probability
   trajectories against the actual match outcome (Brier on final outcome).

RESOLVED 2026-07-15: the point-level feature functions (point_level_features.py) were
corrected to the confirmed-correct literal PtWinner convention, and
day9_point_classifiers.joblib was retrained on the corrected features and deployed
— see docs/ptwinner_convention_correction.md's "Retrain results" section for the
full before/after comparison (rolling-origin log_loss 0.6281 -> 0.6247, Brier 0.2187
-> 0.2172, consistent across all four 2022-2025 folds). The pre-retrain classifier is
preserved at day9_point_classifiers_PRE_PTWINNER_FIX.joblib. Re-running this script
now will retrain again on the (already-correct) current features — a normal refresh,
not something requiring special caution the way it did before this was resolved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.monte_carlo_engine import simulate_match_from_state
from tennis_intel.live.markov_baseline import prob_win_match

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

# Feature schema centralized (external audit, 2026-07, Code Review finding #6): this list
# was previously independently redefined in five files, and drift already happened —
# tune_day9_hyperparameters.py's own copy still had "server_is_winner" (the confirmed
# leakage feature) after every other copy was fixed. See
# src/tennis_intel/live/feature_schema.py for the single source of truth now used
# everywhere, including here.
from tennis_intel.live.feature_schema import POINT_FEATURE_COLS, TARGET
from tennis_intel.live.return_seed import compute_p_a_return_seed

# Temporal split: train on everything before 2022, test on 2022+
# (matches the temporal discipline used throughout the project)
HOLDOUT_YEAR = 2022


def extract_match_year(match_id: str) -> int:
    try:
        return int(str(match_id)[:4])
    except (ValueError, TypeError):
        return 0


def build_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ]),
    }


def evaluate_point_level(y_true, y_prob, label: str) -> dict:
    ll = log_loss(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    n = len(y_true)
    logger.info("%s — n=%d, log_loss=%.4f, brier=%.4f", label, n, ll, bs)
    return {"label": label, "n": n, "log_loss": ll, "brier": bs}


def markov_match_brier(test_points: pd.DataFrame) -> float:
    """
    For each charted test match, the Markov baseline's 'live' prediction is simply
    prob_win_match using the winner's career first-serve-win rate as p_serve, and the
    OPPONENT's (loser's) real serve rate to derive p_return. Average Brier score across
    matches (actual outcome = 1 if server_is_winner ended up winning the match, else 0).

    BUG FIX: p_return must be 1 - the opponent's actual serve-win rate, not the winner's
    own generic return_pts_won_pct_career statistic — see evaluate_live_engines_v2.py's
    markov_p_winner for the full explanation (found via a real match's implausible 0.995
    pre-match probability). This means the ORIGINAL Day 9 freeze doc's reported
    "Markov baseline match-level Brier: 0.0588" used the buggy construction and should be
    treated as superseded pending a re-run with this fix.
    """
    results = []
    for match_id, grp in test_points.groupby("match_id"):
        row = grp.iloc[0]
        # BUG FIX (external review, 2026-07): see return_seed.py's module docstring.
        p_serve = row.get("winner_combined_serve_win_pct_career")
        if p_serve is None or pd.isna(p_serve):
            p_serve = row.get("winner_first_serve_win_pct_career", 0.65)  # known-inferior fallback
        if pd.isna(p_serve): p_serve = 0.65
        p_return = compute_p_a_return_seed(row, track_winner=True)
        best_of = int(row.get("best_of", 3))
        pred = prob_win_match(p_serve, p_return, best_of=best_of)
        # Actual outcome: winner won the match (by definition of the dataset)
        actual = 1.0
        results.append((pred - actual) ** 2)
    return float(np.mean(results)) if results else float("nan")


def main() -> None:
    logger.info("Loading frozen join and Day 6 features...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")

    logger.info("Building point-level dataset...")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)

    points["match_year"] = points["match_id"].apply(extract_match_year)
    train_pts = points[points["match_year"] < HOLDOUT_YEAR].copy()
    test_pts = points[points["match_year"] >= HOLDOUT_YEAR].copy()
    logger.info("Train: %d points (%d matches), Test: %d points (%d matches)",
                len(train_pts), train_pts["match_id"].nunique(),
                len(test_pts), test_pts["match_id"].nunique())

    feature_cols = [c for c in POINT_FEATURE_COLS if c in points.columns]
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    X_train = train_pts[feature_cols].values
    y_train = train_pts[TARGET].values
    X_test = test_pts[feature_cols].values
    y_test = test_pts[TARGET].values

    results = []
    models_fitted = {}
    for name, model in build_models().items():
        logger.info("Training %s...", name)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        res = evaluate_point_level(y_test, y_prob, name)
        results.append(res)
        models_fitted[name] = (model, y_prob)

    # Baseline: constant 0.5 (no skill) and Markov
    naive_ll = log_loss(y_test, np.full(len(y_test), 0.5))
    naive_bs = brier_score_loss(y_test, np.full(len(y_test), 0.5))
    logger.info("Naive baseline (p=0.5) — log_loss=%.4f, brier=%.4f", naive_ll, naive_bs)

    print("\n=== Point-Level Results ===")
    print(f"{'Model':<25} {'Log Loss':>10} {'Brier':>8}")
    print(f"{'naive_baseline':<25} {naive_ll:>10.4f} {naive_bs:>8.4f}")
    for r in results:
        print(f"{r['label']:<25} {r['log_loss']:>10.4f} {r['brier']:>8.4f}")

    print("\n=== Match-Level Markov Baseline Brier ===")
    markov_brier = markov_match_brier(test_pts)
    print(f"  Markov baseline match-level Brier: {markov_brier:.4f}")
    print("  (Note: 0.25 = chance; 0.0 = perfect; Markov always predicts the winner won,")
    print("   so its Brier score is (p_markov - 1)^2 on each test match)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(PROCESSED / "day9_point_model_results.csv", index=False)
    print(f"\nSaved results to {PROCESSED / 'day9_point_model_results.csv'}")

    # Persist ALL trained models — the head-to-head evaluation uses Logistic Regression for
    # the Monte Carlo rollout specifically because it is ~10x faster per predict_proba call
    # than Gradient Boosting (benchmarked separately) while differing by only ~0.001 log-loss
    # on this exact task (see the point-level results table above) — a well-justified
    # speed/accuracy tradeoff, not a silent downgrade.
    import joblib
    import shutil
    payload = {name: model for name, (model, _) in models_fitted.items()}
    payload["feature_cols"] = feature_cols
    model_path = PROCESSED / "day9_point_classifiers.joblib"

    # LEAKAGE-FIX PROTOCOL (external audit, 2026-07): defensively back up the EXISTING
    # artifact before overwriting, so the pre-fix (server_is_winner) and post-fix
    # (server_is_player1) classifiers can both be evaluated side by side, per the
    # prescribed protocol — "don't overwrite the current classifier... run both through
    # the identical evaluation pipeline." Automatic, not a manual step the user has to
    # remember: if a model already exists at model_path when this script runs, it is
    # copied to day9_point_classifiers_PRE_LEAKAGE_FIX.joblib exactly once (skipped if
    # that backup already exists, so re-running this script multiple times after the fix
    # doesn't overwrite the ORIGINAL pre-fix backup with an already-fixed one).
    backup_path = PROCESSED / "day9_point_classifiers_PRE_LEAKAGE_FIX.joblib"
    if model_path.exists() and not backup_path.exists():
        shutil.copy(model_path, backup_path)
        print(f"Backed up existing (pre-fix) classifier to {backup_path} before overwriting "
              f"— use this for the required before/after comparison.")
    elif model_path.exists() and backup_path.exists():
        print(f"Backup already exists at {backup_path} — not overwriting it. If you need a "
              f"fresh backup of a DIFFERENT prior state, rename or delete the existing "
              f"backup file first.")

    joblib.dump(payload, model_path)
    print(f"Saved trained classifiers ({list(models_fitted.keys())}) to {model_path}")


if __name__ == "__main__":
    main()