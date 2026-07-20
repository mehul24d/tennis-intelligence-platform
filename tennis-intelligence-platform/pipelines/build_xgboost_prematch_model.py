"""
build_xgboost_prematch_model.py — trains an XGBoost (gradient-boosted trees) classifier to
predict MATCH-LEVEL pre-match win probability, using the exact same pre-match feature set
already in use elsewhere in this project (Elo, surface Elo, Elo confidence, career
serve/return stats, surface-conditioned career stats, H2H, tournament-specific H2H and
form) — reusing the ALREADY-PROVEN, ALREADY-TESTED symmetric dataset construction in
build_symmetric_dataset.py (diff features, hash-determined player_1/player_2 assignment,
verified swap-symmetric) rather than reinventing it.

WHY MATCH-LEVEL, NOT POINT-LEVEL: pre-match prediction is fundamentally a match-outcome
question ("given these two players' profiles, who wins the match"), not a point-outcome
question. Training on points (like the existing Day 9 classifier) would let the same match
appear hundreds of times with a near-constant label, which is appropriate for THAT model's
purpose (predicting the next point) but not for this one's (predicting the match before it
starts) — this model is trained one row per match instead.

TEMPORAL SPLIT: identical train(<2022)/test(>=2022) boundary as build_day9_point_model.py,
for direct comparability and to avoid inventing a third, inconsistent convention.

REQUIRES: pip install xgboost (not a dependency of the rest of this project; installed
separately since it's the specific library requested for this new engine).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from tennis_intel.modeling.build_symmetric_dataset import build_symmetric_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

HOLDOUT_YEAR = 2022  # identical boundary to build_day9_point_model.py

# Expanded feature-pair list covering every pre-match feature currently in use across the
# project (Elo redesign + surface-conditioned serve stats + H2H/tournament features),
# passed into the EXISTING, unmodified build_symmetric_dataset() via its feature_pairs
# parameter — no changes to that frozen, tested module were needed.
PREMATCH_FEATURE_PAIRS: list[tuple[str, str, str]] = [
    ("elo_pre_match_winner", "elo_pre_match_loser", "elo"),
    ("elo_surface_pre_match_winner", "elo_surface_pre_match_loser", "elo_surface"),
    ("elo_matches_played_pre_winner", "elo_matches_played_pre_loser", "elo_matches_played"),
    ("winner_win_pct_last10", "loser_win_pct_last10", "win_pct_last10"),
    ("winner_surface_win_pct_last10", "loser_surface_win_pct_last10", "surface_win_pct_last10"),
    ("winner_first_serve_in_pct_career", "loser_first_serve_in_pct_career", "first_serve_in_pct_career"),
    ("winner_first_serve_win_pct_career", "loser_first_serve_win_pct_career", "first_serve_win_pct_career"),
    ("winner_first_serve_win_pct_surface_career", "loser_first_serve_win_pct_surface_career",
     "first_serve_win_pct_surface_career"),
    ("winner_bp_saved_pct_career", "loser_bp_saved_pct_career", "bp_saved_pct_career"),
    ("winner_h2h_wins_pre_match", "loser_h2h_wins_pre_match", "h2h_wins"),
    ("winner_tourney_h2h_wins_pre_match", "loser_tourney_h2h_wins_pre_match", "tourney_h2h_wins"),
    ("winner_tourney_win_pct_last10", "loser_tourney_win_pct_last10", "tourney_win_pct_last10"),
]

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, random_state=42, eval_metric="logloss",
)


def extract_match_year(match_id_or_tourney_id: str) -> int:
    s = str(match_id_or_tourney_id)
    for token in s.split("-"):
        if len(token) == 4 and token.isdigit():
            return int(token)
    try:
        return int(s[:4])
    except (ValueError, TypeError):
        return 0


def main(model_factory=None) -> None:
    """
    model_factory: optional callable() -> sklearn-compatible classifier, for testing this
    pipeline's surrounding logic with a stand-in classifier when xgboost itself is not the
    thing under test. Defaults to a real XGBClassifier with XGB_PARAMS.
    """
    if model_factory is None:
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "This module requires xgboost, which is not installed in this "
                "environment. Install it with:\n\n    pip install xgboost\n\n"
                "then re-run this script. (The rest of this project uses scikit-learn's "
                "GradientBoostingClassifier, which does not require this extra "
                "dependency — xgboost is used here specifically because it was "
                "requested for this new engine.)"
            ) from e
        model_factory = lambda: XGBClassifier(**XGB_PARAMS)

    logger.info("Loading Day 6 features (match-level, includes all pre-match columns)...")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    logger.info("Loaded %d matches", len(day6))

    day6["match_year"] = day6["tourney_id"].apply(extract_match_year)
    if (day6["match_year"] == 0).all():
        # Fall back to tourney_date if tourney_id doesn't embed a year
        day6["match_year"] = pd.to_datetime(day6["tourney_date"]).dt.year

    logger.info("Building symmetric (player_1/player_2, diff-featured) dataset via the "
                "existing, proven build_symmetric_dataset()...")
    symmetric = build_symmetric_dataset(day6, feature_pairs=PREMATCH_FEATURE_PAIRS)
    symmetric["match_year"] = day6["match_year"].values

    feature_cols = [f"{name}_diff" for _, _, name in PREMATCH_FEATURE_PAIRS]
    feature_cols = [c for c in feature_cols if c in symmetric.columns]
    logger.info("Using %d diff features: %s", len(feature_cols), feature_cols)

    train = symmetric[symmetric["match_year"] < HOLDOUT_YEAR].copy()
    test = symmetric[symmetric["match_year"] >= HOLDOUT_YEAR].copy()
    logger.info("Train: %d matches (<%d), Test: %d matches (>=%d)",
                len(train), HOLDOUT_YEAR, len(test), HOLDOUT_YEAR)

    if len(train) == 0 or len(test) == 0:
        raise SystemExit("Empty train or test split — check match_year extraction against "
                          "the real tourney_id/tourney_date format.")

    X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train["label"].values
    X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_test = test["label"].values

    logger.info("Training pre-match model...")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", model_factory()),
    ])
    model.fit(X_train, y_train)

    test_pred = model.predict_proba(X_test)[:, 1]
    ll = log_loss(y_test, test_pred)
    brier = brier_score_loss(y_test, test_pred)

    naive_pred = np.full(len(y_test), 0.5)
    naive_ll = log_loss(y_test, naive_pred)
    naive_brier = brier_score_loss(y_test, naive_pred)

    print("\n=== Pre-match XGBoost model: held-out test set (>=2022) ===")
    print(f"{'Model':<20} {'Log Loss':>10} {'Brier':>10}")
    print(f"{'Naive (p=0.5)':<20} {naive_ll:>10.4f} {naive_brier:>10.4f}")
    print(f"{'XGBoost pre-match':<20} {ll:>10.4f} {brier:>10.4f}")
    print(f"\n(For reference: Day 9's frozen point-level Markov match-level Brier, "
          f"computed after the p_return fix, was 0.2460 — a different methodology and "
          f"evaluation set, not directly comparable row-for-row, but useful context for "
          f"whether this pre-match model is finding real signal beyond chance.)")

    import joblib
    out_path = PROCESSED / "xgboost_prematch_model.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols,
                "feature_pairs": PREMATCH_FEATURE_PAIRS}, out_path)
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    main()