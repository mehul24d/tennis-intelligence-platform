"""
check_deciding_set_importance.py — the 30-second check flagged before drawing any
conclusion from deciding_set's near-zero effect on the points-remaining-binned log loss
table: does deciding_set ALSO come back near-zero in permutation importance (real
evidence the information isn't useful beyond what's already captured), or does it show
meaningful importance despite not moving the confirmatory table (evidence the classifier
found the signal but a bare binary flag was too coarse for it to help downstream)?

Reuses the exact same permutation-importance approach already proven correct in
audit_feature_leakage_sweep.py (which found server_is_winner ranking #2 overall, ~13x
above genuine pre-match features) — same sampling, same random state, same method.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_live_engines_v2 import HOLDOUT_YEAR, POINT_FILES, PROCESSED
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.feature_schema import TARGET


def main() -> None:
    print("Loading trained classifier and rebuilding the point-level dataset...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

    features_to_check = [
        "deciding_set", "points_played_so_far_in_match", "sets_played_so_far_in_match",
        "winner_bp_return_win_pct_career", "loser_bp_return_win_pct_career",
        "winner_second_serve_win_pct_career",
        "p1_games_streak", "p1_in_match_serve_rate", "p1_in_match_return_rate",
        "points_streak_x_break_point", "pressure_index_x_momentum10",
    ]
    missing = [f for f in features_to_check if f not in feature_cols]
    if missing:
        raise SystemExit(
            f"{missing} not in this classifier's feature_cols — the model needs to be "
            f"retrained (python pipelines/build_day9_point_model.py, after also running "
            f"python pipelines/build_day6_features.py if the BP features are new) before "
            f"this check means anything for those specific features."
        )

    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["match_year"] = points["match_id"].str[:4].astype(int)
    test_pts = points[points["match_year"] >= HOLDOUT_YEAR].copy()

    X_test = test_pts[feature_cols].apply(pd.to_numeric, errors="coerce").values
    y_test = test_pts[TARGET].values

    rng = np.random.RandomState(42)
    n_sample = min(10000, len(X_test))
    sample_idx = rng.choice(len(X_test), size=n_sample, replace=False)
    X_sample, y_sample = X_test[sample_idx], y_test[sample_idx]

    print(f"Running permutation importance on {n_sample:,} held-out points...")
    result = permutation_importance(
        model, X_sample, y_sample, scoring="neg_log_loss",
        n_repeats=5, random_state=42, n_jobs=1,
    )
    perm_df = pd.DataFrame({
        "feature": feature_cols,
        "perm_importance_mean": result.importances_mean,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    print("\n--- Top 15 features by permutation importance ---")
    print(perm_df.head(15).to_string(index=False))

    for feat in features_to_check:
        if feat not in feature_cols:
            print(f"\n--- {feat}: NOT in this classifier's feature_cols, skipping ---")
            continue
        row = perm_df[perm_df["feature"] == feat]
        rank = row.index[0] + 1
        importance = row["perm_importance_mean"].iloc[0]
        print(f"\n--- {feat} ---")
        print(f"Rank: {rank} of {len(perm_df)}")
        print(f"Importance: {importance:.6f}")

    print("\nInterpretation:")
    print("- If importance is near-zero (comparable to the bottom of the ranking, e.g.")
    print("  similar magnitude to genuinely weak features), that's real evidence the")
    print("  classifier found nothing useful in the flag — consistent with the")
    print("  fatigue/quality-gap explanations being about MAGNITUDE, not a yes/no switch,")
    print("  which a binary flag can't express regardless of whether the underlying")
    print("  effect is real.")
    print("- If importance is meaningfully positive and non-trivial (even if it didn't")
    print("  move the points-remaining-binned table much), that suggests the classifier")
    print("  DID find something in it, but it wasn't enough to close a gap of this size")
    print("  on its own -- worth keeping the feature regardless while pursuing a graded")
    print("  fatigue proxy alongside it, rather than concluding deciding_set alone was")
    print("  a dead end.")


if __name__ == "__main__":
    main()