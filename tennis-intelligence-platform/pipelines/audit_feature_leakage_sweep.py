"""
audit_feature_leakage_sweep.py — Step 1 of the leakage remediation protocol: a systematic
sweep for the SAME failure pattern that made server_is_winner leakage (a feature that
cannot be computed without already knowing the full match outcome), across every feature
in POINT_FEATURE_COLS, before touching the model at all.

Two independent checks, cross-referenced:
  1. MANUAL CLASSIFICATION (this file's own documentation, verified against the actual
     source code for each feature family — see the classification table below and its
     accompanying comments for exactly which file/lines were checked for each category).
  2. EMPIRICAL CROSS-CHECK: reruns permutation importance (reusing the logic already in
     diagnose_feature_importance.py) and flags any feature classified SAFE that shows
     implausibly high importance relative to what a genuine pre-match/rolling-in-match
     feature should be able to achieve — the same signal that caught server_is_winner.

This script does not modify the model or retrain anything — it is purely diagnostic,
per the prescribed protocol's Step 1 (audit before retrain).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_day9_point_model import POINT_FEATURE_COLS, TARGET
from tennis_intel.live.build_point_dataset import build_point_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW_MCP = PROJECT_ROOT / "data" / "raw" / "tennis_MatchChartingProject"
POINT_FILES = [
    RAW_MCP / "charting-m-points-to-2009.csv",
    RAW_MCP / "charting-m-points-2010s.csv",
    RAW_MCP / "charting-m-points-2020s.csv",
]
HOLDOUT_YEAR = 2022

# --- Manual classification, verified against actual source code for each family ---
# Test applied to every feature: "can this be computed at the moment this point is about
# to be played, using ONLY information that existed before this point's outcome is known —
# with NO dependence, direct or indirect, on the match's eventual final result?"
FEATURE_CLASSIFICATION = {
    # In-match score-state flags: computed purely from the CURRENT score, before this
    # point's outcome is known. Verified: these derive from Set1/Set2/Gm1/Gm2/point score
    # at the moment of the point, never from PtWinner or any post-point information.
    "is_tiebreak_game": ("SAFE", "current score state only"),
    "is_break_point": ("SAFE", "current score state only"),
    "is_set_point": ("SAFE", "current score state only"),
    "is_match_point": ("SAFE", "current score state only"),
    "is_second_serve_point": ("SAFE", "serve number is known before the point is played, "
                                       "not derived from its outcome"),

    # Momentum: VERIFIED directly in point_level_features.py's compute_in_match_momentum —
    # shift(1) is applied BEFORE the rolling window, so the window for point i covers only
    # points i-1, i-2, ..., strictly excluding the current point's own outcome. No
    # off-by-one leak found.
    "p1_momentum_last10": ("SAFE", "verified shift(1)-before-rolling in point_level_features.py"),
    "p2_momentum_last10": ("SAFE", "verified shift(1)-before-rolling in point_level_features.py"),
    "p1_momentum_last20": ("SAFE", "verified shift(1)-before-rolling in point_level_features.py"),
    "p2_momentum_last20": ("SAFE", "verified shift(1)-before-rolling in point_level_features.py"),

    # Elo: genuine pre-match facts, computed from matches strictly before this one. Already
    # independently audited in Phase 4 of the external audit (chronological pre/post
    # extraction, no leakage found).
    "elo_pre_match_winner": ("SAFE", "genuine pre-match fact, audited in Phase 4"),
    "elo_pre_match_loser": ("SAFE", "genuine pre-match fact, audited in Phase 4"),
    "elo_surface_pre_match_winner": ("SAFE", "same chronological construction as overall Elo"),
    "elo_surface_pre_match_loser": ("SAFE", "same chronological construction as overall Elo"),
    "elo_matches_played_pre_winner": ("SAFE", "count of matches strictly before this one"),
    "elo_matches_played_pre_loser": ("SAFE", "count of matches strictly before this one"),

    # Rolling form (last-10 matches, career aggregates): VERIFIED directly in
    # serve_return_features.py's compute_rolling_serve_return_features — shift(1) applied
    # BEFORE expanding()/rolling(), so "career" and "last N matches" both strictly exclude
    # the current match's own outcome.
    "winner_win_pct_last10": ("SAFE", "verified shift(1)-before-rolling in feature_engineering_day5.py"),
    "loser_win_pct_last10": ("SAFE", "verified shift(1)-before-rolling in feature_engineering_day5.py"),
    "winner_surface_win_pct_last10": ("SAFE", "same shift-then-roll pattern, surface-conditioned"),
    "loser_surface_win_pct_last10": ("SAFE", "same shift-then-roll pattern, surface-conditioned"),
    "winner_first_serve_in_pct_career": ("SAFE", "verified shift(1)-before-expanding in serve_return_features.py"),
    "loser_first_serve_in_pct_career": ("SAFE", "verified shift(1)-before-expanding in serve_return_features.py"),
    "winner_first_serve_win_pct_career": ("SAFE", "verified shift(1)-before-expanding in serve_return_features.py"),
    "loser_first_serve_win_pct_career": ("SAFE", "verified shift(1)-before-expanding in serve_return_features.py"),
    "winner_bp_saved_pct_career": ("SAFE", "verified shift(1)-before-expanding in serve_return_features.py"),
    "loser_bp_saved_pct_career": ("SAFE", "verified shift(1)-before-expanding in serve_return_features.py"),
    "winner_first_serve_win_pct_surface_career": ("SAFE", "same shift-then-expand pattern, surface-conditioned"),
    "loser_first_serve_win_pct_surface_career": ("SAFE", "same shift-then-expand pattern, surface-conditioned"),

    # H2H / tournament features: explicitly "_pre_match" in name, already independently
    # audited in Phase 4 ("no evidence match result leaks into its own pre-match H2H
    # features").
    "winner_h2h_wins_pre_match": ("SAFE", "audited in Phase 4, pre-match by construction"),
    "loser_h2h_wins_pre_match": ("SAFE", "audited in Phase 4, pre-match by construction"),
    "winner_tourney_h2h_wins_pre_match": ("SAFE", "same pre-match construction pattern"),
    "loser_tourney_h2h_wins_pre_match": ("SAFE", "same pre-match construction pattern"),
    "winner_tourney_win_pct_last10": ("SAFE", "same shift-then-roll pattern, tournament-conditioned"),
    "loser_tourney_win_pct_last10": ("SAFE", "same shift-then-roll pattern, tournament-conditioned"),

    # THE CONFIRMED LEAK: cannot be computed without already knowing which player wins the
    # ENTIRE match — "winner" here refers to the match's final, eventual outcome, not any
    # pre-point-observable quantity. Fails the test explicitly.
    "server_is_winner": ("UNSAFE — CONFIRMED LEAK", "requires knowing the match's final "
                          "outcome; cross-validated against permutation importance ranking "
                          "it #2 overall, far above genuine pre-match features"),
}


def main() -> None:
    print("=== Manual classification of every feature in POINT_FEATURE_COLS ===\n")
    missing_classification = [c for c in POINT_FEATURE_COLS if c not in FEATURE_CLASSIFICATION]
    if missing_classification:
        print(f"WARNING: {len(missing_classification)} feature(s) in POINT_FEATURE_COLS have "
              f"NO manual classification recorded — this audit is INCOMPLETE until these "
              f"are explicitly reasoned through: {missing_classification}\n")

    unsafe = [c for c, (verdict, _) in FEATURE_CLASSIFICATION.items() if "UNSAFE" in verdict]
    print(f"{len(FEATURE_CLASSIFICATION)} features classified, {len(unsafe)} flagged UNSAFE:\n")
    for feat, (verdict, reason) in FEATURE_CLASSIFICATION.items():
        marker = "  [UNSAFE]" if "UNSAFE" in verdict else "  [safe] "
        print(f"{marker} {feat:<50} {reason}")

    print("\n\n=== Empirical cross-check: fresh permutation importance ===")
    print("Loading trained classifier and rebuilding the point-level dataset...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

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

    print("\n--- Cross-check: any feature classified SAFE ranking suspiciously high? ---")
    top10_features = set(perm_df.head(10)["feature"])
    safe_features_in_top10 = [
        f for f in top10_features
        if f in FEATURE_CLASSIFICATION and "UNSAFE" not in FEATURE_CLASSIFICATION[f][0]
    ]
    if safe_features_in_top10:
        print(f"Features classified SAFE but appearing in the top 10 by importance "
              f"(worth a second look, though high importance alone isn't proof of leakage "
              f"for genuinely predictive pre-match signals like Elo):")
        for f in safe_features_in_top10:
            rank = perm_df[perm_df["feature"] == f].index[0] + 1
            val = perm_df[perm_df["feature"] == f]["perm_importance_mean"].iloc[0]
            print(f"  {f}: rank {rank}, importance={val:.5f}")
    else:
        print("None — no SAFE-classified feature appears in the top 10, consistent with "
              "the classification above (only the already-confirmed server_is_winner leak "
              "should show implausibly high importance for a supposedly-safe feature).")

    unsafe_ranks = perm_df[perm_df["feature"].isin(unsafe)]
    print(f"\n--- Confirming the known leak's rank for reference ---")
    print(unsafe_ranks.to_string(index=False))


if __name__ == "__main__":
    main()