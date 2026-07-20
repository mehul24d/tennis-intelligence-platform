"""Diagnoses compute_ml_pre_match_probability's 0.870 estimate for the Sinner-Alcaraz
2025 Roland Garros final: (1) confirms whether real pre-match features are loading
correctly, ruling that out as the cause, then (2) tests the rollout's own stability by
re-running it with multiple random seeds — if the ROLLOUT MECHANISM itself is unstable/
noisy even with identical, correct inputs, that's direct evidence the problem is the
Monte Carlo rollout's suitability as a pre-match estimator, not a data-loading bug."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "pipelines")
import random
import pandas as pd
import numpy as np
import joblib
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.monte_carlo_engine import batch_simulate_dynamic
from pipelines.generate_publication_trajectory import PREMATCH_FEATURE_NAMES

RAW_MCP = "data/raw/tennis_MatchChartingProject"
POINT_FILES = [f"{RAW_MCP}/charting-m-points-to-2009.csv",
               f"{RAW_MCP}/charting-m-points-2010s.csv",
               f"{RAW_MCP}/charting-m-points-2020s.csv"]
MATCH_ID = "20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz"

frozen_join = pd.read_parquet("data/processed/joined_matches_m.parquet")
day6 = pd.read_parquet("data/processed/matches_with_day6_features.parquet")
points = build_point_dataset(POINT_FILES, frozen_join, day6)
points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]
row = points[points["match_id"] == MATCH_ID].iloc[0].to_dict()

print("=" * 70)
print("STEP 1: Are the real pre-match features actually loading correctly?")
print("=" * 70)
for name in ["elo_pre_match_winner", "elo_pre_match_loser",
             "elo_surface_pre_match_winner", "elo_surface_pre_match_loser",
             "winner_h2h_wins_pre_match", "loser_h2h_wins_pre_match",
             "winner_tourney_h2h_wins_pre_match", "loser_tourney_h2h_wins_pre_match",
             "winner_tourney_win_pct_last10", "loser_tourney_win_pct_last10"]:
    print(f"  {name:45s} = {row.get(name)}")
print("\n(winner_* = Alcaraz's real stats, loser_* = Sinner's real stats, per this")
print(" project's established convention)")

payload = joblib.load("data/processed/day9_point_classifiers.joblib")
model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

def predict_fn(fm):
    return model.predict_proba(fm)[:, 1]

p1_is_winner = bool(row["player1_is_winner"])
static_features = {}
for name in PREMATCH_FEATURE_NAMES:
    if name in feature_cols:
        val = row.get(name)
        static_features[name] = float(val) if pd.notna(val) else np.nan
first_server_is_a = (row.get("Svr", 1) == 1) == p1_is_winner
if "server_is_winner" in feature_cols:
    first_server_is_p1 = (row.get("Svr", 1) == 1)
    static_features["server_is_winner"] = first_server_is_p1 if p1_is_winner else not first_server_is_p1
best_of = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3

print("\n" + "=" * 70)
print("STEP 2: Is the rollout itself STABLE across different random seeds?")
print("(same inputs, same features, only the RNG seed changes)")
print("=" * 70)
results = []
for seed in range(10):
    p_a_wins = batch_simulate_dynamic(
        (0, 0, 0, 0, 0, 0, first_server_is_a, False),
        static_features, feature_cols, predict_fn, best_of=best_of,
        player1_is_winner=p1_is_winner,
        seed_momentum={"p1_momentum_last10": 0.5, "p1_momentum_last20": 0.5},
        n_simulations=200, rng=random.Random(seed),
    )
    results.append(p_a_wins)
    print(f"  seed={seed}: p_a_wins (Alcaraz) = {p_a_wins:.4f}")

results = np.array(results)
print(f"\nMean: {results.mean():.4f}, Std: {results.std():.4f}, "
      f"Range: [{results.min():.4f}, {results.max():.4f}]")