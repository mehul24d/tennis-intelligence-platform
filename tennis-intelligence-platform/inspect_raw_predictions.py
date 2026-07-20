"""Pulls the raw per-point classifier predict_proba output across the same window
(points 61-145) to see whether the ML+MC volatility on the chart traces back to genuinely
volatile raw point-level predictions, or is a Monte-Carlo-rollout artifact on top of
smoother underlying predictions."""
import sys
sys.path.insert(0, "src")
import joblib
import pandas as pd
import numpy as np
from tennis_intel.live.build_point_dataset import build_point_dataset

RAW_MCP = "data/raw/tennis_MatchChartingProject"
POINT_FILES = [f"{RAW_MCP}/charting-m-points-to-2009.csv",
               f"{RAW_MCP}/charting-m-points-2010s.csv",
               f"{RAW_MCP}/charting-m-points-2020s.csv"]

frozen_join = pd.read_parquet("data/processed/joined_matches_m.parquet")
day6 = pd.read_parquet("data/processed/matches_with_day6_features.parquet")
points = build_point_dataset(POINT_FILES, frozen_join, day6)

candidates = points[points["match_id"].str.contains("2023", na=False) &
                    points["match_id"].str.contains("Wimbledon", na=False, case=False) &
                    points["match_id"].str.contains("Djokovic", na=False) &
                    points["match_id"].str.contains("Alcaraz", na=False)]
match_id = candidates["match_id"].iloc[0]
match = candidates.sort_values("Pt").reset_index(drop=True)

payload = joblib.load("data/processed/day9_point_classifiers.joblib")
model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

window = match.iloc[60:145].copy()
available_cols = [c for c in feature_cols if c in window.columns]
missing = [c for c in feature_cols if c not in window.columns]
print(f"Feature cols available: {len(available_cols)}/{len(feature_cols)}")
if missing:
    print(f"Missing (will cause NaN in the feature matrix): {missing}")

X = window[available_cols].apply(pd.to_numeric, errors="coerce").values
raw_preds = model.predict_proba(X)[:, 1]

print(f"\nRaw predict_proba (P(server wins point)) across points 61-145:")
print(f"  min={raw_preds.min():.4f}, max={raw_preds.max():.4f}, std={raw_preds.std():.4f}")
print(f"\nPoint-to-point changes (|pred[i] - pred[i-1]|):")
diffs = np.abs(np.diff(raw_preds))
print(f"  mean={diffs.mean():.4f}, max={diffs.max():.4f}")
print(f"  fraction of consecutive points with change > 0.1: {(diffs > 0.1).mean():.2%}")
print(f"  fraction of consecutive points with change > 0.2: {(diffs > 0.2).mean():.2%}")

print(f"\nFirst 20 raw predictions:")
for i in range(20):
    print(f"  Pt={window['Pt'].iloc[i]}: raw_pred={raw_preds[i]:.4f}")