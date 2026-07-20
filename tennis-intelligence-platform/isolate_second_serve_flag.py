"""Isolates whether is_second_serve_point alone reproduces the observed 0.56<->0.76 jump,
by holding a real row's OTHER features fixed and toggling only this one flag."""
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
match = candidates.sort_values("Pt").reset_index(drop=True)

payload = joblib.load("data/processed/day9_point_classifiers.joblib")
model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

# First, confirm: does is_second_serve_point actually DIFFER between the alternating
# points we saw (61,62 vs 63,64 vs 65...)?
window = match.iloc[60:80]
print("Actual is_second_serve_point values for points 61-80 (matching the earlier window):")
print(window[["Pt", "is_second_serve_point"]].to_string(index=False))

# Now the isolation test: take ONE real row, hold everything fixed, toggle ONLY this flag
base_row = match.iloc[65].copy()
X_base = base_row[feature_cols].apply(pd.to_numeric, errors="coerce").values.reshape(1, -1)
p_base = model.predict_proba(X_base)[0, 1]
print(f"\nBase row (Pt={base_row['Pt']}), is_second_serve_point={base_row['is_second_serve_point']}: "
      f"predict_proba={p_base:.4f}")

idx = feature_cols.index("is_second_serve_point")
X_toggled = X_base.copy()
X_toggled[0, idx] = 1.0 - X_toggled[0, idx]
p_toggled = model.predict_proba(X_toggled)[0, 1]
print(f"SAME row, is_second_serve_point TOGGLED to {X_toggled[0, idx]}: predict_proba={p_toggled:.4f}")
print(f"\nDifference from toggling ONLY this one flag: {abs(p_toggled - p_base):.4f}")
print(f"(Compare to the ~0.20 swings observed in the real point-by-point sequence)")