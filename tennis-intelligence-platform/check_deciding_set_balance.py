"""Two-line check: is the deciding-set match-level log loss gap explained simply by
deciding sets being closer to genuinely 50/50, independent of any model quality issue?"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "pipelines")
import pandas as pd
import numpy as np
from pipelines.evaluate_live_engines_v2 import tracked_player_is_winner, HOLDOUT_YEAR, POINT_FILES, PROCESSED
from tennis_intel.live.build_point_dataset import build_point_dataset

frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
points = build_point_dataset(POINT_FILES, frozen_join, day6)
points["match_year"] = points["match_id"].str[:4].astype(int)
test_points = points[points["match_year"] >= HOLDOUT_YEAR].copy()
test_points["player1_is_winner"] = (test_points["Svr"] == 1) == test_points["server_is_winner"]

def is_deciding(row):
    best_of = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3
    sets_needed = (best_of // 2) + 1
    return row["Set1"] == row["Set2"] == sets_needed - 1

test_points["is_deciding"] = test_points.apply(is_deciding, axis=1)
test_points["target"] = test_points["match_id"].map(
    lambda m: 1.0 if tracked_player_is_winner(m) else 0.0
)

for deciding, label in [(False, "non-deciding"), (True, "deciding")]:
    sub = test_points[test_points["is_deciding"] == deciding]
    balance = sub["target"].mean()
    print(f"{label:<14} n={len(sub):>7}  target balance={balance:.4f}  "
          f"(distance from 0.5: {abs(balance-0.5):.4f})")