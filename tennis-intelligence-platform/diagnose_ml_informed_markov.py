"""Checks whether ML-Informed Markov's dramatic improvement is genuine accuracy or
overconfident sharpness compounding through the recursion — by comparing its predictions
directly against ML+MC's on the SAME points, and checking sharpness/extremity."""
import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/ml_informed_markov_predictions.parquet")
day11 = pd.read_parquet("data/processed/day11_head_to_head_v2_predictions.parquet")

merged = df.merge(day11[["match_id", "Pt", "ml_pred", "markov_pred"]],
                  on=["match_id", "Pt"], suffixes=("", "_day11"))
print(f"Merged points: {len(merged):,} (should be close to 25,881 if match_id/Pt align)")

print(f"\nSharpness (mean |p - 0.5|):")
print(f"  Pure Markov:          {(merged['markov_pred'] - 0.5).abs().mean():.4f}")
print(f"  ML+MC (Day 11):       {(merged['ml_pred'] - 0.5).abs().mean():.4f}")
print(f"  ML-Informed Markov:   {(merged['ml_informed_pred'] - 0.5).abs().mean():.4f}")

print(f"\nFraction of predictions in [0, 0.05] or [0.95, 1.0] (extreme):")
for name, col in [("Pure Markov", "markov_pred"), ("ML+MC", "ml_pred"),
                   ("ML-Informed Markov", "ml_informed_pred")]:
    extreme = ((merged[col] < 0.05) | (merged[col] > 0.95)).mean()
    print(f"  {name}: {extreme:.1%}")

print(f"\nCorrelation between ML-Informed Markov and ML+MC predictions: "
      f"{merged['ml_informed_pred'].corr(merged['ml_pred']):.4f}")

# When ML-Informed Markov is WRONG (confidently), how wrong is it?
wrong_confident = merged[((merged["ml_informed_pred"] > 0.9) & (merged["target"] == 0)) |
                          ((merged["ml_informed_pred"] < 0.1) & (merged["target"] == 1))]
print(f"\nPoints where ML-Informed Markov was >90% confident and WRONG: {len(wrong_confident)} "
      f"({100*len(wrong_confident)/len(merged):.2f}% of all points)")