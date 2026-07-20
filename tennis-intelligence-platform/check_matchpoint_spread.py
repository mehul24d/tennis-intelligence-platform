"""Checks the DISTRIBUTION (not just the mean) of Markov predictions at real match-point
rows, directly from the saved parquet -- resolves whether the earlier flat mean_p (0.536)
reflects genuinely sharp-but-symmetric predictions, or genuinely muted ones."""
import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/day11_head_to_head_v2_predictions.parquet")
mp = df[df["is_match_point"] == True]

print(f"n = {len(mp)}")
print(f"\nDistribution of markov_pred at real match-point rows:")
print(mp["markov_pred"].describe())

print(f"\nSharpness (mean |p - 0.5|): {(mp['markov_pred'] - 0.5).abs().mean():.4f}")
print(f"Fraction with markov_pred > 0.9 or < 0.1 (genuinely sharp): "
      f"{((mp['markov_pred'] > 0.9) | (mp['markov_pred'] < 0.1)).mean():.3f}")
print(f"Fraction with markov_pred between 0.4 and 0.6 (genuinely muted): "
      f"{((mp['markov_pred'] > 0.4) & (mp['markov_pred'] < 0.6)).mean():.3f}")

# Accuracy check: when Markov predicts >0.9 at match point, does target=1 that often?
sharp_high = mp[mp["markov_pred"] > 0.9]
sharp_low = mp[mp["markov_pred"] < 0.1]
print(f"\nWhen markov_pred > 0.9 (n={len(sharp_high)}): actual target mean = "
      f"{sharp_high['target'].mean():.3f} (should be close to 1.0 if well-calibrated)")
print(f"When markov_pred < 0.1 (n={len(sharp_low)}): actual target mean = "
      f"{sharp_low['target'].mean():.3f} (should be close to 0.0 if well-calibrated)")