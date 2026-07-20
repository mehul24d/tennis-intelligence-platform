"""
diagnose_markov_by_score_state.py — the decisive experiment: buckets every evaluated point
by how decisive its score state is (early / mid-match / serving-for-set / set-point /
match-point), then computes log loss, Brier, and mean predicted probability PER BUCKET for
both engines.

This distinguishes two hypotheses about Markov's near-chance-level aggregate performance
(Day 11): (A) the recursion is correct and simply inherits weak pre-match parameters, in
which case predictions should still correctly sharpen toward 0/1 at match points even if
early-match predictions are mediocre; or (B) predictions never become appropriately
extreme even at genuine match points, which would point to a remaining implementation
issue in the recursion or state construction, not just weak inputs.

Uses the ALREADY-VALIDATED is_break_point/is_set_point/is_match_point flags from Day 7
rather than approximating "decisiveness" with a new heuristic.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

PROCESSED = "data/processed"
PRED_FILE = f"{PROCESSED}/day11_head_to_head_v2_predictions.parquet"


def log_loss_safe(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier_safe(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def bucket_row(row) -> str:
    if bool(row.get("is_match_point", False)):
        return "5_match_point"
    if bool(row.get("is_set_point", False)):
        return "4_set_point"
    games_played = int(row.get("Gm1", 0)) + int(row.get("Gm2", 0))
    max_games_gap = abs(int(row.get("Gm1", 0)) - int(row.get("Gm2", 0)))
    if max(int(row.get("Gm1", 0)), int(row.get("Gm2", 0))) >= 5 and max_games_gap >= 1:
        return "3_serving_for_set"
    if games_played <= 3:
        return "1_match_start"
    return "2_mid_match"


def main() -> None:
    df = pd.read_parquet(PRED_FILE)
    print(f"Total points: {len(df):,}\n")

    df["bucket"] = df.apply(bucket_row, axis=1)

    print(f"{'Bucket':<20} {'n':>8} {'Markov LL':>10} {'ML+MC LL':>10} "
          f"{'Markov mean_p':>14} {'ML+MC mean_p':>14} {'target_mean':>12}")
    for bucket in sorted(df["bucket"].unique()):
        g = df[df["bucket"] == bucket]
        y = g["target"].values
        mk = g["markov_pred"].values
        ml = g["ml_pred"].values
        mk_ll = log_loss_safe(y, mk)
        ml_ll = log_loss_safe(y, ml)
        print(f"{bucket:<20} {len(g):>8} {mk_ll:>10.4f} {ml_ll:>10.4f} "
              f"{mk.mean():>14.4f} {ml.mean():>14.4f} {y.mean():>12.4f}")

    print("\n=== Interpretation guide ===")
    print("Hypothesis A (recursion correct, only pre-match parameters are weak):")
    print("  Markov's mean predicted probability should progressively sharpen toward")
    print("  extremes as buckets move from match_start -> match_point, and match_point")
    print("  log loss should be LOW (predictions genuinely close to correct extremes).")
    print()
    print("Hypothesis B (a remaining issue beyond weak parameters):")
    print("  Markov's mean predicted probability stays muted (e.g. never exceeds ~0.75)")
    print("  even at match_point, and/or match_point log loss remains high despite the")
    print("  score state being maximally informative.")
    print()
    print("Directly check: does Markov's match_point mean_p approach target_mean")
    print("(which should be close to 1.0 at match point, since the tracked-and-favored")
    print("player usually — though not always — goes on to win from there)?")


if __name__ == "__main__":
    main()