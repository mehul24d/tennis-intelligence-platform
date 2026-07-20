"""
diagnose_day10_runtime.py — Task 6: empirical investigation of the v1 evaluation's
throughput degradation (6.1 -> 1.5 pts/sec) and max_points non-termination warnings,
using the per-point runtimes saved in day10_head_to_head_predictions.parquet.

Hypotheses under test (from docs/day10_head_to_head_freeze.md):
  H1: best-of-5 points are several times slower to evaluate than best-of-3 points (each
      simulated continuation is much longer), and date-sorted match_ids cluster Slam
      (bo5) matches together — producing sustained slow regions late in the run rather
      than uniform slowness.
  H2: the flat max_points=400 cap was genuinely reachable by realistic bo5 continuations,
      so many "non-terminating" warnings were expected behavior on Slams, not a bug.
  H3 (residual): any slowdown NOT explained by best_of composition — e.g. thermal
      throttling on a fanless laptop over a 4.5-hour run — shows up as the same
      best_of/points-remaining profile getting slower in later wall-clock order.

Usage:
    python pipelines/diagnose_day10_runtime.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_FILE = PROJECT_ROOT / "data" / "processed" / "day10_head_to_head_predictions.parquet"


def main() -> None:
    df = pd.read_parquet(PRED_FILE)
    df = df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
    df["eval_order"] = np.arange(len(df))  # v1 processed rows in exactly this order
    df["ms"] = df["ml_runtime_sec"] * 1000

    print(f"Points: {len(df):,}   matches: {df['match_id'].nunique()}\n")

    # --- H1: runtime by format ---
    print("=== Mean ML+MC runtime per point, by best_of (H1) ===")
    by_bo = df.groupby(df["best_of"].fillna(3).astype(int))["ms"].agg(["count", "mean", "median"])
    print(by_bo.to_string())
    if 5 in by_bo.index and 3 in by_bo.index:
        ratio = by_bo.loc[5, "mean"] / by_bo.loc[3, "mean"]
        print(f"\nbo5 / bo3 mean-runtime ratio: {ratio:.1f}x")

    # --- H1 continued: are slow regions where the bo5 matches sit in eval order? ---
    print("\n=== Runtime and bo5 share by position in the run (deciles of eval order) ===")
    df["decile"] = pd.qcut(df["eval_order"], 10, labels=False)
    dec = df.groupby("decile").agg(
        mean_ms=("ms", "mean"),
        share_bo5=("best_of", lambda s: (s.fillna(3).astype(int) == 5).mean()),
        pts_per_sec=("ms", lambda s: 1000 / s.mean()),
    )
    print(dec.to_string(float_format=lambda x: f"{x:.2f}"))
    corr = dec["mean_ms"].corr(dec["share_bo5"])
    print(f"\nCorrelation(decile mean runtime, decile bo5 share): {corr:.3f}")
    print("High positive correlation supports H1 (format clustering) as the main driver;")
    print("slowdown in late deciles beyond what bo5 share explains supports H3 (thermal).")

    # --- H3: same-format runtime drift over the run ---
    print("\n=== bo3-only runtime by decile (isolates drift from format mix, H3) ===")
    bo3 = df[df["best_of"].fillna(3).astype(int) == 3]
    if len(bo3):
        drift = bo3.groupby("decile")["ms"].mean()
        print(drift.to_string(float_format=lambda x: f"{x:.1f}"))
        if len(drift) > 1 and drift.iloc[0] > 0:
            print(f"\nbo3-only slowdown, last vs first populated decile: "
                  f"{drift.iloc[-1] / drift.iloc[0]:.2f}x")
            print("A ratio well above 1 for the SAME format = system-level slowdown "
                  "(thermal throttling / background load), since the work per point "
                  "is format-determined and unchanged.")

    # --- H2: how far from terminating were bo5 continuations under a 400 cap? ---
    print("\n=== H2 context: points remaining when evaluated, by format ===")
    # Points remaining in the real match approximates required simulation depth
    df["pts_in_match"] = df.groupby("match_id")["Pt"].transform("max")
    df["pts_remaining"] = df["pts_in_match"] - df["Pt"]
    rem = df.groupby(df["best_of"].fillna(3).astype(int))["pts_remaining"].describe(
        percentiles=[0.5, 0.9, 0.99])[["50%", "90%", "99%", "max"]]
    print(rem.to_string(float_format=lambda x: f"{x:.0f}"))
    print("\nIf bo5 'max' approaches or exceeds 400: the v1 cap was reachable by REAL")
    print("match lengths, so cap-hits on bo5 were expected behavior (fixed in")
    print("batch_simulate_dynamic: cap now 350 bo3 / 700 bo5).")


if __name__ == "__main__":
    main()