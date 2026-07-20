"""
generate_trajectories.py — Task 3: publication-quality win-probability trajectory plots.

Reads the saved per-point predictions from the head-to-head evaluation (no recomputation —
trajectories come from the exact predictions that produced the reported metrics, so plots
and tables can never silently disagree) and renders, for each selected match:

    x = point index, y = P(tracked player wins), one line per engine,
    horizontal reference at the actual outcome (1 or 0), 0.5 guide line.

Match selection is configurable:
  --match-ids ID [ID ...]   plot exactly these matches
  --n-auto K                otherwise: the K most "interesting" matches, ranked by the
                            volatility of the ML+MC trajectory (sum of absolute
                            point-to-point probability changes) — comebacks and swingy
                            matches rank highest, which is what a reader wants to see.

Usage:
    python pipelines/generate_trajectories.py                # 4 auto-selected matches
    python pipelines/generate_trajectories.py --n-auto 6
    python pipelines/generate_trajectories.py --match-ids 20220605-M-Roland_Garros-F-...
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never requires a display
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "docs" / "trajectories"

PREDICTIONS_FILE = PROCESSED / "day11_head_to_head_v2_predictions.parquet"
# Fallback to the v1 file if v2 hasn't been run yet (v1 lacks the mixed target; its
# trajectories are still valid P(winner) curves, just with target always 1).
FALLBACK_FILE = PROCESSED / "day10_head_to_head_predictions.parquet"
# ML-Informed Markov (corrected Elo/H2H-inverted prior) predictions — a SEPARATE file
# from evaluate_ml_informed_markov.py, merged in below by (match_id, Pt). Optional: older
# runs of this script, or environments that haven't run that evaluation yet, still work
# with just Markov + ML+MC — the third line is added only when this file is present.
ML_INFORMED_FILE = PROCESSED / "ml_informed_markov_predictions.parquet"


def load_predictions() -> tuple[pd.DataFrame, bool]:
    if PREDICTIONS_FILE.exists():
        df = pd.read_parquet(PREDICTIONS_FILE)
        has_target = True
    else:
        logger.warning("v2 predictions not found; falling back to v1 file %s", FALLBACK_FILE)
        df = pd.read_parquet(FALLBACK_FILE)
        has_target = False

    if ML_INFORMED_FILE.exists():
        ml_informed = pd.read_parquet(ML_INFORMED_FILE)[["match_id", "Pt", "ml_informed_pred"]]
        n_before = len(df)
        df = df.merge(ml_informed, on=["match_id", "Pt"], how="left")
        if len(df) != n_before:
            raise AssertionError(
                f"Row count changed during merge with {ML_INFORMED_FILE.name}: "
                f"{n_before:,} -> {len(df):,}. This indicates a non-unique merge key — "
                f"do not trust the resulting plots until investigated."
            )
        n_matched = df["ml_informed_pred"].notna().sum()
        logger.info("Merged ML-Informed Markov predictions: %d/%d points matched "
                   "(unmatched points — e.g. matches not in evaluate_ml_informed_markov.py's "
                   "150-match sample — will simply not show that line)", n_matched, len(df))
    else:
        logger.warning("%s not found — trajectories will show Markov + ML+MC only. Run "
                       "evaluate_ml_informed_markov.py first to include the third line.",
                       ML_INFORMED_FILE.name)
        df["ml_informed_pred"] = float("nan")

    return df, has_target


def rank_by_volatility(df: pd.DataFrame) -> pd.Series:
    """Sum of absolute point-to-point changes in the ML+MC trajectory, per match —
    a simple, defensible 'how dramatic was this match' score."""
    def vol(g: pd.DataFrame) -> float:
        return g.sort_values("Pt")["ml_pred"].diff().abs().sum()
    return df.groupby("match_id").apply(vol).sort_values(ascending=False)


def plot_match(g: pd.DataFrame, match_id: str, has_target: bool, out_path: Path) -> None:
    g = g.sort_values("Pt")
    target = float(g["target"].iloc[0]) if has_target and "target" in g.columns else 1.0

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(g["Pt"], g["markov_pred"], label="Markov (analytical)",
            lw=1.6, color="#1f77b4")
    ax.plot(g["Pt"], g["ml_pred"], label="ML + Monte Carlo",
            lw=1.6, color="#d62728", alpha=0.9)
    if "ml_informed_pred" in g.columns and g["ml_informed_pred"].notna().any():
        ax.plot(g["Pt"], g["ml_informed_pred"], label="ML-Informed Markov (corrected prior)",
                lw=1.6, color="#17becf", alpha=0.95)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.axhline(target, color="black", lw=1.0, ls="--",
               label=f"Actual outcome ({int(target)})")

    # Shade set boundaries where available (Set1+Set2 increments)
    if {"Set1", "Set2"}.issubset(g.columns):
        sets_total = (g["Set1"].fillna(0) + g["Set2"].fillna(0)).astype(int)
        changes = g["Pt"][sets_total.diff() > 0]
        for x in changes:
            ax.axvline(x, color="grey", lw=0.6, alpha=0.5)

    ax.set_xlabel("Point index")
    ax.set_ylabel("P(tracked player wins match)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(match_id.replace("_", " "), fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-ids", nargs="*", default=None)
    parser.add_argument("--n-auto", type=int, default=4)
    args = parser.parse_args()

    df, has_target = load_predictions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.match_ids:
        selected = [m for m in args.match_ids if m in set(df["match_id"])]
        missing = set(args.match_ids) - set(selected)
        if missing:
            logger.warning("Not in predictions file (skipped): %s", sorted(missing))
    else:
        selected = rank_by_volatility(df).head(args.n_auto).index.tolist()
        logger.info("Auto-selected %d most volatile matches", len(selected))

    for match_id in selected:
        g = df[df["match_id"] == match_id]
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in match_id)
        plot_match(g, match_id, has_target, OUT_DIR / f"trajectory_{safe_name}.png")

    logger.info("Done: %d trajectory plot(s) in %s", len(selected), OUT_DIR)


if __name__ == "__main__":
    main()