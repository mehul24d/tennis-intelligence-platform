"""
test_hybrid_vs_ml_informed_prematch.py — sanity check comparing the Hybrid (fixed-weight
Markov/ML+MC) engine's pre-match prediction against ML-Informed Markov's pre-match
prediction, for BOTH players (not just the winner), on several real matches.

Prints, for each match:
  - Hybrid pre-match: P(Player1 wins), P(Player2 wins)
  - ML-Informed Markov pre-match: P(Player1 wins), P(Player2 wins)
  - the real final outcome, for eyeballing plausibility

This is a sanity check, not a formal evaluation — the point is to see whether the two
numbers move together sensibly (same favorite, roughly comparable magnitude) or whether
one is behaving in a way that doesn't make sense, on a small, human-inspectable sample.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_publication_trajectory import (
    compute_markov_pre_match_probability,
    compute_ml_pre_match_probability,
    compute_composite_prematch_probability,
)
from tennis_intel.live.hybrid_engine import hybrid_predict
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.ml_informed_markov import build_pretrained_prior
from tennis_intel.live.markov_baseline import prob_win_match
from tennis_intel.live.build_point_dataset import build_point_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW_MCP = PROJECT_ROOT / "data" / "raw" / "tennis_MatchChartingProject"
POINT_FILES = [
    RAW_MCP / "charting-m-points-to-2009.csv",
    RAW_MCP / "charting-m-points-2010s.csv",
    RAW_MCP / "charting-m-points-2020s.csv",
]

N_MATCHES = 8
SEED = 77


def main() -> None:
    print("Loading model and building point dataset...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]

    all_match_ids = sorted(points["match_id"].unique())
    rng = random.Random(SEED)
    selected = rng.sample(all_match_ids, min(N_MATCHES, len(all_match_ids)))

    print(f"\nRandomly selected {len(selected)} matches (seed={SEED}):\n")
    header = (f"{'Match':<55} {'Hybrid P1':>10} {'Hybrid P2':>10} "
              f"{'MLInf P1':>10} {'MLInf P2':>10} {'Sum(H)':>8} {'Sum(ML)':>8} {'Winner':>8}")
    print(header)
    print("-" * len(header))

    for match_id in selected:
        match_df = points[points["match_id"] == match_id].sort_values("Pt")
        row0 = match_df.iloc[0].to_dict()
        p1_is_winner = bool(row0["player1_is_winner"])
        best_of_val = int(row0["best_of"]) if pd.notna(row0.get("best_of")) else 3

        # --- Hybrid: fixed-weight blend of Markov's and ML+MC's OWN pre-match values ---
        pre_markov_p1 = compute_markov_pre_match_probability(row0)
        pre_ml_for_winner = compute_ml_pre_match_probability(row0, model, feature_cols)
        pre_ml_p1 = pre_ml_for_winner if p1_is_winner else (1.0 - pre_ml_for_winner)
        hybrid_p1 = hybrid_predict(markov_p=pre_markov_p1, ml_mc_p=pre_ml_p1)
        hybrid_p2 = hybrid_predict(markov_p=1.0 - pre_markov_p1, ml_mc_p=1.0 - pre_ml_p1)

        # --- ML-Informed Markov: composite prior inverted through the recursion ---
        p0_a_wins_composite = compute_composite_prematch_probability(row0)
        # BUG FIX (external review, 2026-07): see return_seed.py's module docstring.
        p_a_return_seed = compute_p_a_return_seed(row0, track_winner=True)
        elo_matches_played_a = row0.get("elo_matches_played_pre_winner")

        p_serve0, _, p_return0, _ = build_pretrained_prior(
            p0_a_wins_composite, p_a_return_seed, best_of_val,
            elo_matches_played_a=elo_matches_played_a,
        )
        ml_informed_for_winner = prob_win_match(p_serve0, p_return0, best_of=best_of_val)
        ml_informed_p1 = ml_informed_for_winner if p1_is_winner else (1.0 - ml_informed_for_winner)
        ml_informed_p2 = 1.0 - ml_informed_p1

        winner_label = "P1" if p1_is_winner else "P2"
        label = match_id[:53]
        print(f"{label:<55} {hybrid_p1:>10.4f} {hybrid_p2:>10.4f} "
              f"{ml_informed_p1:>10.4f} {ml_informed_p2:>10.4f} "
              f"{hybrid_p1 + hybrid_p2:>8.4f} {ml_informed_p1 + ml_informed_p2:>8.4f} "
              f"{winner_label:>8}")

    print("\nSanity checks to eyeball:")
    print("  - Sum(H) and Sum(ML) should both be ~1.0000 (P1 + P2 should sum to 1 for")
    print("    any single engine's own pre-match estimate — a genuine deviation would")
    print("    indicate the two players' probabilities aren't being computed consistently)")
    print("  - Hybrid and ML-Informed Markov should usually agree on WHICH player is")
    print("    favored, even if the exact magnitude differs — a sign disagreement between")
    print("    them on the same match is worth investigating directly, not assumed to be fine")
    print("  - Neither should be flatly ~0.50/0.50 for every match regardless of the real")
    print("    matchup — that would suggest one of the pre-match inputs isn't varying")


if __name__ == "__main__":
    main()