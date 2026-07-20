"""
test_prematch_probability_sanity.py — computes the GENUINE pre-match win probability
(not point-1, the real thing: build_pretrained_prior's Elo/surface-Elo/H2H/tournament-
form-informed estimate, inverted through the Markov recursion) for 10 randomly selected
real matches, as a quick sanity check that the numbers look plausible before committing
further effort to optimizing this specific engine.

For each match, prints:
  - the real players and match context (surface, best_of)
  - p0_a_wins: the underlying feature-rich pre-match estimate (compute_ml_pre_match_probability)
  - the inverted point-level prior (p_serve0, n0_serve, p_return0, n0_return)
  - the REAL final outcome, so you can eyeball whether the favorite (by this pre-match
    estimate) actually tended to win — NOT a formal accuracy claim from just 10 matches,
    purely a sanity check that nothing looks obviously broken.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_publication_trajectory import compute_ml_pre_match_probability
from tennis_intel.live.ml_informed_markov import build_pretrained_prior
from tennis_intel.live.return_seed import compute_p_a_return_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW_MCP = PROJECT_ROOT / "data" / "raw" / "tennis_MatchChartingProject"
POINT_FILES = [
    RAW_MCP / "charting-m-points-to-2009.csv",
    RAW_MCP / "charting-m-points-2010s.csv",
    RAW_MCP / "charting-m-points-2020s.csv",
]

N_MATCHES = 10
SEED = 123


def main() -> None:
    from tennis_intel.live.build_point_dataset import build_point_dataset

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
    print(f"{'Match':<70} {'p0(winner)':>11} {'p_serve0':>10} {'n0_serve':>9} {'Real result':>14}")
    print("-" * 118)

    for match_id in selected:
        match_df = points[points["match_id"] == match_id].sort_values("Pt")
        first_row = match_df.iloc[0].to_dict()
        p1_is_winner = bool(first_row["player1_is_winner"])

        p0_a_wins = compute_ml_pre_match_probability(first_row, model, feature_cols)

        # BUG FIX (external review, 2026-07): see return_seed.py's module docstring.
        p_a_return_seed = compute_p_a_return_seed(first_row, track_winner=True)

        elo_matches_played_a = first_row.get("elo_matches_played_pre_winner")
        best_of_val = int(first_row["best_of"]) if pd.notna(first_row.get("best_of")) else 3

        p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
            p0_a_wins, p_a_return_seed, best_of_val, elo_matches_played_a=elo_matches_played_a,
        )

        # p0_a_wins is P(the tracked winner, "A", wins) — always correct by construction,
        # since we're just re-stating who actually won. What's INFORMATIVE is p_serve0
        # and n0, and whether p0_a_wins itself looks like a SANE probability (not 0.5-flat
        # regardless of matchup, not degenerate at exactly 0 or 1 for an ordinary match).
        winner_name = "Player1" if p1_is_winner else "Player2"
        label = f"{match_id[:66]}"
        print(f"{label:<70} {p0_a_wins:>11.4f} {p_serve0:>10.4f} {n0_serve:>9.1f} "
              f"{'winner=' + winner_name:>14}")

    print("\nSanity checks to eyeball:")
    print("  - p0(winner) should mostly sit above 0.5 (it's re-stating the real winner's")
    print("    own pre-match estimate) but should VARY meaningfully across matches, not")
    print("    cluster at one extreme or sit flat at ~0.5 for every match regardless of")
    print("    the real matchup.")
    print("  - p_serve0 should look like a plausible serve-win rate (roughly 0.55-0.85 for")
    print("    most real tour matches) — a value near 0 or 1 would indicate the inversion")
    print("    or its inputs are behaving unexpectedly for that specific match.")
    print("  - n0_serve should vary with elo_matches_played (more experienced players ->")
    print("    higher n0), not be identical for every match.")


if __name__ == "__main__":
    main()