"""
build_symmetric_dataset.py — converts the winner_*/loser_* feature dataset (Days 1-5,
frozen) into a symmetric player_1/player_2 modeling dataset suitable for training a
classifier.

WHY THIS STEP IS NECESSARY (read before touching anything downstream):
Every row in matches_with_day5_features.parquet has features named winner_X / loser_X. If
fed directly to a classifier with label="did winner win" the label is constant (always 1)
— there is nothing to predict, and any attempt to relabel it naively risks the model
learning "column identity" rather than genuine signal. The standard, correct formulation
for a win-probability model is:

    - Randomly assign each match's two players to player_1 / player_2 slots
    - label = 1 if player_1 is the actual winner, else 0
    - Features are built as DIFFERENCES (player_1's value minus player_2's value) for every
      paired numeric feature, since a difference is naturally antisymmetric: swapping which
      player is "player_1" flips both the sign of every diff feature AND the label,
      preserving the model's symmetry. This is verified explicitly in
      tests/unit/test_build_symmetric_dataset.py::test_swap_symmetry.

ASSIGNMENT DETERMINISM: player_1/player_2 assignment is derived from a hash of
(tourney_id, match_num) — NOT from row order or a global RNG seed — so it is stable and
reproducible regardless of how the input is sorted or re-run.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# (winner_column, loser_column, output_diff_name). Every pair here becomes a single
# `{name}_diff` feature = player_1's value - player_2's value.
FEATURE_PAIRS: list[tuple[str, str, str]] = [
    ("elo_pre_match_winner", "elo_pre_match_loser", "elo"),
    ("winner_win_pct_last5", "loser_win_pct_last5", "win_pct_last5"),
    ("winner_win_pct_last10", "loser_win_pct_last10", "win_pct_last10"),
    ("winner_win_pct_last20", "loser_win_pct_last20", "win_pct_last20"),
    ("winner_surface_win_pct_last10", "loser_surface_win_pct_last10", "surface_win_pct_last10"),
    ("winner_avg_game_diff_last10", "loser_avg_game_diff_last10", "avg_game_diff_last10"),
    ("winner_surface_avg_game_diff_last10", "loser_surface_avg_game_diff_last10", "surface_avg_game_diff_last10"),
    ("winner_opponent_elo_mean_last10", "loser_opponent_elo_mean_last10", "opponent_elo_mean_last10"),
    ("winner_win_streak_entering_match", "loser_win_streak_entering_match", "win_streak"),
    ("winner_loss_streak_entering_match", "loser_loss_streak_entering_match", "loss_streak"),
    ("winner_rest_days", "loser_rest_days", "rest_days"),
    ("winner_straight_set_rate_last10", "loser_straight_set_rate_last10", "straight_set_rate_last10"),
]

# Contextual (non-diffed) columns carried through as-is for reference/stratification, not
# necessarily all used as model inputs — model training selects from these explicitly.
CONTEXT_COLS = ["tourney_date", "tourney_id", "tourney_name", "surface", "tourney_level",
                 "round", "best_of"]


def _assignment_bit(df: pd.DataFrame) -> pd.Series:
    """Deterministic player_1/player_2 assignment from a hash of stable match identifiers —
    NOT row order, NOT a global RNG — so this is reproducible across any re-run or re-sort."""
    key = df["tourney_id"].astype(str) + "_" + df["match_num"].astype(str)
    hashed = pd.util.hash_pandas_object(key, index=False)
    return (hashed % 2).astype(int)


def build_symmetric_dataset(matches: pd.DataFrame, feature_pairs: list[tuple[str, str, str]] | None = None) -> pd.DataFrame:
    df = matches.copy()
    assign_p1_is_winner = _assignment_bit(df) == 0  # arbitrary convention, deterministic
    pairs = feature_pairs if feature_pairs is not None else FEATURE_PAIRS

    out = pd.DataFrame(index=df.index)
    out["label"] = assign_p1_is_winner.astype(int)  # 1 if player_1 is the actual winner

    for winner_col, loser_col, name in pairs:
        if winner_col not in df.columns or loser_col not in df.columns:
            logger.warning("Skipping feature pair '%s' — column(s) not found in input.", name)
            continue
        p1_val = np.where(assign_p1_is_winner, df[winner_col], df[loser_col])
        p2_val = np.where(assign_p1_is_winner, df[loser_col], df[winner_col])
        out[f"{name}_diff"] = p1_val - p2_val

    for col in CONTEXT_COLS:
        if col in df.columns:
            out[col] = df[col]

    out["player_1_id"] = np.where(assign_p1_is_winner, df["winner_id"], df["loser_id"])
    out["player_2_id"] = np.where(assign_p1_is_winner, df["loser_id"], df["winner_id"])

    return out