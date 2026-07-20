"""
head_to_head_features.py — overall and tournament-specific head-to-head record, computed
on the WIDE (match-level) dataframe since H2H inherently needs both players in the same
row, unlike the long-format per-player rolling stats in feature_engineering_day5.py.

PROVENANCE: the core H2H logic here is extracted and adapted from
src/tennis_intel/features/rolling_stats.py, which implemented this correctly (verified by
inspection: proper pre-match state, sequential update pattern identical to Day 4's Elo and
Day 5's rolling stats) but was NEVER WIRED into the actual pipeline — no entrypoint called
it, and its own referenced test file (tests/unit/test_rolling_stats.py) does not exist.
Rather than resurrect that module wholesale (its win_pct/surface_win_pct/rest_days logic
would collide with feature_engineering_day5.py's already-frozen, already-tested versions of
the same features), only the H2H-specific piece is extracted here, then extended with a
tournament-specific variant.

LEAKAGE DISCIPLINE: identical pattern to every other sequential feature in this project —
for each match, both players' pre-match head-to-head counts are read from state accumulated
strictly BEFORE this match, then state is updated AFTER. Verified explicitly in
tests/unit/test_head_to_head_features.py.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from tennis_intel.ratings.processor import _round_rank

logger = logging.getLogger(__name__)


def add_head_to_head_features(
    matches: pd.DataFrame,
    winner_id_col: str = "winner_id",
    loser_id_col: str = "loser_id",
    tourney_name_col: str = "tourney_name",
) -> pd.DataFrame:
    """
    Adds four PRE-MATCH columns:
      winner_h2h_wins_pre_match / loser_h2h_wins_pre_match — overall career head-to-head
        win counts between these two specific players, before this match.
      winner_tourney_h2h_wins_pre_match / loser_tourney_h2h_wins_pre_match — head-to-head
        win counts between these two players RESTRICTED to matches previously played at
        this SAME tournament (e.g. "these two players' record specifically at Wimbledon").

    If tourney_name_col is missing from the input, the tournament-specific columns are
    filled with NaN and a warning is logged — general H2H is still computed normally,
    since it doesn't depend on tournament identity.
    """
    df = matches.copy()

    missing_ids = df[winner_id_col].isna().sum() + df[loser_id_col].isna().sum()
    if missing_ids:
        logger.warning(
            "%d row(s) have a missing winner_id/loser_id and will be dropped before "
            "head-to-head processing.", missing_ids
        )
        df = df.dropna(subset=[winner_id_col, loser_id_col])

    df["_round_rank"] = _round_rank(df["round"])
    sort_cols = ["tourney_date", "_round_rank", "match_num", "tourney_id"]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    has_tourney_name = tourney_name_col in df.columns
    if not has_tourney_name:
        logger.warning(
            "'%s' not found in input — tournament-specific head-to-head will be all-NaN. "
            "General (overall) head-to-head is unaffected.", tourney_name_col
        )

    # pair_key: unordered pair of player IDs, so A-vs-B and B-vs-A share the same H2H state.
    # tourney_pair_key: the same, further keyed by tournament NAME (not tourney_id, since
    # the same recurring tournament — e.g. Wimbledon — has a DIFFERENT tourney_id every
    # year in TML's schema; tourney_name is the stable identifier across years).
    h2h_wins: dict[frozenset, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tourney_h2h_wins: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    winner_h2h, loser_h2h = [], []
    winner_tourney_h2h, loser_tourney_h2h = [], []

    for row in df.itertuples(index=False):
        winner_id = getattr(row, winner_id_col)
        loser_id = getattr(row, loser_id_col)
        pair_key = frozenset({winner_id, loser_id})

        winner_h2h.append(h2h_wins[pair_key][winner_id])
        loser_h2h.append(h2h_wins[pair_key][loser_id])

        if has_tourney_name:
            tourney_name = getattr(row, tourney_name_col)
            tourney_pair_key = (pair_key, tourney_name)
            winner_tourney_h2h.append(tourney_h2h_wins[tourney_pair_key][winner_id])
            loser_tourney_h2h.append(tourney_h2h_wins[tourney_pair_key][loser_id])
        else:
            winner_tourney_h2h.append(float("nan"))
            loser_tourney_h2h.append(float("nan"))

        # Update state AFTER reading this match's pre-match values
        h2h_wins[pair_key][winner_id] += 1
        if has_tourney_name:
            tourney_h2h_wins[tourney_pair_key][winner_id] += 1

    df["winner_h2h_wins_pre_match"] = winner_h2h
    df["loser_h2h_wins_pre_match"] = loser_h2h
    df["winner_tourney_h2h_wins_pre_match"] = winner_tourney_h2h
    df["loser_tourney_h2h_wins_pre_match"] = loser_tourney_h2h

    return df.drop(columns=["_round_rank"])