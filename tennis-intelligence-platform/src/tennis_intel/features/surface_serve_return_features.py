"""
surface_serve_return_features.py — surface-conditioned extension to Day 6's serve/return
features, built following the exact same non-destructive pattern as the Elo redesign
(src/tennis_intel/ratings/surface_elo.py): a NEW function reusing the frozen Day 6 module's
already-validated stats-loading and player-ID-resolution logic, adding surface-conditioning
on top rather than modifying anything in serve_return_features.py itself.

MOTIVATION (evidence-based, not speculative): direct data analysis during this project's
Elo-redesign work found that surface-blind career first-serve-win% is a weak predictor of
match outcome (37.6% of match winners had a LOWER career first-serve-win% than the loser).
Surface-specific Elo, built to address the analogous weakness in Elo, was subsequently
CONFIRMED via permutation importance to be genuinely used by the trained classifier and to
outperform both the surface-blind Elo and the existing surface_win_pct_last10 rolling
feature. This module applies the same fix to the OTHER career-aggregate features that share
the identical diagnosed weakness (first_serve_win_pct_career, first_serve_in_pct_career,
bp_saved_pct_career, etc.) — same logic, same evidence, same non-destructive approach.

LEAKAGE: uses the identical shift(1)-then-expanding/rolling idiom as the frozen
compute_rolling_serve_return_features, just with a (player_id, surface) grouping key
instead of player_id alone — mirroring exactly how Day 5's _add_rolling_surface groups by
(player_id, surface) for win_pct. Leakage safety is inherited from this proven idiom, not
re-derived.
"""

from __future__ import annotations

import logging

import pandas as pd

from tennis_intel.ratings.processor import _round_rank

logger = logging.getLogger(__name__)

DEFAULT_SURFACE_WINDOWS = (10,)

SURFACE_RATE_COLS = [
    "first_serve_in_pct", "first_serve_win_pct", "second_serve_win_pct",
    "ace_rate", "df_rate", "bp_saved_pct", "return_pts_won_pct", "bp_converted_pct",
]


def attach_surface(stats_with_ids: pd.DataFrame, frozen_join: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a `surface` column to the already-ID-resolved stats dataframe, via a fresh merge
    against frozen_join's tml_surface — does NOT modify attach_player_ids_and_chronology
    itself, so the frozen Day 6 module's existing behavior and any code depending on its
    exact output shape is completely unaffected by this addition.
    """
    surface_lookup = frozen_join[["mcp_match_id", "tml_surface"]].drop_duplicates(
        subset="mcp_match_id"
    ).rename(columns={"tml_surface": "surface"})
    merged = stats_with_ids.merge(surface_lookup, left_on="match_id",
                                   right_on="mcp_match_id", how="left")
    n_missing = merged["surface"].isna().sum()
    if n_missing:
        logger.warning("%d row(s) could not be matched to a surface — will be excluded from "
                       "surface-conditioned rolling stats for those rows.", n_missing)
    return merged.drop(columns=["mcp_match_id"], errors="ignore")


def compute_rolling_surface_serve_return_features(
    stats_with_surface: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_SURFACE_WINDOWS
) -> pd.DataFrame:
    """
    Leakage-safe rolling (last-N and career-to-date) serve/return rates, conditioned on
    surface — i.e. "this player's career first-serve-win% specifically on clay", not
    blended across all surfaces. Same shift(1)-then-expanding/rolling idiom as the frozen
    compute_rolling_serve_return_features, with (player_id, surface) as the grouping key
    instead of player_id alone.

    Output columns are named {col}_surface_career and {col}_surface_last{w} — distinct
    names from the existing surface-blind {col}_career/{col}_last{w}, so both can coexist
    without collision.
    """
    df = stats_with_surface.copy()
    df["_round_rank"] = _round_rank(df["tml_round"])
    df = df.sort_values(
        ["player_id", "surface", "tml_tourney_date", "_round_rank", "tml_match_num", "tml_tourney_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    key = [df["player_id"], df["surface"]]
    g = df.groupby(key)
    for col in SURFACE_RATE_COLS:
        if col not in df.columns:
            continue
        shifted = g[col].shift(1)
        df[f"{col}_surface_career"] = shifted.groupby(key).expanding(min_periods=1).mean().reset_index(drop=True)
        for w in windows:
            df[f"{col}_surface_last{w}"] = shifted.groupby(key).rolling(w, min_periods=1).mean().reset_index(drop=True)

    return df.drop(columns=["_round_rank"])