"""
feature_engineering_day5.py — leakage-safe rolling performance features, built on top of the
FROZEN Day 4 Elo pipeline (matches_with_elo.parquet). Do not modify Day 1-4 outputs.

DESIGN: long-format, vectorized, shift-then-roll
-------------------------------------------------
Every match is expanded into two rows (one per player's perspective) in a "long" dataframe,
sorted per-player by the same chronology key frozen in Day 4:
(tourney_date, round_order, match_num, tourney_id).

For every rolling feature, the pattern is: `group['x'].shift(1).rolling(window).agg(...)`.
LEAKAGE PROOF: `.shift(1)` moves every value down by one row within each player's group
BEFORE the rolling window is computed, so a window ending at row i can only ever contain
values from rows < i for that player — the current match's own outcome is structurally
excluded from its own feature, not just filtered out after the fact. This is the same
class of proof used for compute_ratings() in Day 4, applied via vectorized pandas ops
instead of an explicit loop (this module intentionally avoids per-row Python loops for
performance across ~198k matches / ~400k long-format rows).

CHRONOLOGY CAVEAT (inherited from Day 4, not re-litigated here): TML has no per-match date,
only tournament start date. "Matches in last N days" and "rest days" are therefore proxies
based on tourney_date, not exact match timestamps — documented, not silently assumed exact.

SCOPE (see accompanying freeze/summary doc for full reasoning):
  Built now: overall rolling form, surface-specific rolling form, opponent strength
             (via Day 4 Elo, not recomputed), momentum (streaks/rest/recent-match-count),
             tournament context (previous level/round reached).
  Deferred, explicitly (not silently dropped): rolling surface-specific ELO (a distinct
             rating system, belongs with ratings/surface_elo.py), travel proxy (no
             geolocation data available), rolling same-tournament/Grand-Slam-specific
             performance, and serve/return stats from MCP (proposed as Day 6 — a genuine
             separate data-integration task, not a Day 5 add-on).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from tennis_intel.ratings.processor import _round_rank
from tennis_intel.features.score_parser import parse_score
from tennis_intel.features.head_to_head_features import add_head_to_head_features

logger = logging.getLogger(__name__)

DEFAULT_WINDOWS = (5, 10, 20)
OPPONENT_STRENGTH_WINDOW = 10
RECENT_MATCH_WINDOWS_DAYS = (7, 14, 30)


@dataclass
class Day5FeaturesResult:
    augmented: pd.DataFrame
    diagnostics: dict


def _sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_round_rank"] = _round_rank(df["round"])
    df = df.sort_values(
        ["tourney_date", "_round_rank", "match_num", "tourney_id"], kind="mergesort"
    ).reset_index(drop=True)
    df["_match_key"] = df.index
    return df


def _parse_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Applies score_parser to every match once, attaching winner-perspective game/set stats
    as columns. Rows that fail to parse get NaN — logged as a diagnostic, not silently
    dropped, since the match itself is still valid for win/loss and Elo purposes even if its
    score string couldn't be parsed."""
    parsed = df["score"].apply(parse_score)
    df = df.copy()
    df["_w_sets_won"] = [p.sets_won for p in parsed]
    df["_w_sets_lost"] = [p.sets_lost for p in parsed]
    df["_w_games_won"] = [p.games_won for p in parsed]
    df["_w_games_lost"] = [p.games_lost for p in parsed]
    df["_n_sets_played"] = [p.n_sets_played for p in parsed]
    df["_straight_sets"] = [p.straight_sets for p in parsed]
    df["_score_parse_ok"] = [p.parse_ok for p in parsed]
    return df


def _build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Expands each match into two player-perspective rows. See module docstring for the
    leakage-safety argument this structure enables."""
    common_cols = ["_match_key", "tourney_date", "_round_rank", "match_num", "tourney_id",
                   "surface", "tourney_level", "round", "best_of", "minutes",
                   "elo_pre_match_winner", "elo_pre_match_loser", "tourney_name"]
    common_cols = [c for c in common_cols if c in df.columns]

    winner_rows = df[common_cols + [
        "winner_id", "loser_id", "_w_sets_won", "_w_sets_lost",
        "_w_games_won", "_w_games_lost", "_n_sets_played", "_straight_sets",
    ]].copy()
    winner_rows = winner_rows.rename(columns={
        "winner_id": "player_id", "loser_id": "opponent_id",
        "elo_pre_match_loser": "opponent_elo_pre",
    })
    winner_rows["win"] = 1
    winner_rows["games_won"] = winner_rows["_w_games_won"]
    winner_rows["games_lost"] = winner_rows["_w_games_lost"]
    winner_rows["sets_won"] = winner_rows["_w_sets_won"]
    winner_rows["sets_lost"] = winner_rows["_w_sets_lost"]

    loser_rows = df[common_cols + [
        "winner_id", "loser_id", "_w_sets_won", "_w_sets_lost",
        "_w_games_won", "_w_games_lost", "_n_sets_played", "_straight_sets",
    ]].copy()
    loser_rows = loser_rows.rename(columns={
        "loser_id": "player_id", "winner_id": "opponent_id",
        "elo_pre_match_winner": "opponent_elo_pre",
    })
    loser_rows["win"] = 0
    # Loser's perspective: games/sets won/lost are the MIRROR of the winner-perspective values
    loser_rows["games_won"] = loser_rows["_w_games_lost"]
    loser_rows["games_lost"] = loser_rows["_w_games_won"]
    loser_rows["sets_won"] = loser_rows["_w_sets_lost"]
    loser_rows["sets_lost"] = loser_rows["_w_sets_won"]

    long_df = pd.concat([winner_rows, loser_rows], ignore_index=True)
    long_df = long_df.drop(columns=["_w_games_won", "_w_games_lost", "_w_sets_won", "_w_sets_lost"])
    long_df["game_diff"] = long_df["games_won"] - long_df["games_lost"]
    long_df["three_set_win_or_loss"] = (long_df["_n_sets_played"] == 3) & (long_df["best_of"] == 3)
    long_df["five_set_win_or_loss"] = (long_df["_n_sets_played"] == 5) & (long_df["best_of"] == 5)

    # Per-player chronological order for rolling ops. _match_key is a global chronological
    # tiebreak (already unique and monotonic from the Day 4 sort), so sorting by
    # (player_id, _match_key) gives correct per-player chronology without re-deriving it.
    long_df = long_df.sort_values(["player_id", "_match_key"], kind="mergesort").reset_index(drop=True)
    return long_df


def _add_rolling_overall(long_df: pd.DataFrame, windows: tuple[int, ...]) -> None:
    g = long_df.groupby("player_id")
    for w in windows:
        shifted_win = g["win"].shift(1)
        long_df[f"win_pct_last{w}"] = shifted_win.groupby(long_df["player_id"]).rolling(w, min_periods=1).mean().reset_index(drop=True)
        long_df[f"n_matches_last{w}"] = shifted_win.groupby(long_df["player_id"]).rolling(w, min_periods=1).count().reset_index(drop=True)

        for col, out_name in [
            ("games_won", "avg_games_won"), ("games_lost", "avg_games_lost"),
            ("game_diff", "avg_game_diff"), ("sets_won", "avg_sets_won"),
            ("sets_lost", "avg_sets_lost"), ("_straight_sets", "straight_set_rate"),
            ("three_set_win_or_loss", "three_set_rate"), ("five_set_win_or_loss", "five_set_rate"),
            ("minutes", "avg_duration"),
        ]:
            shifted = g[col].shift(1)
            long_df[f"{out_name}_last{w}"] = shifted.groupby(long_df["player_id"]).rolling(w, min_periods=1).mean().reset_index(drop=True)


def _add_rolling_surface(long_df: pd.DataFrame, windows: tuple[int, ...]) -> None:
    key = [long_df["player_id"], long_df["surface"]]
    g = long_df.groupby(key)
    for w in windows:
        shifted_win = g["win"].shift(1)
        long_df[f"surface_win_pct_last{w}"] = shifted_win.groupby(key).rolling(w, min_periods=1).mean().reset_index(drop=True)
        long_df[f"surface_n_matches_last{w}"] = shifted_win.groupby(key).rolling(w, min_periods=1).count().reset_index(drop=True)
        shifted_diff = g["game_diff"].shift(1)
        long_df[f"surface_avg_game_diff_last{w}"] = shifted_diff.groupby(key).rolling(w, min_periods=1).mean().reset_index(drop=True)


def _add_rolling_tournament(long_df: pd.DataFrame, windows: tuple[int, ...]) -> None:
    """
    Tournament-specific rolling form: "this player's win rate over their last N times
    playing THIS SPECIFIC recurring tournament" (e.g. Djokovic's Wimbledon win rate over
    his last 10 Wimbledon appearances) — distinct from surface_win_pct_last{w} (Grass-wide,
    not Wimbledon-specific) and from previous_tourney_level (tracks the PRIOR tournament
    generically, not performance at THIS recurring event across years).

    Explicitly deferred in this module's original scope note (see module docstring) —
    picked up now following the same (player_id, key) grouping pattern already proven for
    surface-specific form above, just with tourney_name as the key instead of surface.

    If tourney_name is missing from the input, these columns are silently all-NaN (same
    graceful-degradation convention as head_to_head_features.py) rather than raising.
    """
    if "tourney_name" not in long_df.columns:
        logger.warning("'tourney_name' not found — tournament-specific rolling form will "
                       "be all-NaN. Other Day 5 features are unaffected.")
        for w in windows:
            long_df[f"tourney_win_pct_last{w}"] = float("nan")
            long_df[f"tourney_n_matches_last{w}"] = float("nan")
        return

    key = [long_df["player_id"], long_df["tourney_name"]]
    g = long_df.groupby(key)
    for w in windows:
        shifted_win = g["win"].shift(1)
        long_df[f"tourney_win_pct_last{w}"] = shifted_win.groupby(key).rolling(w, min_periods=1).mean().reset_index(drop=True)
        long_df[f"tourney_n_matches_last{w}"] = shifted_win.groupby(key).rolling(w, min_periods=1).count().reset_index(drop=True)


def _add_opponent_strength(long_df: pd.DataFrame, window: int) -> None:
    g = long_df.groupby("player_id")
    shifted_opp_elo = g["opponent_elo_pre"].shift(1)
    rolling = shifted_opp_elo.groupby(long_df["player_id"]).rolling(window, min_periods=1)
    long_df[f"opponent_elo_mean_last{window}"] = rolling.mean().reset_index(drop=True)
    long_df[f"opponent_elo_median_last{window}"] = rolling.median().reset_index(drop=True)
    long_df[f"opponent_elo_max_last{window}"] = rolling.max().reset_index(drop=True)
    long_df[f"opponent_elo_min_last{window}"] = rolling.min().reset_index(drop=True)


def _add_momentum(long_df: pd.DataFrame, day_windows: tuple[int, ...]) -> None:
    g = long_df.groupby("player_id")

    # Streaks: compute POST-match streak via a standard run-length trick, then shift by one
    # row (per player) to get the PRE-match streak — i.e. "streak entering this match".
    sign = long_df["win"] * 2 - 1  # +1 win, -1 loss
    change = sign.ne(sign.groupby(long_df["player_id"]).shift()).astype(int)
    run_id = change.groupby(long_df["player_id"]).cumsum()
    run_len = long_df.groupby(["player_id", run_id]).cumcount() + 1
    current_streak = sign * run_len
    win_streak_post = current_streak.clip(lower=0)
    loss_streak_post = (-current_streak).clip(lower=0)

    long_df["win_streak_entering_match"] = win_streak_post.groupby(long_df["player_id"]).shift(1).fillna(0).astype(int)
    long_df["loss_streak_entering_match"] = loss_streak_post.groupby(long_df["player_id"]).shift(1).fillna(0).astype(int)

    # Rest days: days since this player's previous match (tourney_date proxy)
    prev_date = g["tourney_date"].shift(1)
    long_df["rest_days"] = (long_df["tourney_date"] - prev_date).dt.days

    # Matches in the last N days: per-player, time-indexed rolling count with closed="left"
    # so the current match itself is excluded from its own count (leakage-safe by
    # construction, same idiom as everything else in this module — just using pandas' native
    # time-window support instead of shift+integer-rolling, since day-windows aren't a fixed
    # row count). Implemented per-group via groupby.apply — this is a per-GROUP operation
    # (~7,561 groups), not a per-ROW Python loop, so it stays broadly vectorized.
    def _count_in_window(group: pd.DataFrame, days: int) -> pd.Series:
        s = pd.Series(1, index=pd.DatetimeIndex(group["tourney_date"]))
        counts = s.rolling(f"{days}D", closed="left").count()
        return pd.Series(counts.values, index=group.index)

    for days in day_windows:
        col = f"matches_last_{days}d"
        long_df[col] = 0
        for _, group in long_df.groupby("player_id", sort=False):
            long_df.loc[group.index, col] = _count_in_window(group, days).values


def _add_tournament_context(long_df: pd.DataFrame) -> None:
    g = long_df.groupby("player_id")
    long_df["previous_tourney_level"] = g["tourney_level"].shift(1)
    long_df["previous_round_reached"] = g["round"].shift(1)


def _pivot_to_match_level(long_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    winner_side = long_df[long_df["win"] == 1][["_match_key"] + feature_cols].copy()
    winner_side.columns = ["_match_key"] + [f"winner_{c}" for c in feature_cols]
    loser_side = long_df[long_df["win"] == 0][["_match_key"] + feature_cols].copy()
    loser_side.columns = ["_match_key"] + [f"loser_{c}" for c in feature_cols]
    return winner_side.merge(loser_side, on="_match_key", how="inner")


def compute_day5_features(
    matches_with_elo: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> Day5FeaturesResult:
    df = _sort_chronologically(matches_with_elo)

    n_missing_score = df["score"].isna().sum()
    df = _parse_all_scores(df)
    n_unparseable = (~df["_score_parse_ok"]).sum() - n_missing_score
    if n_unparseable > 0:
        logger.warning("%d score(s) present but unparseable — game/set features will be "
                        "NaN for these rows.", n_unparseable)

    long_df = _build_long_format(df)

    _add_rolling_overall(long_df, windows)
    _add_rolling_surface(long_df, windows)
    _add_rolling_tournament(long_df, windows)
    _add_opponent_strength(long_df, OPPONENT_STRENGTH_WINDOW)
    _add_momentum(long_df, RECENT_MATCH_WINDOWS_DAYS)
    _add_tournament_context(long_df)

    feature_cols = [c for c in long_df.columns if c not in {
        "_match_key", "tourney_date", "_round_rank", "match_num", "tourney_id", "surface",
        "tourney_level", "round", "best_of", "minutes", "opponent_elo_pre", "player_id",
        "opponent_id", "win", "games_won", "games_lost", "sets_won", "sets_lost",
        "game_diff", "_n_sets_played", "_straight_sets", "three_set_win_or_loss",
        "five_set_win_or_loss", "tourney_name",
    }]

    pivoted = _pivot_to_match_level(long_df, feature_cols)
    result_df = df.merge(pivoted, on="_match_key", how="left")
    result_df = result_df.drop(columns=["_round_rank", "_match_key"], errors="ignore")

    # Head-to-head (overall + tournament-specific): computed on the WIDE dataframe, since
    # H2H inherently needs both players in the same row — see head_to_head_features.py for
    # the full leakage-safety argument (adapted from the previously-unused rolling_stats.py).
    n_before_h2h = len(result_df)
    result_df = add_head_to_head_features(result_df)
    if len(result_df) != n_before_h2h:
        raise AssertionError(
            f"Row count changed while adding head-to-head features: "
            f"{n_before_h2h:,} -> {len(result_df):,}. Investigate before trusting this output."
        )

    diagnostics = {
        "processed_matches": len(result_df),
        "players_tracked": long_df["player_id"].nunique(),
        "score_missing": int(n_missing_score),
        "score_unparseable": int(n_unparseable),
        "score_parse_rate": float(df["_score_parse_ok"].mean()),
    }

    return Day5FeaturesResult(augmented=result_df, diagnostics=diagnostics)