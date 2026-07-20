"""
serve_return_features.py — Day 6: leakage-safe rolling serve/return statistics from MCP
point-level charting data, joined against the FROZEN Day 1-2 TML<->MCP match join.

SCOPE LIMITATION (real, not hidden): point-level features can only exist for the 5,988
matches already in the frozen join (79.1% of MCP's charted matches — see
docs/join_pipeline_v1_freeze.md). This is a hard ceiling on how many of the 198,062 total
TML matches can ever have serve/return features attached; everywhere else they are NaN by
necessity, not a bug. Rolling windows are also necessarily much sparser here than Day 5's
match-level windows, since a player typically has far fewer CHARTED matches than total
matches — both last-10 AND full career-to-date rolling averages are computed for this
reason (a last-10 window may have very few real observations for most players).

SCHEMA NOTE: charting-m-stats-Overview.csv's `bk_pts`/`bp_saved` columns are assumed to
mean "break points faced while serving" / "of those, how many were saved" — consistent with
TML/Sackmann's standard bpFaced/bpSaved convention. This assumption is checked explicitly
in the pipeline diagnostics (tour-average break-point-save rate should land near the
historically-known ~60-65% range) rather than silently trusted.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tennis_intel.ratings.processor import _round_rank

logger = logging.getLogger(__name__)

DEFAULT_WINDOWS = (10,)  # last-10 only for point-level (see sparsity note above);
                         # career-to-date is computed unconditionally alongside it


def load_and_prepare_key_points_stats(
    serve_stats_path, return_stats_path,
) -> pd.DataFrame:
    """
    Loads charting-m-stats-KeyPointsServe.csv and charting-m-stats-KeyPointsReturn.csv,
    filters each to the break-point-specific row (confirmed via direct inspection of real
    sample rows, not assumed: 'BP' = break points faced while serving, in KeyPointsServe;
    'BPO' = break point opportunities while returning, in KeyPointsReturn — distinct from
    'GP'/'GPF' [game points] and 'Deuce'/'DeuceR', and from 'STotal'/'RTotal', which are
    match-wide rollups, not break-point-specific), and computes the per-player-per-match
    break-point-serve and break-point-return win rates.

    LEAKAGE NOTE (same standard as load_and_prepare_stats above): this function computes
    a RAW, match-level rate from THIS match's own charted points — exactly like
    first_serve_win_pct, bp_saved_pct, etc. above. It carries no leakage risk on its own;
    leakage safety comes entirely from the DOWNSTREAM shift(1)-then-expanding() step in
    compute_rolling_serve_return_features, which excludes the current match from its own
    rolling/career window — this function's output must be fed into that same pipeline,
    not used as a raw per-match feature directly.

    A player who never faced a break point while serving (or never had a break-point
    opportunity while returning) in a given match has pts=0 for that row, producing a
    0/0 -> NaN rate — handled naturally by pandas division, not a bug requiring a
    fallback here (the existing bp_saved_pct above has the identical property when
    bk_pts=0).
    """
    serve_raw = pd.read_csv(serve_stats_path)
    return_raw = pd.read_csv(return_stats_path)

    serve_bp = serve_raw[serve_raw["row"] == "BP"].copy()
    return_bpo = return_raw[return_raw["row"] == "BPO"].copy()

    for name, df in [("KeyPointsServe (BP)", serve_bp), ("KeyPointsReturn (BPO)", return_bpo)]:
        n_dupes = df.duplicated(subset=["match_id", "player"]).sum()
        if n_dupes:
            logger.warning(
                "%d duplicate (match_id, player) '%s' row(s) found — keeping first "
                "occurrence only, same handling as load_and_prepare_stats's Total-row "
                "duplicates.", n_dupes, name,
            )

    serve_bp = serve_bp.drop_duplicates(subset=["match_id", "player"], keep="first")
    return_bpo = return_bpo.drop_duplicates(subset=["match_id", "player"], keep="first")

    serve_bp["bp_serve_win_pct"] = serve_bp["pts_won"] / serve_bp["pts"]
    return_bpo["bp_return_win_pct"] = return_bpo["pts_won"] / return_bpo["pts"]

    # Merge serve-side and return-side onto one row per (match_id, player) — a player's
    # break-point-serving rate and break-point-returning rate are logically independent
    # quantities from the same match, both wanted on the same downstream row.
    merged = serve_bp[["match_id", "player", "pts", "bp_serve_win_pct"]].merge(
        return_bpo[["match_id", "player", "pts", "bp_return_win_pct"]],
        on=["match_id", "player"], how="outer", suffixes=("_serve", "_return"),
    )
    return merged


def load_and_prepare_stats(stats_path) -> pd.DataFrame:
    """Loads charting-m-stats-Overview.csv, keeps only the per-match Total row (not the
    per-set breakdown), and computes the per-player-per-match rate features.

    KNOWN DATA-QUALITY GAP (confirmed via direct inspection, not assumed): a small number
    of (match_id, player) combinations have more than one "Total" row in the raw MCP file
    (24 out of 15,116 rows, ~0.16% — likely a charting/data-entry duplicate upstream in
    MCP's own file, not something introduced by this pipeline). Deduplicated deterministically
    by keeping the first occurrence, logged explicitly rather than silently dropped, since
    an undeduplicated key here previously caused a real merge fan-out bug (see
    build_day6_features.py's row-count safety assertion, which caught it).
    """
    raw = pd.read_csv(stats_path)
    totals = raw[raw["set"] == "Total"].copy()

    n_dupes = totals.duplicated(subset=["match_id", "player"]).sum()
    if n_dupes:
        logger.warning(
            "%d duplicate (match_id, player) Total row(s) found in the raw stats file "
            "(%.2f%% of %d rows) — keeping first occurrence only. This is an upstream MCP "
            "data-quality characteristic, not a bug introduced here.",
            n_dupes, 100 * n_dupes / len(totals), len(totals),
        )
        totals = totals.drop_duplicates(subset=["match_id", "player"], keep="first")

    totals["first_serve_in_pct"] = totals["first_in"] / totals["serve_pts"]
    totals["first_serve_win_pct"] = totals["first_won"] / totals["first_in"]
    totals["second_serve_win_pct"] = totals["second_won"] / totals["second_in"]
    totals["ace_rate"] = totals["aces"] / totals["serve_pts"]
    totals["df_rate"] = totals["dfs"] / totals["serve_pts"]
    totals["bp_saved_pct"] = totals["bp_saved"] / totals["bk_pts"]
    totals["return_pts_won_pct"] = totals["return_pts_won"] / totals["return_pts"]

    # Break points CONVERTED (while returning) = opponent's break points faced minus
    # opponent's break points saved, in the same match. Requires pairing both players'
    # rows for a given match_id.
    per_match = totals.set_index(["match_id", "player"])
    bp_converted = []
    for (match_id, player), row in per_match.iterrows():
        opponents = totals[(totals["match_id"] == match_id) & (totals["player"] != player)]
        if len(opponents) != 1:
            bp_converted.append(np.nan)
            continue
        opp = opponents.iloc[0]
        opp_bk_pts = opp["bk_pts"]
        bp_converted.append((opp_bk_pts - opp["bp_saved"]) / opp_bk_pts if opp_bk_pts > 0 else np.nan)
    totals["bp_converted_pct"] = bp_converted

    return totals


def attach_player_ids_and_chronology(
    stats: pd.DataFrame, frozen_join: pd.DataFrame
) -> pd.DataFrame:
    """Maps each stats row's raw `player` name string to a player_id and attaches the TML
    chronology fields, using ONLY the frozen join's already-resolved name->id mapping — no
    new fuzzy matching is introduced here, reusing Day 3's frozen resolution exactly."""
    name_to_id: dict[tuple[str, str], str] = {}
    for _, row in frozen_join.iterrows():
        name_to_id[(row["mcp_match_id"], row["mcp_Player 1"])] = (
            row["tml_winner_id"] if row["mcp_player1_norm"] == row["tml_winner_name_norm"] else row["tml_loser_id"]
        )
        name_to_id[(row["mcp_match_id"], row["mcp_Player 2"])] = (
            row["tml_winner_id"] if row["mcp_player2_norm"] == row["tml_winner_name_norm"] else row["tml_loser_id"]
        )

    stats = stats.copy()
    stats["player_id"] = stats.apply(
        lambda r: name_to_id.get((r["match_id"], r["player"])), axis=1
    )

    chronology_cols = ["mcp_match_id", "tml_tourney_id", "tml_tourney_date", "tml_round", "tml_match_num"]
    chronology = frozen_join[chronology_cols].drop_duplicates(subset="mcp_match_id")
    stats = stats.merge(chronology, left_on="match_id", right_on="mcp_match_id", how="inner")

    n_unresolved = stats["player_id"].isna().sum()
    if n_unresolved:
        logger.warning("%d stats row(s) could not be mapped to a player_id — dropping.", n_unresolved)
        stats = stats.dropna(subset=["player_id"])

    return stats


def compute_elo_trend_features(
    day5: pd.DataFrame, windows: tuple[int, ...] = (10, 20, 50),
) -> pd.DataFrame:
    """
    Elo-trend features (elo_change_lastN): a player's Elo change over their last N
    matches, distinct from static elo_pre_match_winner/loser (which the model already
    has) — captures whether a player is currently trending up or down, not just their
    absolute current strength.

    OPERATES DIRECTLY on day5's TML-native structure (winner_id/loser_id/tourney_date
    already present), NOT via attach_player_ids_and_chronology — that function exists
    specifically to resolve MCP's raw player-name strings to TML ids using the frozen
    join, which is unnecessary here since day5 already has TML ids natively (Elo is
    computed directly from TML match data, never touches MCP at all).

    Carries tourney_id/match_num, winner_id/loser_id (BOTH, regardless of which side
    player_id represents), and an explicit is_winner_row boolean, through into the
    returned long-form table (alongside player_id/tourney_date) — is_winner_row lets the
    caller split cleanly back into winner-side/loser-side without ambiguity (the table
    itself only has player_id, not separate winner_id/loser_id columns to compare
    against, since each row already represents one specific player's perspective).
    winner_id/loser_id BOTH being present lets the caller merge this back onto day5
    using the FULL, already-established (tourney_id, match_num, winner_id, loser_id)
    key used everywhere else in this pipeline — an EARLIER version of this function
    carried only tourney_id/match_num and assumed that pair alone was a unique match
    identifier, which was NEVER actually verified and caused a real, caught fan-out bug
    (198,062 -> 198,894 rows) the first time this ran against the real dataset. Do not
    narrow the merge key again without first directly confirming uniqueness on the real
    data, not assuming it from a subset of the established key.

    LEAKAGE SAFETY — a DIFFERENT argument than every other _career/_last10 feature in
    this file, worth being explicit about rather than assuming the same shift(1)
    reasoning applies unchanged: elo_pre_match_winner/loser for a given match is, by
    construction, ALREADY the rating entering that match — not that match's own
    outcome. It is not leaky even used directly (already a confirmed, existing feature
    in the model). Therefore elo_pre_match[i] - elo_pre_match[i-N] is a difference of
    TWO already-prior, already-valid values — genuinely leakage-safe WITHOUT an
    additional shift(1), unlike the rate features in this file, whose raw per-match
    input WAS that match's own outcome and required shift(1) specifically to exclude it.
    """
    id_cols = ["tourney_id", "match_num", "winner_id", "loser_id"]
    as_winner = day5[["winner_id", "tourney_date", "elo_pre_match_winner"] + [c for c in id_cols if c != "winner_id"]].copy()
    as_winner["player_id"] = as_winner["winner_id"]
    as_winner = as_winner.rename(columns={"elo_pre_match_winner": "elo_pre_match"})
    as_winner["is_winner_row"] = True

    as_loser = day5[["loser_id", "tourney_date", "elo_pre_match_loser"] + [c for c in id_cols if c != "loser_id"]].copy()
    as_loser["player_id"] = as_loser["loser_id"]
    as_loser = as_loser.rename(columns={"elo_pre_match_loser": "elo_pre_match"})
    as_loser["is_winner_row"] = False

    long_form = pd.concat([as_winner, as_loser], ignore_index=True)
    long_form = long_form.sort_values(
        ["player_id", "tourney_date"], kind="mergesort"
    ).reset_index(drop=True)

    for window in windows:
        col = f"elo_change_last{window}"
        long_form[col] = long_form.groupby("player_id")["elo_pre_match"].transform(
            lambda s: s - s.shift(window)
        )

    return long_form


def compute_rolling_serve_return_features(
    stats_with_ids: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_WINDOWS,
    rate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Leakage-safe rolling (last-N and career-to-date) serve/return rates per player, using
    the exact same shift-then-roll idiom proven in Day 5 (shift(1) excludes the current
    match from its own window by construction).

    rate_cols: which raw, per-match rate columns to roll. Defaults to the original
    Overview.csv-derived set (unchanged, so every existing caller behaves identically
    without passing this argument). Pass a different list — e.g. the new
    bp_serve_win_pct/bp_return_win_pct from load_and_prepare_key_points_stats — to reuse
    this exact same leakage-safe sorting/shift/roll logic on a different stats source,
    rather than duplicating it in a second function."""
    df = stats_with_ids.copy()
    df["_round_rank"] = _round_rank(df["tml_round"])
    df = df.sort_values(
        ["player_id", "tml_tourney_date", "_round_rank", "tml_match_num", "tml_tourney_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if rate_cols is None:
        rate_cols = ["first_serve_in_pct", "first_serve_win_pct", "second_serve_win_pct",
                     "ace_rate", "df_rate", "bp_saved_pct", "return_pts_won_pct", "bp_converted_pct"]

    g = df.groupby("player_id")
    for col in rate_cols:
        shifted = g[col].shift(1)
        # Career-to-date (expanding, min_periods=1)
        df[f"{col}_career"] = shifted.groupby(df["player_id"]).expanding(min_periods=1).mean().reset_index(drop=True)
        for w in windows:
            df[f"{col}_last{w}"] = shifted.groupby(df["player_id"]).rolling(w, min_periods=1).mean().reset_index(drop=True)

    return df.drop(columns=["_round_rank"])