"""
point_level_features.py — Day 7: builds the leakage-safe point-by-point state dataset from
MCP's raw point-charting files.

Pipeline: load + sort (match_id, Pt) -> parse score state -> compute break/set/match-point
flags -> compute leakage-safe in-match momentum. Score-notation grounding is documented in
point_score_parser.py's module docstring (verified against real data, not assumed).

LEAKAGE DISCIPLINE: in-match momentum uses shift(1) + rolling within each match_id group —
the same idiom proven in Day 5, but critically DIFFERENT in scope: this resets per match
(momentum is meaningless carried across different matches), whereas Day 5's rolling form
was deliberately carried ACROSS a player's match history. Conflating the two would be a
real, distinct leakage-adjacent bug — documented explicitly here so it isn't repeated.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tennis_intel.features.point_score_parser import (
    parse_pts, is_break_point, is_set_point, is_match_point,
)

logger = logging.getLogger(__name__)

MOMENTUM_WINDOWS = (10, 20)


def load_and_sort_points(paths: list) -> pd.DataFrame:
    """Loads one or more charting-*-points-*.csv files and sorts by (match_id, Pt) — the
    raw files are NOT chronologically sorted by default (confirmed via direct inspection),
    so this step is mandatory before any leakage-safe processing."""
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["match_id", "Pt"], kind="mergesort").reset_index(drop=True)
    return df


def _vectorized_parse_pts(pts_series: pd.Series, is_tb_series: pd.Series,
                           svr_series: pd.Series) -> pd.DataFrame:
    """
    Vectorizes parse_pts across the whole dataframe by computing unique (Pts, is_tiebreak)
    combinations once, then broadcasting — avoids calling parse_pts 1.28M times in Python.
    Same logic as parse_pts(), just applied via pandas string ops instead of per-row calls.

    TIEBREAK NOTATION FIX (found via single-match replay + validated at scale, 2026-07):
    within a tiebreak, `Pts` is SERVER-FIRST notation (first number = the CURRENT SERVER's
    count, second = the receiver's), NOT fixed player1/player2 notation. The original
    implementation treated it as fixed player1/player2, which produced physically
    impossible score jumps (e.g. 6-0 -> 0-7 in one point) whenever the server changed.

    Validated across the FULL dataset (not just one match) by cross-checking against
    PtWinner, which is unambiguous regardless of Pts convention: fixed player1/player2
    interpretation matches PtWinner-implied transitions only 50.4% of the time (chance
    level — confirms it was wrong); server-first interpretation matches 97.7% of the time
    (44,012 real transitions checked), with the residual ~2.3% attributable to genuine
    charting errors, consistent with the error rate already documented elsewhere in this
    dataset (192 unparseable regular-game scores, ~0.02%). Regular-game scores are NOT
    affected — Day 7's original fixed player1/player2 decoding for regular games was
    independently validated against PtWinner and remains correct; this fix is tiebreak-only.
    """
    # Regular game point ordinals
    ordinal = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}

    pts_clean = pts_series.astype(str).str.strip()

    # Split on '-' to get the two printed numbers (meaning depends on regular vs tiebreak)
    split = pts_clean.str.split("-", n=1, expand=True)
    split.columns = ["first_raw", "second_raw"]

    # Regular game scores: fixed player1/player2, unaffected by this fix
    p1_reg = split["first_raw"].map(ordinal)
    p2_reg = split["second_raw"].map(ordinal)

    # Tiebreak scores: server-first. first_raw = server's count, second_raw = receiver's.
    first_tb = pd.to_numeric(split["first_raw"], errors="coerce")
    second_tb = pd.to_numeric(split["second_raw"], errors="coerce")
    svr_is_p1 = (svr_series == 1)
    # If server is P1: p1=first (server's count), p2=second (receiver's count).
    # If server is P2: p1=second (receiver's count), p2=first (server's count).
    p1_tb = first_tb.where(svr_is_p1, second_tb)
    p2_tb = second_tb.where(svr_is_p1, first_tb)

    parse_ok_reg = ~is_tb_series & p1_reg.notna() & p2_reg.notna()
    parse_ok_tb = is_tb_series & p1_tb.notna() & p2_tb.notna()
    parse_ok = parse_ok_reg | parse_ok_tb

    p1_points = pd.Series(index=pts_series.index, dtype="Float64")
    p2_points = pd.Series(index=pts_series.index, dtype="Float64")
    tb_p1 = pd.Series(index=pts_series.index, dtype="Float64")
    tb_p2 = pd.Series(index=pts_series.index, dtype="Float64")

    p1_points[parse_ok_reg] = p1_reg[parse_ok_reg]
    p2_points[parse_ok_reg] = p2_reg[parse_ok_reg]
    tb_p1[parse_ok_tb] = p1_tb[parse_ok_tb]
    tb_p2[parse_ok_tb] = p2_tb[parse_ok_tb]

    return pd.DataFrame({
        "p1_points": p1_points,
        "p2_points": p2_points,
        "tb_p1_points": tb_p1,
        "tb_p2_points": tb_p2,
        "score_parse_ok": parse_ok,
    }, index=pts_series.index)


def _vectorized_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorizes break-point, set-point, and match-point detection using pandas column ops.
    All flag logic mirrors point_score_parser.py exactly — same rules, no Python loops.
    """
    p1 = df["p1_points"].fillna(-1).astype(int)
    p2 = df["p2_points"].fillna(-1).astype(int)
    tb_p1 = df["tb_p1_points"].fillna(-1).astype(int)
    tb_p2 = df["tb_p2_points"].fillna(-1).astype(int)
    tb = df["is_tiebreak_game"]
    ok = df["score_parse_ok"]
    svr_p1 = df["Svr"] == 1
    gm1 = df["Gm1"].fillna(0).astype(int)
    gm2 = df["Gm2"].fillna(0).astype(int)
    sets1 = df["Set1"].fillna(0).astype(int)
    sets2 = df["Set2"].fillna(0).astype(int)
    best_of = df["best_of"].fillna(3).astype(int)
    sets_needed = (best_of // 2) + 1

    # --- Break point (server perspective) ---
    # Returner at advantage (4) or at 40 (3) with server below 40
    ret_p = p2.where(svr_p1, p1)  # returner's points
    srv_p = p1.where(svr_p1, p2)  # server's points
    ret_tb = tb_p2.where(svr_p1, tb_p1)
    bp = ok & ~tb & (
        (ret_p == 4) |  # returner at advantage
        ((ret_p == 3) & (srv_p < 3))  # 40 to server's 0/15/30
    )

    # --- Would win game next point? ---
    # Regular game: server at 40 (3) with returner < 40, or server at AD (4)
    srv_wins_game_reg = ok & ~tb & (
        (srv_p == 4) |
        ((srv_p == 3) & (ret_p < 3))
    )
    ret_wins_game_reg = ok & ~tb & (
        (ret_p == 4) |
        ((ret_p == 3) & (srv_p < 3))
    )
    # Tiebreak: either player at (target-1)+ with 2-clear, next point wins
    srv_tb_pts = tb_p1.where(svr_p1, tb_p2)
    ret_tb_pts = tb_p2.where(svr_p1, tb_p1)
    srv_wins_game_tb = ok & tb & (
        ((srv_tb_pts + 1) >= 7) & ((srv_tb_pts + 1 - ret_tb_pts) >= 2)
    )
    ret_wins_game_tb = ok & tb & (
        ((ret_tb_pts + 1) >= 7) & ((ret_tb_pts + 1 - srv_tb_pts) >= 2)
    )
    srv_wins_game = srv_wins_game_reg | srv_wins_game_tb
    ret_wins_game = ret_wins_game_reg | ret_wins_game_tb

    # --- Would win set by winning game? ---
    # Server's current games
    srv_games = gm1.where(svr_p1, gm2)
    ret_games = gm2.where(svr_p1, gm1)
    srv_games_after = srv_games + 1
    ret_games_after = ret_games + 1

    def wins_set(games_after, opp_games):
        return (
            ((games_after >= 6) & ((games_after - opp_games) >= 2)) |
            ((games_after == 7) & (opp_games == 6))
        )

    srv_wins_set_by_game = srv_wins_game & wins_set(srv_games_after, ret_games)
    ret_wins_set_by_game = ret_wins_game & wins_set(ret_games_after, srv_games)

    # --- Set point: either player one point from winning game AND set ---
    sp = srv_wins_set_by_game | ret_wins_set_by_game

    # --- Match point: one point from winning game, set, AND match ---
    srv_sets = sets1.where(svr_p1, sets2)
    ret_sets = sets2.where(svr_p1, sets1)
    mp = (
        (srv_wins_set_by_game & ((srv_sets + 1) >= sets_needed)) |
        (ret_wins_set_by_game & ((ret_sets + 1) >= sets_needed))
    )

    # NEW (pressure_index, 2026-07): additive columns, not touched by any existing
    # caller of _vectorized_flags — reuses the SAME srv_wins_game intermediate already
    # computed above for set-point detection, rather than recomputing the "would the
    # server win this game" logic a second time.
    is_server_game_point = srv_wins_game & ~bp & ~sp & ~mp
    # Deuce-level: both players have reached at least 40 (3 in the 0/15/30/40 point
    # count) without either side yet being one point from winning the game — i.e. deuce
    # itself, or advantage-server/advantage-returner in a regular (non-tiebreak) game.
    is_deuce_level = ok & ~tb & (ret_p >= 3) & (srv_p >= 3) & ~srv_wins_game & ~ret_wins_game

    return pd.DataFrame({
        "is_break_point": bp.fillna(False).astype(bool),
        "is_set_point": sp.fillna(False).astype(bool),
        "is_match_point": mp.fillna(False).astype(bool),
        "is_server_game_point": is_server_game_point.fillna(False).astype(bool),
        "is_deuce_level": is_deuce_level.fillna(False).astype(bool),
    }, index=df.index)


def compute_point_state(df: pd.DataFrame, best_of_map: dict) -> pd.DataFrame:
    """
    Adds parsed score state and situational flags to every point row.
    Fully vectorized — no Python loops over rows.

    best_of_map: {match_id: best_of} — required for match-point detection, sourced from
    the frozen match-level data (charting-m-matches.csv / TML), not re-derived here.
    """
    df = df.copy()

    df["is_tiebreak_game"] = (df["Gm1"] == 6) & (df["Gm2"] == 6)
    df["best_of"] = df["match_id"].map(best_of_map)

    # NEW (2026-07, following the points-remaining-controlled investigation into the
    # deciding-set log loss gap): a genuine, structural finding survived conditioning on
    # points_remaining — at MATCHED points-remaining, deciding-set points are still
    # substantially harder to call correctly across all three engines (e.g. the 25-50
    # points-remaining bin: 0.146 non-deciding vs 0.681 deciding log loss for the
    # smoothed engine, a ~4.6x gap, at the SAME horizon length). This means deciding-set
    # status carries real predictive information the classifier currently has no way to
    # use, since no feature previously told it "this is a deciding set" — every prior
    # and posterior was built purely from career rates blind to this context. Added here,
    # vectorized, using the exact same formula already validated across every diagnostic
    # script in that investigation (is_deciding_set / classify_point's is_deciding).
    sets_needed = (df["best_of"] // 2) + 1
    df["deciding_set"] = (df["Set1"] == df["Set2"]) & (df["Set1"] == (sets_needed - 1))

    # NEW (2026-07, following the deciding_set permutation-importance null result —
    # rank 23 of 35, importance -0.000002, indistinguishable from zero): a binary
    # deciding-set flag gave the classifier nothing to exploit, which is consistent with
    # fatigue and same-day-quality-gap effects being fundamentally about MAGNITUDE, not a
    # yes/no switch a binary flag can express. Adding graded, continuous fatigue proxies
    # instead, using EXACT derivations rather than an approximation:
    #   - points_played_so_far_in_match: Pt - 1, directly from the point's own
    #     already-sorted sequential index (see load_and_sort_points) — exact, not
    #     positional/cumcount-based, so it stays correct even if this function is ever
    #     called on a differently-ordered subset of rows.
    #   - sets_played_so_far_in_match: Set1 + Set2, exact count of completed sets.
    # Deliberately NOT attempting an exact games_played_so_far_in_match: completed sets
    # (Set1/Set2) don't record how many games each one actually contained (a 6-0 set and
    # a 7-6 set both just increment the set count by one), so an exact count isn't
    # derivable from currently-available columns without approximation. points_played
    # arguably captures effort more faithfully anyway, since it reflects long deuce games
    # that a games-only count would treat identically to quick ones.
    df["points_played_so_far_in_match"] = df["Pt"] - 1
    df["sets_played_so_far_in_match"] = df["Set1"] + df["Set2"]

    # Score parsing — vectorized across all unique (Pts, is_tiebreak) combinations
    parsed = _vectorized_parse_pts(df["Pts"], df["is_tiebreak_game"], df["Svr"])
    df = pd.concat([df, parsed], axis=1)

    n_bad = (~df["score_parse_ok"]).sum()
    if n_bad:
        logger.warning("%d point(s) had unparseable score notation — flags will be False/NaN.",
                       n_bad)

    # Flag computation — vectorized
    flags = _vectorized_flags(df)
    df = pd.concat([df, flags], axis=1)

    df["is_second_serve_point"] = df["2nd"].notna()

    # NEW (pressure_index, 2026-07): a single, ordinal "how much is riding on this point"
    # variable, replacing the need to separately weigh several binary flags. Priority
    # order matters — a point can satisfy multiple conditions at once (e.g. a match
    # point is ALSO technically a set point and a break point by construction), so this
    # assigns each point to its SINGLE highest tier via np.select's first-match-wins
    # evaluation order, rather than summing flags (which would double-count and make the
    # scale meaningless — a match point should not score higher just because it happens
    # to also be a break point, since "match point" already implies that).
    #
    # Tier weights are a hand-chosen ordinal scale, not derived from data — deliberately
    # simple and monotonic (higher stakes = higher number) rather than an attempt at a
    # precisely calibrated leverage measure. A genuinely calibrated version (e.g. actual
    # measured win-probability swing per tier) is a natural future refinement once this
    # simpler version is validated via permutation importance, matching the same
    # discipline used for every other feature added this project (build simple, measure,
    # only add complexity if the simple version proves it's worth it).
    pressure_conditions = [
        df["is_match_point"],
        df["is_set_point"],
        df["is_break_point"],
        df["is_server_game_point"],
        df["is_deuce_level"],
    ]
    pressure_values = [10, 8, 5, 3, 2]
    df["pressure_index"] = np.select(pressure_conditions, pressure_values, default=1)

    return df


def compute_in_match_momentum(df: pd.DataFrame, windows: tuple = MOMENTUM_WINDOWS) -> pd.DataFrame:
    """
    Leakage-safe rolling point-win-rate for player 1, WITHIN each match (resets per match,
    unlike Day 5's across-match rolling form). Player 2's rate for the same window is
    exactly 1 - player 1's rate, since for every point exactly one of them wins it — no
    separate computation needed, but exposed explicitly for clarity of downstream use.

    CONVENTION, SETTLED 2026-07 (see ml_informed_markov.py's ml_informed_markov_predict
    docstring and docs/ptwinner_convention_correction.md for the full investigation):
    PtWinner is LITERAL, fixed-player-relative — PtWinner==1 means player 1 won the
    point, PERIOD, independent of who served. A same-day "fix" here previously claimed
    the opposite (server-relative), citing check_ptwinner_disagreement_at_scale.py's
    "0.00% disagreement" — that script only checks internal self-consistency between
    PtWinner and fixed-player Pts on INTERIOR (non-game-boundary) points, which cannot
    distinguish "PtWinner is server-relative" from "PtWinner is literal" (the two are
    mirror images that only diverge at Svr==2, exactly what that script never examines).
    Checked directly against Gm1/Gm2 at real game boundaries instead: literal PtWinner
    matches at 99.91% corpus-wide, symmetric across Svr==1/2; server-relative matches
    only ~51% (chance) at boundaries. `p1_won_point` is therefore simply
    `PtWinner == 1` — no Svr cross-reference needed or wanted.
    """
    df = df.copy()
    df["p1_won_point"] = (df["PtWinner"] == 1).astype(int)

    g = df.groupby("match_id")
    for w in windows:
        shifted = g["p1_won_point"].shift(1)
        df[f"p1_momentum_last{w}"] = shifted.groupby(df["match_id"]).rolling(w, min_periods=1).mean().reset_index(drop=True)
        df[f"p2_momentum_last{w}"] = 1 - df[f"p1_momentum_last{w}"]

    return df


def _signed_run_length(shifted: pd.Series, group_key: pd.Series) -> pd.Series:
    """
    Shared run-length core, factored out of compute_consecutive_points_streak so every
    streak variant (points, serve-only, return-only, games) reuses the SAME validated
    logic rather than re-deriving it with fresh chances to reintroduce a subtle bug.

    shifted: a 0/1/NaN series, ALREADY shift(1)-ed and ALREADY restricted to whichever
    subset of rows this streak variant cares about (e.g. only server-is-p1 rows for a
    serve streak) — this function does not know or care what the subset represents, it
    only computes a signed run-length over whatever series it's given.
    group_key: the grouping key (typically match_id) at the SAME row alignment as
    shifted, used only to detect group boundaries — a group boundary always forces a new
    streak, regardless of whether the raw value happened to repeat across it.
    """
    is_new_group = group_key != group_key.shift(1)
    changed = (shifted != shifted.shift(1)) | is_new_group
    streak_id = changed.cumsum()
    raw_length = shifted.groupby(streak_id).cumcount() + 1
    has_no_history = shifted.isna()
    streak_length = raw_length.where(~has_no_history, 0)
    sign = np.where(shifted.fillna(0) >= 0.5, 1, -1)
    return (streak_length * sign).astype(int)


def compute_consecutive_points_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leakage-safe consecutive-points-won streak (points_streak), a SIGNED, single-column
    feature: positive N means player 1 has won their last N consecutive points entering
    this point; negative N means player 2 has (i.e. player 1 is on an N-point losing
    streak). Resets per match, same scope as in-match momentum. Mixes serve and return
    points together — see compute_split_points_streak below for the serve/return-
    separated variant.

    DISTINCT FROM p1/p2_momentum_last10/20 (already in the model): momentum is a rolling
    WIN RATE over a fixed window (e.g. "won 7 of the last 10 points"), which can be high
    even with an interrupted, back-and-forth sequence. Streak captures something momentum
    cannot: an UNBROKEN run's actual length ("won the last 4 points in a row"), which is
    the more literal notion of "currently hot" a human would describe watching the match.

    LEAKAGE SAFETY: shift(1)-then-run-length via the shared _signed_run_length helper —
    the streak entering point i is computed from points STRICTLY BEFORE i (shift(1)
    excludes point i's own not-yet-known outcome). The first point of each match has no
    prior points, so its streak is 0 (a real, correct "no streak yet" value, not a leaky
    default) — NOT NaN, since a streak length of zero points is genuinely, unambiguously
    zero, unlike momentum's NaN-until-enough-history (a rate over zero points is
    genuinely undefined).

    CONVENTION (same as compute_in_match_momentum above, settled 2026-07 — see that
    function's own docstring, ml_informed_markov.py, and
    docs/ptwinner_convention_correction.md for the full investigation): PtWinner is
    LITERAL, fixed-player-relative (PtWinner==1 means player 1 won, period). A same-day
    "fix" here previously combined it with Svr on the (now-reverted) assumption that
    PtWinner was server-relative; that assumption was refuted against Gm1/Gm2 at real
    game boundaries (99.91% match for literal vs. ~51%/chance for server-relative).
    """
    df = df.copy()
    df["p1_won_point"] = (df["PtWinner"] == 1).astype(int)
    shifted = df.groupby("match_id")["p1_won_point"].shift(1)
    df["points_streak"] = _signed_run_length(shifted, df["match_id"])
    return df


def compute_split_points_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Serve-streak and return-streak for player 1, SEPARATED — unlike points_streak
    (compute_consecutive_points_streak above), which mixes serve and return points into
    one signed count. A player on a hot SERVING streak and a player on a hot RETURNING
    streak are different situations (serve is the higher-base-rate activity — most
    players win 55-65% of serve points — so a serve streak of length 4 is less
    surprising than a return streak of length 4); collapsing them into one number
    discards this distinction.

    Adds: p1_serve_streak, p1_return_streak (player 2's are always the negation, same
    convention as points_streak).

    DESIGN — a genuinely subtle bug was caught here before shipping, worth documenting
    in full since a naive version looks plausible and passes casual inspection: a
    single-series forward-fill is WRONG. p1_serve_streak at a SERVE point's own row must
    use shift(1) (excluding that point's own not-yet-known outcome, same as every other
    leakage-safe feature). But p1_serve_streak at a LATER, non-serve (return) row must
    reflect the OUTCOME of the most recent serve point, which by then has ALREADY
    happened and is known — using the same shift(1)'d value there is WRONG, since it
    describes the state entering the serve point, not the state resulting from it. This
    was caught directly by a permanent test (test_forward_fill_persists_through_
    opposite_type_points): a single win at the first serve point produced
    p1_serve_streak=0 at every subsequent return point (correct would be +1) with the
    naive single-series design, because it was propagating the "entering the serve
    point" value instead of the "resulting from the serve point" value.

    FIX: two separate run-length series are computed on the SAME filtered subset — one
    shift(1)'d (used only at the serve rows' own feature values), one NOT shifted (used
    only for forward-filling onto later, non-serve rows, placed starting one row after
    each serve point via a match-grouped shift(1) on the "after" series itself).

    LEAKAGE SAFETY of the "after" (un-shifted) series is NOT a violation despite not
    being shift(1)'d: it is placed and forward-filled starting strictly AFTER the serve
    point it describes (via the match-grouped shift(1) on placement, not on the streak
    value itself) — every row that ends up seeing a given "after" value is
    chronologically LATER than the serve point that produced it, so no row ever sees
    its own or a future point's outcome.
    """
    df = df.copy()
    # CONVENTION (same as compute_in_match_momentum above, settled 2026-07 — see that
    # function's own docstring for the full investigation): PtWinner is LITERAL,
    # fixed-player-relative. A same-day "fix" here previously combined it with Svr on
    # the now-reverted server-relative assumption.
    p1_won = (df["PtWinner"] == 1).astype(int)
    p1_serving = df["Svr"] == 1

    for label, mask in [("serve", p1_serving), ("return", ~p1_serving)]:
        subset_idx = df.index[mask]
        subset_won = p1_won[mask]
        subset_match = df.loc[mask, "match_id"]

        # Value AT the subset row itself: "entering" (shift(1)'d), leakage-safe re:
        # that row's own not-yet-known outcome.
        shifted_for_own = subset_won.groupby(subset_match).shift(1)
        streak_at_own = _signed_run_length(shifted_for_own, subset_match)

        # Value to forward-fill onto LATER, opposite-type rows: NOT shift(1)'d, since
        # it already correctly incorporates that subset row's own, by-then-known
        # outcome — safe because of WHERE it gets placed (below), not because the
        # value itself is somehow exempt from the leakage discipline.
        streak_after = _signed_run_length(subset_won, subset_match)

        own_col = pd.Series(index=df.index, dtype="float64")
        own_col.loc[subset_idx] = streak_at_own.values

        after_col = pd.Series(index=df.index, dtype="float64")
        after_col.loc[subset_idx] = streak_after.values
        df["_after_tmp"] = after_col
        # Shift the "after" value forward by one row's worth of PLACEMENT (grouped by
        # match) so it starts propagating from the row immediately following each
        # subset row, not from the subset row itself.
        df["_after_shifted"] = df.groupby("match_id")["_after_tmp"].shift(1)
        df["_after_ffilled"] = df.groupby("match_id")["_after_shifted"].transform(
            lambda g: g.ffill()
        )

        # combine_first prioritizes own_col (populated only at subset rows) and falls
        # back to the ffilled "after" value everywhere else.
        final = own_col.combine_first(df["_after_ffilled"])
        df[f"p1_{label}_streak"] = final.fillna(0).astype(int)
        df = df.drop(columns=["_after_tmp", "_after_shifted", "_after_ffilled"])

    return df


def compute_games_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    p1_games_streak: a signed, consecutive-GAMES-won run-length — the same idea as
    points_streak, but one level coarser. A player can lose their points_streak on a
    single lost point while still dominating the game overall (e.g. winning it 4-1 on
    points); games_streak captures a different, complementary momentum signal that a
    point-level streak can miss entirely.

    GAME-BOUNDARY DETECTION: a new game has just started at row i when Gm1+Gm2 (the
    total games played so far, ENTERING row i's point — same established convention as
    detect_set_boundaries) differs from row i-1's value, within the same match. Which
    player won the PRECEDING (just-completed) game is determined by whether Gm1 or Gm2
    specifically increased — verified directly against a hand-traced synthetic sequence
    with three real game transitions before being used here.

    DESIGN reuses the EXACT two-series forward-fill pattern proven (and bug-fixed) in
    compute_split_points_streak: a game-won event is sparse (occurs only at the first
    point of each new game), so the streak value must be forward-filled across every
    point WITHIN a game, only updating when the next game actually completes. The same
    "own row" (shift(1)'d) vs. "after" (un-shifted, placed one row later) distinction
    applies for the identical reason — a point mid-game must see the OUTCOME of the
    most recently COMPLETED game, not the state entering that game's own first point.
    """
    df = df.copy()
    games_sum = df["Gm1"].fillna(0) + df["Gm2"].fillna(0)
    is_new_game = (games_sum != games_sum.shift(1)) & (df["match_id"] == df["match_id"].shift(1))
    p1_won_game = (df["Gm1"] > df["Gm1"].shift(1)).astype(int)

    game_boundary_idx = df.index[is_new_game]
    subset_won = p1_won_game[is_new_game]
    subset_match = df.loc[is_new_game, "match_id"]

    # "Own row" value: the streak AT each game-boundary row itself (the first point of
    # the new game), shift(1)'d to exclude that row's own not-yet-known point outcome —
    # this reflects the streak as of the game that JUST completed, which is exactly
    # what a point at the start of a new game should see.
    shifted_for_own = subset_won.groupby(subset_match).shift(1)
    streak_at_own = _signed_run_length(shifted_for_own, subset_match)

    # "After" value: NOT shift(1)'d, used only for forward-filling onto LATER points
    # within the remainder of that same new game (which come chronologically after
    # the game-boundary row and can safely reflect that boundary's own now-known
    # implication for the running streak).
    streak_after = _signed_run_length(subset_won, subset_match)

    own_col = pd.Series(index=df.index, dtype="float64")
    own_col.loc[game_boundary_idx] = streak_at_own.values

    after_col = pd.Series(index=df.index, dtype="float64")
    after_col.loc[game_boundary_idx] = streak_after.values
    df["_after_tmp"] = after_col
    df["_after_shifted"] = df.groupby("match_id")["_after_tmp"].shift(1)
    df["_after_ffilled"] = df.groupby("match_id")["_after_shifted"].transform(
        lambda g: g.ffill()
    )

    final = own_col.combine_first(df["_after_ffilled"])
    df["p1_games_streak"] = final.fillna(0).astype(int)
    df = df.drop(columns=["_after_tmp", "_after_shifted", "_after_ffilled"])

    return df


def compute_in_match_serve_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    p1_in_match_serve_rate, p1_in_match_return_rate: a player's RAW, cumulative
    (expanding, not fixed-window) serve/return win rate SO FAR IN THIS MATCH — distinct
    from both p1_momentum_last10/20 (a fixed-window rolling rate, mixing serve and
    return points together) and the career-level _career rate features (which never
    see today's actual in-match performance at all).

    MOTIVATION: the Beta-Binomial smoothed engine already blends a career-level prior
    against in-match evidence via its own posterior mechanism — but the POINT-LEVEL
    CLASSIFIER itself, upstream of that blend, has never had direct visibility into
    "how has this player actually served/returned in THIS match so far," only their
    career baseline. Giving the classifier this signal directly lets it learn how much
    to trust today's in-match form versus career history contextually (e.g. a big
    deviation from career rate partway through a match may itself be informative,
    which the classifier can only learn if it can see BOTH quantities side by side).

    DESIGN reuses the exact two-series forward-fill pattern proven for
    compute_split_points_streak, with an EXPANDING MEAN in place of a run-length: the
    "own row" value (shift(1)'d, used at a serve/return point's own row) and the
    "after" value (un-shifted, forward-filled onto later points of the opposite type)
    are both expanding means over the same filtered subset, differing only in whether
    that specific row's own outcome is included.

    NaN, NOT a default like 0 or 0.5, when NO serve (or return) points have occurred
    yet in the match — a rate over zero observations is genuinely undefined, the same
    reasoning already established for compute_in_match_momentum's own NaN-until-
    enough-history behavior. This is DELIBERATELY DIFFERENT from the streak features'
    convention (where 0 is a meaningful "no streak yet" value) — a rate and a
    run-length have different natural "no data" semantics, and forcing this rate to a
    fake default would misrepresent genuine absence of evidence as a genuine 50/50 or
    zero rate.
    """
    df = df.copy()
    # CONVENTION (same as compute_in_match_momentum above, settled 2026-07 — see that
    # function's own docstring for the full investigation): PtWinner is LITERAL,
    # fixed-player-relative. A same-day "fix" here previously combined it with Svr on
    # the now-reverted server-relative assumption.
    p1_won = (df["PtWinner"] == 1).astype(int)
    p1_serving = df["Svr"] == 1

    for label, mask in [("serve", p1_serving), ("return", ~p1_serving)]:
        subset_idx = df.index[mask]
        subset_won = p1_won[mask]
        subset_match = df.loc[mask, "match_id"]

        shifted_for_own = subset_won.groupby(subset_match).shift(1)
        own_expanding = shifted_for_own.groupby(subset_match).expanding(min_periods=1).mean().reset_index(level=0, drop=True)

        after_expanding = subset_won.groupby(subset_match).expanding(min_periods=1).mean().reset_index(level=0, drop=True)

        own_col = pd.Series(index=df.index, dtype="float64")
        own_col.loc[subset_idx] = own_expanding.values

        after_col = pd.Series(index=df.index, dtype="float64")
        after_col.loc[subset_idx] = after_expanding.values
        df["_after_tmp"] = after_col
        df["_after_shifted"] = df.groupby("match_id")["_after_tmp"].shift(1)
        df["_after_ffilled"] = df.groupby("match_id")["_after_shifted"].transform(
            lambda g: g.ffill()
        )

        final = own_col.combine_first(df["_after_ffilled"])
        df[f"p1_in_match_{label}_rate"] = final
        df = df.drop(columns=["_after_tmp", "_after_shifted", "_after_ffilled"])

    return df


def compute_in_match_serve_return_rate_rolling(
    df: pd.DataFrame, windows: tuple[int, ...] = (10, 15),
) -> pd.DataFrame:
    """
    p1_in_match_serve_rate_last{N}, p1_in_match_return_rate_last{N}: a FIXED-WINDOW
    (last N serve/return points, not whole-match-so-far) version of
    compute_in_match_serve_return_rate — added specifically because that expanding
    version was confirmed the single strongest new feature family this session
    (ranks 3-4 by permutation importance, 4-7x every career-stat feature), and a
    fixed window may separate "current form within the match" from "match-long
    average" the same way winner_win_pct_last10 exists alongside career win rate —
    the whole-match expanding rate can be slow to reflect a real, recent shift (e.g.
    a player who served poorly for two sets then found their rhythm), which only a
    shorter window would pick up quickly.

    DESIGN is IDENTICAL to compute_in_match_serve_return_rate's two-series
    forward-fill pattern, with ONE change: .expanding(min_periods=1).mean() replaced
    by .rolling(window, min_periods=1).mean() — everything else (the own-row vs.
    after-row distinction, the match-grouped forward-fill, the NaN-not-a-default
    semantics for zero observations) carries over unchanged, since none of that
    reasoning depended on expanding vs. rolling specifically.

    Multiple windows computed in one pass (default 10 and 15) since the caller may
    want to test several side by side, same rationale as compute_in_match_momentum's
    own multi-window support.

    CONVENTION (same as compute_in_match_momentum above, settled 2026-07 — see that
    function's own docstring for the full investigation): PtWinner is LITERAL,
    fixed-player-relative. A same-day "fix" here previously combined it with Svr on
    the now-reverted server-relative assumption.
    """
    df = df.copy()
    p1_won = (df["PtWinner"] == 1).astype(int)
    p1_serving = df["Svr"] == 1

    for window in windows:
        for label, mask in [("serve", p1_serving), ("return", ~p1_serving)]:
            subset_idx = df.index[mask]
            subset_won = p1_won[mask]
            subset_match = df.loc[mask, "match_id"]

            shifted_for_own = subset_won.groupby(subset_match).shift(1)
            own_rolling = shifted_for_own.groupby(subset_match).rolling(
                window, min_periods=1
            ).mean().reset_index(level=0, drop=True)

            after_rolling = subset_won.groupby(subset_match).rolling(
                window, min_periods=1
            ).mean().reset_index(level=0, drop=True)

            own_col = pd.Series(index=df.index, dtype="float64")
            own_col.loc[subset_idx] = own_rolling.values

            after_col = pd.Series(index=df.index, dtype="float64")
            after_col.loc[subset_idx] = after_rolling.values
            df["_after_tmp"] = after_col
            df["_after_shifted"] = df.groupby("match_id")["_after_tmp"].shift(1)
            df["_after_ffilled"] = df.groupby("match_id")["_after_shifted"].transform(
                lambda g: g.ffill()
            )

            final = own_col.combine_first(df["_after_ffilled"])
            df[f"p1_in_match_{label}_rate_last{window}"] = final
            df = df.drop(columns=["_after_tmp", "_after_shifted", "_after_ffilled"])

    return df


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit interaction terms between already-validated features, computed as a
    simple multiplicative product — tree-based models CAN partially discover
    interactions between raw features on their own, but explicit interaction features
    often still help, especially between a high-cardinality feature (a streak or
    momentum value) and a rare-event binary flag (like is_break_point), where the tree
    may not have enough examples at a given leaf to reliably find the right split from
    the raw features alone.

    MUST run AFTER compute_point_state, compute_in_match_momentum, AND
    compute_consecutive_points_streak have all already populated the columns this
    function depends on (is_break_point, points_streak, pressure_index,
    p1_momentum_last10) — this function does not compute any of those itself.

    LEAKAGE SAFETY is trivial and inherited, not independently re-derived: the product
    of two columns that are EACH already leakage-safe (computed only from information
    strictly prior to the current point) is itself leakage-safe by construction —
    multiplication is a deterministic function of already-safe inputs and cannot, on
    its own, introduce any dependence on future information. No new leakage-safety
    test is needed for this property specifically; only correctness of the
    multiplication itself is checked.
    """
    df = df.copy()
    df["points_streak_x_break_point"] = df["points_streak"] * df["is_break_point"].astype(int)
    df["pressure_index_x_momentum10"] = df["pressure_index"] * df["p1_momentum_last10"]
    return df