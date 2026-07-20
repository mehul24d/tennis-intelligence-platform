"""
build_point_dataset.py — Day 9: joins Day 6 match-level features (Elo, rolling form,
serve/return) onto the point-by-point sequences from Day 7, producing the full training
dataset for the point-outcome classifier.

DESIGN NOTE: pre-match features are STATIC per match (broadcast to every point row).
In-match state features (score, momentum) vary point-by-point (built in Day 7). This
separation is deliberate — mixing them in one pipeline would obscure which features are
forward-looking.

JOIN KEY: mcp_match_id links point files to the frozen join, which links to day6's
(tourney_id, match_num, winner_id, loser_id). Reuses the same mcp_match_id-anchored join
pattern proven safe in Day 6's bug-fix (avoids the NaN match_num fan-out issue).

TARGET: PtWinner (1 or 2) — whether the server won the point. Reframed as a binary:
server_wins_point = (PtWinner == Svr), so the classifier predicts from the server's
perspective regardless of which player is "Player 1" or "Player 2" in MCP's convention.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tennis_intel.features.point_level_features import (
    load_and_sort_points, compute_point_state, compute_in_match_momentum,
    compute_consecutive_points_streak, compute_split_points_streak, compute_games_streak,
    compute_in_match_serve_return_rate,
    compute_interaction_features,
)

logger = logging.getLogger(__name__)

# Pre-match feature columns to broadcast to every point in a match.
# All are diff features (player_1 - player_2 perspective from build_symmetric_dataset)
# OR raw winner_/loser_ features we'll convert to server_/returner_ perspective below.
PREMATCH_FEATURE_COLS = [
    "elo_pre_match_winner", "elo_pre_match_loser",
    "winner_win_pct_last10", "loser_win_pct_last10",
    "winner_surface_win_pct_last10", "loser_surface_win_pct_last10",
    "winner_first_serve_in_pct_career", "loser_first_serve_in_pct_career",
    "winner_first_serve_win_pct_career", "loser_first_serve_win_pct_career",
    # NEW (2026-07, following the Sinner-Alcaraz return-seed investigation): needed to
    # construct a properly-weighted COMBINED (first+second) serve-win rate — see the
    # combined_serve_win_pct_career computation below for the full explanation of why
    # first_serve_win_pct_career alone was being incorrectly used as a stand-in for the
    # opponent's TRUE overall serve strength in both pure Markov's own ps/pr construction
    # and the ML-Informed Markov engine's return-seed construction. Verified present at
    # the match-level feature stage via the same shift-then-roll leakage-safe idiom as
    # every other _career column here (serve_return_features.py's
    # compute_rolling_serve_return_features, rate_cols list) — this is not a new
    # leakage-safety question of its own, just a previously-unmerged column from an
    # already-validated pipeline.
    "winner_second_serve_win_pct_career", "loser_second_serve_win_pct_career",
    # NOTE (2026-07, corrected): loser_second_serve_win_pct_career was briefly removed
    # from this list, which broke build_point_dataset.py — the
    # combined_serve_win_pct_career loop below (for _prefix in ("winner", "loser"))
    # unconditionally references {_prefix}_second_serve_win_pct_career for BOTH
    # prefixes, and that computation feeds the return-seed/Beta-Binomial seeding, a
    # completely different and load-bearing consumer from the point-level
    # classifier. RESTORED HERE. The actual, correct fix for the confirmed negative
    # classifier importance (see feature_schema.py's POINT_FEATURE_COLS for the full
    # derivation) is to exclude loser_second_serve_win_pct_career from
    # feature_schema.py's POINT_FEATURE_COLS ONLY — this list (PREMATCH_FEATURE_COLS)
    # controls what gets MERGED from the Day 6 parquet file, which is a genuinely
    # different concern from what the CLASSIFIER is trained on, and this specific
    # column is still required here regardless of whether the classifier uses it.
    "winner_bp_saved_pct_career", "loser_bp_saved_pct_career",
    "winner_return_pts_won_pct_career", "loser_return_pts_won_pct_career",
    # NEW (Elo redesign, 2026-07): surface-specific Elo + a match-count confidence signal
    # for the overall Elo. Same naming pattern as elo_pre_match_winner/loser above (suffix,
    # not prefix, since these come from build_elo.py directly rather than Day 5/6's
    # winner_/loser_-prefixed rolling-stat convention).
    "elo_surface_pre_match_winner", "elo_surface_pre_match_loser",
    "elo_matches_played_pre_winner", "elo_matches_played_pre_loser",
    # NEW (surface-conditioned career serve/return stats, 2026-07): same evidence base as
    # the surface Elo addition above (37.6% of match winners had a LOWER surface-blind
    # career first-serve-win% than the loser). Starting with first_serve_win_pct_career
    # specifically, since that's the exact feature this evidence was gathered against —
    # first_serve_in_pct_career and bp_saved_pct_career are candidates for the same
    # treatment once this one is validated via permutation importance, per the same
    # discipline used for surface Elo (test, measure, don't assume).
    "winner_first_serve_win_pct_surface_career", "loser_first_serve_win_pct_surface_career",
    # BUG FIX (2026-07, found via check_deciding_set_importance.py's guard flagging
    # these as missing from the trained classifier's feature_cols despite being
    # correctly listed in feature_schema.py's POINT_FEATURE_COLS): this list
    # (build_point_dataset.py's OWN PREMATCH_FEATURE_COLS) is the ACTUAL list
    # controlling what gets merged from the Day 6 parquet file into the point-level
    # dataframe — feature_schema.py's POINT_FEATURE_COLS only controls what the
    # REMOVED (2026-07, shortly after the fix above landed): the fix that added this
    # column was correct and confirmed the data pipeline itself works end-to-end, but
    # a subsequent permutation-importance retrain + check_second_serve_correlation.py
    # showed this specific column is substantially correlated with its own career-level
    # counterpart (r=0.78 winner, r=0.69 loser) — largely redundant, unlike
    # career-level second-serve (r≈0.35-0.38 vs first-serve, genuinely distinct, kept).
    # Removed from both this list and feature_schema.py's POINT_FEATURE_COLS together.
    # NEW (break-point-specific serve/return career rates, 2026-07): motivated by the
    # deciding-set investigation's real, structural residual gap that survived
    # conditioning on points-remaining, a graded fatigue proxy, and a binary deciding-set
    # NEW (break-point-specific return career rate, 2026-07): motivated by the
    # deciding-set investigation's real, structural residual gap that survived
    # conditioning on points-remaining, a graded fatigue proxy, and a binary deciding-set
    # flag. Return-side only — the analogous serve-side feature
    # (bp_serve_win_pct_career, from KeyPointsServe's row=='BP') was built, tested via
    # permutation importance, and REMOVED after direct verification: its correlation with
    # the already-existing bp_saved_pct_career was measured at r=1.0000 across 347,093
    # points — not merely redundant, but mathematically identical (a break point is
    # "saved" if and only if the server wins that point, so both ratios are computed
    # over the same numerator/denominator, just sourced from two different MCP files).
    # The return-side feature does NOT have this property: measured correlation with the
    # existing return_pts_won_pct_career was only r=0.45-0.51 — related but genuinely
    # distinct (break-point-opportunity performance specifically, not routine-point
    # return performance) — and it showed real, non-trivial permutation importance,
    # unlike its removed serve-side counterpart. See
    # serve_return_features.py's load_and_prepare_key_points_stats docstring for the
    # full derivation and leakage-safety argument.
    "winner_bp_return_win_pct_career", "loser_bp_return_win_pct_career",
    # NEW (Elo-trend features, 2026-07): captures whether a player is currently
    # trending up or down in strength, distinct from static elo_pre_match_winner/loser
    # (which the model already has as an absolute-level feature). See
    # serve_return_features.py's compute_elo_trend_features docstring for the full
    # derivation and leakage-safety argument (elo_pre_match is already a prior-only
    # value by construction — verified directly by perturbation testing).
    "winner_elo_change_last10", "loser_elo_change_last10",
    "winner_elo_change_last20", "loser_elo_change_last20",
    "winner_elo_change_last50", "loser_elo_change_last50",
    # NEW (H2H + tournament features, 2026-07): see build_day9_point_model.py's
    # POINT_FEATURE_COLS for the full justification.
    "winner_h2h_wins_pre_match", "loser_h2h_wins_pre_match",
    "winner_tourney_h2h_wins_pre_match", "loser_tourney_h2h_wins_pre_match",
    "winner_tourney_win_pct_last10", "loser_tourney_win_pct_last10",
    # NOTE: "best_of" deliberately excluded here — the point-level dataset already has its
    # own correctly-computed `best_of` column (from compute_point_state, via best_of_map
    # derived from frozen_join's "mcp_Best of"). Including it again here would collide
    # during the merge below and get silently renamed to best_of_x/best_of_y by pandas,
    # which is exactly the bug this comment exists to prevent from being reintroduced.
]


def build_point_dataset(
    point_paths: list[Path],
    frozen_join: pd.DataFrame,
    day6_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns one row per point for all charted matches that exist in BOTH the frozen join
    (so they have player IDs) AND the day6 feature dataset (so they have pre-match context).
    The MCP Player 1 / Player 2 slots are preserved as-is; the server/returner perspective
    is derived from the `Svr` column (1 or 2) in the point data.
    """
    logger.info("Loading and sorting point files...")
    points = load_and_sort_points([str(p) for p in point_paths])
    logger.info("Loaded %d points across %d matches",
                len(points), points["match_id"].nunique())

    # Restrict to matches in the frozen join only (those with player IDs + TML context)
    valid_ids = set(frozen_join["mcp_match_id"])
    points = points[points["match_id"].isin(valid_ids)].copy()
    logger.info("After filtering to frozen-join matches: %d points, %d matches",
                len(points), points["match_id"].nunique())

    # Build best_of_map from frozen join
    best_of_map = dict(zip(frozen_join["mcp_match_id"],
                           frozen_join["mcp_Best of"].fillna(3).astype(int)))

    logger.info("Computing point state (score parsing, situational flags)...")
    points = compute_point_state(points, best_of_map)

    logger.info("Computing in-match momentum...")
    points = compute_in_match_momentum(points)

    logger.info("Computing consecutive-points streak...")
    points = compute_consecutive_points_streak(points)

    logger.info("Computing serve/return-split streak...")
    points = compute_split_points_streak(points)

    logger.info("Computing games streak...")
    points = compute_games_streak(points)

    logger.info("Computing in-match serve/return rate...")
    points = compute_in_match_serve_return_rate(points)

    # REMOVED (2026-07): compute_in_match_serve_return_rate_rolling was called here,
    # producing last10/last15 windows. Both landed negative or negligible in
    # permutation importance (see feature_schema.py's own comment for the exact
    # numbers) — most likely a too-small sample size per window (roughly 5-7 actual
    # serve points within a last-10-points window) to estimate a rate reliably,
    # unlike the expanding (whole-match) version above, which stays dominant. The
    # function itself is NOT deleted (still correct, tested, may be useful with a
    # different window in the future) — only this call, to stop paying for unused
    # computation.

    logger.info("Computing interaction features...")
    points = compute_interaction_features(points)

    # Build the match-level feature lookup: mcp_match_id -> pre-match feature row
    # Join frozen_join (has mcp_match_id -> tml keys) to day6 (has tml keys -> features)
    join_keys = frozen_join[["mcp_match_id", "tml_tourney_id", "tml_match_num",
                              "tml_winner_id", "tml_loser_id"]].drop_duplicates("mcp_match_id")

    match_features = join_keys.merge(
        day6_matches[["tourney_id", "match_num", "winner_id", "loser_id"]
                     + [c for c in PREMATCH_FEATURE_COLS if c in day6_matches.columns]],
        left_on=["tml_tourney_id", "tml_match_num", "tml_winner_id", "tml_loser_id"],
        right_on=["tourney_id", "match_num", "winner_id", "loser_id"],
        how="inner",
    ).drop(columns=["tourney_id", "match_num", "winner_id", "loser_id"])

    logger.info("Match feature lookup: %d matches with pre-match context",
                len(match_features))

    # Broadcast match-level features onto every point row
    points = points.merge(match_features, left_on="match_id",
                          right_on="mcp_match_id", how="inner")

    # Construct server/returner perspective features
    # MCP convention: Svr=1 means "Player 1" is serving; Svr=2 means "Player 2" is serving.
    player1_is_winner_map = dict(zip(
        frozen_join["mcp_match_id"],
        frozen_join["mcp_player1_norm"] == frozen_join["tml_winner_name_norm"],
    ))

    points["player1_is_winner"] = points["match_id"].map(player1_is_winner_map)

    # LEAKAGE FIX (found via external audit, 2026-07, Phase 4 "Critical" finding,
    # cross-validated against permutation importance ranking this feature #2 overall by a
    # ~13x margin over every genuinely pre-match feature): server_is_winner encodes "does
    # the current server go on to win the ENTIRE match" — a quantity that cannot be
    # computed without already knowing this match's final outcome, unlike every other
    # winner_/loser_-labeled feature in this project (Elo, H2H, career stats), which are
    # genuine pre-match facts that exist independent of this specific match's result.
    # Feeding this directly to the point classifier as a training feature is target
    # leakage: a live, in-progress match never has access to its own final outcome.
    #
    # server_is_winner is KEPT here (not deleted) because other code legitimately depends
    # on it for a DIFFERENT purpose — mapping "the tracked winner" onto Player1/Player2 for
    # ORIENTATION when reading pre-match winner_/loser_-labeled feature columns (see
    # _row_to_match_state implementations, ml_informed_point_probabilities). That usage is
    # NOT leakage: it's a bookkeeping label applied consistently across an entire backtest,
    # not a feature the model is trained to directly read. What WAS wrong is using it AS a
    # trained-on feature — see build_day9_point_model.py's POINT_FEATURE_COLS and
    # ml_informed_point_probabilities's hypothetical-row construction, both fixed
    # separately to use server_is_player1 (below) as the actual model-facing feature.
    #
    # server_is_player1: genuinely safe — a real-time, directly-observable fact (who is
    # physically serving right now) with zero dependence on who wins the match.
    points["server_is_winner"] = (
        (points["Svr"] == 1) == points["player1_is_winner"]
    )
    points["server_is_player1"] = (points["Svr"] == 1)

    # Binary target: did the server win this point?
    points["server_wins_point"] = (points["PtWinner"] == points["Svr"]).astype(int)

    # NEW (2026-07, following the Sinner-Alcaraz return-seed investigation): a properly-
    # weighted combined (first+second) serve-win rate, since first_serve_win_pct_career
    # ALONE was being used as a stand-in for a player's true overall serve strength in
    # both pure Markov's own serve/return construction and the ML-Informed Markov
    # engine's return-seed construction — systematically UNDERSTATING it, since it
    # ignores second-serve points (won at a meaningfully lower rate) entirely. This is a
    # deterministic arithmetic combination of three already-merged, already-leakage-safe
    # columns (see PREMATCH_FEATURE_COLS above) — no new leakage-safety question of its
    # own, since it introduces no new information source beyond what's already validated.
    # NaN-safe: if any of the three inputs is missing for a given match, the combined
    # rate is also NaN, and callers already have established NaN-fallback handling
    # (e.g. return_seed.py's own fallback chain) rather than silently guessing here.
    for _prefix in ("winner", "loser"):
        points[f"{_prefix}_combined_serve_win_pct_career"] = (
            points[f"{_prefix}_first_serve_in_pct_career"]
            * points[f"{_prefix}_first_serve_win_pct_career"]
            + (1.0 - points[f"{_prefix}_first_serve_in_pct_career"])
            * points[f"{_prefix}_second_serve_win_pct_career"]
        )

    # Drop internal join/temp columns
    drop_cols = ["mcp_match_id", "tml_tourney_id", "tml_match_num",
                 "tml_winner_id", "tml_loser_id", "player1_is_winner"]
    points = points.drop(columns=[c for c in drop_cols if c in points.columns])

    n_ok = points["score_parse_ok"].sum()
    logger.info("Final dataset: %d points, %d with parsed score (%.1f%%)",
                len(points), n_ok, 100 * n_ok / len(points))

    return points