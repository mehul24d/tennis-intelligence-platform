"""
feature_schema.py — the single, canonical source of truth for the Day 9 point
classifier's feature schema, extracted per the external audit's Code Review finding #6:
this exact list was independently redefined in five separate files
(build_day9_point_model.py, evaluate_live_engines_v2.py, generate_publication_trajectory.py,
replay_match.py, tune_day9_hyperparameters.py), each with a comment acknowledging the drift
risk ("must exactly match POINT_FEATURE_COLS in build_day9_point_model.py") but no
mechanism actually preventing it.

This was not a hypothetical risk: at the time of this centralization,
tune_day9_hyperparameters.py's own copy still contained "server_is_winner" — the confirmed
leakage feature already removed from every other copy after the Phase 4 audit finding. If
that script were run again as-is, it would have silently retrained a new model with the
leaky feature reintroduced. Centralizing here means there is now exactly one place this
can go wrong, not five.

PREMATCH_FEATURE_NAMES is deliberately DERIVED from POINT_FEATURE_COLS minus
IN_MATCH_ONLY_FEATURES, not maintained as a separately-copied list — this was verified to
already be an exact match across all three of its previous independent definitions
(STATIC_FEATURE_NAMES in evaluate_live_engines_v2.py and replay_match.py,
PREMATCH_FEATURE_NAMES in generate_publication_trajectory.py) before being consolidated
this way, so deriving it programmatically cannot silently produce a different list than
what was already in use.
"""

from __future__ import annotations

POINT_FEATURE_COLS: list[str] = [
    "is_tiebreak_game", "is_break_point", "is_set_point", "is_match_point",
    "is_second_serve_point",
    # NEW (pressure_index, 2026-07): a single ordinal "how much is riding on this point"
    # variable (match/set/break/server-game/deuce/routine, 10/8/5/3/2/1), built on top of
    # the existing break/set/match-point flags via priority-ordered np.select (a point
    # gets its single HIGHEST tier, not a sum). See point_level_features.py's
    # compute_point_state for the full derivation, tier-boundary tests, and the
    # explicit note that these weights are a hand-chosen ordinal scale, not a precisely
    # calibrated leverage measure — a natural future refinement once this simpler
    # version is validated via permutation importance.
    "pressure_index",
    # NEW (points_streak, 2026-07): a signed, consecutive-points-won run-length,
    # distinct from momentum_last10/20 (a rolling WIN RATE, which stays high through an
    # interrupted sequence) — captures an unbroken run's actual length instead. See
    # point_level_features.py's compute_consecutive_points_streak for the full
    # derivation and leakage-safety argument (same shift(1)-then-groupby(match_id)
    # discipline as in-match momentum, verified directly against a match-boundary case
    # to confirm no streak leaks from one match into the next).
    "points_streak",
    # NEW (serve/return-split streak, 2026-07): p1_serve_streak and p1_return_streak
    # separate points_streak into serve-only and return-only run-lengths — a player on
    # a hot SERVING streak and a player on a hot RETURNING streak are different
    # situations, since serve is the higher-base-rate activity, so a serve streak of a
    # given length is less surprising than a return streak of the same length. See
    # point_level_features.py's compute_split_points_streak for the full derivation,
    # including a genuinely subtle bug (caught by a permanent test, not by inspection)
    # in an earlier draft's single-series forward-fill design, and the two-series fix.
    "p1_serve_streak", "p1_return_streak",
    # NEW (games streak, 2026-07): p1_games_streak is a signed consecutive-GAMES-won
    # run-length, one level coarser than points_streak — a player can lose their
    # point-level streak on a single dropped point while still dominating the game
    # overall, so this captures a different, complementary momentum signal. See
    # point_level_features.py's compute_games_streak for the full derivation, reusing
    # the same two-series forward-fill design already proven for the serve/return
    # split, applied to game-boundary events instead of serve/return-type events.
    "p1_games_streak",
    # NEW (in-match serve/return rate, 2026-07): p1_in_match_serve_rate and
    # p1_in_match_return_rate, a RAW, cumulative (expanding, not fixed-window) in-match
    # rate — distinct from momentum (fixed window, mixes serve/return) and from the
    # career-level rates (never see today's actual performance). NaN, not a default,
    # when no serve/return points have occurred yet — a rate over zero observations is
    # genuinely undefined. See point_level_features.py's
    # compute_in_match_serve_return_rate for the full derivation.
    "p1_in_match_serve_rate", "p1_in_match_return_rate",
    # NEW (rolling-window in-match serve/return rate, 2026-07): a fixed-window sibling
    # of p1_in_match_serve_rate/return_rate above. REMOVED (2026-07) after direct
    # measurement: both windows (10 and 15) landed NEGATIVE or negligible in
    # permutation importance (serve_last10=-0.000019, serve_last15=-0.000011,
    # return_last10=+0.000002, return_last15=+0.000025 — only marginally positive
    # for one variant, not enough to justify keeping any of the four), while the
    # expanding (whole-match) version stayed dominant at rank 3-4. Most likely
    # explanation: a 10-15 point window is too small a sample to estimate a rate
    # reliably (roughly 5-7 actual serve points within a last-10-points window,
    # since only about half of points are this player's serve points) — the same
    # small-sample-noise principle already established in this project's calibration
    # work (small n-buckets producing unstable estimates), now showing up in feature
    # engineering instead of evaluation. See point_level_features.py's
    # compute_in_match_serve_return_rate_rolling — the FUNCTION ITSELF IS NOT
    # REMOVED (it's correct, tested, and may be useful with a differently-tuned
    # window in the future), only these two specific window choices are no longer
    # requested here.
    # NEW (interaction terms, 2026-07): explicit multiplicative interactions between
    # already-validated features, motivated by the idea that a tree model may not
    # reliably find these splits on its own from raw features alone, especially
    # between a high-cardinality streak/momentum value and a rare-event flag. Leakage
    # safety is inherited trivially from the already-safe input columns. See
    # point_level_features.py's compute_interaction_features for the full derivation.
    "points_streak_x_break_point", "pressure_index_x_momentum10",
    # deciding-set points are genuinely harder to call correctly at MATCHED horizon
    # length across all three engines, and no prior feature told the classifier "this is
    # a deciding set." See point_level_features.py's compute_point_state for the full
    # explanation and the exact formula (validated against the same test cases used
    # throughout that investigation's diagnostic scripts).
    "deciding_set",
    # NEW (2026-07, following deciding_set's null permutation-importance result): graded
    # fatigue proxies, since a binary flag gave the classifier nothing to work with. See
    # point_level_features.py's compute_point_state for the exact derivations and why an
    # approximated games_played_so_far was deliberately avoided in favor of these two
    # exact quantities.
    "points_played_so_far_in_match", "sets_played_so_far_in_match",
    "p1_momentum_last10", "p2_momentum_last10",
    "p1_momentum_last20", "p2_momentum_last20",
    "elo_pre_match_winner", "elo_pre_match_loser",
    "winner_win_pct_last10", "loser_win_pct_last10",
    "winner_surface_win_pct_last10", "loser_surface_win_pct_last10",
    "winner_first_serve_in_pct_career", "loser_first_serve_in_pct_career",
    "winner_first_serve_win_pct_career", "loser_first_serve_win_pct_career",
    # NEW (second-serve rates, 2026-07): the entire seeding-bug investigation this
    # session was about first_serve_win_pct systematically understating true serve
    # strength by ignoring second-serve points — the FIX was applied to the
    # Beta-Binomial seeding (combined_serve_win_pct_career, see return_seed.py), but
    # the classifier itself had NEVER been given direct visibility into second-serve
    # performance at all, despite is_second_serve_point being the single dominant
    # feature by a wide margin (importance ~0.048, next-highest ~0.001). The
    # classifier knows WHEN it's a second-serve point but had no career baseline for
    # how well either player performs on them specifically.
    "winner_second_serve_win_pct_career",
    # REMOVED loser_second_serve_win_pct_career (2026-07): consistently negative
    # importance across four separate retrains (-0.000011, then -0.000031, then
    # -0.000047 — growing in magnitude, not noise around zero). Confirmed via
    # check_loser_second_serve_sparsity.py: loser-side rows have systematically
    # thinner career history than winner-side (mean elo_matches_played_pre_loser=
    # 277.3 vs. pre_winner=357.2, an 80-match gap across the whole holdout,
    # unconditional on this feature's presence) — a real, structural selection
    # effect in the winner/loser labeling itself: a shallow-history player is
    # disproportionately likely to be the underdog and therefore the "loser," but
    # "winner" has no symmetric correlation with career depth. This makes the
    # loser-side second-serve estimate noisier, and for THIS SPECIFIC feature (where
    # the underlying true effect size is apparently small), the added noise was
    # enough to tip it net-harmful (permuting it IMPROVED held-out performance).
    #
    # STANDING CAVEAT, not just a one-off fix: every loser_*_career feature in this
    # schema is drawn from the same systematically-thinner-history population and is
    # therefore a candidate for the same effect. Most still show real, positive
    # importance (e.g. loser_first_serve_win_pct_career, loser_bp_return_win_pct_
    # career), meaning their true signal is strong enough to outweigh the extra
    # noise — this was NOT a reason to re-litigate every loser_* feature individually.
    # But if any loser_*_career feature's importance looks anomalously low or
    # negative in a FUTURE run, this sparsity effect is the first hypothesis to
    # check before assuming the feature itself is flawed or starting a fresh
    # investigation.
    "winner_bp_saved_pct_career", "loser_bp_saved_pct_career",
    "server_is_player1",
    "elo_surface_pre_match_winner", "elo_surface_pre_match_loser",
    "elo_matches_played_pre_winner", "elo_matches_played_pre_loser",
    "winner_first_serve_win_pct_surface_career", "loser_first_serve_win_pct_surface_career",
    # REMOVED (surface second-serve, 2026-07): winner_/loser_second_serve_win_pct_
    # surface_career were added, measured via permutation importance, and REMOVED
    # after direct verification via check_second_serve_correlation.py that each is
    # substantially correlated with its own career-level counterpart (r=0.78 winner,
    # r=0.69 loser) — largely redundant, unlike career-level second-serve (only
    # r≈0.35-0.38 vs first-serve, genuinely distinct, kept). This is the same
    # collinearity-driven permutation-importance instability mechanism already
    # confirmed once this session for bp_serve_win_pct_career (there, r=1.0000 with
    # bp_saved_pct_career — an exact duplicate; here, high but not exact correlation,
    # a partial-redundancy case rather than a literal duplicate, but still enough to
    # produce unstable, sign-flipping importance across a retrain).
    # NEW (break-point-specific return career rate, 2026-07): see build_point_dataset.py's
    # PREMATCH_FEATURE_COLS for the full derivation, including why the serve-side
    # counterpart was built, tested, and removed (confirmed r=1.0000 with the already-
    # existing bp_saved_pct_career — mathematically identical, not merely redundant).
    "winner_bp_return_win_pct_career", "loser_bp_return_win_pct_career",
    # NEW (Elo-trend features, 2026-07): see build_point_dataset.py's
    # PREMATCH_FEATURE_COLS for the full derivation.
    "winner_elo_change_last10", "loser_elo_change_last10",
    "winner_elo_change_last20", "loser_elo_change_last20",
    "winner_elo_change_last50", "loser_elo_change_last50",
    "winner_h2h_wins_pre_match", "loser_h2h_wins_pre_match",
    "winner_tourney_h2h_wins_pre_match", "loser_tourney_h2h_wins_pre_match",
    "winner_tourney_win_pct_last10", "loser_tourney_win_pct_last10",
]

TARGET = "server_wins_point"

IN_MATCH_ONLY_FEATURES: set[str] = {
    "is_tiebreak_game", "is_break_point", "is_set_point", "is_match_point",
    "is_second_serve_point", "deciding_set", "pressure_index", "points_streak",
    "p1_serve_streak", "p1_return_streak", "p1_games_streak",
    "p1_in_match_serve_rate", "p1_in_match_return_rate",
    "points_streak_x_break_point", "pressure_index_x_momentum10",
    "points_played_so_far_in_match", "sets_played_so_far_in_match",
    "p1_momentum_last10", "p2_momentum_last10", "p1_momentum_last20", "p2_momentum_last20",
    "server_is_player1",
}

PREMATCH_FEATURE_NAMES: list[str] = [
    c for c in POINT_FEATURE_COLS if c not in IN_MATCH_ONLY_FEATURES
]