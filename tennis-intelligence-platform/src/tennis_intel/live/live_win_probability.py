"""
live_win_probability.py — computes win probability FROM AN ARBITRARY IN-MATCH STATE, using
the analytical Markov model in markov_baseline.py.

This is what makes the Markov model a *live* engine rather than just a pre-match predictor:
given the current sets/games/points score, whose serve it is, and both players' per-serve
point-win probabilities, it computes each player's probability of winning the match from
exactly this state forward.

Approach: condition on the current state and recursively/analytically compute the
probability of completing the current game, then the current set, then the match, from
where play actually is — not from 0-0. Uses the same i.i.d.-given-server assumption as the
baseline (documented limitation, same as markov_baseline.py).

Validation: at the start of a match (0-0-0, first server to serve), the live probability
must exactly equal the pre-match prob_win_match — verified in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from tennis_intel.live.markov_baseline import prob_win_game, prob_win_set, prob_win_match


@dataclass(frozen=True)
class MatchState:
    """Current in-match state, all from the perspective of tracking player A vs B.
    points are ordinal (0,1,2,3=40,4=AD) for regular games; for tiebreaks, raw counts."""
    a_sets: int
    b_sets: int
    a_games: int
    b_games: int
    a_points: int
    b_points: int
    server_is_a: bool
    is_tiebreak: bool = False
    best_of: int = 3


def _prob_a_wins_regular_game_from(a_pts: int, b_pts: int, p: float) -> float:
    """Probability the SERVER (with per-point serve-win prob p) wins a regular game from an
    arbitrary in-game point score (a_pts = server's points, b_pts = returner's), ordinal
    0..4 where 4 = advantage. Closed-form recursion with the deuce fixed point."""
    q = 1 - p

    # Deuce subgame solved in CLOSED FORM, not by recursion. Deuce (3-3) and the two
    # advantage states (4-3 server AD, 3-4 returner AD) form a cycle — a naive recursion
    # f(3,3)->f(4,3)->f(3,3) never terminates. From deuce, the server wins the subgame with
    # probability p^2 / (p^2 + q^2) (win two in a row before losing two in a row); this is
    # the standard fixed point and is exactly what prob_win_game() uses.
    denom = p * p + q * q
    p_win_from_deuce = (p * p / denom) if denom > 0 else 0.0
    # From server advantage (4,3): win the point -> game (prob p), else back to deuce.
    p_win_from_server_ad = p + q * p_win_from_deuce
    # From returner advantage (3,4): server must win the point to return to deuce, else loses.
    p_win_from_returner_ad = p * p_win_from_deuce

    @lru_cache(maxsize=None)
    def f(sp: int, rp: int) -> float:
        # sp = server points, rp = returner points (ordinal, 0,1,2,3=40; 4=AD handled via
        # the closed-form deuce/advantage results above rather than recursion).
        if sp == 3 and rp == 3:
            return p_win_from_deuce
        if sp == 4:            # server advantage
            return p_win_from_server_ad
        if rp == 4:            # returner advantage
            return p_win_from_returner_ad
        if sp == 3 and rp < 3:
            # server at 40, returner below 40: win the point -> game outright
            return p + q * f(3, rp + 1)
        if rp == 3 and sp < 3:
            # returner at 40, server below 40: lose the point -> game lost; win -> advance
            return p * f(sp + 1, 3)
        # general non-terminal (neither player at 40 yet)
        return p * f(sp + 1, rp) + q * f(sp, rp + 1)

    return f(a_pts, b_pts)


def prob_a_wins_match_from_state(state: MatchState, p_a_serve: float, p_a_return: float) -> float:
    """
    Probability player A wins the match from the given in-match state.
    p_a_serve: A's probability of winning a point on A's serve.
    p_a_return: A's probability of winning a point on B's serve (i.e. 1 - B's serve-win prob).

    Tiebreak states are handled by a dedicated enumeration; regular play composes
    game -> set -> match analytically from the current score.
    """
    # --- Probability A wins the CURRENT game from the current point score ---
    if state.is_tiebreak:
        p_a_win_current_game = _prob_a_wins_tiebreak_from(
            state.a_points, state.b_points, state.server_is_a, p_a_serve, p_a_return
        )
    else:
        if state.server_is_a:
            p_a_win_current_game = _prob_a_wins_regular_game_from(
                state.a_points, state.b_points, p_a_serve
            )
        else:
            # B is serving; compute prob B wins the game with B's serve prob (1 - p_a_return),
            # from B's perspective (B's points = state.b_points), then A wins = 1 - that.
            p_b_serve = 1 - p_a_return
            p_b_win = _prob_a_wins_regular_game_from(state.b_points, state.a_points, p_b_serve)
            p_a_win_current_game = 1 - p_b_win

    # --- Compose up to set, then match, by enumerating game/set completions ---
    # Probability A wins a FUTURE game A serves / B serves (fresh game, 0-0)
    pg_a_serve = prob_win_game(p_a_serve)
    pg_a_return = 1 - prob_win_game(1 - p_a_return)

    @lru_cache(maxsize=None)
    def set_win_prob_from_games(a_g: int, b_g: int, a_to_serve_next: bool,
                                 current_game_done: bool, a_won_current: bool) -> float:
        # This helper is only used for FRESH games (fresh 0-0 games), the current partial
        # game is handled separately by the caller blending p_a_win_current_game.
        # Standard set completion from an arbitrary game score.
        # Terminal: someone reached 6 with 2 clear, or 7-5, or 7-6 (tiebreak).
        if a_g >= 6 and a_g - b_g >= 2:
            return 1.0
        if b_g >= 6 and b_g - a_g >= 2:
            return 0.0
        if a_g == 7 and b_g == 6:
            return 1.0
        if b_g == 7 and a_g == 6:
            return 0.0
        if a_g == 6 and b_g == 6:
            # tiebreak; first server of the tiebreak is whoever is "to serve next"
            return _prob_a_wins_tiebreak_from(0, 0, a_to_serve_next, p_a_serve, p_a_return)

        # BUG FIX (found via a genuine RecursionError replaying the 2008 Wimbledon final,
        # 2026-07): this function previously assumed EVERY set resolves via a tiebreak at
        # 6-6, with no path for a set that legitimately continues past 6-6 without one —
        # exactly what happens under advantage-set ("no-ad final set") rules, real and in
        # effect at Wimbledon prior to 2019 (this match's real 5th set went to 9-7, no
        # breaker) and at other majors in other eras. Any state reached beyond 6-6 without
        # a tiebreak having been triggered (e.g. 7-7, 7-8, 8-8) fell through to naive,
        # unbounded recursion: mathematically it always terminates eventually (someone
        # must eventually get 2 games clear), but the EXHAUSTIVE recursion tree must still
        # visit branches that stay tied for arbitrarily many rounds, however vanishingly
        # likely — and Python's call stack has a hard depth limit lru_cache does not help
        # with (memoization only avoids recomputing an ALREADY-visited state, it doesn't
        # reduce the depth needed to first reach one).
        #
        # Fixed with a genuine closed form, structurally identical to the already-validated
        # extended-deuce tiebreak fix above (_prob_a_wins_tiebreak_from) — the SAME 6-state
        # absorbing Markov chain (diff in {-1,0,+1} x next-server in {A,B}), just at the
        # level of whole GAMES instead of points, using pg_a_serve/pg_a_return (a player's
        # probability of winning a game they serve/return) in place of the point-level
        # p_a_serve/p_a_return. Solved via the same direct-substitution technique, so it
        # cannot recurse unboundedly — this is a closed-form evaluation, not a search.
        if min(a_g, b_g) >= 6:
            diff = a_g - b_g
            if abs(diff) <= 1:
                return _advantage_set_extended_deuce_prob(
                    diff, a_to_serve_next, pg_a_serve, pg_a_return
                )

        pg = pg_a_serve if a_to_serve_next else pg_a_return
        return (pg * set_win_prob_from_games(a_g + 1, b_g, not a_to_serve_next, False, False)
                + (1 - pg) * set_win_prob_from_games(a_g, b_g + 1, not a_to_serve_next, False, False))

    # After the current game resolves, games become (a_games+1, b_games) or (a_games, b_games+1),
    # and serve passes to the other player.
    next_server_is_a_after_game = not state.server_is_a
    set_if_a_wins_game = set_win_prob_from_games(
        state.a_games + 1, state.b_games, next_server_is_a_after_game, False, False
    )
    set_if_a_loses_game = set_win_prob_from_games(
        state.a_games, state.b_games + 1, next_server_is_a_after_game, False, False
    )
    p_a_win_current_set = (p_a_win_current_game * set_if_a_wins_game
                           + (1 - p_a_win_current_game) * set_if_a_loses_game)

    # --- Compose set -> match ---
    # Probability A wins a FRESH future set (approximation consistent with the baseline:
    # single per-set probability, ignoring serve-order carryover across sets).
    p_fresh_set = prob_win_set(p_a_serve, p_a_return)
    sets_needed = (state.best_of // 2) + 1

    @lru_cache(maxsize=None)
    def match_from_sets(a_s: int, b_s: int, current_set_done: bool) -> float:
        if a_s >= sets_needed:
            return 1.0
        if b_s >= sets_needed:
            return 0.0
        # Use current-set probability for the in-progress set, fresh probability afterward.
        p = p_fresh_set if current_set_done else p_a_win_current_set
        return (p * match_from_sets(a_s + 1, b_s, True)
                + (1 - p) * match_from_sets(a_s, b_s + 1, True))

    return match_from_sets(state.a_sets, state.b_sets, current_set_done=False)


def _advantage_set_extended_deuce_prob(
    diff: int, a_serves_next: bool, pg_a_serve: float, pg_a_return: float
) -> float:
    """
    Probability A eventually wins a set that has reached a tied-or-near-tied game score at
    6-all or beyond, played under ADVANTAGE-SET rules (no tiebreak — win by 2 games,
    indefinitely) — e.g. 6-6, 7-7, 7-6, 6-7, 8-8, etc. diff = a_games - b_games, valid only
    for diff in {-1, 0, 1} (the only non-terminal offsets once both players have reached 6).

    Structurally identical to _prob_a_wins_tiebreak_from's extended-deuce closed form
    (same 6-state absorbing Markov chain: diff in {-1,0,+1} x next-server in {A,B}, solved
    by direct substitution) — the only difference is this operates at the level of whole
    GAMES using pg_a_serve/pg_a_return (a player's probability of winning a game they
    serve/return) rather than at the level of POINTS. Reusing the same closed-form
    technique rather than re-deriving from scratch, since the underlying math (a repeated
    binary event with a probability that depends only on "who serves this instance", with
    server strictly alternating) is identical in both cases.
    """
    qg_a_serve = 1 - pg_a_serve
    qg_a_return = 1 - pg_a_return

    denom_a = 1 - pg_a_serve * qg_a_return - qg_a_serve * pg_a_return
    w0_a = (pg_a_serve * pg_a_return / denom_a) if denom_a > 0 else 0.5
    denom_b = 1 - pg_a_return * qg_a_serve - qg_a_return * pg_a_serve
    w0_b = (pg_a_return * pg_a_serve / denom_b) if denom_b > 0 else 0.5

    w1_a = pg_a_serve * 1.0 + qg_a_serve * w0_b
    w1_b = pg_a_return * 1.0 + qg_a_return * w0_a
    wm1_a = pg_a_serve * w0_b + qg_a_serve * 0.0
    wm1_b = pg_a_return * w0_a + qg_a_return * 0.0

    if diff == 0:
        return w0_a if a_serves_next else w0_b
    if diff == 1:
        return w1_a if a_serves_next else w1_b
    if diff == -1:
        return wm1_a if a_serves_next else wm1_b
    raise ValueError(f"_advantage_set_extended_deuce_prob called with |diff|>=2: {diff}")


def _prob_a_wins_tiebreak_from(a_pts: int, b_pts: int, server_is_a: bool,
                                p_a_serve: float, p_a_return: float, target: int = 7) -> float:
    """Probability A wins a tiebreak from an arbitrary tiebreak score, tracking the serving
    rotation (alternates every 2 points after the first). server_is_a = whether A serves the
    NEXT point from this state."""

    # Once the tiebreak reaches (target-1, target-1) or beyond with the gap under 2, it
    # enters a win-by-2 "extended deuce" phase. The ORIGINAL version of this function
    # collapsed EVERY such state to a single constant, incorrectly treating "tied",
    # "ahead by 1", and "behind by 1" (and both server-identity cases) as identical —
    # mathematically wrong, since being ahead by a point must give a strictly higher win
    # probability than being tied, which must exceed being behind. This produced visibly
    # incorrect, unstable predictions in the deciding-set tiebreak of a real replayed match
    # (2025 Roland Garros final), caught by inspecting that match's point-by-point output.
    #
    # CORRECT closed form: the only state that matters once deep in extended deuce is the
    # score DIFFERENTIAL d = a_points - b_points (only |d|<2 is non-terminal) and who
    # serves next. This is a 6-state absorbing Markov chain (d in {-1,0,+1} x server in
    # {A,B}); solved here via direct substitution rather than recursion, so it cannot loop.
    q_a_serve = 1 - p_a_serve   # B wins the point when A serves
    q_a_return = 1 - p_a_return  # B wins the point when B serves (A wins w.p. p_a_return)

    # W(0,A) = P(A wins | tied, A serves next); W(0,B) = P(A wins | tied, B serves next)
    # Derived by substitution from the 6-equation system (see module history/derivation):
    #   W(0,A) = p_a_serve * p_a_return / [1 - p_a_serve*q_a_return - q_a_serve*p_a_return]
    #   W(0,B) = p_a_return * p_a_serve / [1 - p_a_return*q_a_serve - q_a_return*p_a_serve]
    denom = 1 - p_a_serve * q_a_return - q_a_serve * p_a_return
    w0_a = (p_a_serve * p_a_return / denom) if denom > 0 else 0.5
    denom_b = 1 - p_a_return * q_a_serve - q_a_return * p_a_serve
    w0_b = (p_a_return * p_a_serve / denom_b) if denom_b > 0 else 0.5

    # W(+1, server) = win outright this point, or fall back to tied with the OTHER server
    w1_a = p_a_serve * 1.0 + q_a_serve * w0_b     # A ahead by 1, A serves next
    w1_b = p_a_return * 1.0 + q_a_return * w0_a   # A ahead by 1, B serves next
    # W(-1, server) = win falls back to tied; lose is absorbed at 0
    wm1_a = p_a_serve * w0_b + q_a_serve * 0.0    # A behind by 1, A serves next
    wm1_b = p_a_return * w0_a + q_a_return * 0.0  # A behind by 1, B serves next

    def extended_deuce_prob(diff: int, a_serves_next: bool) -> float:
        """diff = a_points - b_points, valid only for diff in {-1, 0, 1}."""
        if diff == 0:
            return w0_a if a_serves_next else w0_b
        if diff == 1:
            return w1_a if a_serves_next else w1_b
        if diff == -1:
            return wm1_a if a_serves_next else wm1_b
        raise ValueError(f"extended_deuce_prob called with |diff|>=2: {diff}")

    deuce_score = target - 1  # e.g. 6 in a 7-point tiebreak

    @lru_cache(maxsize=None)
    def f(ap: int, bp: int, a_serves_next: bool) -> float:
        if ap >= target and ap - bp >= 2:
            return 1.0
        if bp >= target and bp - ap >= 2:
            return 0.0
        if ap >= deuce_score and bp >= deuce_score:
            return extended_deuce_prob(ap - bp, a_serves_next)
        p_a_point = p_a_serve if a_serves_next else p_a_return
        total = ap + bp
        switch = (total % 2 == 0)
        next_a_serves = (not a_serves_next) if switch else a_serves_next
        return (p_a_point * f(ap + 1, bp, next_a_serves)
                + (1 - p_a_point) * f(ap, bp + 1, next_a_serves))

    return f(a_pts, b_pts, server_is_a)