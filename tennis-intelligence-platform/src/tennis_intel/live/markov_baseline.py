"""
markov_baseline.py — analytical closed-form tennis win-probability model (Day 8).

Foundational assumption (Klaassen & Magnus, "Are Points in Tennis Independent and
Identically Distributed?", and the classic tennis-probability literature): points are
i.i.d. GIVEN who is serving. Each player has a constant probability of winning a point on
their own serve; from those two numbers alone, the probability of winning a game, set, and
match follow analytically.

This is deliberately the SIMPLE, interpretable baseline the Day 9 ML+simulation approach
will be measured against — exactly the role logistic regression played against tree models
in Milestone 5. Its known limitation (real points are NOT perfectly i.i.d. — momentum,
fatigue, and pressure effects exist) is the entire motivation for building something richer,
and is stated plainly rather than hidden.

All functions are pure and closed-form (no simulation, no iteration beyond small finite
sums), so they are fast and exactly reproducible. Validated against reference values from
the tennis-analytics literature in tests/unit/test_markov_baseline.py.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb


def prob_win_game(p: float) -> float:
    """
    Probability the SERVER wins a service game, given p = probability of winning a single
    point on serve. Standard closed form.

    A game is won by the first player to 4 points with a 2-point margin. The server wins
    outright (to love, 15, or 30) via the binomial paths, plus the deuce branch: reaching
    3-3 (deuce) then winning the deuce subgame, which has probability p^2 / (p^2 + (1-p)^2).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    q = 1 - p

    # Win without reaching deuce: server wins 4 points while loser gets 0, 1, or 2.
    # P(win to 40-k for k in 0,1,2) = C(3+k, k) * p^4 * q^k
    win_before_deuce = sum(comb(3 + k, k) * (p ** 4) * (q ** k) for k in range(3))

    # Reach deuce (3-3): C(6,3) * p^3 * q^3
    prob_deuce = comb(6, 3) * (p ** 3) * (q ** 3)
    # Win the deuce subgame (first to +2): p^2 / (p^2 + q^2), guarding p=q=0.5 etc.
    denom = p ** 2 + q ** 2
    win_deuce_subgame = (p ** 2 / denom) if denom > 0 else 0.0

    return win_before_deuce + prob_deuce * win_deuce_subgame


def prob_win_tiebreak(p_serve: float, p_return: float) -> float:
    """
    Probability that the player who serves FIRST in the tiebreak wins it, given their
    point-win probability on their own serve (p_serve) and on return (p_return).

    A tiebreak is first to 7 points, win by 2, with a fixed serving rotation (first server
    serves point 1, then serve alternates every 2 points). Computed by enumerating all
    reachable non-deuce end states plus a closed-form 6-6 deuce branch.
    """
    # Serving schedule for the first server (player A): point 1 A serves, then B serves
    # points 2-3, A serves 4-5, B 6-7, ... i.e. point index (0-based) i is served by A iff
    # ((i + 1) // 2) is even.
    def a_serves(i: int) -> bool:
        return ((i + 1) // 2) % 2 == 0

    @lru_cache(maxsize=None)
    def state_prob(a_pts: int, b_pts: int) -> float:
        """Probability of reaching exactly (a_pts, b_pts) before either has won, tracked
        from A's perspective (A the first server)."""
        if a_pts == 0 and b_pts == 0:
            return 1.0
        total = 0.0
        # Came from (a-1, b) via A winning the (a+b-1)-th point, or (a, b-1) via B winning.
        i = a_pts + b_pts - 1  # 0-based index of the just-played point
        if a_pts > 0:
            prev = state_prob(a_pts - 1, b_pts)
            p_a_wins = p_serve if a_serves(i) else p_return
            total += prev * p_a_wins
        if b_pts > 0:
            prev = state_prob(a_pts, b_pts - 1)
            # B wins the point: if A served it, B wins on return (1 - p_serve); if B served
            # it, B wins on serve. B's serve-win prob = 1 - p_return (since p_return is A's
            # return-win prob = 1 - B's serve-win prob).
            p_b_wins = (1 - p_serve) if a_serves(i) else (1 - p_return)
            total += prev * p_b_wins
        return total

    prob = 0.0
    # Non-deuce wins for A: A reaches 7, B has 0..5
    for b in range(6):
        i = 7 + b - 1
        p_a_final = p_serve if a_serves(i) else p_return
        prob += state_prob(6, b) * p_a_final

    # Deuce branch at 6-6: from 6-6, it becomes first-to-+2. Over any two consecutive points
    # the serving is one A-serve and one B-serve (given the rotation), so per "pair" A wins
    # both with p_serve*p_return-style terms. Use the standard two-point cycle:
    prob_6_6 = state_prob(6, 6)
    # In the deuce phase, consider blocks of 2 points where serve alternates. P(A wins a
    # given deuce point when A serves) = p_serve; when B serves = p_return. Across a 2-point
    # block (one A serve, one B serve), A wins the block outright with p_serve*p_return,
    # B wins outright with (1-p_serve)*(1-p_return), else back to deuce.
    a_block = p_serve * p_return
    b_block = (1 - p_serve) * (1 - p_return)
    denom = a_block + b_block
    win_deuce = (a_block / denom) if denom > 0 else 0.0
    prob += prob_6_6 * win_deuce

    return prob


def prob_win_set(p_serve: float, p_return: float, server_serves_first: bool = True) -> float:
    """
    Probability of winning a set, given the player's point-win probability on their own
    serve (p_serve) and on return (p_return). A set is first to 6 games, win by 2, with a
    tiebreak at 6-6. Games alternate serve.

    server_serves_first: whether the player in question serves the first game of the set.
    """
    pg_on_serve = prob_win_game(p_serve)          # P(win a game the player serves)
    pg_on_return = 1 - prob_win_game(1 - p_return)  # P(win a game the OPPONENT serves)

    @lru_cache(maxsize=None)
    def game_state_prob(my_games: int, opp_games: int) -> float:
        if my_games == 0 and opp_games == 0:
            return 1.0
        games_played = my_games + opp_games - 1
        # Whether the player served the game just played depends on who served first and parity.
        if server_serves_first:
            i_served_that_game = (games_played % 2 == 0)
        else:
            i_served_that_game = (games_played % 2 == 1)
        pg = pg_on_serve if i_served_that_game else pg_on_return
        total = 0.0
        if my_games > 0:
            total += game_state_prob(my_games - 1, opp_games) * pg
        if opp_games > 0:
            total += game_state_prob(my_games, opp_games - 1) * (1 - pg)
        return total

    prob = 0.0
    # Win 6-0 .. 6-4
    for opp in range(5):
        games_played = 6 + opp - 1
        if server_serves_first:
            i_served = (games_played % 2 == 0)
        else:
            i_served = (games_played % 2 == 1)
        pg = pg_on_serve if i_served else pg_on_return
        prob += game_state_prob(5, opp) * pg

    # Win 7-5: reach 5-5, win a game, then win another
    # reach 6-5 from 5-5 then win the 12th game... handle via reaching 5-5 then two games
    p_5_5 = game_state_prob(5, 5)
    # From 5-5: need to win game 11 and game 13-equivalent (i.e. 6-5 then 7-5)
    games_played_11 = 10  # 0-indexed the 11th game
    if server_serves_first:
        i_served_11 = (games_played_11 % 2 == 0)
        i_served_12 = ((games_played_11 + 1) % 2 == 0)
    else:
        i_served_11 = (games_played_11 % 2 == 1)
        i_served_12 = ((games_played_11 + 1) % 2 == 1)
    pg_11 = pg_on_serve if i_served_11 else pg_on_return
    pg_12 = pg_on_serve if i_served_12 else pg_on_return
    prob += p_5_5 * pg_11 * pg_12  # win both game 11 and 12 -> 7-5

    # Tiebreak at 6-6: reach 6-6 (5-5 then split the next two, or reach 6-6 other ways).
    # Simplest correct route: P(reach 6-6). From 5-5, split games (win one, lose one, either
    # order) -> 6-6.
    p_6_6 = p_5_5 * (pg_11 * (1 - pg_12) + (1 - pg_11) * pg_12)
    # In the tiebreak, the player who did NOT serve the 12th game serves first in the breaker.
    # Whoever served first in the set serves game 1,3,5,7,9,11 (odd games, 1-indexed). Game 13
    # (the tiebreak) is served first by the player who served game 1 iff... standard rule: the
    # player who received in game 12 serves first in the tiebreak. Equivalent: the set's first
    # server serves the tiebreak first (since 12 games played, serve returns to game-1 server).
    tb = prob_win_tiebreak(p_serve, p_return) if server_serves_first else (1 - prob_win_tiebreak(1 - p_return, 1 - p_serve))
    prob += p_6_6 * tb

    return prob


def prob_win_match(p_serve: float, p_return: float, best_of: int = 3,
                    server_serves_first: bool = True) -> float:
    """
    Probability of winning the match, given per-serve point-win probabilities. Assumes
    (per the i.i.d. model) that each set is independent with the same per-set win
    probability. Best-of-3 (first to 2 sets) or best-of-5 (first to 3 sets).

    NOTE: server_serves_first only affects the first set's serve order; across sets the
    effect on the aggregate is second-order and, consistent with the standard analytical
    model, we use a single per-set probability. This is a documented simplification of the
    baseline, not an oversight — the whole point of this model is analytical simplicity.
    """
    p_set = prob_win_set(p_serve, p_return, server_serves_first)
    sets_needed = (best_of // 2) + 1

    # P(win the match) = P(win `sets_needed` sets before opponent does), each set i.i.d.
    prob = 0.0
    for opp_sets in range(sets_needed):
        # Win the final (deciding) set, having reached (sets_needed-1, opp_sets) before it
        ways = comb(sets_needed - 1 + opp_sets, opp_sets)
        prob += ways * (p_set ** sets_needed) * ((1 - p_set) ** opp_sets)
    return prob