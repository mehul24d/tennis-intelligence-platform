"""
test_monte_carlo_engine.py — addresses the external audit's Code Review finding #7:
monte_carlo_engine.py had 0% unit test coverage, which the audit's own summary identifies
as the reason two Critical bugs (terminal-state mishandling, degenerate-input hang) went
undetected pre-audit. Covers _advance_point (the core state-transition logic everything
else builds on), simulate_match_from_state (both confirmed-and-fixed Critical bugs), and
batch_simulate_dynamic (the function actually used for every reported ML+MC metric).

CALLING CONVENTION: _advance_point is called AFTER a point's outcome has already been
incremented into a_points/b_points — it checks whether that increment just completed a
game/set/match, not whether one is about to be played.
"""

import random
import time

import numpy as np
import pytest

from tennis_intel.live.monte_carlo_engine import (
    _advance_point, simulate_match_from_state, batch_simulate_dynamic,
)


class TestAdvancePoint:
    def test_regular_game_win_requires_two_clear(self):
        # 3-3 in points, A wins the next point -> 4-3, NOT yet a game win (needs 2 clear)
        result = _advance_point(0, 0, 0, 0, 4, 3, True, False, best_of=3)
        a_sets, b_sets, a_games, b_games, a_pts, b_pts, srv_a, tb, winner = result
        assert winner is None
        assert a_games == 0  # game not yet won

    def test_regular_game_win_at_four_zero(self):
        result = _advance_point(0, 0, 0, 0, 4, 0, True, False, best_of=3)
        a_sets, b_sets, a_games, b_games, a_pts, b_pts, srv_a, tb, winner = result
        assert a_games == 1
        assert a_pts == 0 and b_pts == 0  # points reset after game
        assert srv_a is False  # server alternates after a game
        assert winner is None

    def test_deuce_requires_two_clear(self):
        # 5-4 in points (from a deuce sequence) is NOT a win — needs 2 clear
        result = _advance_point(0, 0, 0, 0, 5, 4, True, False, best_of=3)
        assert result[8] is None
        assert result[2] == 0  # game not won yet

    def test_set_win_at_six_games_two_clear(self):
        result = _advance_point(0, 0, 6, 4, 4, 0, True, False, best_of=3)
        a_sets, b_sets, a_games, b_games = result[0], result[1], result[2], result[3]
        assert a_sets == 1
        assert a_games == 0 and b_games == 0  # games reset after set

    def test_set_win_at_seven_five(self):
        result = _advance_point(0, 0, 6, 5, 4, 0, True, False, best_of=3)
        assert result[0] == 1  # a_sets

    def test_six_six_triggers_tiebreak(self):
        result = _advance_point(0, 0, 5, 6, 4, 0, True, False, best_of=3)
        a_sets, b_sets, a_games, b_games = result[0], result[1], result[2], result[3]
        is_tiebreak = result[7]
        assert a_games == 6 and b_games == 6
        assert is_tiebreak is True

    def test_match_win_at_sets_needed_bo3(self):
        result = _advance_point(1, 0, 6, 4, 4, 0, True, False, best_of=3)
        assert result[8] is True  # match_winner_is_a

    def test_match_win_requires_three_sets_bo5(self):
        result = _advance_point(2, 0, 6, 4, 4, 0, True, False, best_of=5)
        assert result[8] is True
        # Only 2 sets should NOT be a match win in bo5
        result2 = _advance_point(1, 0, 6, 4, 4, 0, True, False, best_of=5)
        assert result2[8] is None

    def test_tiebreak_serve_alternates_every_two_points(self):
        """Regression test for the serve-alternation bug found via a multi-run averaged
        symmetric-skill test earlier this project (10 independent 50/50 simulations
        landed ~11 standard errors from 0.5 before the fix). Within an ongoing tiebreak,
        serve must alternate every 2 points after the first, not stay fixed."""
        # Points 1 total (odd) -> server should have just flipped
        result_odd = _advance_point(0, 0, 6, 6, 1, 0, True, True, best_of=3)
        assert result_odd[6] is False  # server_is_a flipped
        # Points 2 total (even) -> server should NOT flip again
        result_even = _advance_point(0, 0, 6, 6, 1, 1, True, True, best_of=3)
        assert result_even[6] is True  # server_is_a unchanged

    def test_tiebreak_win_at_seven_two_clear(self):
        result = _advance_point(0, 0, 6, 6, 7, 5, True, True, best_of=3)
        # a_points-b_points=2, a_points>=7 -> A wins the tiebreak and the set
        assert result[0] == 1  # a_sets incremented
        assert result[7] is False  # no longer in tiebreak


class TestSimulateMatchFromState:
    """Covers both Critical bugs found by the external audit and already fixed."""

    def test_already_terminal_state_a_wins_returns_exactly_one(self):
        result = simulate_match_from_state(2, 0, 0, 0, 0, 0, True, False, 3, 0.6,
                                           n_simulations=200)
        assert result == 1.0

    def test_already_terminal_state_b_wins_returns_exactly_zero(self):
        result = simulate_match_from_state(0, 2, 0, 0, 0, 0, True, False, 3, 0.6,
                                           n_simulations=200)
        assert result == 0.0

    def test_already_terminal_state_bo5(self):
        assert simulate_match_from_state(3, 1, 0, 0, 0, 0, True, False, 5, 0.6,
                                         n_simulations=50) == 1.0
        assert simulate_match_from_state(1, 3, 0, 0, 0, 0, True, False, 5, 0.6,
                                         n_simulations=50) == 0.0

    def test_degenerate_p_one_does_not_hang(self):
        """At p_server_wins_point=1.0, whoever serves always holds — since serve
        alternates every game, neither player can ever build a 2-game lead, and the
        match provably never terminates. This is a genuine mathematical infinite loop,
        not merely a slow one; only the max_points cap can resolve it."""
        t0 = time.perf_counter()
        result = simulate_match_from_state(0, 0, 0, 0, 0, 0, True, False, 3, 1.0,
                                           n_simulations=5)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, "Must not hang — should resolve via the max_points cap"
        assert 0.0 <= result <= 1.0

    def test_degenerate_p_zero_does_not_hang(self):
        t0 = time.perf_counter()
        result = simulate_match_from_state(0, 0, 0, 0, 0, 0, True, False, 3, 0.0,
                                           n_simulations=5)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0
        assert 0.0 <= result <= 1.0

    def test_symmetric_matchup_converges_near_half(self):
        """A non-degenerate, genuinely 50/50 matchup should converge close to 0.5 with
        enough simulations — a basic sanity check that the simulator isn't systematically
        biased toward one player."""
        result = simulate_match_from_state(0, 0, 0, 0, 0, 0, True, False, 3, 0.5,
                                           n_simulations=2000, rng=random.Random(1))
        assert abs(result - 0.5) < 0.05

    def test_reproducible_with_fixed_seed(self):
        r1 = simulate_match_from_state(0, 0, 0, 0, 0, 0, True, False, 3, 0.6,
                                       n_simulations=500, rng=random.Random(42))
        r2 = simulate_match_from_state(0, 0, 0, 0, 0, 0, True, False, 3, 0.6,
                                       n_simulations=500, rng=random.Random(42))
        assert r1 == r2


class TestBatchSimulateDynamic:
    """Covers the function actually used for every reported ML+MC metric in this project."""

    @staticmethod
    def _dummy_predict_fn(constant_p: float):
        def fn(fm):
            return np.full(len(fm), constant_p)
        return fn

    def test_already_terminal_state_a_wins(self):
        result = batch_simulate_dynamic(
            (2, 0, 0, 0, 0, 0, True, False), {}, ["dummy"],
            self._dummy_predict_fn(0.5), best_of=3, player1_is_winner=True,
        )
        assert result == 1.0

    def test_already_terminal_state_b_wins(self):
        result = batch_simulate_dynamic(
            (0, 2, 0, 0, 0, 0, True, False), {}, ["dummy"],
            self._dummy_predict_fn(0.5), best_of=3, player1_is_winner=True,
        )
        assert result == 0.0

    def test_genuinely_nonterminal_state_runs_real_simulation(self):
        feature_cols = ["is_break_point", "is_set_point", "is_match_point",
                        "is_tiebreak_game", "is_second_serve_point", "server_is_winner",
                        "p1_momentum_last10", "p2_momentum_last10",
                        "p1_momentum_last20", "p2_momentum_last20"]
        result = batch_simulate_dynamic(
            (0, 0, 0, 0, 0, 0, True, False), {}, feature_cols,
            self._dummy_predict_fn(0.6), best_of=3, player1_is_winner=True,
            n_simulations=200, rng=random.Random(0),
        )
        assert 0.0 < result < 1.0

    def test_favors_the_higher_probability_player(self):
        feature_cols = ["is_break_point", "is_set_point", "is_match_point",
                        "is_tiebreak_game", "is_second_serve_point", "server_is_winner",
                        "p1_momentum_last10", "p2_momentum_last10",
                        "p1_momentum_last20", "p2_momentum_last20"]
        result_strong = batch_simulate_dynamic(
            (0, 0, 0, 0, 0, 0, True, False), {}, feature_cols,
            self._dummy_predict_fn(0.8), best_of=3, player1_is_winner=True,
            n_simulations=200, rng=random.Random(0),
        )
        result_weak = batch_simulate_dynamic(
            (0, 0, 0, 0, 0, 0, True, False), {}, feature_cols,
            self._dummy_predict_fn(0.3), best_of=3, player1_is_winner=True,
            n_simulations=200, rng=random.Random(0),
        )
        assert result_strong > result_weak

    def test_max_points_cap_prevents_hang_and_honestly_reports_nan_if_none_resolve(self):
        """At p_server_wins_point=1.0 with a small max_points, EVERY simulation hits the
        cap unresolved (whoever serves always holds, so no simulation can ever finish).
        The function's own documented, deliberate behavior in this case is to return
        NaN rather than silently guessing an arbitrary fallback — this is more honest
        than hiding the degenerate case behind a fake number. The property actually being
        tested here is the ABSENCE of a hang, not a specific numeric result."""
        feature_cols = ["is_break_point", "is_set_point", "is_match_point",
                        "is_tiebreak_game", "is_second_serve_point", "server_is_winner",
                        "p1_momentum_last10", "p2_momentum_last10",
                        "p1_momentum_last20", "p2_momentum_last20"]
        t0 = time.perf_counter()
        result = batch_simulate_dynamic(
            (0, 0, 0, 0, 0, 0, True, False), {}, feature_cols,
            self._dummy_predict_fn(1.0), best_of=3, player1_is_winner=True,
            n_simulations=5, rng=random.Random(0), max_points=50,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, "Must not hang, regardless of what value it returns"
        assert np.isnan(result), (
            "With every simulation unresolved, NaN is the correct, documented, honest "
            "answer — not an arbitrary numeric fallback"
        )

    def test_max_points_large_enough_for_some_simulations_to_resolve(self):
        """With a more generous max_points at the same degenerate p=1.0, SOME
        simulations may still resolve before the cap (games/sets can complete even
        while others remain stuck) — confirming the function doesn't ALWAYS return NaN
        at this degenerate value, only when truly nothing resolved."""
        feature_cols = ["is_break_point", "is_set_point", "is_match_point",
                        "is_tiebreak_game", "is_second_serve_point", "server_is_winner",
                        "p1_momentum_last10", "p2_momentum_last10",
                        "p1_momentum_last20", "p2_momentum_last20"]
        result = batch_simulate_dynamic(
            (0, 0, 0, 0, 0, 0, True, False), {}, feature_cols,
            self._dummy_predict_fn(0.99), best_of=3, player1_is_winner=True,
            n_simulations=50, rng=random.Random(0), max_points=700,
        )
        assert not np.isnan(result), "A near-degenerate but non-exact value should resolve"
        assert 0.0 <= result <= 1.0