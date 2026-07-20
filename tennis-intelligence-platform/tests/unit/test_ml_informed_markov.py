import pytest

from tennis_intel.live.live_win_probability import MatchState
from tennis_intel.live.ml_informed_markov import ServeReturnPosterior, ml_informed_markov_predict


def _mid_match_state(server_is_a: bool) -> MatchState:
    """An arbitrary non-terminal, non-extreme in-match state so the recursion's output
    is sensitive to the posterior update (not saturated near 0 or 1)."""
    return MatchState(
        a_sets=0, b_sets=0, a_games=2, b_games=2, a_points=1, b_points=1,
        server_is_a=server_is_a, is_tiebreak=False, best_of=3,
    )


def _neutral_posterior() -> ServeReturnPosterior:
    return ServeReturnPosterior.from_pretrained_prior(
        p_serve0=0.62, n0_serve=20.0, p_return0=0.38, n0_return=20.0,
    )


def _row(svr: int, pt_winner: int, player1_is_winner: bool) -> dict:
    """A minimal synthetic point row. server_is_player1 mirrors the real
    build_point_dataset.py convention: True iff player 1 is physically serving
    (svr == 1)."""
    return {
        "Svr": svr,
        "PtWinner": pt_winner,
        "player1_is_winner": player1_is_winner,
        "server_is_player1": (svr == 1),
    }


class _StubModel:
    """A model stub whose predict_proba is irrelevant to this test — only the
    posterior UPDATE (derived from real PtWinner, not the classifier) is under
    test here. Returns a fixed, mid-range probability regardless of input."""

    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.4, 0.6]] * len(X))


FEATURE_COLS = ["server_is_player1"]


@pytest.mark.parametrize(
    "server_is_a, svr, pt_winner, player1_is_winner, expected_a_won",
    [
        # PtWinner is LITERAL, fixed-player-relative: PtWinner==1 means player 1 won,
        # PERIOD -- independent of who served. See ml_informed_markov.py's
        # ml_informed_markov_predict docstring/comment for the full investigation that
        # settled this (a same-day false "fix" to server-relative was traced and
        # reverted; docs/ptwinner_convention_correction.md has the complete record).
        #
        # A serves, A is player1, PtWinner==1 (player1 won) -> A won.
        (True, 1, 1, True, True),
        # A serves, A is player1, PtWinner==2 (player2 won) -> A lost.
        (True, 1, 2, True, False),
        # A returns (player2 serves), A is still player1, PtWinner==1 (player1 won,
        # i.e. the RETURNER won -- a break) -> A won. Svr is irrelevant to this.
        (False, 2, 1, True, True),
        # A returns, A is player1, PtWinner==2 (player2/the server won, a hold) -> A lost.
        (False, 2, 2, True, False),
        # A is player2 this time (player1_is_winner=False) -- PtWinner==2 means A won,
        # regardless of server_is_a/Svr.
        (True, 2, 2, False, True),
        (False, 1, 2, False, True),
    ],
)
def test_a_won_this_point_is_literal_player_relative_not_server_relative(
    server_is_a, svr, pt_winner, player1_is_winner, expected_a_won,
):
    """Regression test for PtWinner's convention: it is LITERAL and fixed-player-
    relative (PtWinner==1 means player 1 won, full stop) -- NOT server-relative.
    Verified end-to-end via the posterior's observed mean update rather than reaching
    into private state, so this test would have caught the same-day server-relative
    mis-fix that was traced and reverted (see the docstring in ml_informed_markov.py's
    ml_informed_markov_predict, and docs/ptwinner_convention_correction.md)."""
    state = _mid_match_state(server_is_a)
    row = _row(svr, pt_winner, player1_is_winner)
    posterior = _neutral_posterior()

    _, updated = ml_informed_markov_predict(state, row, _StubModel(), FEATURE_COLS, posterior)

    if server_is_a:
        mean_before, mean_after = posterior.mean_serve(), updated.mean_serve()
    else:
        mean_before, mean_after = posterior.mean_return(), updated.mean_return()

    if expected_a_won:
        assert mean_after > mean_before, "posterior mean should rise when A wins the point"
    else:
        assert mean_after < mean_before, "posterior mean should fall when A loses the point"


def test_posterior_update_direction_matches_across_serve_and_return():
    """End-to-end sanity check spanning both serve and return: a posterior fed a long
    run of real A-wins on serve should have a higher serve mean than one fed a long run
    of A-losses, and likewise for return -- confirming update_serve/update_return are
    driven by the correct, literal-player-relative outcome regardless of who's serving."""
    posterior = _neutral_posterior()

    # A serves and wins every point (svr=1, PtWinner=1, A=player1 -> A, the server, won).
    for _ in range(15):
        state = _mid_match_state(server_is_a=True)
        row = _row(svr=1, pt_winner=1, player1_is_winner=True)
        _, posterior = ml_informed_markov_predict(state, row, _StubModel(), FEATURE_COLS, posterior)
    serve_mean_after_wins = posterior.mean_serve()

    # A returns (opponent, player 2, serves) and A wins every point: PtWinner=1 (player1,
    # i.e. A the returner, won -- a break every time).
    for _ in range(15):
        state = _mid_match_state(server_is_a=False)
        row = _row(svr=2, pt_winner=1, player1_is_winner=True)
        _, posterior = ml_informed_markov_predict(state, row, _StubModel(), FEATURE_COLS, posterior)
    return_mean_after_wins = posterior.mean_return()

    baseline = _neutral_posterior()
    assert serve_mean_after_wins > baseline.mean_serve()
    assert return_mean_after_wins > baseline.mean_return()
