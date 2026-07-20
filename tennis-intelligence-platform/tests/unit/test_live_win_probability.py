import pytest

from tennis_intel.live.markov_baseline import prob_win_match
from tennis_intel.live.live_win_probability import MatchState, prob_a_wins_match_from_state


PS, PR = 0.65, 0.40


class TestConsistencyWithPreMatch:
    def test_live_at_start_equals_prematch(self):
        # The single most important test: at 0-0-0 with the first server serving, the live
        # engine must reproduce the pre-match analytical model exactly.
        start = MatchState(0, 0, 0, 0, 0, 0, server_is_a=True, best_of=3)
        live = prob_a_wins_match_from_state(start, PS, PR)
        prematch = prob_win_match(PS, PR, best_of=3, server_serves_first=True)
        assert live == pytest.approx(prematch, abs=1e-9)


class TestIntuition:
    def test_match_point_near_one(self):
        mp = MatchState(1, 0, 5, 3, 3, 0, server_is_a=True, best_of=3)
        assert prob_a_wins_match_from_state(mp, PS, PR) > 0.97

    def test_facing_match_point_near_zero(self):
        facing = MatchState(0, 1, 3, 5, 0, 3, server_is_a=False, best_of=3)
        assert prob_a_wins_match_from_state(facing, PS, PR) < 0.03

    def test_monotonic_within_game(self):
        probs = []
        for a_pts, b_pts in [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)]:
            s = MatchState(0, 0, 3, 3, a_pts, b_pts, server_is_a=True, best_of=3)
            probs.append(prob_a_wins_match_from_state(s, PS, PR))
        assert all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))


class TestTiebreakDoesNotHang:
    def test_tiebreak_lead_resolves(self):
        tb = MatchState(0, 0, 6, 6, 5, 3, server_is_a=True, is_tiebreak=True, best_of=3)
        p = prob_a_wins_match_from_state(tb, PS, PR)
        assert 0.5 < p < 1.0

    def test_tiebreak_deuce_phase_resolves(self):
        # 6-6 in the tiebreak is the win-by-2 deuce phase — must resolve via closed form,
        # not recurse forever (this was a real bug caught during development).
        tb = MatchState(0, 0, 6, 6, 6, 6, server_is_a=True, is_tiebreak=True, best_of=3)
        p = prob_a_wins_match_from_state(tb, PS, PR)
        assert 0.0 < p < 1.0