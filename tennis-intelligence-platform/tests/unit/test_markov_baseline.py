import pytest

from tennis_intel.live.markov_baseline import (
    prob_win_game, prob_win_set, prob_win_match, prob_win_tiebreak,
)


class TestProbWinGame:
    @pytest.mark.parametrize("p,expected", [
        (0.50, 0.5000), (0.60, 0.7357), (0.70, 0.9008),
    ])
    def test_literature_reference_values(self, p, expected):
        # Canonical values from the tennis-probability literature (p=0.60 -> 0.7357 is the
        # textbook figure). Tight tolerance because these are exact closed-form results.
        assert prob_win_game(p) == pytest.approx(expected, abs=1e-3)

    def test_symmetry(self):
        assert prob_win_game(0.5) == pytest.approx(0.5, abs=1e-9)

    def test_monotonic(self):
        vals = [prob_win_game(p) for p in [0.5, 0.55, 0.6, 0.65, 0.7]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            prob_win_game(1.5)


class TestProbWinTiebreak:
    def test_symmetry(self):
        assert prob_win_tiebreak(0.5, 0.5) == pytest.approx(0.5, abs=1e-9)

    def test_stronger_player_favored(self):
        assert prob_win_tiebreak(0.7, 0.5) > 0.5


class TestProbWinSet:
    def test_symmetry(self):
        assert prob_win_set(0.5, 0.5) == pytest.approx(0.5, abs=1e-6)

    def test_stronger_server_favored(self):
        assert prob_win_set(0.68, 0.38) > 0.5


class TestProbWinMatch:
    def test_symmetry_both_formats(self):
        assert prob_win_match(0.5, 0.5, best_of=3) == pytest.approx(0.5, abs=1e-6)
        assert prob_win_match(0.5, 0.5, best_of=5) == pytest.approx(0.5, abs=1e-6)

    def test_best_of_five_amplifies_favorite(self):
        # A real, known tennis phenomenon: more sets -> less variance -> favorite's edge grows
        p3 = prob_win_match(0.68, 0.38, best_of=3)
        p5 = prob_win_match(0.68, 0.38, best_of=5)
        assert p5 > p3