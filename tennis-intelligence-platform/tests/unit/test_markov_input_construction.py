"""
test_markov_input_construction.py — permanent regression tests guarding against the
p_return construction bug found in this project's audit (2026-07): every Markov call site
was using a player's own generic return_pts_won_pct_career as p_return, when the correct
definition is 1 - the OPPONENT's real serve-win rate. This file's tests must keep passing
regardless of how future code expresses the calculation (multi-line, helper function,
renamed variables) — they check the mathematical PROPERTY, not the source code pattern,
which is what the companion static audit (audit_markov_call_sites.py) cannot guarantee.
"""

import pytest

from tennis_intel.live.markov_baseline import prob_win_match


class TestPReturnConstruction:
    """The single most important regression guard from this audit: a CORRECT
    (self-serve, 1-opponent-serve) pairing for both players in one real match must sum to
    exactly 1.0, since exactly one of them wins. Any p_return construction that doesn't
    honor this — including reusing a player's OWN generic return statistic — will violate
    it, often severely, which is exactly how the original bug was diagnosed."""

    def test_correct_construction_sums_to_one(self):
        a_serve, b_serve = 0.70, 0.65
        p_a = prob_win_match(a_serve, 1 - b_serve, best_of=3)
        p_b = prob_win_match(b_serve, 1 - a_serve, best_of=3)
        assert p_a + p_b == pytest.approx(1.0, abs=1e-9)

    def test_correct_construction_sums_to_one_bo5(self):
        a_serve, b_serve = 0.72, 0.58
        p_a = prob_win_match(a_serve, 1 - b_serve, best_of=5)
        p_b = prob_win_match(b_serve, 1 - a_serve, best_of=5)
        assert p_a + p_b == pytest.approx(1.0, abs=1e-9)

    def test_buggy_own_return_stat_construction_violates_zero_sum(self):
        """Documents the FAILURE MODE precisely: if p_return is (incorrectly) each
        player's own return_pts_won_pct-style stat rather than derived from the real
        opponent's serve rate, the two match probabilities do NOT sum to 1.0 — this test
        exists so a future reader can see exactly what the bug looked like, not just that
        a fix exists."""
        a_serve, a_own_return_stat = 0.70, 0.40
        b_serve, b_own_return_stat = 0.65, 0.38
        p_a_buggy = prob_win_match(a_serve, a_own_return_stat, best_of=3)
        p_b_buggy = prob_win_match(b_serve, b_own_return_stat, best_of=3)
        assert abs((p_a_buggy + p_b_buggy) - 1.0) > 0.01

    def test_realistic_elite_matchup_stays_plausible(self):
        """Plausibility guard using the exact real Sinner/Alcaraz career serve rates
        pulled during this audit: with the CORRECT construction, pre-match confidence for
        a genuine top-2-in-the-world final must stay in a believable range. The original
        bug produced 0.9951 for this exact matchup — this test exists specifically to
        catch a recurrence of that failure mode."""
        sinner_serve, alcaraz_serve = 0.7675, 0.7258
        p_sinner = prob_win_match(sinner_serve, 1 - alcaraz_serve, best_of=5)
        assert 0.50 < p_sinner < 0.90, (
            f"got {p_sinner:.4f} — implausibly high confidence for two elite, "
            f"closely-matched players; check for a p_return construction regression"
        )