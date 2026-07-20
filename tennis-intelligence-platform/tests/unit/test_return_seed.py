"""
test_return_seed.py — permanent regression tests for return_seed.py.

Deliberately does NOT test for reproducing the old 0.26 value as a "passing" case (that
number was the confirmed BUG this module fixes, not a target behavior) — see
return_seed.py's module docstring for the full history, including a real near-miss where
an earlier version of this fix reintroduced a DIFFERENT, already-identified-and-rejected
regression (using the player's own generic return average, losing opponent-conditioning).
"""

import pandas as pd
import pytest

from tennis_intel.live.return_seed import compute_p_a_return_seed, DEFAULT_RETURN_SEED


class TestComputePAReturnSeed:
    def test_primary_path_uses_opponent_combined_rate(self):
        row = {
            "loser_combined_serve_win_pct_career": 0.6564,
            "loser_first_serve_win_pct_career": 0.74,  # must be ignored
        }
        result = compute_p_a_return_seed(row, track_winner=True)
        assert result == pytest.approx(1.0 - 0.6564)

    def test_fallback_to_surface_first_serve_only(self):
        row = {
            "loser_combined_serve_win_pct_career": None,
            "loser_first_serve_win_pct_surface_career": 0.74,
        }
        result = compute_p_a_return_seed(row, track_winner=True)
        assert result == pytest.approx(1.0 - 0.74)

    def test_fallback_to_career_first_serve_only(self):
        row = {
            "loser_combined_serve_win_pct_career": None,
            "loser_first_serve_win_pct_surface_career": None,
            "loser_first_serve_win_pct_career": 0.70,
        }
        result = compute_p_a_return_seed(row, track_winner=True)
        assert result == pytest.approx(1.0 - 0.70)

    def test_final_default_fallback(self):
        result = compute_p_a_return_seed({}, track_winner=True)
        assert result == DEFAULT_RETURN_SEED

    def test_track_winner_false_swaps_prefix(self):
        row = {"winner_combined_serve_win_pct_career": 0.68}
        result = compute_p_a_return_seed(row, track_winner=False)
        assert result == pytest.approx(1.0 - 0.68)

    def test_does_not_reproduce_old_flawed_value(self):
        """Regression guard: the corrected seed, given a realistic combined rate, must be
        materially higher than the confirmed-flawed 0.26 that the old first-serve-only
        construction produced on real Sinner-Alcaraz data."""
        row = {"loser_combined_serve_win_pct_career": 0.6564}
        result = compute_p_a_return_seed(row, track_winner=True)
        assert result > 0.30, (
            "Should be materially higher than the old flawed 0.26 — if this fails, the "
            "fix has regressed back toward the first-serve-only understatement bug."
        )

    def test_nan_treated_as_missing(self):
        """pd.NA / float('nan') must fall through to the next fallback, not be treated as
        a valid (garbage) value."""
        row = {
            "loser_combined_serve_win_pct_career": float("nan"),
            "loser_first_serve_win_pct_career": 0.70,
        }
        result = compute_p_a_return_seed(row, track_winner=True)
        assert result == pytest.approx(1.0 - 0.70)