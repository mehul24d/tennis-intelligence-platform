"""
test_in_match_serve_return_rate.py — permanent regression tests for
compute_in_match_serve_return_rate, a raw, cumulative (expanding) in-match serve/return
win rate — distinct from both the fixed-window momentum features and the career-level
_career rate features.

Reuses the exact two-series forward-fill design already proven for
compute_split_points_streak, with an expanding mean in place of a run-length. The key
semantic difference from the streak features, tested explicitly: a rate with zero
observations is genuinely undefined (NaN), not a meaningful default like 0.
"""

import pandas as pd
import pytest

from tennis_intel.features.point_level_features import (
    compute_in_match_serve_return_rate, compute_in_match_serve_return_rate_rolling,
)


class TestInMatchServeReturnRate:
    def test_serve_rate_hand_traced(self):
        """NOTE: unaffected by the separate PtWinner-is-relative-to-server fix (see
        test_return_rate_hand_traced's own note) — every SERVE point in this
        sequence has Svr=1, so server_is_p1=True throughout, making the fixed
        formula reduce to the original one for these particular rows. Confirmed
        directly, not assumed."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 8,
            "Svr":      [1, 1, 2, 2, 1, 1, 2, 1],
            "PtWinner": [1, 1, 1, 2, 1, 2, 2, 1],
        })
        result = compute_in_match_serve_return_rate(df)
        actual = result["p1_in_match_serve_rate"].tolist()
        expected = [None, 1.00, 1.00, 1.00, 1.00, 1.00, 0.75, 0.75]
        for a, e in zip(actual, expected):
            if e is None:
                assert pd.isna(a)
            else:
                assert abs(a - e) < 1e-9

    def test_return_rate_hand_traced(self):
        """RE-DERIVED AGAIN (2026-07): PtWinner's convention was settled as LITERAL,
        fixed-player-relative — see test_split_points_streak.py's
        test_return_streak_hand_traced for the full re-derivation notes and
        docs/ptwinner_convention_correction.md for the complete investigation. Return
        points are indices 2,3,6 (Svr==2); PtWinner there is [1,2,2] -> player1
        literally won only index 2. Entering index 3: 1 win/1 return = 1.0. After
        index 3 resolves (a loss): 1 win/2 returns = 0.5, forward-filled from index 4
        through index 6's own (consistent) value. After index 6 resolves (a loss):
        1 win/3 returns = 1/3, forward-filled onto index 7."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 8,
            "Svr":      [1, 1, 2, 2, 1, 1, 2, 1],
            "PtWinner": [1, 1, 1, 2, 1, 2, 2, 1],
        })
        result = compute_in_match_serve_return_rate(df)
        actual = result["p1_in_match_return_rate"].tolist()
        expected = [None, None, None, 1.0, 0.5, 0.5, 0.5, 1 / 3]
        for a, e in zip(actual, expected):
            if e is None:
                assert pd.isna(a)
            else:
                assert abs(a - e) < 1e-9

    def test_no_data_is_nan_not_a_default(self):
        """A genuinely important semantic difference from the streak features: a rate
        with zero prior observations is undefined and must be NaN, not forced to 0 or
        0.5 — forcing a fake default would misrepresent genuine absence of evidence."""
        df = pd.DataFrame({
            "match_id": ["m1"],
            "Svr": [1],
            "PtWinner": [1],
        })
        result = compute_in_match_serve_return_rate(df)
        assert pd.isna(result["p1_in_match_serve_rate"].iloc[0])
        assert pd.isna(result["p1_in_match_return_rate"].iloc[0])

    def test_no_leak_across_match_boundary(self):
        df = pd.DataFrame({
            "match_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "Svr":      [1,    1,    1,    2,    2,    1],
            "PtWinner": [1,    1,    2,    2,    1,    1],
        })
        result = compute_in_match_serve_return_rate(df)
        m2_serve = result[result["match_id"] == "m2"]["p1_in_match_serve_rate"].tolist()
        assert pd.isna(m2_serve[0]) and pd.isna(m2_serve[1])

    def test_no_leakage_from_future_point_outcomes(self):
        base = {
            "match_id": ["m1"] * 8,
            "Svr":      [1, 1, 2, 2, 1, 1, 2, 1],
            "PtWinner": [1, 1, 1, 2, 1, 2, 2, 1],
        }
        df_a = pd.DataFrame(base)
        df_b = pd.DataFrame(base)
        df_b.loc[7, "PtWinner"] = 2

        result_a = compute_in_match_serve_return_rate(df_a)
        result_b = compute_in_match_serve_return_rate(df_b)
        serve_a = result_a["p1_in_match_serve_rate"].iloc[:7].tolist()
        serve_b = result_b["p1_in_match_serve_rate"].iloc[:7].tolist()
        for a, b in zip(serve_a, serve_b):
            if pd.isna(a):
                assert pd.isna(b)
            else:
                assert abs(a - b) < 1e-9


class TestInMatchServeReturnRateRolling:
    """Permanent regression tests for compute_in_match_serve_return_rate_rolling — a
    fixed-window sibling of the expanding version above, sharing the identical
    two-series forward-fill design with .rolling() in place of .expanding()."""

    def test_rolling_diverges_from_expanding_after_a_recent_shift(self):
        """The core value proposition, verified directly: a real recent slump (three
        straight serve losses after an earlier hot streak) should be immediately,
        fully reflected in a short rolling window, while the expanding (whole-match)
        rate stays diluted by the earlier good stretch — independently hand-traced
        before being encoded here."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 10,
            "Svr":      [1] * 10,
            "PtWinner": [1, 1, 1, 2, 2, 2, 1, 1, 1, 1],
        })
        result_expanding = compute_in_match_serve_return_rate(df)
        result_rolling = compute_in_match_serve_return_rate_rolling(df, windows=(3,))
        # At idx6 (right after three straight losses), rolling should show 0.0
        # (all three most recent points lost) while expanding still shows 0.5
        assert abs(result_rolling["p1_in_match_serve_rate_last3"].iloc[6] - 0.0) < 1e-9
        assert abs(result_expanding["p1_in_match_serve_rate"].iloc[6] - 0.5) < 1e-9

    def test_no_leak_across_match_boundary(self):
        df = pd.DataFrame({
            "match_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "Svr":      [1,    1,    1,    2,    2,    1],
            "PtWinner": [1,    1,    2,    2,    1,    1],
        })
        result = compute_in_match_serve_return_rate_rolling(df, windows=(3,))
        m2_serve = result[result["match_id"] == "m2"]["p1_in_match_serve_rate_last3"].tolist()
        assert pd.isna(m2_serve[0]) and pd.isna(m2_serve[1])

    def test_no_leakage_from_future_point_outcomes(self):
        base = {
            "match_id": ["m1"] * 8,
            "Svr":      [1, 1, 2, 2, 1, 1, 2, 1],
            "PtWinner": [1, 1, 1, 2, 1, 2, 2, 1],
        }
        df_a = pd.DataFrame(base)
        df_b = pd.DataFrame(base)
        df_b.loc[7, "PtWinner"] = 2

        result_a = compute_in_match_serve_return_rate_rolling(df_a, windows=(3,))
        result_b = compute_in_match_serve_return_rate_rolling(df_b, windows=(3,))
        serve_a = result_a["p1_in_match_serve_rate_last3"].iloc[:7].tolist()
        serve_b = result_b["p1_in_match_serve_rate_last3"].iloc[:7].tolist()
        for a, b in zip(serve_a, serve_b):
            if pd.isna(a):
                assert pd.isna(b)
            else:
                assert abs(a - b) < 1e-9

    def test_multiple_windows_computed_in_one_pass(self):
        df = pd.DataFrame({
            "match_id": ["m1"] * 6,
            "Svr":      [1] * 6,
            "PtWinner": [1, 1, 2, 1, 2, 1],
        })
        result = compute_in_match_serve_return_rate_rolling(df, windows=(2, 4))
        assert "p1_in_match_serve_rate_last2" in result.columns
        assert "p1_in_match_serve_rate_last4" in result.columns
        assert "p1_in_match_return_rate_last2" in result.columns
        assert "p1_in_match_return_rate_last4" in result.columns