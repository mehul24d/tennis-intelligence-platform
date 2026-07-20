"""
test_points_streak.py — permanent regression tests for compute_consecutive_points_streak,
a signed run-length feature distinct from the existing momentum window features.

Verified directly, not just reasoned about: the hand-traced building/breaking/flipping
sequence, and the critical match-boundary case (a streak must never leak from one match
into the next).
"""

import pandas as pd
import pytest

from tennis_intel.features.point_level_features import compute_consecutive_points_streak


class TestConsecutivePointsStreak:
    def test_hand_traced_sequence(self):
        """P1 builds a 3-point streak, P2 breaks it and builds their own — every
        transition (build, break, flip) verified against a hand-traced expectation.
        Svr=1 (P1 serving) throughout, so PtWinner==1 directly means "P1 won" here —
        isolates this test to the run-length logic itself, not the separate
        PtWinner-is-relative-to-server fix (covered by its own dedicated test
        below)."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 6,
            "PtWinner": [1, 1, 1, 2, 2, 1],
            "Svr": [1] * 6,
        })
        result = compute_consecutive_points_streak(df)
        assert result["points_streak"].tolist() == [0, 1, 2, 3, -1, -2]

    def test_first_point_of_match_is_zero_not_nan(self):
        df = pd.DataFrame({"match_id": ["m1"], "PtWinner": [1], "Svr": [1]})
        result = compute_consecutive_points_streak(df)
        assert result["points_streak"].iloc[0] == 0
        assert not pd.isna(result["points_streak"].iloc[0])

    def test_no_leakage_across_match_boundary(self):
        """A streak ending one match must NOT carry into the next match's first point —
        the single most important property of this feature."""
        df = pd.DataFrame({
            "match_id": ["m1", "m1", "m1", "m2", "m2"],
            "PtWinner": [1, 1, 1, 2, 2],
            "Svr": [1, 1, 1, 1, 1],
        })
        result = compute_consecutive_points_streak(df)
        m2 = result[result["match_id"] == "m2"]["points_streak"].tolist()
        assert m2 == [0, -1]

    def test_long_unbroken_streak(self):
        """A long, unbroken streak counts up correctly without drift or off-by-one
        errors accumulating over many points."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 10,
            "PtWinner": [1] * 10,
            "Svr": [1] * 10,
        })
        result = compute_consecutive_points_streak(df)
        assert result["points_streak"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_match_boundary_reset_even_when_pattern_repeats(self):
        """The HARDER version of the match-boundary test: match 1 ends on a P1 win, and
        match 2 ALSO starts with P1 wins — the raw p1_won value doesn't change across
        the boundary, so only is_new_match's explicit inclusion in the 'changed'
        condition correctly forces a reset. A version of this logic missing that OR
        clause would silently extend match 1's streak into match 2 in exactly this
        scenario, while passing the simpler (non-repeating-pattern) boundary test."""
        df = pd.DataFrame({
            "match_id": ["m1", "m1", "m2", "m2", "m2"],
            "PtWinner": [1, 1, 1, 1, 2],
            "Svr": [1, 1, 1, 1, 1],
        })
        result = compute_consecutive_points_streak(df)
        m2_streaks = result[result["match_id"] == "m2"]["points_streak"].tolist()
        assert m2_streaks == [0, 1, 2]

    def test_no_leakage_from_future_point_outcomes(self):
        """Direct future-perturbation test, the same rigor applied to the Elo-trend
        feature: changing a LATER point's outcome must not affect any EARLIER point's
        streak value."""
        base = {
            "match_id": ["m1"] * 8,
            "PtWinner": [1, 1, 1, 2, 2, 1, 1, 1],
            "Svr": [1] * 8,
        }
        df_a = pd.DataFrame(base)
        df_b = pd.DataFrame(base)
        df_b.loc[7, "PtWinner"] = 2

        result_a = compute_consecutive_points_streak(df_a)
        result_b = compute_consecutive_points_streak(df_b)
        assert result_a["points_streak"].iloc[:7].tolist() == result_b["points_streak"].iloc[:7].tolist()

    def test_ptwinner_is_literal_player_relative_not_server_relative(self):
        """Direct regression test for PtWinner's convention: it is LITERAL, fixed-
        player-relative (PtWinner==1 means player 1 won, PERIOD, independent of who
        served) -- NOT server-relative. See point_level_features.py's
        compute_in_match_momentum docstring and docs/ptwinner_convention_correction.md
        for the full investigation: a same-day "fix" briefly claimed the opposite
        (server-relative), citing check_ptwinner_disagreement_at_scale.py's "0.00%
        disagreement" -- that script only checks internal self-consistency between
        PtWinner and fixed-player Pts on interior, non-game-boundary points, which
        cannot distinguish server-relative from literal (they coincide whenever
        Svr==1 and are mirror opposites whenever Svr==2, exactly what that script
        never examines). Checked directly against Gm1/Gm2 at real game boundaries
        instead: literal PtWinner matches at 99.91% corpus-wide, symmetric across
        Svr==1/2; server-relative matches only ~51% (chance) at boundaries. The
        same-day mis-fix was traced and reverted.

        Here: player 2 serves every point (Svr=2) but PtWinner==2 throughout --
        under the literal convention this means PLAYER 2 wins every point (not the
        receiver), so player 1's signed streak should be escalating NEGATIVE."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 4,
            "Svr": [2, 2, 2, 2],  # player 2 serving throughout
            "PtWinner": [2, 2, 2, 2],  # player 2 wins every point, literally
        })
        result = compute_consecutive_points_streak(df)
        # Player 1 should show an ESCALATING NEGATIVE streak, since player 2 (not the
        # receiver -- player 2 IS the server here) wins every single point.
        assert result["points_streak"].tolist() == [0, -1, -2, -3]