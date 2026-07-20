"""
test_games_streak.py — permanent regression tests for compute_games_streak, a
signed consecutive-GAMES-won run-length, one level coarser than points_streak.

Reuses the exact game-boundary detection and two-series forward-fill design already
proven (and bug-fixed once) for compute_split_points_streak. Every property tested here
mirrors that file's discipline: hand-traced correctness across multiple game
transitions (including a sign-flip and the delayed appearance of a streak value one
row after the boundary that produced it), match-boundary isolation, and a direct
future-perturbation leakage check.
"""

import pandas as pd
import pytest

from tennis_intel.features.point_level_features import compute_games_streak


class TestGamesStreak:
    def test_hand_traced_three_games(self):
        """Three consecutive game completions (P1, P1, P2), independently manually
        traced before being encoded here — see the conversation history for the
        row-by-row derivation."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 10,
            "Gm1": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
            "Gm2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        })
        result = compute_games_streak(df)
        assert result["p1_games_streak"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 2]

    def test_sign_flip_and_delayed_streak_value(self):
        """A genuinely subtle property, caught by an incorrect first test assertion
        during development: the value reflecting a just-completed game's OWN outcome
        only appears at the NEXT game boundary, one row after the boundary that
        produced it — not at the boundary row itself, which reflects the PRECEDING
        game only."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 15,
            "Gm1": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
            "Gm2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        })
        result = compute_games_streak(df)
        tail = result["p1_games_streak"].iloc[9:].tolist()
        assert tail == [2, -1, -1, -1, -2, -2]

    def test_no_leak_across_match_boundary(self):
        df = pd.DataFrame({
            "match_id": ["m1", "m1", "m1", "m1", "m2", "m2", "m2", "m2"],
            "Gm1": [0, 0, 1, 1, 0, 0, 0, 0],
            "Gm2": [0, 0, 0, 0, 0, 0, 0, 0],
        })
        result = compute_games_streak(df)
        m2 = result[result["match_id"] == "m2"]["p1_games_streak"].tolist()
        assert m2 == [0, 0, 0, 0]

    def test_no_leakage_from_future_game_outcomes(self):
        """Direct future-perturbation test: changing the LAST row's game outcome must
        not affect any earlier row's games-streak value."""
        base = {
            "match_id": ["m1"] * 9,
            "Gm1": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "Gm2": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
        df_a = pd.DataFrame(base)
        df_b = pd.DataFrame(base)
        df_b.loc[8, "Gm2"] = 0
        df_b.loc[8, "Gm1"] = 3

        result_a = compute_games_streak(df_a)
        result_b = compute_games_streak(df_b)
        assert result_a["p1_games_streak"].iloc[:8].tolist() == result_b["p1_games_streak"].iloc[:8].tolist()