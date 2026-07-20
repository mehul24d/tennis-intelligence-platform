"""
test_trajectory_events.py — permanent regression test for detect_set_boundaries,
specifically the score_str derivation bug found via a real replay checkpoint table
showing "After Set 1 (5-4)" for a set that actually ended 6-4.

Root cause: prev_row's Gm1/Gm2 is the score ENTERING the set-deciding point (this
project's established convention — a row's own Set/Gm columns describe the state before
that row's own point is played), one game short of the set's real final tally. Fixed by
deriving the true final score: prev_row's count with +1 added to whichever player
actually won the set (correct for both a regular game-clinched set and a tiebreak-
clinched set, since tiebreak games don't increment Gm1/Gm2 until the tiebreak concludes).
"""

import pandas as pd
import pytest

from tennis_intel.viz.trajectory_events import detect_set_boundaries


class TestDetectSetBoundaries:
    def test_regular_game_clinched_set_score(self):
        """A set won via a regular game (not a tiebreak) must show the TRUE final score,
        not the score entering the set-deciding game."""
        df = pd.DataFrame({
            "point_index": [1, 2, 3, 4, 5],
            "Set1": [0, 0, 0, 0, 1], "Set2": [0, 0, 0, 0, 0],
            "Gm1": [5, 5, 5, 5, 0], "Gm2": [4, 4, 4, 4, 0],
        })
        boundaries = detect_set_boundaries(df)
        assert boundaries[0].score_str == "6-4"
        assert boundaries[0].winner_is_p1 is True

    def test_tiebreak_clinched_set_score(self):
        """A set won via a tiebreak must ALSO show the true final score — tiebreak
        points don't increment Gm1/Gm2 until the tiebreak concludes, so the same +1
        derivation must still apply correctly."""
        df = pd.DataFrame({
            "point_index": [1, 2, 3, 4, 5],
            "Set1": [0, 0, 0, 0, 0], "Set2": [0, 0, 0, 0, 1],
            "Gm1": [6, 6, 6, 6, 0], "Gm2": [6, 6, 6, 6, 0],
        })
        boundaries = detect_set_boundaries(df)
        assert boundaries[0].score_str == "6-7"
        assert boundaries[0].winner_is_p1 is False

    def test_point_index_still_correct_after_fix(self):
        """The score_str fix must not have disturbed point_index, which correctly points
        to the first point of the NEXT set (verified separately as correct, matching
        this project's 'score describes state before this point' convention) —
        regression guard against re-breaking this while fixing score_str."""
        df = pd.DataFrame({
            "point_index": [1, 2, 3, 4, 5],
            "Set1": [0, 0, 0, 0, 1], "Set2": [0, 0, 0, 0, 0],
            "Gm1": [5, 5, 5, 5, 0], "Gm2": [4, 4, 4, 4, 0],
        })
        boundaries = detect_set_boundaries(df)
        assert boundaries[0].point_index == 5

    def test_multiple_sets(self):
        """Two consecutive sets, each with a real game-clinching score, both derived
        correctly, not just the first boundary in a match."""
        df = pd.DataFrame({
            "point_index": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Set1": [0, 0, 0, 0, 1, 1, 1, 1, 2],
            "Set2": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Gm1":  [5, 5, 5, 5, 0, 6, 6, 6, 0],
            "Gm2":  [4, 4, 4, 4, 0, 4, 4, 4, 0],
        })
        boundaries = detect_set_boundaries(df)
        assert len(boundaries) == 2
        assert boundaries[0].score_str == "6-4"
        assert boundaries[1].score_str == "7-4"