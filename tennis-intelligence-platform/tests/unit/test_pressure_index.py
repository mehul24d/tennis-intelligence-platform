"""
test_pressure_index.py — permanent regression tests for pressure_index, the ordinal
"how much is riding on this point" feature built on top of the existing, already-
validated is_break_point/is_set_point/is_match_point flags.

Each tier is tested in clean isolation (a real, valid score state that triggers exactly
one condition, verified by hand before writing the assertion — see the corresponding
manual trace in the diagnostic history for each case), plus one test confirming the
priority order correctly resolves overlapping conditions (a point that is simultaneously
a break/set/match point) to the single HIGHEST tier, not a sum or the wrong precedence.
"""

import pandas as pd
import pytest

from tennis_intel.features.point_level_features import compute_point_state


def _make_df(**overrides):
    base = {
        "match_id": ["m1"], "Pt": [1], "Svr": [1],
        "Set1": [0], "Set2": [0], "Gm1": [0], "Gm2": [0],
        "Pts": ["0-0"], "best_of": [3], "2nd": [None],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestPressureIndex:
    def test_routine_point(self):
        df = _make_df(Pts=["30-30"], Gm1=[2], Gm2=[2])
        result = compute_point_state(df, best_of_map={"m1": 3})
        assert result["pressure_index"].iloc[0] == 1

    def test_deuce_level(self):
        df = _make_df(Pts=["40-40"], Gm1=[5], Gm2=[3])
        result = compute_point_state(df, best_of_map={"m1": 3})
        assert result["is_deuce_level"].iloc[0] == True
        assert result["pressure_index"].iloc[0] == 2

    def test_server_game_point(self):
        df = _make_df(Pts=["40-30"], Gm1=[2], Gm2=[2])
        result = compute_point_state(df, best_of_map={"m1": 3})
        assert result["is_server_game_point"].iloc[0] == True
        assert result["is_break_point"].iloc[0] == False
        assert result["pressure_index"].iloc[0] == 3

    def test_break_point(self):
        df = _make_df(Pts=["0-40"], Gm1=[5], Gm2=[3])
        result = compute_point_state(df, best_of_map={"m1": 3})
        assert result["is_break_point"].iloc[0] == True
        assert result["pressure_index"].iloc[0] == 5

    def test_set_point_not_match_point(self):
        """Winning this game would win the set (6-3) but not the match (only 1 of 2
        needed sets)."""
        df = _make_df(Pts=["40-15"], Gm1=[5], Gm2=[3], Set1=[0], Set2=[0])
        result = compute_point_state(df, best_of_map={"m1": 3})
        assert result["is_set_point"].iloc[0] == True
        assert result["is_match_point"].iloc[0] == False
        assert result["pressure_index"].iloc[0] == 8

    def test_match_point_takes_priority_over_break_and_set_point(self):
        """A point that is simultaneously a break point, a set point, AND a match point
        must resolve to the single highest tier (10), not a sum (23) or the wrong
        precedence."""
        df = _make_df(Pts=["0-40"], Set1=[0], Set2=[1], Gm1=[3], Gm2=[5])
        result = compute_point_state(df, best_of_map={"m1": 3})
        assert result["is_break_point"].iloc[0] == True
        assert result["is_set_point"].iloc[0] == True
        assert result["is_match_point"].iloc[0] == True
        assert result["pressure_index"].iloc[0] == 10