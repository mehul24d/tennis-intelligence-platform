"""
test_interaction_features.py — permanent regression tests for
compute_interaction_features, explicit multiplicative interaction terms between
already-validated features.

Leakage safety is trivial and inherited (the product of two already-leakage-safe
columns is itself leakage-safe by construction), so these tests check only
correctness of the multiplication and sensible NaN propagation, not leakage.
"""

import pandas as pd
import pytest

from tennis_intel.features.point_level_features import compute_interaction_features


class TestInteractionFeatures:
    def test_points_streak_x_break_point(self):
        df = pd.DataFrame({
            "points_streak": [3, -2, 0, 5],
            "is_break_point": [True, False, True, False],
            "pressure_index": [10, 5, 1, 8],
            "p1_momentum_last10": [0.7, 0.3, 0.5, 0.5],
        })
        result = compute_interaction_features(df)
        assert result["points_streak_x_break_point"].tolist() == [3, 0, 0, 0]

    def test_pressure_index_x_momentum10(self):
        df = pd.DataFrame({
            "points_streak": [0, 0, 0],
            "is_break_point": [False, False, False],
            "pressure_index": [10, 5, 8],
            "p1_momentum_last10": [0.7, 0.3, 0.5],
        })
        result = compute_interaction_features(df)
        expected = [7.0, 1.5, 4.0]
        for a, e in zip(result["pressure_index_x_momentum10"].tolist(), expected):
            assert abs(a - e) < 1e-9

    def test_nan_propagates_sensibly(self):
        """A missing momentum value must produce a missing interaction value, not a
        silently misleading default like 0."""
        df = pd.DataFrame({
            "points_streak": [0],
            "is_break_point": [False],
            "pressure_index": [1],
            "p1_momentum_last10": [None],
        })
        result = compute_interaction_features(df)
        assert pd.isna(result["pressure_index_x_momentum10"].iloc[0])