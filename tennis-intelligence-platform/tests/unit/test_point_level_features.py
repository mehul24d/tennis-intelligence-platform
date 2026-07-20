import pandas as pd
import pytest

from tennis_intel.features.point_level_features import (
    load_and_sort_points, compute_point_state, compute_in_match_momentum,
)


def _write_points_csv(tmp_path):
    path = tmp_path / "points.csv"
    # Deliberately out of Pt order, matching the real MCP file's actual row ordering
    pd.DataFrame([
        {"match_id": "M1", "Pt": 3, "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0, "Pts": "30-40",
         "Gm#": 1, "TbSet": True, "Svr": 1, "1st": "x", "2nd": None, "Notes": None, "PtWinner": 2},
        {"match_id": "M1", "Pt": 1, "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0, "Pts": "0-15",
         "Gm#": 1, "TbSet": True, "Svr": 1, "1st": "x", "2nd": None, "Notes": None, "PtWinner": 2},
        {"match_id": "M1", "Pt": 2, "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0, "Pts": "0-30",
         "Gm#": 1, "TbSet": True, "Svr": 1, "1st": "x", "2nd": "y", "Notes": None, "PtWinner": 1},
    ]).to_csv(path, index=False)
    return path


class TestLoadAndSortPoints:
    def test_sorts_by_match_and_pt_despite_unsorted_file(self, tmp_path):
        df = load_and_sort_points([_write_points_csv(tmp_path)])
        assert df["Pt"].tolist() == [1, 2, 3]


class TestComputePointState:
    def test_break_point_flag(self, tmp_path):
        df = load_and_sort_points([_write_points_csv(tmp_path)])
        df = compute_point_state(df, best_of_map={"M1": 3})
        assert df[df["Pt"] == 1]["is_break_point"].iloc[0] == False
        assert df[df["Pt"] == 3]["is_break_point"].iloc[0] == True

    def test_second_serve_flag_from_column_presence(self, tmp_path):
        df = load_and_sort_points([_write_points_csv(tmp_path)])
        df = compute_point_state(df, best_of_map={"M1": 3})
        assert df[df["Pt"] == 2]["is_second_serve_point"].iloc[0] == True
        assert df[df["Pt"] == 1]["is_second_serve_point"].iloc[0] == False


class TestInMatchMomentum:
    def test_no_history_on_first_point_is_nan(self, tmp_path):
        df = load_and_sort_points([_write_points_csv(tmp_path)])
        df = compute_point_state(df, best_of_map={"M1": 3})
        momentum = compute_in_match_momentum(df)
        assert pd.isna(momentum[momentum["Pt"] == 1]["p1_momentum_last10"].iloc[0])

    def test_momentum_reflects_exact_prior_points(self, tmp_path):
        df = load_and_sort_points([_write_points_csv(tmp_path)])
        df = compute_point_state(df, best_of_map={"M1": 3})
        momentum = compute_in_match_momentum(df)
        # Before point 3: point 1 (p2 won), point 2 (p1 won) -> p1 momentum = 1/2
        pt3 = momentum[momentum["Pt"] == 3]["p1_momentum_last10"].iloc[0]
        assert pt3 == pytest.approx(0.5)

    def test_leakage_removing_future_point_does_not_change_past_momentum(self, tmp_path):
        df = load_and_sort_points([_write_points_csv(tmp_path)])
        df = compute_point_state(df, best_of_map={"M1": 3})
        full = compute_in_match_momentum(df)
        truncated = compute_in_match_momentum(df.iloc[:2].copy())

        pt2_full = full[full["Pt"] == 2]["p1_momentum_last10"].iloc[0]
        pt2_trunc = truncated[truncated["Pt"] == 2]["p1_momentum_last10"].iloc[0]
        assert pt2_full == pt2_trunc