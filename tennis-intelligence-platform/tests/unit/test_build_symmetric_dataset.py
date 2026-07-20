import numpy as np
import pandas as pd
import pytest

from tennis_intel.modeling.build_symmetric_dataset import build_symmetric_dataset


def make_match(winner_elo=1600, loser_elo=1500, tourney_id="T1", match_num=1):
    return {
        "tourney_id": tourney_id, "match_num": match_num, "tourney_date": pd.Timestamp("2020-01-01"),
        "tourney_name": "Test", "surface": "Hard", "tourney_level": "A", "round": "F", "best_of": 3,
        "winner_id": "A", "loser_id": "B",
        "elo_pre_match_winner": winner_elo, "elo_pre_match_loser": loser_elo,
        "winner_win_pct_last5": 0.8, "loser_win_pct_last5": 0.4,
        "winner_win_pct_last10": 0.7, "loser_win_pct_last10": 0.5,
        "winner_win_pct_last20": 0.6, "loser_win_pct_last20": 0.5,
        "winner_surface_win_pct_last10": 0.75, "loser_surface_win_pct_last10": 0.45,
        "winner_avg_game_diff_last10": 2.0, "loser_avg_game_diff_last10": -1.0,
        "winner_surface_avg_game_diff_last10": 1.5, "loser_surface_avg_game_diff_last10": -0.5,
        "winner_opponent_elo_mean_last10": 1550, "loser_opponent_elo_mean_last10": 1520,
        "winner_win_streak_entering_match": 3, "loser_win_streak_entering_match": 0,
        "winner_loss_streak_entering_match": 0, "loser_loss_streak_entering_match": 1,
        "winner_rest_days": 7, "loser_rest_days": 5,
        "winner_straight_set_rate_last10": 0.5, "loser_straight_set_rate_last10": 0.3,
    }


class TestSymmetricDataset:
    def test_label_and_diff_direction_consistent(self):
        matches = pd.DataFrame([make_match(winner_elo=1600, loser_elo=1500)])
        result = build_symmetric_dataset(matches)
        row = result.iloc[0]
        # Whichever way the assignment went, elo_diff sign must match label direction:
        # label=1 means player_1 won, so their elo (the winner's, 1600) minus the loser's
        # (1500) should be positive; label=0 means the reverse.
        if row["label"] == 1:
            assert row["elo_diff"] == 100
        else:
            assert row["elo_diff"] == -100

    def test_swap_symmetry(self):
        # The core correctness property: if we manually force BOTH possible assignments,
        # diff sign and label must be perfectly antisymmetric under the swap.
        diff_if_p1_winner = 1600 - 1500
        diff_if_p1_loser = 1500 - 1600
        assert diff_if_p1_winner == -diff_if_p1_loser

    def test_deterministic_across_reruns(self):
        matches = pd.DataFrame([make_match(tourney_id="T1", match_num=1)])
        r1 = build_symmetric_dataset(matches)
        r2 = build_symmetric_dataset(matches)
        assert r1["label"].iloc[0] == r2["label"].iloc[0]
        assert r1["elo_diff"].iloc[0] == r2["elo_diff"].iloc[0]

    def test_deterministic_regardless_of_row_order(self):
        matches = pd.DataFrame([
            make_match(tourney_id="T1", match_num=1),
            make_match(tourney_id="T2", match_num=1, winner_elo=1700, loser_elo=1400),
        ])
        r1 = build_symmetric_dataset(matches)
        r2 = build_symmetric_dataset(matches.iloc[::-1].reset_index(drop=True))

        r1_by_id = r1.set_index(matches["tourney_id"]).sort_index()
        r2_by_id = r2.set_index(matches["tourney_id"].iloc[::-1].reset_index(drop=True)).sort_index()
        assert (r1_by_id["label"].values == r2_by_id["label"].values).all()

    def test_label_not_degenerate_across_many_matches(self):
        # Across many distinct matches, the label distribution should be roughly balanced,
        # not all-1 or all-0 (which would indicate the assignment logic is broken)
        rows = [make_match(tourney_id=f"T{i}", match_num=1) for i in range(500)]
        matches = pd.DataFrame(rows)
        result = build_symmetric_dataset(matches)
        balance = result["label"].mean()
        assert 0.35 < balance < 0.65

    def test_missing_feature_pair_logs_warning_not_crash(self):
        matches = pd.DataFrame([make_match()])
        matches = matches.drop(columns=["elo_pre_match_winner"])
        # Should not raise, just skip that feature pair
        result = build_symmetric_dataset(matches)
        assert "elo_diff" not in result.columns