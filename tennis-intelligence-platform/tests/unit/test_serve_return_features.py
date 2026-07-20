import pandas as pd
import pytest

from tennis_intel.features.serve_return_features import (
    load_and_prepare_stats, attach_player_ids_and_chronology, compute_rolling_serve_return_features,
)


def _write_stats_csv(tmp_path):
    path = tmp_path / "stats.csv"
    pd.DataFrame([
        {"match_id": "M1", "player": "A", "set": "Total", "serve_pts": 80, "aces": 8, "dfs": 4,
         "first_in": 50, "first_won": 35, "second_in": 30, "second_won": 15,
         "bk_pts": 10, "bp_saved": 6, "return_pts": 70, "return_pts_won": 30,
         "winners": 20, "winners_fh": 12, "winners_bh": 8, "unforced": 15, "unforced_fh": 8, "unforced_bh": 7},
        {"match_id": "M1", "player": "B", "set": "Total", "serve_pts": 70, "aces": 3, "dfs": 6,
         "first_in": 40, "first_won": 25, "second_in": 30, "second_won": 12,
         "bk_pts": 12, "bp_saved": 8, "return_pts": 80, "return_pts_won": 40,
         "winners": 15, "winners_fh": 9, "winners_bh": 6, "unforced": 20, "unforced_fh": 10, "unforced_bh": 10},
        {"match_id": "M1", "player": "A", "set": "1", "serve_pts": 40, "aces": 4, "dfs": 2,
         "first_in": 25, "first_won": 18, "second_in": 15, "second_won": 8,
         "bk_pts": 5, "bp_saved": 3, "return_pts": 35, "return_pts_won": 15,
         "winners": 10, "winners_fh": 6, "winners_bh": 4, "unforced": 8, "unforced_fh": 4, "unforced_bh": 4},
    ]).to_csv(path, index=False)
    return path


class TestLoadAndPrepareStats:
    def test_only_total_rows_kept(self, tmp_path):
        stats = load_and_prepare_stats(_write_stats_csv(tmp_path))
        assert (stats["set"] == "Total").all()
        assert len(stats) == 2  # per-set row for A excluded

    def test_bp_saved_pct(self, tmp_path):
        stats = load_and_prepare_stats(_write_stats_csv(tmp_path))
        a = stats[stats["player"] == "A"].iloc[0]
        assert a["bp_saved_pct"] == pytest.approx(6 / 10)

    def test_bp_converted_pct_cross_references_opponent(self, tmp_path):
        stats = load_and_prepare_stats(_write_stats_csv(tmp_path))
        a = stats[stats["player"] == "A"].iloc[0]
        # A's bp_converted = opponent B's (bk_pts - bp_saved) / bk_pts = (12-8)/12
        assert a["bp_converted_pct"] == pytest.approx((12 - 8) / 12)


class TestChronologyAndRolling:
    def test_player_id_mapping_reuses_frozen_join(self, tmp_path):
        stats = load_and_prepare_stats(_write_stats_csv(tmp_path))
        frozen_join = pd.DataFrame([{
            "mcp_match_id": "M1", "mcp_Player 1": "A", "mcp_Player 2": "B",
            "mcp_player1_norm": "a", "mcp_player2_norm": "b",
            "tml_winner_name_norm": "a", "tml_loser_name_norm": "b",
            "tml_winner_id": "PA", "tml_loser_id": "PB",
            "tml_tourney_id": "T1", "tml_tourney_date": pd.Timestamp("2020-01-01"),
            "tml_round": "F", "tml_match_num": 1,
        }])
        result = attach_player_ids_and_chronology(stats, frozen_join)
        assert result[result["player"] == "A"]["player_id"].iloc[0] == "PA"
        assert result[result["player"] == "B"]["player_id"].iloc[0] == "PB"

    def test_no_history_before_first_charted_match_is_nan(self, tmp_path):
        stats = load_and_prepare_stats(_write_stats_csv(tmp_path))
        frozen_join = pd.DataFrame([{
            "mcp_match_id": "M1", "mcp_Player 1": "A", "mcp_Player 2": "B",
            "mcp_player1_norm": "a", "mcp_player2_norm": "b",
            "tml_winner_name_norm": "a", "tml_loser_name_norm": "b",
            "tml_winner_id": "PA", "tml_loser_id": "PB",
            "tml_tourney_id": "T1", "tml_tourney_date": pd.Timestamp("2020-01-01"),
            "tml_round": "F", "tml_match_num": 1,
        }])
        with_ids = attach_player_ids_and_chronology(stats, frozen_join)
        rolling = compute_rolling_serve_return_features(with_ids)
        a_row = rolling[rolling["player_id"] == "PA"].iloc[0]
        assert pd.isna(a_row["bp_saved_pct_career"])