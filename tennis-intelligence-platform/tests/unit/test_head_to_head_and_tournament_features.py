import pandas as pd
import pytest

from tennis_intel.features.head_to_head_features import add_head_to_head_features
from tennis_intel.features.feature_engineering_day5 import compute_day5_features


def make_match(winner_id, loser_id, date, tourney_name, round_="F", match_num=1, tourney_id="T1"):
    return {"tourney_id": tourney_id, "tourney_date": pd.Timestamp(date), "round": round_,
            "match_num": match_num, "winner_id": winner_id, "loser_id": loser_id,
            "tourney_name": tourney_name}


class TestHeadToHead:
    def test_overall_h2h_matches_hand_computation(self):
        matches = pd.DataFrame([
            make_match("A", "B", "2020-01-01", "Wimbledon", tourney_id="T1"),
            make_match("A", "B", "2020-06-01", "US Open", tourney_id="T2"),
            make_match("B", "A", "2021-01-01", "Wimbledon", tourney_id="T3"),
        ])
        result = add_head_to_head_features(matches)
        h2h = result[["winner_h2h_wins_pre_match", "loser_h2h_wins_pre_match"]].values.tolist()
        assert h2h[0] == [0, 0]
        assert h2h[1] == [1, 0]
        assert h2h[2] == [0, 2]

    def test_tournament_h2h_differs_from_overall(self):
        matches = pd.DataFrame([
            make_match("A", "B", "2020-01-01", "Wimbledon", tourney_id="T1"),
            make_match("A", "B", "2020-06-01", "US Open", tourney_id="T2"),
            make_match("B", "A", "2021-01-01", "Wimbledon", tourney_id="T3"),
        ])
        result = add_head_to_head_features(matches)
        tourney_h2h = result[["winner_tourney_h2h_wins_pre_match", "loser_tourney_h2h_wins_pre_match"]].values.tolist()
        assert tourney_h2h[0] == [0, 0]
        # US Open H2H must be 0 even though overall H2H is 1 (that win was at Wimbledon)
        assert tourney_h2h[1] == [0, 0]
        assert tourney_h2h[2] == [0, 1]

    def test_leakage_safety(self):
        full = pd.DataFrame([
            make_match("A", "B", "2020-01-01", "Wimbledon", tourney_id="T1"),
            make_match("A", "B", "2020-06-01", "US Open", tourney_id="T2"),
            make_match("B", "A", "2021-01-01", "Wimbledon", tourney_id="T3"),
        ])
        result_full = add_head_to_head_features(full)
        result_trunc = add_head_to_head_features(full.iloc[:2])
        pre_full = result_full.iloc[:2][["winner_h2h_wins_pre_match", "loser_h2h_wins_pre_match"]].reset_index(drop=True)
        pre_trunc = result_trunc[["winner_h2h_wins_pre_match", "loser_h2h_wins_pre_match"]].reset_index(drop=True)
        assert pre_full.equals(pre_trunc)

    def test_missing_tourney_name_degrades_gracefully(self):
        matches = pd.DataFrame([make_match("A", "B", "2020-01-01", "Wimbledon")]).drop(columns=["tourney_name"])
        result = add_head_to_head_features(matches)
        assert result["winner_tourney_h2h_wins_pre_match"].isna().all()
        assert not result["winner_h2h_wins_pre_match"].isna().any()


class TestTournamentForm:
    def _make_row(self, winner_id, loser_id, date, tourney_id):
        return {"tourney_id": tourney_id, "tourney_date": pd.Timestamp(date), "round": "F",
                "match_num": 1, "surface": "Hard", "tourney_level": "G", "best_of": 3,
                "minutes": 90, "score": "6-4 6-3", "winner_id": winner_id, "loser_id": loser_id,
                "elo_pre_match_winner": 1700.0, "elo_pre_match_loser": 1600.0,
                "elo_surface_pre_match_winner": 1700.0, "elo_surface_pre_match_loser": 1600.0,
                "elo_matches_played_pre_winner": 0, "elo_matches_played_pre_loser": 0,
                "is_retirement": False, "is_walkover": False, "tourney_name": "Wimbledon"}

    def test_tournament_form_matches_hand_computation(self):
        rows = [
            self._make_row("P0", "P1", "2015-01-01", "T1"),
            self._make_row("P1", "P0", "2016-01-01", "T2"),
            self._make_row("P0", "P1", "2017-01-01", "T3"),
        ]
        df = pd.DataFrame(rows)
        result = compute_day5_features(df)
        aug = result.augmented.sort_values("tourney_date")
        vals = aug[["winner_tourney_win_pct_last10", "loser_tourney_win_pct_last10"]].values.tolist()
        assert pd.isna(vals[0][0]) and pd.isna(vals[0][1])
        assert vals[1][0] == pytest.approx(0.0) and vals[1][1] == pytest.approx(1.0)
        assert vals[2][0] == pytest.approx(0.5) and vals[2][1] == pytest.approx(0.5)

    def test_row_count_preserved_with_h2h_and_tournament_features(self):
        rows = [
            self._make_row("P0", "P1", "2015-01-01", "T1"),
            self._make_row("P1", "P0", "2016-01-01", "T2"),
            self._make_row("P0", "P1", "2017-01-01", "T3"),
        ]
        df = pd.DataFrame(rows)
        result = compute_day5_features(df)
        assert len(result.augmented) == len(df)