import pandas as pd
import pytest

from tennis_intel.features.feature_engineering_day5 import compute_day5_features


def m(winner, loser, date, tourney_id, w_elo=1500, l_elo=1500, surface="Hard",
      score="6-4 6-3", best_of=3, minutes=90, tourney_level="A", round_="F"):
    return {
        "tourney_id": tourney_id, "tourney_date": pd.Timestamp(date), "round": round_,
        "match_num": 1, "winner_id": winner, "loser_id": loser, "surface": surface,
        "tourney_level": tourney_level, "best_of": best_of, "minutes": minutes, "score": score,
        "elo_pre_match_winner": w_elo, "elo_pre_match_loser": l_elo,
    }


def get_feature(augmented: pd.DataFrame, row_idx: int, player_id: str, col: str):
    row = augmented.iloc[row_idx]
    return row[f"winner_{col}"] if row["winner_id"] == player_id else row[f"loser_{col}"]


class TestRollingWinPct:
    def test_w_w_l_w_l_sequence(self):
        # User's exact example: W, W, L, W, L -> win_pct before 5th match = 3/4 = 0.75
        matches = pd.DataFrame([
            m("A", "X1", "2020-01-01", "T1"),
            m("A", "X2", "2020-01-08", "T2"),
            m("X3", "A", "2020-01-15", "T3"),
            m("A", "X4", "2020-01-22", "T4"),
            m("X5", "A", "2020-01-29", "T5"),
        ])
        df = compute_day5_features(matches).augmented
        assert get_feature(df, 4, "A", "win_pct_last5") == pytest.approx(0.75)
        assert get_feature(df, 4, "A", "n_matches_last5") == 4

    def test_first_match_has_no_history(self):
        matches = pd.DataFrame([m("A", "B", "2020-01-01", "T1")])
        df = compute_day5_features(matches).augmented
        assert pd.isna(get_feature(df, 0, "A", "win_pct_last5"))
        assert get_feature(df, 0, "A", "n_matches_last5") == 0


class TestSurfaceSpecific:
    def test_hard_hard_clay_grass_isolation(self):
        # User's exact example: Hard, Hard, Clay, Grass -> surface stats must isolate by surface
        matches = pd.DataFrame([
            m("A", "X1", "2020-01-01", "T1", surface="Hard"),
            m("A", "X2", "2020-01-08", "T2", surface="Hard"),
            m("X3", "A", "2020-01-15", "T3", surface="Clay"),
            m("A", "X4", "2020-01-22", "T4", surface="Grass"),
            m("X5", "A", "2020-01-29", "T5", surface="Hard"),
        ])
        df = compute_day5_features(matches).augmented
        # Before the 5th match (Hard), only matches 1&2 (both Hard) should count
        assert get_feature(df, 4, "A", "surface_win_pct_last5") == pytest.approx(1.0)
        assert get_feature(df, 4, "A", "surface_n_matches_last5") == 2


class TestOpponentStrength:
    def test_mean_median_max_min_1000_1200_1400(self):
        # User's exact example: opponent Elo sequence 1000, 1200, 1400
        matches = pd.DataFrame([
            m("A", "O1", "2020-01-01", "T1", w_elo=1500, l_elo=1000),
            m("A", "O2", "2020-01-08", "T2", w_elo=1500, l_elo=1200),
            m("A", "O3", "2020-01-15", "T3", w_elo=1500, l_elo=1400),
            m("A", "O4", "2020-01-22", "T4", w_elo=1500, l_elo=1300),
        ])
        df = compute_day5_features(matches).augmented
        assert get_feature(df, 3, "A", "opponent_elo_mean_last10") == 1200.0
        assert get_feature(df, 3, "A", "opponent_elo_median_last10") == 1200.0
        assert get_feature(df, 3, "A", "opponent_elo_max_last10") == 1400.0
        assert get_feature(df, 3, "A", "opponent_elo_min_last10") == 1000.0


class TestStreaks:
    def test_streak_sequence_matches_hand_derivation(self):
        matches = pd.DataFrame([
            m("A", "X1", "2020-01-01", "T1"),  # W
            m("A", "X2", "2020-01-08", "T2"),  # W
            m("X3", "A", "2020-01-15", "T3"),  # L
            m("A", "X4", "2020-01-22", "T4"),  # W
            m("X5", "A", "2020-01-29", "T5"),  # L
        ])
        df = compute_day5_features(matches).augmented
        expected = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 0)]
        actual = [
            (get_feature(df, i, "A", "win_streak_entering_match"),
             get_feature(df, i, "A", "loss_streak_entering_match"))
            for i in range(5)
        ]
        assert actual == expected


class TestLeakageProtection:
    def test_removing_future_matches_does_not_change_past_features(self):
        matches = pd.DataFrame([
            m("A", "O1", "2020-01-01", "T1", w_elo=1500, l_elo=1000),
            m("A", "O2", "2020-01-08", "T2", w_elo=1500, l_elo=1200),
            m("A", "O3", "2020-01-15", "T3", w_elo=1500, l_elo=1400),
        ])
        full = compute_day5_features(matches).augmented
        truncated = compute_day5_features(matches.iloc[:2]).augmented

        for col in ["winner_win_pct_last5", "winner_opponent_elo_mean_last10", "winner_rest_days"]:
            full_val = full.iloc[1][col]
            trunc_val = truncated.iloc[1][col]
            assert (pd.isna(full_val) and pd.isna(trunc_val)) or full_val == trunc_val


class TestReproducibility:
    def test_identical_output_on_repeat_run(self):
        matches = pd.DataFrame([
            m("A", "B", "2020-01-01", "T1"),
            m("A", "C", "2020-01-08", "T2"),
        ])
        r1 = compute_day5_features(matches).augmented
        r2 = compute_day5_features(matches).augmented
        assert r1.equals(r2)


class TestGamesSetsAndDuration:
    def test_averages_and_three_set_rate(self):
        matches = pd.DataFrame([
            m("A", "X1", "2020-01-01", "T1", score="6-4 6-3", minutes=80),   # 12 games won, straight sets
            m("A", "X2", "2020-01-03", "T2", score="6-2 3-6 7-5", minutes=150),  # 16 games won, 3 sets
            m("A", "X3", "2020-01-05", "T3", score="6-4 6-3", minutes=90),
        ])
        df = compute_day5_features(matches).augmented
        assert get_feature(df, 2, "A", "avg_games_won_last5") == pytest.approx(14.0)
        assert get_feature(df, 2, "A", "avg_duration_last5") == pytest.approx(115.0)
        assert get_feature(df, 2, "A", "three_set_rate_last5") == pytest.approx(0.5)


class TestTournamentContext:
    def test_previous_level_and_round(self):
        matches = pd.DataFrame([
            m("A", "X1", "2020-01-01", "T1", tourney_level="A", round_="QF"),
            m("A", "X2", "2020-01-03", "T2", tourney_level="M", round_="SF"),
            m("A", "X3", "2020-01-05", "T3", tourney_level="G", round_="F"),
        ])
        df = compute_day5_features(matches).augmented
        assert get_feature(df, 2, "A", "previous_tourney_level") == "M"
        assert get_feature(df, 2, "A", "previous_round_reached") == "SF"