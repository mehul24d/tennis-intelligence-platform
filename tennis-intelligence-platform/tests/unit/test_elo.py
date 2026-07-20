import pandas as pd
import pytest

from tennis_intel.ratings.elo import EloRating
from tennis_intel.ratings.processor import compute_ratings


def make_match(winner_id, loser_id, date, round_="F", match_num=1, tourney_id="T1"):
    return {
        "tourney_id": tourney_id, "tourney_date": pd.Timestamp(date), "round": round_,
        "match_num": match_num, "winner_id": winner_id, "loser_id": loser_id,
    }


class TestEloMath:
    """Mathematical correctness, not just 'does it run'."""

    def test_two_new_players_k32(self):
        # Test 1: two 1500-rated players, A beats B, K=32 -> 1516 / 1484
        matches = pd.DataFrame([make_match("A", "B", "2020-01-01")])
        result = compute_ratings(matches, EloRating(), k=32)
        row = result.augmented.iloc[0]
        assert row["elo_post_match_winner"] == pytest.approx(1516, abs=0.01)
        assert row["elo_post_match_loser"] == pytest.approx(1484, abs=0.01)

    def test_repeated_wins_monotonically_increase(self):
        # Test 2: A beats B three times in a row -> A's rating strictly increases each time
        matches = pd.DataFrame([
            make_match("A", "B", "2020-01-01", tourney_id="T1"),
            make_match("A", "B", "2020-01-02", tourney_id="T2"),
            make_match("A", "B", "2020-01-03", tourney_id="T3"),
        ])
        result = compute_ratings(matches, EloRating(), k=32)
        a_ratings = result.augmented["elo_post_match_winner"].tolist()
        assert a_ratings[0] < a_ratings[1] < a_ratings[2]

    def test_upset_produces_large_update(self):
        # Test 3: 1500 beats 2000 -> large rating change
        elo = EloRating()
        new_w, _ = elo.update_ratings(1500, 2000, k=32)
        assert (new_w - 1500) > 25

    def test_expected_result_produces_small_update(self):
        # Test 4: 2000 beats 1500 -> small rating change
        elo = EloRating()
        new_w, _ = elo.update_ratings(2000, 1500, k=32)
        assert (new_w - 2000) < 5


class TestReproducibility:
    def test_identical_output_on_repeat_run(self):
        # Test 5: running the pipeline twice on the same input is byte-identical
        matches = pd.DataFrame([
            make_match("A", "B", "2020-01-01", tourney_id="T1"),
            make_match("A", "B", "2020-01-02", tourney_id="T2"),
        ])
        result_a = compute_ratings(matches, EloRating(), k=32)
        result_b = compute_ratings(matches, EloRating(), k=32)
        assert result_a.augmented.equals(result_b.augmented)


class TestChronology:
    def test_shuffled_input_produces_same_result(self):
        # Test 6: the pipeline must internally re-sort — input row order must not matter
        matches = pd.DataFrame([
            make_match("A", "B", "2020-01-01", tourney_id="T1"),
            make_match("A", "B", "2020-01-02", tourney_id="T2"),
            make_match("A", "B", "2020-01-03", tourney_id="T3"),
        ])
        unshuffled = compute_ratings(matches, EloRating(), k=32).augmented
        shuffled_input = matches.sample(frac=1, random_state=42).reset_index(drop=True)
        shuffled_result = compute_ratings(shuffled_input, EloRating(), k=32).augmented
        assert unshuffled.reset_index(drop=True).equals(shuffled_result.reset_index(drop=True))


class TestLeakageProtection:
    def test_pre_match_elo_unaffected_by_future_matches(self):
        # Test 7: removing future matches must not change any earlier match's pre-match Elo.
        # This is the test that would catch an accidental look-ahead bug.
        full = pd.DataFrame([
            make_match("A", "B", "2020-01-01", tourney_id="T1"),
            make_match("A", "C", "2020-01-02", tourney_id="T2"),
            make_match("B", "C", "2020-01-03", tourney_id="T3"),
        ])
        result_full = compute_ratings(full, EloRating(), k=32)

        truncated = full.iloc[:2]
        result_truncated = compute_ratings(truncated, EloRating(), k=32)

        pre_full = result_full.augmented.iloc[:2][["elo_pre_match_winner", "elo_pre_match_loser"]].reset_index(drop=True)
        pre_trunc = result_truncated.augmented[["elo_pre_match_winner", "elo_pre_match_loser"]].reset_index(drop=True)
        assert pre_full.equals(pre_trunc)


class TestRoundOrdering:
    def test_earlier_rounds_processed_before_later_rounds_same_date(self):
        # Two matches with the SAME tourney_date but different rounds — R32 must be
        # processed before F even though pandas' default row order might disagree.
        matches = pd.DataFrame([
            make_match("A", "B", "2020-01-01", round_="F", match_num=1, tourney_id="T1"),
            make_match("C", "D", "2020-01-01", round_="R32", match_num=50, tourney_id="T1"),
        ])
        result = compute_ratings(matches, EloRating(), k=32)
        rounds_in_order = result.augmented["round"].tolist()
        assert rounds_in_order == ["R32", "F"]