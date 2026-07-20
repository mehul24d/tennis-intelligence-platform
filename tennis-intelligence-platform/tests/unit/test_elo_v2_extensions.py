import pandas as pd
import pytest

from tennis_intel.ratings.elo import EloRating
from tennis_intel.ratings.processor import compute_ratings, default_dynamic_k
from tennis_intel.ratings.surface_elo import compute_surface_ratings


def make_match(winner_id, loser_id, date, round_="F", match_num=1, tourney_id="T1", **extra):
    row = {"tourney_id": tourney_id, "tourney_date": pd.Timestamp(date), "round": round_,
           "match_num": match_num, "winner_id": winner_id, "loser_id": loser_id}
    row.update(extra)
    return row


class TestBackwardCompatibility:
    """The pre-existing 7-test suite in test_elo.py must keep passing unchanged — these
    are additional spot-checks that new optional params don't alter default behavior."""

    def test_fixed_k_unaffected_by_new_optional_params(self):
        matches = pd.DataFrame([make_match("A", "B", "2020-01-01")])
        result = compute_ratings(matches, EloRating(), k=32)
        row = result.augmented.iloc[0]
        assert row["elo_post_match_winner"] == pytest.approx(1516, abs=0.01)
        assert row["elo_post_match_loser"] == pytest.approx(1484, abs=0.01)


class TestDynamicKFactor:
    def test_default_dynamic_k_boundaries(self):
        assert default_dynamic_k(0) == 40.0
        assert default_dynamic_k(29) == 40.0
        assert default_dynamic_k(30) == 32.0
        assert default_dynamic_k(1000) == 32.0

    def test_k_steps_down_after_threshold_in_full_pipeline(self):
        matches = pd.DataFrame([
            make_match("A", f"opp{i}", f"2020-{(i % 12) + 1:02d}-01", tourney_id=f"T{i}")
            for i in range(35)
        ])
        result = compute_ratings(matches, EloRating(), k_fn=lambda mp: default_dynamic_k(mp))
        k_used = result.augmented["k_factor_used"].tolist()
        assert all(k == 40.0 for k in k_used[:30])
        assert all(k == 32.0 for k in k_used[30:])


class TestRetirementHandling:
    def test_retirement_halves_the_update(self):
        base = make_match("A", "B", "2020-01-01", is_retirement=True)
        normal = make_match("A", "B", "2020-01-01", is_retirement=False)
        r_ret = compute_ratings(pd.DataFrame([base]), EloRating(), k=32, retirement_col="is_retirement")
        r_normal = compute_ratings(pd.DataFrame([normal]), EloRating(), k=32, retirement_col="is_retirement")
        delta_ret = r_ret.augmented.iloc[0]["elo_delta"]
        delta_normal = r_normal.augmented.iloc[0]["elo_delta"]
        assert delta_ret == pytest.approx(delta_normal * 0.5, abs=0.01)

    def test_no_retirement_col_means_no_special_handling(self):
        # Without retirement_col specified, behavior must be identical to plain Elo
        matches = pd.DataFrame([make_match("A", "B", "2020-01-01")])
        result = compute_ratings(matches, EloRating(), k=32)
        assert result.augmented.iloc[0]["elo_delta"] == pytest.approx(16.0, abs=0.01)


class TestWalkoverHandling:
    def test_walkover_produces_zero_delta(self):
        matches = pd.DataFrame([make_match("A", "B", "2020-01-01", is_walkover=True)])
        result = compute_ratings(matches, EloRating(), k=32, walkover_col="is_walkover")
        row = result.augmented.iloc[0]
        assert row["elo_pre_match_winner"] == row["elo_post_match_winner"]
        assert row["elo_delta"] == 0.0
        assert result.diagnostics["walkovers_skipped"] == 1

    def test_walkover_does_not_count_toward_matches_played(self):
        matches = pd.DataFrame([
            make_match("A", "B", "2020-01-01", tourney_id="T1", is_walkover=True),
            make_match("A", "C", "2020-01-02", tourney_id="T2", is_walkover=False),
        ])
        result = compute_ratings(matches, EloRating(), k=32, walkover_col="is_walkover")
        assert result.augmented.iloc[1]["elo_matches_played_pre_winner"] == 0


class TestConfidenceSignal:
    def test_matches_played_increments_leakage_safely(self):
        matches = pd.DataFrame([
            make_match("A", f"opp{i}", f"2020-{i+1:02d}-01", tourney_id=f"T{i}")
            for i in range(5)
        ])
        result = compute_ratings(matches, EloRating(), k=32)
        assert result.augmented["elo_matches_played_pre_winner"].tolist() == [0, 1, 2, 3, 4]


class TestSurfaceElo:
    def test_surfaces_are_independent(self):
        matches = pd.DataFrame([
            make_match("A", "B", "2020-01-01", surface="Hard", tourney_id="T1"),
            make_match("A", "B", "2020-01-02", surface="Hard", tourney_id="T2"),
            make_match("A", "B", "2020-01-03", surface="Hard", tourney_id="T3"),
            make_match("A", "C", "2020-01-04", surface="Clay", tourney_id="T4"),
        ])
        result = compute_surface_ratings(matches, lambda: EloRating())
        clay_row = result[result["surface"] == "Clay"].iloc[0]
        assert clay_row["elo_surface_pre_match_winner"] == 1500.0

    def test_leakage_safety_within_surface(self):
        full = pd.DataFrame([
            make_match("A", "B", "2020-01-01", surface="Clay", tourney_id="T1"),
            make_match("A", "C", "2020-01-02", surface="Clay", tourney_id="T2"),
            make_match("B", "C", "2020-01-03", surface="Clay", tourney_id="T3"),
        ])
        result_full = compute_surface_ratings(full, lambda: EloRating())
        result_trunc = compute_surface_ratings(full.iloc[:2], lambda: EloRating())
        pre_full = result_full.iloc[:2][["elo_surface_pre_match_winner", "elo_surface_pre_match_loser"]].reset_index(drop=True)
        pre_trunc = result_trunc[["elo_surface_pre_match_winner", "elo_surface_pre_match_loser"]].reset_index(drop=True)
        assert pre_full.equals(pre_trunc)

    def test_unrated_surface_left_as_nan_not_dropped(self):
        matches = pd.DataFrame([
            make_match("A", "B", "2020-01-01", surface="Hard", tourney_id="T1"),
            make_match("A", "B", "2020-01-02", surface="Carpet", tourney_id="T2"),
        ])
        result = compute_surface_ratings(matches, lambda: EloRating())
        assert len(result) == 2
        carpet_row = result[result["surface"] == "Carpet"].iloc[0]
        assert pd.isna(carpet_row["elo_surface_pre_match_winner"])

    def test_row_count_preserved(self):
        matches = pd.DataFrame([
            make_match("A", "B", "2020-01-01", surface="Hard", tourney_id="T1"),
            make_match("A", "C", "2020-01-02", surface="Clay", tourney_id="T2"),
            make_match("B", "C", "2020-01-03", surface="Grass", tourney_id="T3"),
        ])
        result = compute_surface_ratings(matches, lambda: EloRating())
        assert len(result) == len(matches)