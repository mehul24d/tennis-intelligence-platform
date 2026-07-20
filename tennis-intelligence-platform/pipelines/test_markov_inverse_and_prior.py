import pandas as pd
import pytest

from tennis_intel.live.ml_informed_markov import build_pretrained_prior, ServeReturnPosterior
from tennis_intel.live.markov_baseline import prob_win_match
from tennis_intel.live.markov_inverse import invert_prematch_probability


class TestMarkovInverse:
    def test_round_trip_recovers_target_exactly(self):
        for target_p0, p_a_return, best_of in [
            (0.60, 0.35, 3), (0.75, 0.40, 5), (0.50, 0.50, 3), (0.90, 0.30, 5),
        ]:
            p_a_serve = invert_prematch_probability(target_p0, p_a_return, best_of)
            recovered = prob_win_match(p_a_serve, p_a_return, best_of=best_of)
            assert recovered == pytest.approx(target_p0, abs=1e-4)

    def test_monotonically_increasing_in_target(self):
        vals = [invert_prematch_probability(p0, 0.40, 3) for p0 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_extreme_target_hits_boundary_not_crash(self):
        p = invert_prematch_probability(0.999, 0.05, best_of=3)
        assert 0.9 < p <= 0.999

    def test_exactly_zero_or_one_does_not_crash(self):
        """Regression test: found on real data, 2026-07 — compute_ml_pre_match_probability
        can legitimately return EXACTLY 1.0 (a finite 200-trial Monte Carlo rollout where
        the favorite wins every simulated trial, given a large enough skill gap), which
        crashed evaluate_ml_informed_markov.py partway through a real 150-match run."""
        p_one = invert_prematch_probability(1.0, 0.35, best_of=5)
        p_zero = invert_prematch_probability(0.0, 0.35, best_of=5)
        assert 0.0 < p_one < 1.0
        assert 0.0 < p_zero < 1.0
        assert p_one > p_zero  # sanity: target=1.0 must need a higher serve rate than target=0.0


class TestBuildPretrainedPrior:
    def test_i_posterior_mean_at_point_zero_equals_prior_exactly(self):
        """Document requirement (i): at point 0, output equals the pre-match prior exactly."""
        p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
            p0_a_wins=0.65, p_a_return_seed=0.35, best_of=3, elo_matches_played_a=100,
        )
        posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)
        assert posterior.mean_serve() == pytest.approx(p_serve0, abs=1e-9)
        assert posterior.mean_return() == pytest.approx(p_return0, abs=1e-9)

    def test_inverted_prior_reproduces_original_p0(self):
        p0_target = 0.65
        p_serve0, _, p_return0, _ = build_pretrained_prior(
            p0_a_wins=p0_target, p_a_return_seed=0.35, best_of=3, elo_matches_played_a=100,
        )
        recovered = prob_win_match(p_serve0, p_return0, best_of=3)
        assert recovered == pytest.approx(p0_target, abs=1e-4)

    def test_ii_n0_scales_with_confidence(self):
        """Document requirement (ii): n0 correctly determines confidence scaling."""
        _, n0_low, _, _ = build_pretrained_prior(0.65, 0.35, 3, elo_matches_played_a=0)
        _, n0_high, _, _ = build_pretrained_prior(0.65, 0.35, 3, elo_matches_played_a=300)
        assert n0_low < n0_high

    def test_iii_lopsided_run_shifts_posterior_by_expected_magnitude_given_n0(self):
        """Document requirement (iii): a lopsided run of points measurably shifts the
        posterior in the expected direction and magnitude given n0 — specifically, a
        WEAKER prior (lower n0) must be swayed MORE by the same evidence than a STRONGER
        prior (higher n0)."""
        p_serve0, _, p_return0, _ = build_pretrained_prior(0.65, 0.35, 3, elo_matches_played_a=100)
        posterior_low = ServeReturnPosterior.from_pretrained_prior(p_serve0, 20.0, p_return0, 20.0)
        posterior_high = ServeReturnPosterior.from_pretrained_prior(p_serve0, 60.0, p_return0, 20.0)

        for _ in range(10):
            posterior_low = posterior_low.update_serve(a_won_point=False)
            posterior_high = posterior_high.update_serve(a_won_point=False)

        shift_low = p_serve0 - posterior_low.mean_serve()
        shift_high = p_serve0 - posterior_high.mean_serve()
        assert shift_low > shift_high > 0  # both move down (lost points), low-n0 moves further

    def test_missing_confidence_signal_falls_back_to_base_n0(self):
        _, n0, _, _ = build_pretrained_prior(0.65, 0.35, 3, elo_matches_played_a=None,
                                             base_n0=20.0)
        assert n0 == pytest.approx(20.0)

    def test_n0_return_uses_opponent_match_count_not_own(self):
        """Regression test (found via code review, 2026-07): n0_return_a must be derived
        from elo_matches_played_b (the OPPONENT's match count), not elo_matches_played_a
        (A's own) — because p_a_return_seed's VALUE is itself derived from the opponent's
        serve rate, so confidence in that value should track the opponent's own sample
        size, not A's. Before this fix, elo_matches_played_b was silently unused and
        n0_serve_a == n0_return_a always, regardless of the opponent's experience."""
        result_veteran_a = build_pretrained_prior(
            0.65, 0.35, 3, elo_matches_played_a=300, elo_matches_played_b=5,
        )
        p_serve0, n0_serve_a, p_return0, n0_return_a = result_veteran_a
        assert n0_serve_a > n0_return_a, (
            "A is a veteran (high n0_serve expected) but B is a newcomer "
            "(low n0_return expected, since n0_return tracks B's sample size)"
        )

        result_veteran_b = build_pretrained_prior(
            0.65, 0.35, 3, elo_matches_played_a=5, elo_matches_played_b=300,
        )
        _, n0_serve_a2, _, n0_return_a2 = result_veteran_b
        assert n0_return_a2 > n0_serve_a2, (
            "B is now the veteran, so n0_return (tracking B's sample size) should exceed "
            "n0_serve (tracking A's now-small sample size)"
        )

    def test_composite_n0_backward_compatible_without_h2h(self):
        """Regression test (external audit, 2026-07, Architecture Review finding C): the
        upgraded composite n0 (career match count + H2H depth + tournament H2H depth)
        must produce EXACTLY the same output as the original match-count-only formula
        when h2h_meetings/tourney_h2h_meetings are not supplied — no existing caller
        should see any change in behavior unless it explicitly opts into the new signals."""
        result = build_pretrained_prior(
            0.65, 0.35, 3, elo_matches_played_a=100, elo_matches_played_b=50,
        )
        _, n0_serve_a, _, n0_return_a = result

        base_n0, min_n0, max_n0, reference_matches = 20.0, 5.0, 60.0, 150.0
        expected_n0_serve = max(min(min(100 / reference_matches, 1.0) * (max_n0 - base_n0) + base_n0, max_n0), min_n0)
        expected_n0_return = max(min(min(50 / reference_matches, 1.0) * (max_n0 - base_n0) + base_n0, max_n0), min_n0)

        assert abs(n0_serve_a - expected_n0_serve) < 1e-9
        assert abs(n0_return_a - expected_n0_return) < 1e-9

    def test_composite_n0_h2h_depth_raises_confidence(self):
        """A rich head-to-head history should raise n0 beyond what career match count
        alone would give, even for players with thin career stats — matchup-specific
        history is real, independent evidence that raw career experience cannot capture."""
        result_deep_h2h = build_pretrained_prior(
            0.65, 0.35, 3, elo_matches_played_a=10, elo_matches_played_b=10,
            h2h_meetings=10, tourney_h2h_meetings=5,
        )
        result_no_h2h = build_pretrained_prior(
            0.65, 0.35, 3, elo_matches_played_a=10, elo_matches_played_b=10,
        )
        assert result_deep_h2h[1] > result_no_h2h[1]


class TestEloTrendFeatures:
    """Regression tests for compute_elo_trend_features — a rolling Elo-change feature,
    with a leakage-safety argument DIFFERENT from every other rolling feature in this
    project (elo_pre_match is already a prior-only value by construction, not a raw
    per-match outcome, so no additional shift(1) is needed)."""

    def test_change_computed_correctly(self):
        from tennis_intel.features.serve_return_features import compute_elo_trend_features
        day5 = pd.DataFrame({
            "winner_id": ["p1", "p2", "p1", "p3"], "loser_id": ["p2", "p1", "p3", "p1"],
            "tourney_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "elo_pre_match_winner": [1500, 1520, 1550, 1600],
            "elo_pre_match_loser": [1480, 1520, 1500, 1560],
            "tourney_id": ["t1", "t2", "t3", "t4"], "match_num": [1, 1, 1, 1],
        })
        result = compute_elo_trend_features(day5, windows=(2,))
        p1_rows = result[result["player_id"] == "p1"].sort_values("tourney_date")
        match3 = p1_rows[p1_rows["elo_pre_match"] == 1550]
        assert abs(match3["elo_change_last2"].iloc[0] - 50) < 1e-9

    def test_early_matches_are_nan_not_zero(self):
        from tennis_intel.features.serve_return_features import compute_elo_trend_features
        day5 = pd.DataFrame({
            "winner_id": ["p1", "p2"], "loser_id": ["p2", "p1"],
            "tourney_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "elo_pre_match_winner": [1500, 1520], "elo_pre_match_loser": [1480, 1520],
            "tourney_id": ["t1", "t2"], "match_num": [1, 1],
        })
        result = compute_elo_trend_features(day5, windows=(2,))
        assert result["elo_change_last2"].isna().all()

    def test_no_leakage_from_future_matches(self):
        """Perturbing a later match's Elo value must NOT change an earlier match's
        elo_change feature — direct proof, not just reasoning, that this is leakage-safe."""
        from tennis_intel.features.serve_return_features import compute_elo_trend_features
        base = {
            "winner_id": ["p1", "p2", "p1", "p3"], "loser_id": ["p2", "p1", "p3", "p1"],
            "tourney_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "elo_pre_match_winner": [1500, 1520, 1550, 1600],
            "elo_pre_match_loser": [1480, 1520, 1500, 1560],
            "tourney_id": ["t1", "t2", "t3", "t4"], "match_num": [1, 1, 1, 1],
        }
        day5_a = pd.DataFrame(base)
        day5_b = pd.DataFrame(base)
        day5_b.loc[3, "elo_pre_match_loser"] = 9999

        result_a = compute_elo_trend_features(day5_a, windows=(2,))
        result_b = compute_elo_trend_features(day5_b, windows=(2,))
        match3_a = result_a[(result_a["player_id"] == "p1") & (result_a["elo_pre_match"] == 1550)]["elo_change_last2"].iloc[0]
        match3_b = result_b[(result_b["player_id"] == "p1") & (result_b["elo_pre_match"] == 1550)]["elo_change_last2"].iloc[0]
        assert match3_a == match3_b

    def test_no_fan_out_when_tourney_id_and_match_num_repeat(self):
        """Regression test for a real bug caught on the actual pipeline run: (tourney_id,
        match_num) alone is NOT a unique match identifier in the real data — a
        previous version merged using only that 3-column subset, on an unverified
        assumption, causing 198,062 -> 198,894 rows (a real, caught fan-out). Both
        winner_id and loser_id must be present on every long-form row so callers can
        use the FULL, established 4-column key."""
        from tennis_intel.features.serve_return_features import compute_elo_trend_features
        day5 = pd.DataFrame({
            "winner_id": ["p1", "p3"], "loser_id": ["p2", "p4"],
            "tourney_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "elo_pre_match_winner": [1500, 1600], "elo_pre_match_loser": [1480, 1580],
            "tourney_id": ["t1", "t1"], "match_num": [1, 1],
        })
        elo_trend = compute_elo_trend_features(day5, windows=(2,))
        elo_feature_cols = [c for c in elo_trend.columns if c.startswith("elo_change_last")]

        elo_winner_side = elo_trend.loc[
            elo_trend["is_winner_row"], ["tourney_id", "match_num", "winner_id", "loser_id"] + elo_feature_cols
        ].rename(columns={c: f"winner_{c}" for c in elo_feature_cols})
        elo_loser_side = elo_trend.loc[
            ~elo_trend["is_winner_row"], ["tourney_id", "match_num", "winner_id", "loser_id"] + elo_feature_cols
        ].rename(columns={c: f"loser_{c}" for c in elo_feature_cols})

        merge_key = ["tourney_id", "match_num", "winner_id", "loser_id"]
        merged = day5.merge(elo_winner_side, on=merge_key, how="left")
        merged = merged.merge(elo_loser_side, on=merge_key, how="left")
        assert len(merged) == len(day5)


class TestDecidingSetShrinkage:
    """Regression tests for the deciding-set shrinkage mechanism (the fourth and final
    attempt against the confirmed, structural deciding-set log-loss gap) —
    sensitivity_aware_blend's is_deciding_set/deciding_set_shrinkage_factor parameters."""

    def test_default_is_fully_backward_compatible(self):
        from tennis_intel.live.ml_informed_markov import sensitivity_aware_blend
        b1 = sensitivity_aware_blend(0.9, 0.65, 5.0, points_observed=30)
        b2 = sensitivity_aware_blend(0.9, 0.65, 5.0, points_observed=30, is_deciding_set=False)
        b3 = sensitivity_aware_blend(0.9, 0.65, 5.0, points_observed=30,
                                     is_deciding_set=True, deciding_set_shrinkage_factor=1.0)
        assert b1 == b2 == b3

    def test_shrinkage_pulls_toward_raw_classifier(self):
        from tennis_intel.live.ml_informed_markov import sensitivity_aware_blend
        baseline = sensitivity_aware_blend(0.9, 0.65, 5.0, points_observed=30)
        shrunk = sensitivity_aware_blend(0.9, 0.65, 5.0, points_observed=30,
                                         is_deciding_set=True, deciding_set_shrinkage_factor=0.5)
        assert shrunk != baseline
        assert abs(shrunk - 0.9) < abs(baseline - 0.9)

    def test_shrinkage_inactive_when_not_deciding_set(self):
        from tennis_intel.live.ml_informed_markov import sensitivity_aware_blend
        baseline = sensitivity_aware_blend(0.9, 0.65, 5.0, points_observed=30)
        result = sensitivity_aware_blend(0.9, 0.65, 5.0, points_observed=30,
                                         is_deciding_set=False, deciding_set_shrinkage_factor=0.5)
        assert result == baseline


class TestNoLeakyServerIsWinnerFeature:
    """Regression test (external audit, 2026-07, Phase 4 Critical finding): server_is_winner
    encodes 'does the current server go on to win the ENTIRE match' — computable only by
    already knowing this match's final outcome. Cross-validated via permutation importance:
    ranked #2 overall, ~13x higher than any genuinely pre-match feature. Fixed by replacing
    it with server_is_player1 (a real-time, outcome-independent fact) as the actual
    model-facing feature. This test guards against server_is_winner ever being silently
    reintroduced into the classifier's training feature list."""

    def test_server_is_winner_not_in_point_feature_cols(self):
        import sys
        sys.path.insert(0, "pipelines")
        from build_day9_point_model import POINT_FEATURE_COLS
        assert "server_is_winner" not in POINT_FEATURE_COLS, (
            "server_is_winner is a confirmed leakage feature (requires knowing the match's "
            "final outcome) and must never be used as a training feature — see "
            "build_point_dataset.py and the external audit's Phase 4 finding for the full "
            "explanation. Use server_is_player1 instead."
        )

    def test_server_is_player1_in_point_feature_cols(self):
        import sys
        sys.path.insert(0, "pipelines")
        from build_day9_point_model import POINT_FEATURE_COLS
        assert "server_is_player1" in POINT_FEATURE_COLS, (
            "server_is_player1 (the safe replacement for server_is_winner) should be "
            "present as the classifier's server-identity feature."
        )