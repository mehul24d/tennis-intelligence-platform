"""Regression tests for the pre-point/post-point MatchState indexing bug documented in
docs/known_issue_ml_informed_markov_pre_point_state.md.

DERIVATION (see the known-issue doc for the full trace): compute_five_engine_trajectory's
ml_informed_p1[i] is computed at row_i's PRE-POINT state — i.e. ml_informed_p1[i] is
legitimately "the probability just BEFORE point i is played". Point i's real outcome is
only reflected once we look at ml_informed_p1[i+1] (row i+1's pre-point state IS the true
post-point-i state, by construction of consecutive point-dataset rows) or, for the match's
final point, the actual terminal outcome (1.0/0.0).

point_timeline_service.py and match_summary_service.py both built `all_p1 = [prematch_p1]
+ smoothed_p1` and then paired (before, after) for point i as (all_p1[i], all_p1[i+1]) —
i.e. (smoothed_p1[i-1], smoothed_p1[i]) for i>0 — which is the PREVIOUS point's before/after
pair, mislabeled as point i's. The correct pairing is (smoothed_p1[i], smoothed_p1[i+1] or
terminal).

These tests mock compute_five_engine_trajectory with hand-picked, fully deterministic
values so the correct pairing can be verified by hand rather than trusting the real
classifier/posterior pipeline.
"""

from unittest.mock import patch

import pytest

# Four synthetic points. The real swing (0.63 -> 0.90) happens going INTO point 1 (i.e.
# it's point 1's own outcome that causes it) — smoothed_p1[1] = "before point 1" = 0.63,
# smoothed_p1[2] = "before point 2" = "after point 1" = 0.90.
PREMATCH_P1 = 0.60
SMOOTHED_P1 = [0.62, 0.63, 0.90, 0.91]
FINAL_WINNER_IS_P1 = True


def _fake_row(pt: int, svr: int, pt_winner: int) -> dict:
    return {
        "Pt": pt, "Svr": svr, "PtWinner": pt_winner,
        "p1_points": 0, "p2_points": 0,
        "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0,
        "is_tiebreak_game": False, "is_break_point": False,
        "is_set_point": False, "is_match_point": False,
    }


FAKE_RECORDS = [
    _fake_row(1, svr=1, pt_winner=1),
    _fake_row(2, svr=1, pt_winner=1),   # this point's OWN outcome causes the big swing
    _fake_row(3, svr=1, pt_winner=1),
    _fake_row(4, svr=1, pt_winner=1),
]

FAKE_COMPUTED = {
    "records": FAKE_RECORDS,
    "ml_informed_p1": SMOOTHED_P1,
    "ml_informed_prematch_p1": PREMATCH_P1,
    "final_winner_is_p1": FINAL_WINNER_IS_P1,
    "p1_name": "Player One", "p2_name": "Player Two",
}

# Hand-computed CORRECT before/after/swing per point, per the derivation above.
EXPECTED_BEFORE = [0.62, 0.63, 0.90, 0.91]
EXPECTED_AFTER = [0.63, 0.90, 0.91, 1.0]  # last point -> terminal outcome (P1 won)
EXPECTED_SWING = [
    round(abs(a - b), 6) for a, b in zip(EXPECTED_AFTER, EXPECTED_BEFORE)
]  # [0.01, 0.27, 0.01, 0.09]


def test_point_timeline_before_after_pairing_matches_hand_computed_values():
    from tennis_intel.serving.point_timeline_service import get_point_timeline

    with patch(
        "tennis_intel.serving.point_timeline_service.compute_five_engine_trajectory",
        return_value=FAKE_COMPUTED,
    ):
        result = get_point_timeline(ctx=object(), match_id="fake")

    points = result["points"]
    assert len(points) == 4

    got_before = [p["probability_before_p1"] for p in points]
    got_after = [p["probability_after_p1"] for p in points]
    got_swing = [p["probability_swing"] for p in points]

    assert got_before == pytest.approx(EXPECTED_BEFORE, abs=1e-6)
    assert got_after == pytest.approx(EXPECTED_AFTER, abs=1e-6)
    assert got_swing == pytest.approx(EXPECTED_SWING, abs=1e-6)

    # The largest swing (0.27) must be attributed to point 2 (Pt=2, index 1) — the point
    # whose OWN outcome caused it — not point 3 (the pre-fix, off-by-one-shifted answer).
    largest = [p for p in points if p["is_largest_swing"]]
    assert len(largest) == 1
    assert largest[0]["point_index"] == 2


def test_match_summary_largest_swing_attributed_to_correct_point():
    from tennis_intel.serving.match_summary_service import get_match_summary

    with patch(
        "tennis_intel.serving.match_summary_service.compute_five_engine_trajectory",
        return_value=FAKE_COMPUTED,
    ):
        result = get_match_summary(ctx=object(), match_id="fake")

    swing_info = result["largest_probability_swing"]
    assert swing_info["point_index"] == 2  # Pt=2 (index 1), same reasoning as above
    assert swing_info["probability_before"] == pytest.approx(0.63, abs=1e-6)
    assert swing_info["probability_after"] == pytest.approx(0.90, abs=1e-6)
    assert swing_info["swing"] == pytest.approx(0.27, abs=1e-6)
