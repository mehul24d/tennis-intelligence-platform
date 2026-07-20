"""
test_split_points_streak.py — permanent regression tests for compute_split_points_streak,
which separates points_streak into serve-streak and return-streak.

Two properties verified directly, matching the discipline used for every other feature
this project: correctness against a hand-traced sequence, and the specific match-
boundary leak this function's forward-fill step is vulnerable to if not grouped by
match_id (a real bug caught before shipping — an earlier draft used a global,
ungrouped ffill).
"""

import pandas as pd
import pytest

from tennis_intel.features.point_level_features import compute_split_points_streak


class TestSplitPointsStreak:
    def test_serve_streak_hand_traced(self):
        """Expected values re-derived by hand after the two-series design fix was
        found — the original expected values here were themselves computed under the
        same flawed single-series understanding that produced the bug, and were wrong
        for exactly the same reason (see the function's own docstring for the full
        trace of why idx=2's value must be +2, not +1: entering idx=2, P1 has already
        won BOTH of their prior serve points, idx=0 and idx=1).

        NOTE: unaffected by the separate PtWinner-is-relative-to-server fix (see
        test_return_streak_hand_traced's own note on that fix) — every SERVE point
        in this specific sequence has Svr=1, so server_is_p1=True throughout,
        making the fixed formula reduce to the original one for these particular
        rows. Confirmed directly, not assumed."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 8,
            "Svr":      [1, 1, 2, 2, 1, 1, 2, 1],
            "PtWinner": [1, 1, 1, 2, 1, 2, 2, 1],
        })
        result = compute_split_points_streak(df)
        assert result["p1_serve_streak"].tolist() == [0, 1, 2, 2, 2, 3, -1, -1]

    def test_return_streak_hand_traced(self):
        """See test_serve_streak_hand_traced's own note on why serve-streak values
        were re-derived, not carried over, after the two-series design fix.

        RE-DERIVED AGAIN (2026-07): PtWinner's convention was settled as LITERAL,
        fixed-player-relative (PtWinner==1 means player 1 won, period) — see
        point_level_features.py's compute_in_match_momentum docstring and
        docs/ptwinner_convention_correction.md for the full investigation. An
        intermediate version of this test (briefly, same day) assumed the opposite
        (server-relative) and was updated accordingly; that assumption was itself
        traced and reverted, so this expectation is re-derived a second time, by
        hand, against the LITERAL convention: return points are indices 2,3,6
        (Svr==2). PtWinner there is [1,2,2] -> player1 literally won index 2 only.
        Entering index 2 (first return point): no prior return history -> 0.
        Entering index 3: 1 prior return win -> +1. Entering index 6: prior returns
        were win,loss -> last outcome loss -> -1. Forward-filled onto serve/other
        rows starting the row after each return point's own (unshifted) resolved
        value: after idx2 (win) -> +1, placed from idx3 (but idx3 is itself a return
        point with its own value); after idx3 (loss) -> 1 win/2 losses net -1,
        placed from idx4 onward until idx6's own value (also -1, consistent);
        after idx6 (loss) -> -2, placed onto idx7."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 8,
            "Svr":      [1, 1, 2, 2, 1, 1, 2, 1],
            "PtWinner": [1, 1, 1, 2, 1, 2, 2, 1],
        })
        result = compute_split_points_streak(df)
        assert result["p1_return_streak"].tolist() == [0, 0, 0, 1, -1, -1, -1, -2]

    def test_no_leak_across_match_boundary_of_opposite_type(self):
        """The specific bug caught before shipping: match 1 ends with P1 on a serve
        streak; match 2 starts with P1 RETURNING (not serving). A global, ungrouped
        forward-fill would incorrectly carry match 1's serve-streak value into match
        2's early return points. Grouped-by-match forward-fill must show 0 instead,
        since no serve history exists yet in the new match."""
        df = pd.DataFrame({
            "match_id": ["m1", "m1", "m2", "m2", "m2"],
            "Svr":      [1,    1,    2,    2,    1],
            "PtWinner": [1,    1,    2,    1,    1],
        })
        result = compute_split_points_streak(df)
        m2_serve = result[result["match_id"] == "m2"]["p1_serve_streak"].tolist()
        assert m2_serve == [0, 0, 0]

    def test_forward_fill_persists_through_opposite_type_points(self):
        """p1_serve_streak must remain constant through a stretch of return points,
        only updating at the next real serve point — not reset to 0 or NaN during the
        return-point interlude."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 5,
            "Svr":      [1,    2,    2,    2,    1],
            "PtWinner": [1,    1,    2,    1,    1],
        })
        result = compute_split_points_streak(df)
        # After the single serve win at idx0, p1_serve_streak should hold at 1 through
        # ALL the return points (idx1,2,3) until the next serve point (idx4) updates it.
        assert result["p1_serve_streak"].tolist() == [0, 1, 1, 1, 1]

    def test_no_leakage_from_future_point_outcomes(self):
        """Direct future-perturbation test on the two-series design specifically —
        critical to re-verify after the rewrite, not assumed to carry over from the
        earlier, simpler (and buggy) single-series version's leakage argument."""
        base = {
            "match_id": ["m1"] * 8,
            "Svr":      [1, 1, 2, 2, 1, 1, 2, 1],
            "PtWinner": [1, 1, 1, 2, 1, 2, 2, 1],
        }
        df_a = pd.DataFrame(base)
        df_b = pd.DataFrame(base)
        df_b.loc[7, "PtWinner"] = 2

        result_a = compute_split_points_streak(df_a)
        result_b = compute_split_points_streak(df_b)
        assert result_a["p1_serve_streak"].iloc[:7].tolist() == result_b["p1_serve_streak"].iloc[:7].tolist()
        assert result_a["p1_return_streak"].iloc[:7].tolist() == result_b["p1_return_streak"].iloc[:7].tolist()