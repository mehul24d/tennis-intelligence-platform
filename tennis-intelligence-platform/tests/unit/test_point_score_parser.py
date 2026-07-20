from tennis_intel.features.point_score_parser import (
    parse_pts, is_break_point, would_win_game_next_point,
    would_win_set_by_winning_this_game, is_set_point, is_match_point,
)


class TestParsePts:
    def test_regular_scores(self):
        for raw, expected in [("0-15", (0, 1)), ("40-40", (3, 3)), ("AD-40", (4, 3)), ("15-30", (1, 2))]:
            r = parse_pts(raw, is_tiebreak_game=False)
            assert r.parse_ok
            assert (r.p1_points, r.p2_points) == expected

    def test_tiebreak_score(self):
        r = parse_pts("6-5", is_tiebreak_game=True)
        assert r.is_tiebreak_score
        assert r.tb_p1_points == 6 and r.tb_p2_points == 5

    def test_malformed_does_not_raise(self):
        r = parse_pts("garbage", is_tiebreak_game=False)
        assert not r.parse_ok


class TestBreakPoint:
    def test_deuce_is_not_break_point(self):
        assert is_break_point(True, 3, 3, False) is False

    def test_thirty_forty_is_break_point(self):
        assert is_break_point(True, 2, 3, False) is True

    def test_returner_advantage_is_break_point(self):
        assert is_break_point(True, 3, 4, False) is True

    def test_server_advantage_is_not_break_point(self):
        assert is_break_point(True, 4, 3, False) is False

    def test_tiebreak_never_flagged(self):
        assert is_break_point(True, 0, 0, True, tb_p1=5, tb_p2=6) is False


class TestGameAndSetLogic:
    def test_would_win_game_regular(self):
        assert would_win_game_next_point(True, 3, 1, False) is True   # 40-15
        assert would_win_game_next_point(True, 2, 3, False) is False  # 30-40

    def test_would_win_game_tiebreak_needs_two_clear(self):
        assert would_win_game_next_point(True, 0, 0, True, tb_p1=6, tb_p2=5) is True   # -> 7-5
        assert would_win_game_next_point(True, 0, 0, True, tb_p1=6, tb_p2=6) is False  # -> 7-6, no

    def test_would_win_set(self):
        assert would_win_set_by_winning_this_game(True, p1_games=5, p2_games=3) is True   # 6-3
        assert would_win_set_by_winning_this_game(True, p1_games=5, p2_games=5) is False  # 6-5
        assert would_win_set_by_winning_this_game(True, p1_games=6, p2_games=6) is True   # 7-6 via breaker


class TestSetAndMatchPoint:
    def test_set_point_true(self):
        assert is_set_point(True, 3, 1, False, p1_games=5, p2_games=3) is True

    def test_set_point_false_when_game_win_does_not_clinch_set(self):
        assert is_set_point(True, 3, 1, False, p1_games=3, p2_games=2) is False

    def test_match_point_best_of_3_up_one_set(self):
        assert is_match_point(True, 3, 1, False, p1_games=5, p2_games=3,
                               p1_sets=1, p2_sets=0, best_of=3) is True

    def test_not_match_point_when_sets_even(self):
        assert is_match_point(True, 3, 1, False, p1_games=5, p2_games=3,
                               p1_sets=0, p2_sets=0, best_of=3) is False

    def test_match_point_best_of_5_needs_three_sets(self):
        assert is_match_point(True, 3, 1, False, p1_games=5, p2_games=3,
                               p1_sets=2, p2_sets=0, best_of=5) is True