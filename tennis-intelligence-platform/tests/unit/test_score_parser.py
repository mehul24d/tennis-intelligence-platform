from tennis_intel.features.score_parser import parse_score


class TestParseScore:
    def test_straight_sets(self):
        r = parse_score("6-4 6-3")
        assert r.sets_won == 2 and r.sets_lost == 0
        assert r.games_won == 12 and r.games_lost == 7
        assert r.straight_sets is True
        assert r.parse_ok is True

    def test_three_set_match(self):
        r = parse_score("6-4 3-6 7-5")
        assert r.sets_won == 2 and r.sets_lost == 1
        assert r.games_won == 16 and r.games_lost == 15
        assert r.straight_sets is False

    def test_tiebreak_parenthetical_stripped(self):
        r = parse_score("7-6(4) 6-3")
        assert r.sets_won == 2 and r.games_won == 13 and r.games_lost == 9

    def test_retirement_with_incomplete_trailing_set_is_dropped(self):
        # "3-1" never reached a valid finished-set score -> must not count as a won set
        r = parse_score("6-2 3-1 RET")
        assert r.retired is True
        assert r.sets_won == 1 and r.sets_lost == 0
        assert r.games_won == 6 and r.games_lost == 2
        assert r.n_sets_played == 1

    def test_retirement_at_clean_set_boundary_keeps_last_set(self):
        # "6-3" IS a valid finished-set score -> must be kept, not dropped just because it's last
        r = parse_score("6-2 6-3 RET")
        assert r.sets_won == 2 and r.sets_lost == 0
        assert r.games_won == 12 and r.games_lost == 5
        assert r.n_sets_played == 2

    def test_retired_in_first_set_no_completed_sets(self):
        r = parse_score("3-1 RET")
        assert r.sets_won == 0 and r.sets_lost == 0
        assert r.n_sets_played == 0
        assert r.parse_ok is True

    def test_walkover(self):
        r = parse_score("W/O")
        assert r.walkover is True
        assert r.parse_ok is False

    def test_none_input(self):
        r = parse_score(None)
        assert r.walkover is True
        assert r.parse_ok is False

    def test_malformed_input_does_not_raise(self):
        r = parse_score("garbage!!")
        assert r.parse_ok is False