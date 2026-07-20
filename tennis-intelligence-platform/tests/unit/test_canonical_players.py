from tennis_intel.entities.canonical_players import full_name_key, initials_key, query_keys, reorder_comma_name


class TestReorderCommaName:
    def test_reorders_last_comma_first(self):
        assert reorder_comma_name("Nadal, Rafael") == "Rafael Nadal"

    def test_leaves_no_comma_unchanged(self):
        assert reorder_comma_name("Rafael Nadal") == "Rafael Nadal"

    def test_handles_none(self):
        assert reorder_comma_name(None) == ""


class TestFullNameKey:
    def test_matches_across_formats(self):
        # User's exact example: Rafael Nadal / R. Nadal / Nadal, Rafael -> should converge
        # (full_name_key alone converges "Nadal, Rafael" with "Rafael Nadal"; "R. Nadal"
        # converges only via initials_key — see TestQueryKeys below)
        assert full_name_key("Rafael Nadal") == full_name_key("Nadal, Rafael") == "rafael nadal"

    def test_accent_normalization(self):
        # User's exact example: Novak Djoković -> Novak Djokovic
        assert full_name_key("Novak Djoković") == full_name_key("Novak Djokovic") == "novak djokovic"


class TestInitialsKey:
    def test_generates_first_initial_plus_surname(self):
        assert initials_key("rafael nadal") == "r nadal"

    def test_returns_none_for_single_token(self):
        assert initials_key("nadal") is None


class TestQueryKeys:
    def test_r_nadal_matches_rafael_nadal_via_initials(self):
        full = query_keys("Rafael Nadal")
        abbrev = query_keys("R. Nadal")
        assert full["initials"] == abbrev["initials"] == "r nadal"
        # but full names differ, which is exactly why initials is a separate fallback key
        assert full["full_name"] != abbrev["full_name"]