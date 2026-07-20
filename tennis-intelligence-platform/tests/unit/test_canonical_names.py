from tennis_intel.data.canonical_player_names import (
    normalize_player_name,
    normalize_round,
    normalize_tournament_name,
)


class TestNormalizePlayerName:
    def test_strips_accents(self):
        assert normalize_player_name("Félix Auger-Aliassime") == "felix auger-aliassime"

    def test_lowercases(self):
        assert normalize_player_name("Roger Federer") == "roger federer"

    def test_removes_periods(self):
        assert normalize_player_name("J.J. Wolf") == "jj wolf"

    def test_collapses_whitespace(self):
        assert normalize_player_name("Novak   Djokovic") == "novak djokovic"

    def test_handles_none(self):
        assert normalize_player_name(None) == ""

    def test_handles_nan(self):
        assert normalize_player_name(float("nan")) == ""

    def test_strips_suffix(self):
        assert normalize_player_name("John Smith Jr.") == "john smith"

    def test_preserves_hyphens(self):
        assert normalize_player_name("Jean-Julien Rojer") == "jean-julien rojer"


class TestNormalizeTournamentName:
    def test_lowercases_and_strips(self):
        assert normalize_tournament_name("  Wimbledon  ") == "wimbledon"

    def test_handles_none(self):
        assert normalize_tournament_name(None) == ""

    def test_strips_sponsor_noise(self):
        result = normalize_tournament_name("Miami Open presented by Itau")
        assert "presented by" not in result


class TestNormalizeRound:
    def test_maps_short_codes(self):
        assert normalize_round("F") == "F"
        assert normalize_round("SF") == "SF"

    def test_maps_long_form(self):
        assert normalize_round("Final") == "F"
        assert normalize_round("Quarterfinal") == "QF"

    def test_handles_none(self):
        assert normalize_round(None) == ""

    def test_unknown_round_uppercased_not_dropped(self):
        # Unknown round labels should still produce SOMETHING joinable, not silently vanish
        assert normalize_round("weird_round_label") == "WEIRD_ROUND_LABEL"