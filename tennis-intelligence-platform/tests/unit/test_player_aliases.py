import pandas as pd
import pytest

from tennis_intel.entities.player_aliases import resolve_names, resolve_name, AliasResolutionResult


@pytest.fixture
def sample_registry():
    return pd.DataFrame([
        {"player_id": "player_00015", "canonical_name": "Rafael Nadal", "canonical_name_key": "rafael nadal"},
        {"player_id": "player_00003", "canonical_name": "Novak Djokovic", "canonical_name_key": "novak djokovic"},
        {"player_id": "player_00042", "canonical_name": "Robert Smith", "canonical_name_key": "robert smith"},
        {"player_id": "player_00043", "canonical_name": "Ryan Smith", "canonical_name_key": "ryan smith"},
    ])


class TestResolveNames:
    def test_exact_full_name_match(self, sample_registry):
        result = resolve_names(["Rafael Nadal"], sample_registry)
        assert result.name_to_id["Rafael Nadal"] == "player_00015"

    def test_abbreviated_initials_match(self, sample_registry):
        # User's exact example: R. Nadal -> player_00015
        result = resolve_names(["R. Nadal"], sample_registry)
        assert result.name_to_id["R. Nadal"] == "player_00015"

    def test_comma_format_match(self, sample_registry):
        # User's exact example: Nadal, Rafael -> player_00015
        result = resolve_names(["Nadal, Rafael"], sample_registry)
        assert result.name_to_id["Nadal, Rafael"] == "player_00015"

    def test_accented_name_match(self, sample_registry):
        # User's exact example: Novak Djoković -> Novak Djokovic's id
        result = resolve_names(["Novak Djoković"], sample_registry)
        assert result.name_to_id["Novak Djoković"] == "player_00003"

    def test_ambiguous_initials_left_unresolved(self, sample_registry):
        # Two players share the "r smith" initials key — must NOT guess
        result = resolve_names(["R. Smith"], sample_registry)
        assert "R. Smith" not in result.name_to_id
        assert result.resolutions[0].strategy == "unresolved"

    def test_completely_unknown_name_unresolved(self, sample_registry):
        result = resolve_names(["Totally Unknown Player"], sample_registry)
        assert "Totally Unknown Player" not in result.name_to_id

    def test_join_derived_alias_takes_priority(self, sample_registry):
        # Even if a name COULD fuzzy-match, an explicit join-derived alias should be used
        # directly rather than re-deriving it — this is the free, ground-truth path
        aliases = {"Some Weird Spelling": "player_00015"}
        result = resolve_names(["Some Weird Spelling"], sample_registry, join_derived_aliases=aliases)
        assert result.name_to_id["Some Weird Spelling"] == "player_00015"
        assert result.resolutions[0].strategy == "join_derived"

    def test_unresolved_names_helper(self, sample_registry):
        result = resolve_names(["Rafael Nadal", "Nobody Real"], sample_registry)
        assert result.unresolved_names() == ["Nobody Real"]