from pathlib import Path

import pytest

from rag_engine.ingest.match_documents import build_match_documents, DEFAULT_MATCHES_PATH
from rag_engine.ingest.player_documents import build_player_documents, DEFAULT_DAY6_PATH
from rag_engine.ingest.point_documents import build_point_documents

try:
    from rag_engine import _v1_path  # noqa: F401
    from tennis_intel.serving.replay_service import PROCESSED as V1_PROCESSED
    V1_CLASSIFIER_AVAILABLE = (V1_PROCESSED / "day9_point_classifiers.joblib").exists()
except Exception:
    V1_CLASSIFIER_AVAILABLE = False


def test_match_documents_shape():
    if not DEFAULT_MATCHES_PATH.exists():
        pytest.skip("v1 matches_with_elo.parquet not present in this environment")

    docs = list(build_match_documents(limit=5))
    assert len(docs) == 5
    for doc in docs:
        assert doc.doc_id.startswith("match:")
        assert doc.text  # non-empty
        assert doc.metadata["doc_type"] == "match_summary"
        assert doc.metadata["winner"]
        assert doc.metadata["loser"]
        # No NaN/None leaked into metadata (Chroma rejects non-str/int/float/bool).
        for value in doc.metadata.values():
            assert isinstance(value, (str, int, float, bool))


def test_match_document_text_is_grounded_in_row_fields():
    if not DEFAULT_MATCHES_PATH.exists():
        pytest.skip("v1 matches_with_elo.parquet not present in this environment")

    doc = next(build_match_documents(limit=1))
    assert doc.metadata["winner"] in doc.text
    assert doc.metadata["loser"] in doc.text


def test_player_documents_shape():
    if not DEFAULT_DAY6_PATH.exists():
        pytest.skip("v1 matches_with_day6_features.parquet not present in this environment")

    docs = list(build_player_documents(limit=5))
    assert len(docs) == 5
    for doc in docs:
        assert doc.doc_id.startswith("player:")
        assert doc.text
        assert doc.metadata["doc_type"] == "player_profile"
        assert doc.metadata["career_matches"] >= 10  # MIN_CAREER_MATCHES
        for value in doc.metadata.values():
            assert isinstance(value, (str, int, float, bool))


def test_player_document_text_is_grounded_in_profile():
    if not DEFAULT_DAY6_PATH.exists():
        pytest.skip("v1 matches_with_day6_features.parquet not present in this environment")

    doc = next(build_player_documents(limit=1))
    assert doc.metadata["player"] in doc.text


@pytest.fixture(scope="module")
def point_docs():
    """Loads the full v1 ReplayContext once (slow: classifier + full point-level
    dataset) and shares it across the point-document tests in this module. Scans only
    a handful of matches -- each match requires the full 5-engine per-point
    computation (~30-100s/match observed in production use), so match_limit is kept
    small deliberately; notable points (swing >= MIN_SWING) aren't guaranteed in
    every match, so this can legitimately skip if none turn up in the small sample."""
    if not V1_CLASSIFIER_AVAILABLE:
        pytest.skip("v1 day9_point_classifiers.joblib not present in this environment")
    docs = list(build_point_documents(match_limit=3))
    if not docs:
        pytest.skip("no notable points found in the first 3 scanned matches")
    return docs


def test_point_documents_shape(point_docs):
    for doc in point_docs:
        assert doc.doc_id.startswith("point:")
        assert doc.text
        assert doc.metadata["doc_type"] == "notable_point"
        assert doc.metadata["swing"] >= 0.10  # MIN_SWING
        assert doc.metadata["winner"] in (doc.metadata["player1"], doc.metadata["player2"])
        for value in doc.metadata.values():
            assert isinstance(value, (str, int, float, bool))


def test_point_document_text_is_grounded_in_metadata(point_docs):
    doc = point_docs[0]
    assert doc.metadata["player1"] in doc.text
    assert doc.metadata["player2"] in doc.text
    assert doc.metadata["winner"] in doc.text
    # "overall match win probability" is the swing-neutral phrasing this module is
    # built around (see known_issue_after_point_swing_includes_next_point_context.md)
    # -- never assert the point's outcome caused the swing.
    assert "overall match win probability" in doc.text
    assert "swung" not in doc.text.lower()
    assert "caused" not in doc.text.lower()


def test_point_document_hedge_text_is_well_formed(point_docs):
    """direction_matches_winner's hedge sentence ("not attributable solely to this
    point's outcome") must never appear mid-word or malformed, and every document's
    text must be non-empty regardless of whether the hedge fires for it.

    Deliberately does NOT assert a minimum hedge count: whether it fires depends on
    which specific matches get scanned in this fixture's small match_limit=3 sample
    (not guaranteed even with >=20 points, unlike the dedicated ~334-point
    investigation in PROGRESS.md / known_issue_after_point_swing_includes_next_point_context.md,
    which already confirms the hedge fires on a real, substantial fraction of points
    at proper scale -- asserting a minimum here on a tiny sample would be flaky, not
    a real regression signal)."""
    hedge_text = "not attributable solely to this point's outcome"
    for doc in point_docs:
        assert doc.text.strip()
        if hedge_text in doc.text:
            assert doc.text.rstrip().endswith(
                "the difficulty of the next point to be served)."
            )
