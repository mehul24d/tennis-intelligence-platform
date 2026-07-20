import shutil
import tempfile
from pathlib import Path

import pytest

from rag_engine.ingest.types import RagDocument
from rag_engine.index.vector_store import VectorStore

FAKE_DOCS = [
    RagDocument(
        doc_id="match:1",
        text="Carlos Alcaraz defeated Novak Djokovic 6-4 6-4 in the final of Wimbledon (Grass).",
        metadata={"doc_type": "match_summary", "surface": "Grass", "winner": "Carlos Alcaraz"},
    ),
    RagDocument(
        doc_id="match:2",
        text="Iga Swiatek defeated Aryna Sabalenka 6-2 6-3 in the final of Roland Garros (Clay).",
        metadata={"doc_type": "match_summary", "surface": "Clay", "winner": "Iga Swiatek"},
    ),
    RagDocument(
        doc_id="player:1",
        text="Carlos Alcaraz — career profile. 200 matches, 160 wins (80.0% win rate).",
        metadata={"doc_type": "player_profile", "player": "Carlos Alcaraz"},
    ),
]


@pytest.fixture
def store():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        yield VectorStore(persist_dir=tmp_dir, collection_name="test_collection")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_index_and_count(store):
    n = store.build_index(FAKE_DOCS)
    assert n == 3
    assert store.count() == 3


def test_retrieve_returns_topically_relevant_results(store):
    store.build_index(FAKE_DOCS)
    results = store.retrieve("Alcaraz Wimbledon grass court final", k=1)
    assert len(results) == 1
    assert results[0].doc_id == "match:1"


def test_retrieve_respects_metadata_filter(store):
    store.build_index(FAKE_DOCS)
    results = store.retrieve("tennis final", k=5, filters={"doc_type": "player_profile"})
    assert len(results) == 1
    assert results[0].doc_id == "player:1"


def test_build_index_reset_clears_previous_documents(store):
    store.build_index(FAKE_DOCS)
    assert store.count() == 3
    store.build_index(FAKE_DOCS[:1], reset=True)
    assert store.count() == 1
