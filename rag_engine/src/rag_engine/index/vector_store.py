"""vector_store.py — a thin, typed wrapper around a locally-persisted Chroma
collection. build_index() writes RagDocuments; retrieve() reads them back with
optional metadata filters. No external service — everything lives under
rag_engine/data/chroma/ on disk, rebuildable at any time from source parquet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rag_engine.index.embedder import Embedder, get_default_embedder
from rag_engine.ingest.types import RagDocument

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[3] / "data" / "chroma"
DEFAULT_COLLECTION_NAME = "tennis_intelligence"

# Chroma writes/embeds documents in batches — keeps memory bounded when indexing the
# full ~198k-match corpus, rather than encoding every document's text in one call.
BATCH_SIZE = 256


@dataclass(frozen=True)
class RetrievedDocument:
    doc_id: str
    text: str
    metadata: dict
    distance: float


class VectorStore:
    def __init__(
        self,
        persist_dir: Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedder: Embedder | None = None,
    ):
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(collection_name)
        self._embedder = embedder or get_default_embedder()

    def build_index(
        self, documents: Iterable[RagDocument], reset: bool = True, progress_every: int = 0
    ) -> int:
        """Writes documents into the collection, batched. If reset=True (default),
        deletes any existing documents in the collection first, so re-running
        build_index against updated source data doesn't accumulate stale duplicates.
        progress_every > 0 prints a running count every N documents, to give visibility
        into long-running builds over the full ~198k-match corpus."""
        import time

        if reset:
            existing = self._collection.get(include=[])
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])

        batch: list[RagDocument] = []
        n_written = 0
        t0 = time.time()
        for doc in documents:
            batch.append(doc)
            if len(batch) >= BATCH_SIZE:
                self._write_batch(batch)
                n_written += len(batch)
                batch = []
                if progress_every and n_written % progress_every < BATCH_SIZE:
                    rate = n_written / (time.time() - t0)
                    print(f"  ... {n_written} written ({rate:.0f} docs/sec)", flush=True)
        if batch:
            self._write_batch(batch)
            n_written += len(batch)
        return n_written

    def _write_batch(self, batch: list[RagDocument]) -> None:
        texts = [doc.text for doc in batch]
        embeddings = self._embedder.encode(texts)
        self._collection.upsert(
            ids=[doc.doc_id for doc in batch],
            documents=texts,
            metadatas=[doc.metadata for doc in batch],
            embeddings=embeddings,
        )

    def retrieve(
        self, query: str, k: int = 5, filters: dict | None = None
    ) -> list[RetrievedDocument]:
        """filters: a Chroma `where` clause, e.g. {"surface": "Clay"} or
        {"doc_type": "player_profile"} — exact-match metadata filtering, composed with
        semantic similarity (Chroma applies the filter first, then ranks by distance)."""
        query_embedding = self._embedder.encode([query])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding], n_results=k, where=filters,
        )
        out = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
            out.append(RetrievedDocument(doc_id=doc_id, text=text, metadata=meta, distance=dist))
        return out

    def count(self) -> int:
        return self._collection.count()
