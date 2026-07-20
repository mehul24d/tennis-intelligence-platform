"""embed_point_documents_from_disk.py — step 2 of 2. Loads the JSON produced by
generate_point_documents_to_disk.py and appends them to the persisted index. Only
imports rag_engine.index (sentence-transformers/Chroma) -- never touches v1's
stack, by design (see generate_point_documents_to_disk.py's docstring for why)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_engine.index.vector_store import VectorStore
from rag_engine.ingest.types import RagDocument

IN_PATH = Path(__file__).resolve().parents[1] / "data" / "point_documents_batch.json"


def main():
    with open(IN_PATH) as f:
        raw = json.load(f)
    docs = [RagDocument(doc_id=d["doc_id"], text=d["text"], metadata=d["metadata"]) for d in raw]
    print(f"loaded {len(docs)} point documents from {IN_PATH}")

    store = VectorStore()
    print(f"docs before: {store.count()}")
    t0 = time.time()
    n = store.build_index(docs, reset=False, progress_every=200)
    print(f"Indexed {n} point documents in {time.time()-t0:.1f}s")
    print(f"docs after: {store.count()}")


if __name__ == "__main__":
    main()
