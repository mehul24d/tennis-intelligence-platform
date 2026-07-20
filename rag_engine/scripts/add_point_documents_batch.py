"""add_point_documents_batch.py — appends a representative match_limit=100 batch of
point documents to the ALREADY-PERSISTED 22,610-doc index (match+player docs),
without wiping or re-embedding them. reset=False is deliberate: build_index.py's CLI
always resets on its first non-skipped step, which would require re-running the
match+player embed (~unnecessary cost) just to add points -- this script instead
opens the existing persisted collection directly and appends.

Real, measured cost (see conversation): ~25s/match for point-document generation
(v1's full 5-engine per-point computation per match) -- match_limit=100 is a
deliberate, documented partial/representative subset of the 5,981-match frozen-join
corpus (full corpus measured at ~41.6 hours), not a full production deployment --
same scope-decision pattern as the existing 22,610-doc match/player subset.

REAL CRASH FOUND AND FIXED (2026-07-16): the first run of this script segfaulted
(exit 139) with no Python traceback -- combining rag_engine's VectorStore
(PyTorch/sentence-transformers) and v1's ReplayContext (its own model-loading
stack) in one process crashes on macOS without `KMP_DUPLICATE_LIB_OK=TRUE` set, a
known class of failure when multiple libraries bundle conflicting OpenMP runtimes.
Confirmed directly (isolated repro: both load fine together with the env var set,
crash reproduces without it) before retrying the real batch -- run this script with
`KMP_DUPLICATE_LIB_OK=TRUE python3 add_point_documents_batch.py`.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_engine.ingest.point_documents import build_point_documents
from rag_engine.index.vector_store import VectorStore

MATCH_LIMIT = 100


def main():
    store = VectorStore()
    print(f"docs before: {store.count()}")

    t0 = time.time()
    n = store.build_index(
        build_point_documents(match_limit=MATCH_LIMIT), reset=False, progress_every=200
    )
    elapsed = time.time() - t0
    print(f"Indexed {n} point documents in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"docs after: {store.count()}")


if __name__ == "__main__":
    main()
