"""generate_point_documents_to_disk.py — step 1 of 2 for adding point documents to
the live index. Runs ONLY v1's stack (no sentence-transformers/torch/Chroma loaded
at all) and serializes the generated RagDocuments to a JSON file.

WHY TWO SEPARATE PROCESSES: the combined-process approach
(add_point_documents_batch.py) segfaulted (exit 139) twice -- once at the "load
both stacks" point (fixed by KMP_DUPLICATE_LIB_OK=TRUE), and again during the
actual generation+embedding loop even with that fix, meaning the crash is a
deeper runtime conflict between PyTorch/MPS and v1's model-inference stack when
both are ACTIVELY computing in one process, not just present. Splitting into two
single-stack processes sidesteps the conflict entirely rather than chasing a
harder in-process fix -- a real environment finding, not a workaround for a bug
in this project's own code.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_engine.ingest.point_documents import build_point_documents

MATCH_LIMIT = 100
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "point_documents_batch.json"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    docs = list(build_point_documents(match_limit=MATCH_LIMIT))
    elapsed = time.time() - t0
    print(f"Generated {len(docs)} point documents from {MATCH_LIMIT} matches in {elapsed:.1f}s")

    serializable = [{"doc_id": d.doc_id, "text": d.text, "metadata": d.metadata} for d in docs]
    with open(OUT_PATH, "w") as f:
        json.dump(serializable, f)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
