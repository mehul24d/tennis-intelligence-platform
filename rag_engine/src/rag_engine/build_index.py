"""build_index.py — CLI entrypoint to (re)build the persisted Chroma index from the
v1 platform's processed data.

Usage:
    python -m rag_engine.build_index --limit 50          # fast subset, for verification
    python -m rag_engine.build_index                     # full corpus
    python -m rag_engine.build_index --skip-points        # skip the expensive point-doc
                                                           # step (loads the full v1
                                                           # ReplayContext: classifier +
                                                           # point-level dataset)

Prints per-type document counts so index-build failures/coverage gaps are visible
immediately, rather than silently producing a small or empty index.
"""

from __future__ import annotations

import argparse
import time

from rag_engine.ingest.match_documents import build_match_documents
from rag_engine.ingest.player_documents import build_player_documents
from rag_engine.ingest.point_documents import build_point_documents
from rag_engine.index.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of matches AND players ingested (each independently), "
             "unless overridden by --match-limit/--player-limit/--point-match-limit. "
             "Omit all to index the full corpus.",
    )
    parser.add_argument(
        "--match-limit", type=int, default=None,
        help="Cap the number of matches ingested (most recent first). Overrides --limit.",
    )
    parser.add_argument(
        "--player-limit", type=int, default=None,
        help="Cap the number of players ingested (most career matches first). Overrides --limit.",
    )
    parser.add_argument(
        "--point-match-limit", type=int, default=None,
        help="Cap the number of matches SCANNED for notable points (not the number of "
             "point documents produced). Overrides --limit.",
    )
    parser.add_argument(
        "--skip-matches", action="store_true", help="Skip match documents.",
    )
    parser.add_argument(
        "--skip-players", action="store_true", help="Skip player documents.",
    )
    parser.add_argument(
        "--skip-points", action="store_true",
        help="Skip point documents (avoids loading the full v1 ReplayContext, which is "
             "the most expensive step — the trained classifier plus the full point-level "
             "dataset).",
    )
    args = parser.parse_args()
    match_limit = args.match_limit if args.match_limit is not None else args.limit
    player_limit = args.player_limit if args.player_limit is not None else args.limit
    point_match_limit = args.point_match_limit if args.point_match_limit is not None else args.limit

    store = VectorStore()
    reset = True  # only the FIRST build_index call in this run should clear the collection

    if not args.skip_matches:
        t0 = time.time()
        n = store.build_index(
            build_match_documents(limit=match_limit), reset=reset, progress_every=2000
        )
        reset = False
        print(f"Indexed {n} match documents in {time.time() - t0:.1f}s")

    if not args.skip_players:
        t0 = time.time()
        n = store.build_index(
            build_player_documents(limit=player_limit), reset=reset, progress_every=500
        )
        reset = False
        print(f"Indexed {n} player documents in {time.time() - t0:.1f}s")

    if not args.skip_points:
        t0 = time.time()
        n = store.build_index(
            build_point_documents(match_limit=point_match_limit), reset=reset, progress_every=200
        )
        reset = False
        print(f"Indexed {n} point documents in {time.time() - t0:.1f}s")

    print(f"Total documents in collection: {store.count()}")


if __name__ == "__main__":
    main()
