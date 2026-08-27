# Deliberately empty (2026-08): this package used to eagerly re-export
# build_match_documents/build_player_documents/build_point_documents/RagDocument
# here for `from rag_engine.ingest import X` convenience, but nothing in the repo
# actually imported it that way (confirmed via a full grep) -- every real caller
# (build_index.py, tests) already imports from the specific submodule
# (`rag_engine.ingest.match_documents`, etc.). That eager re-export had a real
# cost: importing ANY submodule under rag_engine.ingest runs this __init__.py
# first (how Python package imports work), which pulled in point_documents.py's
# own module-level `from tennis_intel.serving.replay_service import ...` --
# v1's ENTIRE serving module, including joblib and its trained-classifier-loading
# code -- just to reach vector_store.py's `from rag_engine.ingest.types import
# RagDocument` (a plain dataclass with zero need for any of that). Discovered
# directly: v2_serving's public (query-only) deployment failed at query time with
# `ModuleNotFoundError: No module named 'joblib'`, since that lean deployment's
# venv has no reason to install v1's dependencies at all. Import the specific
# submodule you need instead (`from rag_engine.ingest.match_documents import
# build_match_documents`, `from rag_engine.ingest.types import RagDocument`, etc.).
