"""embedder.py — thin wrapper around a local, CPU-only sentence-transformers model.

MODEL CHOICE: all-MiniLM-L6-v2 — small (~80MB), fast on CPU (no GPU available on this
M2), and a standard, well-validated baseline for semantic-search embeddings. Loaded
once and reused (model loading is the expensive part, not individual encode calls).
"""

from __future__ import annotations

from functools import lru_cache

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


@lru_cache(maxsize=1)
def get_default_embedder() -> Embedder:
    """Process-wide singleton — avoids reloading the model once per script/module,
    the same rationale as replay_service.py's load_replay_context() being called once
    at startup rather than per-request."""
    return Embedder()
