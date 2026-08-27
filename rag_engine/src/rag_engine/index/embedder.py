"""embedder.py — turns text into vectors for the Chroma index.

PROVIDER (2026-08): switched from a local, CPU-only sentence-transformers model
(all-MiniLM-L6-v2) to Gemini's embedding API. The local model's own weights are
only ~80MB, but sentence-transformers pulls in torch as a dependency — measured
directly (not assumed) at ~504MB RSS just to construct one Embedder + open an
empty Chroma client, before FastAPI/uvicorn's own overhead, which alone exceeds
Render's 512MB free-tier ceiling this project is deployed under. GeminiEmbedder has no local model to
load — its only per-process cost is the google-genai HTTP client — trading a
network round-trip per encode() call for that entire footprint. Both classes
implement the same encode()/encode_query() interface, so VectorStore doesn't
care which one it's holding; Embedder (local) is kept for offline/no-network use,
not deleted, since rebuilding the index is exactly the kind of one-off script
where a local model's latency advantage still matters and network access can't
be assumed.

ASYMMETRIC EMBEDDING (query vs document): Gemini's embedding API distinguishes
task_type="RETRIEVAL_DOCUMENT" (what gets indexed) from "RETRIEVAL_QUERY" (what
a search query gets encoded as) — using the matched pair measurably improves
retrieval quality over embedding both the same way (Google's own documented
recommendation for this API). encode_query() exists as a separate method
specifically so VectorStore.retrieve() can request the query variant instead of
silently reusing encode()'s document-task embeddings for queries too.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL_NAME = "gemini-embedding-001"
# Matryoshka-truncated from the model's native 3072 dims (Gemini's own documented
# support for this) — keeps the persisted index and per-query payload smaller with
# minimal measured quality loss at this size, per Google's own guidance for RAG-scale
# retrieval use cases (not a arbitrary guess).
GEMINI_OUTPUT_DIMENSIONALITY = 768
# The API's own hard limit (confirmed directly: a 150-item batch returned "400
# INVALID_ARGUMENT ... at most 100 requests can be in one batch"), independent of
# VectorStore's own BATCH_SIZE (256) for writing to Chroma — encode() sub-chunks
# internally so callers can pass any size list without knowing about this ceiling.
GEMINI_MAX_BATCH_SIZE = 100


class Embedder:
    """Local, CPU-only sentence-transformers model — kept for offline/no-network
    use (e.g. a laptop with no Gemini key configured), not the default anymore."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def encode_query(self, text: str) -> list[float]:
        """No query/document distinction locally (MiniLM embeds both the same way) —
        exists only so callers (VectorStore.retrieve) can treat both Embedder
        implementations identically."""
        return self.encode([text])[0]


class MissingAPIKeyError(RuntimeError):
    """Raised at GeminiEmbedder construction time (not on first call) when no key
    is configured — same fail-fast discipline as generate.GeminiClient."""


def _is_retryable(exc: BaseException) -> bool:
    """503 UNAVAILABLE (transient overload) and 429 RESOURCE_EXHAUSTED (free-tier
    per-minute rate limit — real and expected during a bulk index rebuild's many
    consecutive calls, unlike generate.GeminiClient's single-call-per-request
    pattern where a 429 usually means a truly exhausted daily quota instead) are
    both worth retrying here; 400/401/403 are not (waiting doesn't fix a bad
    request or a bad key)."""
    from google.genai.errors import ClientError, ServerError

    if isinstance(exc, ServerError) and getattr(exc, "code", None) == 503:
        return True
    if isinstance(exc, ClientError) and getattr(exc, "code", None) == 429:
        return True
    return False


class GeminiEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_GEMINI_MODEL_NAME,
        api_key: str | None = None,
        output_dimensionality: int = GEMINI_OUTPUT_DIMENSIONALITY,
    ):
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self._api_key = api_key or self._resolve_api_key()

        from google import genai

        self._client = genai.Client(api_key=self._api_key)

    @staticmethod
    def _resolve_api_key() -> str:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "No Gemini API key found. Set GEMINI_API_KEY (preferred) or "
                "GOOGLE_API_KEY before constructing GeminiEmbedder. Get a key at "
                "https://ai.google.dev/gemini-api/docs/api-key ."
            )
        return key

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        last_exc: BaseException | None = None
        for attempt in range(4):  # initial attempt + 3 retries, 1s/2s/4s backoff
            try:
                result = self._client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=self.output_dimensionality,
                    ),
                )
                return [e.values for e in result.embeddings]
            except Exception as exc:  # noqa: BLE001 -- inspected by _is_retryable, reraised otherwise
                if not _is_retryable(exc) or attempt == 3:
                    raise
                last_exc = exc
                time.sleep(2 ** attempt)
        raise last_exc  # unreachable, satisfies type-checkers

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), GEMINI_MAX_BATCH_SIZE):
            out.extend(self._embed_batch(texts[i:i + GEMINI_MAX_BATCH_SIZE], "RETRIEVAL_DOCUMENT"))
        return out

    def encode_query(self, text: str) -> list[float]:
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]


@lru_cache(maxsize=1)
def get_default_embedder() -> GeminiEmbedder:
    """Process-wide singleton — avoids reconstructing the client once per
    script/module, same rationale as replay_service.py's load_replay_context()
    being called once at startup rather than per-request."""
    return GeminiEmbedder()
