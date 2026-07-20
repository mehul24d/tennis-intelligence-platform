"""generate.py — the generation seam: takes retrieved context (from
index/vector_store.py) plus a question and calls Gemini to produce a grounded
answer. This is the ONLY module in rag_engine that talks to the LLM provider —
everything else (ingest/, index/) is provider-agnostic and untouched by this choice.

PROVIDER: Gemini (Google), via the `google-genai` package -- verified current as of
this module's writing (2026-07): `google-genai` reached General Availability in May
2025 and is Google's actively-maintained, unified Gen AI SDK. The older
`google-generativeai` package (`import google.generativeai as genai`) is DEPRECATED
(as of 2025-11-30) and must not be used for new code -- if you're reading this and
`google-genai` has since been superseded, verify current guidance before assuming
this module is still correct (this is exactly the kind of fast-moving API surface
that goes stale silently).

ENV VAR: reads `GEMINI_API_KEY` (Google's preferred, explicit-intent variable name;
`GOOGLE_API_KEY` also works and takes precedence if both happen to be set, per
google-genai's own resolution order, but this module deliberately checks
GEMINI_API_KEY first and only falls back to GOOGLE_API_KEY, to keep intent explicit
rather than relying on ambient precedence rules). Also loads a `.env` file (via
python-dotenv) from the rag_engine project root if one exists, so a key pasted into
`.env` (gitignored — see .gitignore and .env.example) is picked up without needing
to `export` it in the shell first. Never overrides a variable already set in the
real environment (`override=False`) — an explicit `export` always wins over `.env`.

FAILS GRACEFULLY WHEN NO KEY IS PRESENT: constructing GeminiClient with no key
raises a clear, actionable error immediately (at construction, not on first call) --
retrieval/index scripts never import this module, so they run fully whether or not
a key is configured. This is the seam the Phase 2 LLM agent (llm_agent/) will reuse.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Loaded once at import time, not per-call: .env lives at the rag_engine project
# root (sibling of pyproject.toml), independent of the caller's own cwd.
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH, override=False)


class MissingAPIKeyError(RuntimeError):
    """Raised at GeminiClient construction time (not on first call) when neither
    GEMINI_API_KEY nor GOOGLE_API_KEY is set -- fail fast and clearly, rather than
    letting a caller discover this deep inside a generate() call."""


@dataclass(frozen=True)
class RetrievedContext:
    """A single retrieved document, formatted for inclusion in the prompt. Mirrors
    index.vector_store.RetrievedDocument's fields but kept as its own type here so
    generate.py doesn't import chromadb-specific machinery -- callers pass whatever
    they got back from VectorStore.retrieve() (or any other source) through this
    thin, provider-agnostic shape."""

    doc_id: str
    text: str
    metadata: dict


# "Pro"-tier models (e.g. gemini-2.5-pro) return a 429 RESOURCE_EXHAUSTED with a
# free-tier quota of 0 for accounts without billing enabled -- confirmed directly
# against a real key, 2026-07. gemini-3.5-flash is the current flash-tier model
# (verified available via client.models.list()) and works on the free tier.
# Override via GeminiClient(model=...) if your account has billing enabled and you
# want a Pro-tier model instead.
DEFAULT_MODEL = "gemini-3.5-flash"


def _is_retryable_server_error(exc: BaseException) -> bool:
    """True only for 503 UNAVAILABLE (transient overload -- self-heals on retry).
    401/403 (bad key) and 429 (quota exhausted) are NOT retried here: waiting a few
    seconds doesn't fix a bad key or an exhausted quota, so retrying just delays a
    failure the caller should see immediately."""
    from google.genai.errors import ServerError

    return isinstance(exc, ServerError) and getattr(exc, "code", None) == 503


class GeminiClient:
    """Thin wrapper around google-genai's client, scoped to this project's one use
    case: answer a tennis question grounded in retrieved context. Not a general
    chat wrapper -- see build_prompt() for the exact grounding discipline."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self._api_key = api_key or self._resolve_api_key()

        # Import here, not at module level: importing google.genai when no key is
        # configured (and the package may not even be installed, since it's an
        # optional dependency -- see pyproject.toml's `generation` extra) shouldn't
        # break importing the rest of rag_engine. Retrieval/index code never
        # triggers this import at all.
        from google import genai

        self._client = genai.Client(api_key=self._api_key)

    @staticmethod
    def _resolve_api_key() -> str:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "No Gemini API key found. Set GEMINI_API_KEY (preferred) or "
                "GOOGLE_API_KEY in your environment before constructing GeminiClient. "
                "Get a key at https://ai.google.dev/gemini-api/docs/api-key . "
                "Retrieval and index-building do not require this -- only generate()."
            )
        return key

    @retry(
        retry=retry_if_exception(_is_retryable_server_error),
        stop=stop_after_attempt(4),  # initial attempt + 3 retries
        wait=wait_exponential(multiplier=1, min=1, max=4),  # 1s, 2s, 4s
        reraise=True,
    )
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Single-turn generation. temperature defaults low (0.2, not the SDK's own
        default) since this is meant for grounded, factual tennis analysis, not
        creative writing -- callers doing something more exploratory can override.

        Retries up to 3 times (1s/2s/4s backoff) on a 503 UNAVAILABLE (Google's
        transient "high demand" response, observed directly against a real key/model
        combination) -- other errors (bad key, quota exhaustion) propagate immediately,
        since retrying wouldn't change their outcome."""
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return response.text


def build_prompt(question: str, context: list[RetrievedContext]) -> str:
    """Builds a grounded prompt from retrieved context, instructing the model to
    answer ONLY from what's provided and say so explicitly when the context doesn't
    cover the question -- the core RAG discipline, not left to the model's own
    judgment about what counts as "grounded enough."""
    if not context:
        context_block = "(no retrieved context available)"
    else:
        context_block = "\n\n".join(
            f"[{i + 1}] ({doc.metadata.get('doc_type', 'unknown')}) {doc.text}"
            for i, doc in enumerate(context)
        )
    return (
        "You are a tennis analytics assistant. Answer the question below using "
        "ONLY the retrieved context provided. If the context doesn't contain enough "
        "information to answer confidently, say so explicitly rather than guessing "
        "or relying on general knowledge.\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_grounded_answer(
    question: str, context: list[RetrievedContext], client: GeminiClient | None = None,
) -> str:
    """Convenience entrypoint tying build_prompt() + GeminiClient.generate()
    together -- the function callers (including the Phase 2 LLM agent) should reach
    for by default; pass an already-constructed `client` to reuse one across
    multiple calls rather than re-resolving the API key each time."""
    client = client or GeminiClient()
    prompt = build_prompt(question, context)
    return client.generate(prompt)
