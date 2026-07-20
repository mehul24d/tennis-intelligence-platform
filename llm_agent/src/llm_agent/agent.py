"""agent.py — the tennis tactical-analyst agent: fuses a LiveFeatureSnapshot (from v1's
serving layer) with rag_engine-retrieved historical context, and calls Gemini through a
stateful multi-turn chat session. This is the seam the dashboard (v2_dashboard/) will
call into.

MULTI-TURN STATE: uses google-genai's stateful chat API (`client.chats.create(...)` +
`chat.send_message(...)`), NOT a manually-managed messages list — the SDK's Chat object
keeps its own history internally and appends each turn's request/response
automatically, which is the documented, current way to do multi-turn conversations
with this SDK (confirmed against SDK docs, 2026-07; a hand-rolled list of
role/content dicts is not this SDK's shape and would need re-verifying independently
if this module is ever ported to a different provider's SDK).

SOURCES-USED TRANSPARENCY: every live feature and retrieved doc handed to the model is
tagged [L#]/[D#] (see live_features.py and rag_engine's own [n] convention in
generate.py -- this module uses its own D-prefixed numbering rather than reusing
generate.py's build_prompt(), since it needs the tags to survive into the citation
parse below). After each answer, this module regexes the answer text for [L#]/[D#]
citations and reports exactly which sources the model actually cited back to the
caller -- this is a citation audit, not a claim that the model internally "used" only
those sources; it can only report what it cited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from rag_engine.generate import _is_retryable_server_error
from rag_engine.index.vector_store import RetrievedDocument, VectorStore

from llm_agent.live_features import LiveFeatureSnapshot
from llm_agent.system_prompt import SYSTEM_PROMPT

_CITATION_RE = re.compile(r"\[(L\d+|D\d+)\]")


@dataclass(frozen=True)
class SourcesUsed:
    """The citation audit for one answer -- only tags the model actually referenced in
    its text, each resolved back to a human-readable descriptor."""

    live_features: list[str] = field(default_factory=list)
    retrieved_docs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    sources_used: SourcesUsed
    sources_offered: dict  # tag -> descriptor, for every source the model was given (cited or not)


def _build_context_block(
    live: LiveFeatureSnapshot | None, docs: list[RetrievedDocument]
) -> tuple[str, dict[str, str]]:
    """Returns (context text for the prompt, tag -> descriptor map for every source
    offered) -- the descriptor map is what sources_offered / the citation audit resolve
    against."""
    lines: list[str] = []
    tag_map: dict[str, str] = {}

    if live is not None and live.features:
        lines.append(f"LIVE FEATURES ({live.p1_name} vs {live.p2_name}, match {live.match_id}):")
        for tagged_line, feat in zip(live.to_tagged_lines(), live.features):
            lines.append(tagged_line)
            tag_num = tagged_line.split("]")[0].strip("[")
            tag_map[tag_num] = f"{feat.label}: {feat.value}"

    if docs:
        lines.append("\nRETRIEVED CONTEXT:")
        for i, doc in enumerate(docs, start=1):
            tag = f"D{i}"
            doc_type = doc.metadata.get("doc_type", "unknown")
            lines.append(f"[{tag}] ({doc_type}) {doc.text}")
            tag_map[tag] = f"{doc_type}: {doc.text[:100]}{'...' if len(doc.text) > 100 else ''}"

    if not lines:
        lines.append("(no live features or retrieved context available for this question)")

    return "\n".join(lines), tag_map


class TennisAnalystAgent:
    """One agent instance == one ongoing conversation. Construct a new instance to
    start a fresh conversation; reuse the same instance across ask() calls to keep
    multi-turn context (the model can refer back to a previous answer)."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        model: str | None = None,
        k: int = 5,
        gemini_client=None,
    ):
        from rag_engine.generate import DEFAULT_MODEL, GeminiClient

        self._store = vector_store or VectorStore()
        self._k = k
        self._gemini = gemini_client or GeminiClient(model=model or DEFAULT_MODEL)
        self._chat = None  # created lazily on first ask() -- see _ensure_chat()

    def _ensure_chat(self):
        if self._chat is None:
            from google.genai import types

            self._chat = self._gemini._client.chats.create(
                model=self._gemini.model,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, temperature=0.2,
                ),
            )
        return self._chat

    @staticmethod
    @retry(
        retry=retry_if_exception(_is_retryable_server_error),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def _send_with_retry(chat, message: str):
        """Same 503-only retry discipline as GeminiClient.generate() -- see
        rag_engine/generate.py's _is_retryable_server_error for why only 503 is
        retried."""
        return chat.send_message(message)

    def ask(
        self, question: str, live_features: LiveFeatureSnapshot | None = None,
        retrieval_query: str | None = None,
    ) -> AgentResponse:
        """retrieval_query defaults to `question` itself -- pass a different string when
        the natural-language question phrasing wouldn't retrieve well (e.g. it refers
        back to a prior turn: "and how does that compare historically?")."""
        docs = self._store.retrieve(retrieval_query or question, k=self._k)
        context_block, tag_map = _build_context_block(live_features, docs)

        message = (
            f"{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer, citing [L#]/[D#] tags for every factual claim as instructed."
        )

        chat = self._ensure_chat()
        response = self._send_with_retry(chat, message)
        answer = response.text

        cited_tags = set(_CITATION_RE.findall(answer))
        sources_used = SourcesUsed(
            live_features=sorted(
                (tag_map[t] for t in cited_tags if t.startswith("L")), key=str,
            ),
            retrieved_docs=sorted(
                (tag_map[t] for t in cited_tags if t.startswith("D")), key=str,
            ),
        )
        return AgentResponse(answer=answer, sources_used=sources_used, sources_offered=tag_map)
