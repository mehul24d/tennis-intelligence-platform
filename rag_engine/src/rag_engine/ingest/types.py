"""types.py — the single document shape every ingestion module produces, and every
index/retrieval module consumes. Keeping this in one place means match_documents.py,
player_documents.py, and (later) point_documents.py all speak the same contract to
vector_store.py, rather than each inventing their own dict shape."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RagDocument:
    """One retrievable unit: a globally-unique id, the embeddable text, and metadata
    used for filtered retrieval (surface, date, player names, doc_type, ...).

    Metadata values must be str/int/float/bool (Chroma's metadata constraint) — no
    None, no nested structures. Ingestion modules are responsible for coercing NaN/None
    fields to a sentinel (e.g. "" or -1) before constructing a RagDocument.
    """

    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)
