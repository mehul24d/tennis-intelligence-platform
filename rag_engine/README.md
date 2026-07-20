# rag_engine

Retrieval layer over v1's match/player/point dataset, built for v2's LLM tactical
agent. Reuses v1's serving-layer functions (`career_stats_service`,
`point_timeline_service`, etc.) rather than re-deriving stats from raw parquet —
any future fix to those stats propagates here automatically.

## What it does

Turns three kinds of v1 data into embeddable, retrievable documents:

- **Match documents** (`ingest/match_documents.py`) — one per match, sourced from
  `matches_with_elo.parquet` (the full ~198k-match corpus).
- **Player documents** (`ingest/player_documents.py`) — one per player with ≥10
  career matches, sourced via `career_stats_service.get_player_profile`.
- **Point documents** (`ingest/point_documents.py`) — one per notable in-match point
  (live win-probability swing ≥10%), sourced via `point_timeline_service.get_point_timeline`.
  Phrased deliberately swing-neutral: a point's own outcome and its probability
  swing are stated as separate facts, never asserted as cause-and-effect, and an
  automatic hedge fires when the swing direction doesn't match the point winner's
  side — see `tennis-intelligence-platform/docs/known_issue_after_point_swing_includes_next_point_context.md`
  for why (a swing can be dominated by the *next* point's own context, e.g. whether
  it's a second serve, not the traced point's outcome).

Documents are embedded locally (CPU-only, `all-MiniLM-L6-v2` via
`sentence-transformers`) and stored in a Chroma collection persisted under
`data/chroma/` — no external service, rebuildable at any time from source parquet.

## Usage

```bash
# Build the full index (all match/player/point documents)
python -m rag_engine.build_index

# Fast subset, for iteration (caps matches/players scanned; point docs are the
# slowest step — each match requires v1's full 5-engine per-point computation)
python -m rag_engine.build_index --limit 50

# Skip the expensive point-document step (avoids loading v1's full ReplayContext)
python -m rag_engine.build_index --skip-points
```

```python
from rag_engine.index.vector_store import VectorStore

store = VectorStore()
results = store.retrieve("how does Alcaraz perform on clay in deciding sets", k=5)
# optionally filter: store.retrieve(query, k=5, filters={"doc_type": "player_profile"})
```

## Tests

```bash
pytest tests/
```

`tests/test_ingest.py`'s point-document tests load the full v1 `ReplayContext`
(trained classifier + point-level dataset) and are correspondingly slow (~2 min) —
scoped to a small `match_limit` deliberately, since each match requires v1's full
5-engine per-point computation.

## Generation

`generate.py` provides the LLM seam: `GeminiClient` (via `google-genai`, Google's
current unified Gen AI SDK — see the module's own docstring for why not
`google-generativeai`, which is deprecated) plus `build_prompt()`/
`generate_grounded_answer()` to turn `VectorStore.retrieve()` results into a
grounded answer.

```bash
pip install -e ".[generation]"
export GEMINI_API_KEY="your-key-here"   # get one at https://ai.google.dev/gemini-api/docs/api-key
```

```python
from rag_engine.index.vector_store import VectorStore
from rag_engine.generate import RetrievedContext, generate_grounded_answer

store = VectorStore()
results = store.retrieve("how does Alcaraz perform on clay in deciding sets", k=5)
context = [RetrievedContext(r.doc_id, r.text, r.metadata) for r in results]
answer = generate_grounded_answer("How does Alcaraz perform on clay?", context)
```

Retrieval and index-building never import `generate.py` and work fully without any
API key configured; constructing `GeminiClient` with no `GEMINI_API_KEY`/
`GOOGLE_API_KEY` set raises a clear error immediately, not on first use.

## Status

Phase 1 of v2 (see `../PROGRESS.md`). All three document types built, wired into
`build_index.py`, and tested against real data. Generation seam (`generate.py`) also
built, targeting Gemini. Phase 2's `llm_agent/` (multi-turn tool-use agent, not yet
built) will reuse this same client.
