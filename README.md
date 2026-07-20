# Tennis Intelligence Platform

An end-to-end tennis analytics system, built in two phases:

- **v1** — a statistical win-probability engine over 198,000+ historical ATP matches
- **v2** — a multimodal extension adding computer vision, retrieval-augmented generation, and an LLM tactical agent

## Start Here

This project's real value is in the bug-finding and validation trail, not just the
code. Read **[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md)** first — it's the actual
narrative: what was built, what broke, how each break was found and fixed, and the
honest, unrounded evaluation numbers behind every headline claim.

If you want to check those claims yourself rather than take them on faith, go to
**[`VERIFICATION.md`](VERIFICATION.md)** next — a checklist mapping every major claim
to the exact test or script that reproduces it, what to expect, and which numbers
depend on data that isn't in this repo (see below).

`PROGRESS.md` is the full chronological build log underneath both of the above —
long, but it's where every investigation is recorded in real time, including the ones
that were wrong before they were right.

## Repo Structure

```
tennis-intelligence/
├─ tennis-intelligence-platform/       # v1 backend (Python) — ML pipeline, Elo ratings,
│                                        Monte Carlo/Markov win-probability engine
├─ tennis-intelligence-platform-web/   # v1 frontend (Next.js/TypeScript)
├─ cv_pipeline/                        # v2 — video → structured features (YOLOv8, ByteTrack,
│                                        MediaPipe, court/homography detection)
├─ rag_engine/                         # v2 — vector retrieval over v1's match/point dataset
├─ llm_agent/                          # v2 — Gemini-based tactical analysis agent
├─ v2_serving/                         # v2 — FastAPI orchestration layer
└─ v2_dashboard/                       # v2 — React dashboard (video overlay, chat, win-probability chart)
```

## v1 — Statistical Win-Probability Engine

An end-to-end machine learning pipeline over 198,000+ professional tennis matches and
7,500+ point-by-point charted matches.

- 100+ leakage-safe temporal features: dynamic Elo ratings, rolling player form,
  surface-specific performance, momentum, opponent-strength metrics, serve/return
  statistics
- Ensemble of gradient-boosted models (XGBoost, LightGBM, CatBoost) benchmarked under
  rolling-origin (walk-forward) validation
- MLflow experiment tracking, SHAP explainability
- A live, in-match win-probability engine combining Monte Carlo simulation and
  Markov-chain modeling, updated point-by-point
- A real, deep bug (`PtWinner` convention) found, fixed, and retrained around — see
  `RESEARCH_REPORT.md` §4.2

**Status**: complete. Backend in `tennis-intelligence-platform/`, frontend in
`tennis-intelligence-platform-web/`.

## v2 — Multimodal Tactical Analysis

Extends v1 with a computer vision + RAG + LLM layer that analyzes match video and
grounds its output in v1's historical dataset.

```
Match video
    → cv_pipeline/    (YOLOv8 detection, ByteTrack tracking, homography,
                        MediaPipe pose — structured per-clip features)
    → rag_engine/      (retrieval over v1's match/player/point data,
                        local Chroma vector store)
    → llm_agent/       (Gemini agent fusing live CV features + retrieved
                        historical context into grounded analysis)
    → v2_serving/      (FastAPI: async video jobs, RAG+LLM query endpoint,
                        win-probability endpoint wrapping v1's engine)
    → v2_dashboard/    (React: upload/poll, result view, video overlay,
                        chat, win-probability panel)
```

**Status**: all five components built and independently tested (293 tests passing
across the whole project as of the most recent full run — see `VERIFICATION.md`).
Several components are deliberately scoped, not fully complete against the "whole
dataset" ideal — e.g. RAG's index covers a documented subset of v1's full corpus, and
CV pipeline's homography validation covers 1 of 10 clips independently. Every such gap
is stated plainly in `RESEARCH_REPORT.md` §5 ("Honest limitations"), not glossed over.

### Design constraints

- Local development on a MacBook M2 (no dedicated GPU) — CV models are
  pre-trained/inference-only, with light YOLO fine-tuning where tested
- Gemini API used for generation (via `google-genai`); local sentence-transformers
  used for embeddings to avoid unnecessary API cost
- v2's live win-probability output wraps v1's existing engine rather than
  reimplementing it — every v2 component that needs a v1 number calls v1's own
  serving functions directly

## Setup

Each component is independently installable. All are Python (`pyproject.toml` +
`.venv`) except `v2_dashboard` and `tennis-intelligence-platform-web` (Node).

```bash
# v1 backend
cd tennis-intelligence-platform
python3 -m venv .venv && .venv/bin/pip install -e .
PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -q   # expect 211 passed

# cv_pipeline (this venv is also used by llm_agent and v2_serving below —
# they're installed editable into it, not separate environments)
cd cv_pipeline
python3 -m venv .venv && .venv/bin/pip install -e .
PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -q   # expect 29 passed (10 skip without data/tennis/*.mp4 — see below)

# rag_engine
cd rag_engine
python3 -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env   # then paste in a real GEMINI_API_KEY
PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -q   # expect 19 passed

# llm_agent (installed editable into cv_pipeline/.venv above)
cd llm_agent && /path/to/cv_pipeline/.venv/bin/pip install -e .
PYTHONPATH=src /path/to/cv_pipeline/.venv/bin/python3 -m pytest tests/test_agent.py -q   # expect 5 passed

# v2_serving (installed editable into cv_pipeline/.venv above)
cd v2_serving && /path/to/cv_pipeline/.venv/bin/pip install -e .
PYTHONPATH=src /path/to/cv_pipeline/.venv/bin/python3 -m pytest tests/ -q   # expect 29 passed

# v2_dashboard
cd v2_dashboard && npm install && npm run dev

# tennis-intelligence-platform-web
cd tennis-intelligence-platform-web && npm install && npm run dev
```

Root-level `.env.example` (copy to `.env`) covers `ROBOFLOW_API_KEY`, used by one of
`cv_pipeline`'s YOLO fine-tuning scripts.

See **[`VERIFICATION.md`](VERIFICATION.md)** for the exact expected output of every
test suite above, and which specific numbers require data not included in this repo.

## What's not included, and how to supply your own

This repo excludes (see `.gitignore`):

- **Raw video clips** (`data/tennis/*.mp4`, `data/*.mp4`, `data/cv_annotated/videos/`)
  — ~1.6GB of source footage used to validate `cv_pipeline`. The ground-truth
  annotation CSVs measured *against* those clips (`data/cv_annotated/annotations/`)
  **are** included, since they're small and are the actual evidence behind this
  project's CV accuracy numbers — they just can't be re-derived from scratch without
  the matching video. To supply your own: any single-camera tennis clip works for the
  pipeline to run against; matching the exact accuracy numbers in `RESEARCH_REPORT.md`
  specifically requires the original 10 clips, which aren't publicly redistributable.
- **v1's raw/processed match datasets** (`tennis-intelligence-platform/data/{raw,processed}/`)
  — built from Jeff Sackmann's `tennis_atp` and the Match Charting Project (both
  public, linked from `tennis-intelligence-platform/docs/`); not redistributed here
  directly, follow those projects' own terms to obtain a copy.
- **The RAG engine's persisted vector index** (`rag_engine/data/chroma/`, 430MB) —
  rebuildable via `rag_engine/src/rag_engine/build_index.py` once you have v1's data
  (a full rebuild takes multiple hours on CPU-only hardware; see `RESEARCH_REPORT.md`
  §3 for why this project scoped it to a documented subset instead).
- **Model checkpoints** (`*.pt`, `*.joblib`, `runs/`) — YOLO checkpoints are the
  standard pretrained `yolov8n`/`yolov8s` weights (auto-downloadable via `ultralytics`)
  plus a fine-tuned variant reproducible via `cv_pipeline/scripts/`; `.joblib` files
  are v1's trained classifiers, retrainable via `tennis-intelligence-platform/pipelines/`.
- **Ad-hoc debug output** (`cv_pipeline/scratch_output/`) — working images from
  iterative investigation, not needed to follow `PROGRESS.md`'s narrative (which
  describes every finding in prose). One specific reference survives this exclusion:
  `reference_video2_calibration.py`'s docstring cites 4 filenames from this directory
  as the basis for a correction — those images aren't included, but the correction
  itself and its reasoning are fully described in the docstring and in `PROGRESS.md`.

## Limitations

Stated plainly in full in `RESEARCH_REPORT.md` §5 — summarized:

- Far-player CV detection/pose is a confirmed hardware/resolution limit, not a bug
- Only 1 of 10 clips' homographies is independently validated for real-world-distance
  metrics
- RAG's index covers a documented subset of v1's full match corpus, not all ~198k
  matches
- Player identity is never resolved from video — connecting a CV job to a specific
  historical match/player requires a human-supplied ID today
- The LLM agent depends on Gemini's API characteristics (model availability, quota,
  transient errors observed and handled during development, not hypothesized)
