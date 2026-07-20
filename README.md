# Tennis Intelligence Platform

An end-to-end tennis analytics system, built in two phases:

- **v1** — a statistical win-probability engine over historical match data
- **v2** — a multimodal extension adding computer vision, RAG, and an LLM tactical agent (in progress)

## Repo Structure

```
tennis-intelligence/
├─ tennis-intelligence-platform/   # v1 backend (Python) — ML pipeline, Elo ratings,
│                                    Monte Carlo/Markov win-probability engine
├─ tennis-intelligence-web/        # v1 frontend (TypeScript)
├─ cv_pipeline/                    # v2 — video → structured features (YOLOv8, ByteTrack,
│                                    MediaPipe, court/homography detection)
├─ rag_engine/                     # v2 — vector retrieval over v1's match/point dataset
├─ llm_agent/                      # v2 — Gemini-based tactical analysis agent
├─ v2_serving/                     # v2 — FastAPI orchestration layer
└─ v2_dashboard/                   # v2 — React dashboard (video overlay, chat, win-probability chart)
```

## v1 — Statistical Win-Probability Engine

Built an end-to-end machine learning pipeline over 198,000+ professional tennis matches and 7,500+ point-by-point charted matches.

- 100+ leakage-safe temporal features: dynamic Elo ratings, rolling player form, surface-specific performance, momentum, opponent-strength metrics, serve/return statistics
- Benchmarked Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and calibrated ensembles using rolling-origin validation
- MLflow experiment tracking, SHAP explainability, bootstrap confidence intervals
- Live win-probability engine combining Monte Carlo simulation and Markov-based probability models
- Modular, reproducible data pipelines suitable for research-grade ML workflows

**Status:** Complete. See `tennis-intelligence-platform/` (backend) and `tennis-intelligence-web/` (frontend).

## v2 — Multimodal Tactical Analysis (In Progress)

Extends v1 with a computer vision + RAG + LLM layer that analyzes live match video and grounds its output in v1's historical dataset.

**Pipeline:**
```
Match video
    → cv_pipeline/    (player + ball detection, tracking, court homography,
                        pose estimation, feature extraction)
    → rag_engine/      (retrieval over v1's 198k-match dataset: player profiles,
                        head-to-head history, surface performance)
    → llm_agent/       (Gemini agent fuses live features + historical context
                        into grounded tactical analysis)
    → v2_serving/      (FastAPI orchestration)
    → v2_dashboard/    (React UI: video overlay, live metrics, historical
                        comparison, chat interface, win-probability chart)
```

**Status:** In progress, built in phases (see build log below).

### Design constraints
- Local development on MacBook M2 (no dedicated GPU) — CV models are pre-trained/inference-only, no local training beyond light fine-tuning
- Gemini API used for generation (via `google-genai`, Google's current unified SDK); local sentence-transformers used for embeddings to avoid unnecessary API cost
- v2's win-probability output is compared against v1's pre-match-only baseline, not a replacement for it

## Build Log / Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Architecture & repo scaffolding | ⬜ |
| 1 | CV pipeline (video → structured features) | ⬜ |
| 2 | RAG knowledge base (reusing v1 dataset) | ⬜ |
| 3 | LLM tactical agent | ⬜ |
| 4 | API + orchestration | ⬜ |
| 5 | Dashboard | ⬜ |
| 6 | Evaluation & research write-up | ⬜ |

*(Update this table as phases complete — also useful as interview talking points later.)*

## Limitations

- CV-derived stats (serve speed, shot classification) are estimates based on 2D video projection, not radar/sensor-accurate — documented as a known limitation, not presented as ground truth
- RAG retrieval quality is bounded by v1's dataset coverage (some lesser-known players/matchups will have sparse historical context)
- No GPU locally — model choices favor pre-trained inference over custom training; noted in the eval section what would change with GPU access