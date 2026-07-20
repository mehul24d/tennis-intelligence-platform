# Tennis Intelligence Platform

A live tennis win-probability engine: point-by-point probability updates, Monte Carlo match
simulation, and "what-if" scenario analysis — built as a full ML system (not just a notebook),
with an eventual IEEE-style technical paper as the capstone writeup.

**Status:** Week 1 — Setup & EDA (in progress)

## Project Goals
- Predict match / set / game / point win probability, updating live as a match progresses
- Simulate counterfactual scenarios ("what if this break point was saved?")
- Ship it as a real system: FastAPI + dashboard + Docker + CI/CD + MLflow
- Document methodology rigorously enough to become an IEEE conference-style paper

See `docs/blueprint.md` for the full 17-section project blueprint (architecture, roadmap,
feature engineering plan, evaluation methodology, timeline).

## Quickstart (local, macOS)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Get the data (see data/README.md for exact commands)

# 4. Launch Jupyter for EDA
jupyter notebook notebooks/
```

## Project Structure

```
configs/        Hydra-style config files (data, model, simulation, deployment)
data/           raw / interim / processed data (gitignored, DVC-tracked later)
src/tennis_intel/  core library code (data, models, simulation, evaluation, serving, utils)
api/            FastAPI service
dashboard/      Streamlit dashboard
pipelines/      orchestration scripts (data -> features -> train -> eval)
notebooks/      exploratory work only — never the source of truth
tests/          unit + integration tests
docs/           MkDocs source + blueprint + technical report drafts
deployment/     Docker / docker-compose / infra
```

## Data Sources (updated 2026-07-02 — see `data/README.md` for full detail)

Jeff Sackmann's original `tennis_atp`/`tennis_wta`/`tennis_slam_pointbypoint` repos were
removed from GitHub shortly before this project started. Verified live replacements in use:

- **ATP match-level:** `Tennismylife/TML-Database`
- **Point-by-point / shot-level (ATP + WTA):** `JeffSackmann/tennis_MatchChartingProject`
  (still live on his account)
- **WTA match-level:** no live source found — **scope for v1 is ATP only**, documented as a
  deliberate decision, WTA extension is a stretch goal if/when a source appears

## Current Milestone: Week 1 — Setup & EDA
- [x] Repo scaffold
- [ ] Clone TML-Database (ATP match-level) and tennis_MatchChartingProject (point-level)
- [ ] Initial data quality report
- [ ] EDA notebook: match-level and shot-by-shot structure