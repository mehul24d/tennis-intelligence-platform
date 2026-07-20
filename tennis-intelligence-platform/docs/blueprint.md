# Tennis Intelligence Platform — Complete Project Blueprint

*A live win-probability engine, simulation platform, and analytics product built for ML/AI internship and MS-in-AI portfolios.*

---

## Section 1 — Why This Project Is Valuable

### Against typical portfolio projects
Most ML portfolios stop at "train a model, report accuracy, done." A tabular classification project — even a good one — reads as a course assignment. You've already built that (the World Cup project). What separates a candidate who gets shortlisted from one who doesn't is usually **system thinking**: can you take a prediction and turn it into something that runs, updates, and is used?

This project is different on three axes:

1. **Temporal/sequential modeling** — win probability isn't a single prediction, it's a *process* that evolves point by point, game by game, set by set. This forces you into sequence models, state representations, and calibration under distribution shift (a break of serve changes everything instantly).
2. **A live system, not a notebook** — an API serving probabilities in near-real-time, a dashboard consuming it, a simulation engine behind it. This is what "production ML" actually looks like at a sports analytics company.
3. **Simulation + causal "what-if" reasoning** — Monte Carlo methods and counterfactual scenario analysis are a distinct skill set from supervised learning, and most students never touch them.

### New ML concepts this teaches you
- Deep learning for sequences (PyTorch, LSTMs, Transformers) applied to *state transitions* rather than static feature vectors
- Probabilistic modeling and calibration (a win probability is only useful if it's *well-calibrated*, not just discriminative)
- Monte Carlo simulation and stochastic processes
- Real-time inference serving (FastAPI, latency-aware design)
- Full MLOps loop: experiment tracking, data versioning, CI/CD, containerized deployment, monitoring

### Why recruiters like it
Recruiters skim GitHub profiles in under two minutes. A live demo (deployed dashboard + API) with a clear architecture diagram signals "this person ships things," not just "this person can fit a model." Sports analytics is also a recognizable, explainable domain — a hiring manager doesn't need domain expertise to understand what your project does within 30 seconds of looking at it, which is rare and valuable.

### Why European MS admissions committees like it
Admissions committees (ETH, TU Delft, EPFL, Edinburgh, UCL, etc.) evaluate research potential, not just engineering polish. This project demonstrates:
- Ability to formulate an ambiguous real-world problem as a rigorous ML problem (defining win probability, handling non-stationarity, avoiding leakage)
- Statistical maturity (calibration, confidence intervals, significance testing) — often the actual differentiator in an SOP or portfolio review
- Independent, self-directed scoping of a multi-month research-engineering project, which mirrors what a thesis looks like

---

## Section 2 — Prerequisites

| Area | Required Level | Recommended Resources | Learn During Project? |
|---|---|---|---|
| Python | Intermediate–advanced (OOP, packaging, typing) | *Fluent Python* (Ramalho); Real Python | No — should already have this |
| Statistics | Intermediate (distributions, hypothesis testing, MLE) | *Statistical Rethinking* (McElreath, ch. 1–6); Khan Academy stats | Partially |
| Probability | Intermediate (Markov chains, conditional probability) | MIT 6.041 (OCW); *Introduction to Probability* (Blitzstein) | Yes, Markov chains during Milestone 6 |
| Machine Learning | Advanced (you already have this from Project 1) | — | No |
| Deep Learning / PyTorch | Beginner–intermediate | PyTorch official 60-min blitz; *Deep Learning with PyTorch* (Stevens et al.) | Yes — core learning goal |
| Linear Algebra | Intermediate (matrix ops, eigenvalues for embeddings) | 3Blue1Brown "Essence of Linear Algebra"; Strang MIT 18.06 | Refresh, not deep-dive |
| Time Series | Beginner | *Forecasting: Principles and Practice* (Hyndman, free online) | Yes |
| Git | Intermediate (branching, PRs, rebasing) | Pro Git book (free) | No |
| Docker | Beginner | Docker official docs + "Docker for Beginners" | Yes |
| FastAPI | Beginner | Official FastAPI tutorial (excellent, do the whole thing) | Yes |
| SQL | Intermediate | Mode SQL tutorial; PostgreSQL docs | Refresh |
| Linux | Beginner–intermediate (shell, cron, systemd basics) | *The Linux Command Line* (free book) | Yes, as needed |
| Cloud (AWS/GCP) | Beginner | AWS Free Tier + "AWS for ML" crash course | Yes |
| MLOps | Beginner | *Designing Machine Learning Systems* (Chip Huyen) | Yes — core learning goal |
| CI/CD | Beginner | GitHub Actions official docs | Yes |
| Explainability | Intermediate (you have SHAP already) | Extend to attention visualization | Yes, for Transformer stage |
| Data Engineering | Beginner | dbt fundamentals (optional); basic ETL patterns | Yes, lightly |
| Visualization | Intermediate | Plotly/Dash docs; *Storytelling with Data* (Knaflic) | Refresh |

**Bottom line:** you have the ML and Python foundation already. The genuinely new material is PyTorch sequence modeling, MLOps tooling, and deployment — budget the most learning time there.

---

## Section 3 — Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem fit, your existing fluency |
| Classical ML | scikit-learn, XGBoost, LightGBM, CatBoost | Strong baselines, fast iteration, you already know these |
| Deep Learning | PyTorch | Industry-standard, more transparent than Keras for research-style work, better for custom sequence architectures |
| Probabilistic modeling | `hmmlearn`, `pomegranate`, or hand-rolled Markov chain for point-transition modeling; optionally PyMC for Bayesian components | Point-by-point tennis dynamics map naturally onto Markov/HMM structure |
| Experiment tracking | MLflow (you already use it — stay consistent) | Reduces new tool overhead, lets you compare this project against Project 1 methodology |
| Data versioning | DVC | Pairs with MLflow, git-native, industry standard |
| Config management | Hydra | Clean way to manage the many experiment configs (features, model variants, simulation params) |
| API | FastAPI | Async support (important for streaming point updates), automatic OpenAPI docs, fastest way to look "production-grade" |
| Dashboard | Streamlit for MVP → migrate high-value pages to a Plotly Dash or React+FastAPI frontend if time allows | Streamlit gets you a working demo fast; a stretch goal is a more custom frontend for portfolio polish |
| Database | PostgreSQL (via SQLite for local dev) | Standard relational store for match/point event data |
| Containerization | Docker + docker-compose | Standard, and required for most "production ML" evaluation criteria |
| CI/CD | GitHub Actions | Free, integrates directly with your portfolio repo |
| Testing | pytest, pytest-cov | Standard |
| Formatting/Linting | black, ruff, isort, pre-commit | Enforces the "professional engineer" signal |
| Cloud | AWS (EC2/Lightsail for API, S3 for data/artifacts) or Render/Fly.io for a cheaper always-on deploy | AWS looks better on a resume; Render/Fly.io is cheaper and easier if budget-constrained |
| Monitoring | Prometheus + Grafana (stretch) or simple structured logging + a `/health` and `/metrics` endpoint (MVP) | Full observability stack is a stretch goal, not a blocker |
| Logging | `structlog` or Python `logging` with JSON formatting | Production-style structured logs |
| Documentation | MkDocs Material | Clean docs site, deployable to GitHub Pages, looks professional |

---

## Section 4 — Data Sources

| Source | Advantages | Disadvantages | Difficulty | License | Update Frequency | Recommended Usage |
|---|---|---|---|---|---|---|
| **Jeff Sackmann (tennis_atp / tennis_wta / tennis_slam_pointbypoint GitHub repos)** | Free, structured, includes point-by-point data for many Slam matches since ~2011; widely used and trusted in tennis analytics | Point-by-point coverage limited mostly to Grand Slams; some inconsistency in older years | Medium | CC BY-NC-SA (check current repo license) | Updated periodically, community-maintained | **Primary data source** for both match-level and point-level modeling |
| **Tennis Abstract (Jeff Sackmann's site)** | Rich match stats, serve/return breakdowns, some pre-built ratings (Elo) | Not always in clean tabular form; some scraping required | Medium–High | Check site terms before scraping | Irregular | Secondary source; useful for validating your own Elo implementation |
| **ATP/WTA official sites** | Authoritative rankings, tournament data | No public API, scraping is fragile and against ToS in places | High | Restrictive | Live | Avoid direct scraping; use only for occasional manual cross-checks |
| **Hawk-Eye data** | Extremely rich (ball trajectory, shot-level) | Not publicly available outside partnerships | N/A | Proprietary | N/A | Not accessible — mention in your README as "future work if partnered access obtained" |
| **IBM SlamTracker** | Sets the bar for the *type* of product you're building | Not a data source, it's a product | N/A | Proprietary | N/A | Use as design inspiration only |
| **Kaggle tennis datasets** | Quick to start, pre-cleaned | Often derived from Sackmann's data anyway, less credible for a serious project | Low | Varies | Static | Use only for quick prototyping, not final results |
| **GitHub community repos (e.g., point-by-point parsers)** | Save you data-engineering time | Quality varies, verify before trusting | Medium | Varies | Static | Use as reference implementations, not blind imports |

**Recommendation:** Build your core dataset from **Sackmann's `tennis_slam_pointbypoint` repo** for the sequential point-by-point modeling, and **`tennis_atp`/`tennis_wta`** for match-level features (rankings, surface, head-to-head). This mirrors what real tennis analytics teams actually use, since Hawk-Eye/SlamTracker data isn't publicly accessible.

---

## Section 5 — Project Architecture

```
tennis-intelligence-platform/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── ci.yml                # lint, test, type-check on PR
│       └── cd.yml                # build + push docker image on merge to main
│
├── configs/                      # Hydra configs
│   ├── data/
│   ├── model/
│   ├── simulation/
│   └── deployment/
│
├── data/
│   ├── raw/                      # untouched source data (DVC-tracked)
│   ├── interim/
│   ├── processed/
│   └── dvc.yaml
│
├── src/
│   └── tennis_intel/
│       ├── data/                 # ingestion, cleaning, feature engineering
│       ├── models/               # baseline, GBM, LSTM, Transformer, HMM
│       ├── simulation/           # Monte Carlo engine
│       ├── evaluation/           # calibration, backtesting, metrics
│       ├── serving/              # inference logic shared by API
│       └── utils/
│
├── api/
│   ├── main.py                   # FastAPI app
│   ├── routers/
│   ├── schemas/                  # Pydantic models
│   └── Dockerfile
│
├── dashboard/
│   ├── app.py                    # Streamlit entrypoint
│   └── pages/
│
├── pipelines/                    # orchestration scripts (data → features → train → eval)
│
├── notebooks/                    # exploratory only, never source of truth
│
├── mlruns/                       # MLflow tracking store (gitignored, or remote)
│
├── models/                       # serialized model artifacts (DVC or MLflow registry)
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── docs/                         # MkDocs source
│
├── deployment/
│   ├── docker-compose.yml
│   ├── terraform/                # optional, stretch goal for cloud infra-as-code
│   └── k8s/                      # optional, stretch goal
│
└── utilities/
    └── scripts/                  # one-off maintenance / data refresh scripts
```

---

## Section 6 — Machine Learning Roadmap (Incremental Milestones)

Jumping straight to Transformers on a problem you haven't explored yet is how projects stall for weeks. Each milestone below exists to de-risk the next one.

- **Milestone 1 — EDA & Data Understanding.** Understand point-by-point data structure, missingness patterns, surface/tournament distributions. *Why:* you cannot engineer good features or trust a model on data you don't understand.
- **Milestone 2 — Baseline Logistic Regression** on match-level features (ranking diff, surface, H2H). *Why:* establishes a trustworthy, interpretable floor. If your fancy models can't beat this by a meaningful margin, something's wrong.
- **Milestone 3 — Gradient Boosting (XGBoost/LightGBM)** on the same match-level features. *Why:* you already know this toolchain; it's a fast way to validate feature engineering before investing in deep learning.
- **Milestone 4 — Feature Engineering Expansion** (rolling form, Elo, surface Elo, fatigue, momentum). *Why:* almost all of the predictive lift in sports models comes from features, not model complexity — prove this to yourself before adding neural complexity.
- **Milestone 5 — Calibration.** Apply Platt scaling / isotonic regression, check reliability diagrams. *Why:* a win-probability product is worthless if 70% predictions don't win 70% of the time. This must be solid *before* deep learning, or you won't be able to tell if a fancier model is actually better.
- **Milestone 6 — Sequential Modeling: Markov Chain / HMM for point-to-game-to-set transitions.** *Why:* tennis scoring is a well-defined stochastic process; this is the natural bridge between static features and deep sequence models, and gives you an interpretable, theoretically-grounded baseline for "live" probability.
- **Milestone 7 — LSTM on point sequences** (serve outcome, score state, momentum as inputs). *Why:* introduces PyTorch and sequence modeling on a problem where you already have an HMM baseline to sanity-check against.
- **Milestone 8 — Transformer (small, e.g., a lightweight self-attention encoder over point sequences).** *Why:* only attempt this once the LSTM works and you understand *why* it works — Transformers are much harder to debug blind.
- **Milestone 9 — Simulation Engine (Monte Carlo).** *Why:* turns your point-level model into match-level outcome distributions and "what-if" scenarios — this is what makes the project feel like a product, not a model.
- **Milestone 10 — Deployment (API + Dashboard + Docker + CI/CD).** *Why:* final step — nothing here should block earlier modeling milestones, but nothing is a finished "product" without it.

---

## Section 7 — Feature Engineering Plan

| Feature | Description | Leakage Risk | Mitigation |
|---|---|---|---|
| Player Elo | Standard Elo updated match-by-match | High if computed using future matches | Compute strictly sequentially, walk-forward only |
| Surface Elo | Separate Elo per surface (hard/clay/grass) | Same as above | Same |
| Serve % (1st/2nd in, win %) | Rolling average over trailing N matches | High if using current-match stats to predict current match | Use only *pre-match* rolling stats, never in-match stats as pre-match features |
| Return % | Mirror of serve % | Same | Same |
| Break point conversion % | Rolling average | Same | Same |
| Recent form | Win % over last N matches | Medium | Trailing window only, exclude current match |
| Fatigue | Days since last match, sets played in last N days | Low | Straightforward if using match dates correctly |
| Travel | Distance/timezone shift between tournaments (approx. via tournament location) | Low | Static geographic lookup |
| Rest | Days between matches within a tournament | Low | — |
| Head-to-head | Win/loss record vs. this specific opponent | Medium (sparse for new matchups) | Use a smoothed prior (e.g., Bayesian shrinkage toward overall win rate) for low-count H2H |
| Surface history | Career win % on this surface | Low | — |
| Tournament importance | Categorical (Slam, Masters, ATP250, etc.) | None | Static metadata |
| Weather | Temperature/wind if available (affects ball speed) | Low, but data availability is limited | Optional — only include if reliably sourced |
| Momentum | Recent point-win streak within the match | **Very high** — this is in-match, so only valid for the *live* in-match model, not pre-match | Clearly separate pre-match model features from in-match/live model features |
| Previous set/game score state | Current match state | In-match only | Same separation as above |
| Pressure situations | Break point / set point / match point flags | In-match only | Same |
| Ranking | ATP/WTA ranking and points | Low, use ranking as of match date, not current | Time-index carefully |
| Age | Player age at match date | None | — |
| Handedness | Left/right-handed matchup effects | None | — |
| Court speed | Estimated surface speed index | Low | Static per-tournament lookup |
| Rolling statistics | Any stat computed over a trailing window | High if window includes current match | Strict walk-forward computation, unit-test this explicitly |
| Interaction features | e.g., Elo diff × surface, fatigue × ranking | Inherits risk of components | — |
| Sequence features (in-match) | Point outcome sequence, serve sequence | N/A — this *is* the live model's input | Only used in the live/point-level model, never in pre-match model |

**Temporal validation:** All models must be validated with **strict walk-forward / rolling-origin splits** (train on data before date X, test on matches after date X), exactly as you did in the World Cup project with 2014/2018/2022. For the live point-level model, hold out entire *matches* (not points) to avoid leaking within-match information across train/test.

---

## Section 8 — Model Development

| Model | Advantages | Disadvantages | Difficulty | Training Time | Interpretability |
|---|---|---|---|---|---|
| Logistic Regression | Fast, interpretable, strong calibration baseline | Limited capacity for interactions | Low | Seconds | Very high |
| Random Forest | Handles nonlinearity, minimal tuning | Weaker calibration out-of-box | Low | Minutes | Medium |
| XGBoost/LightGBM/CatBoost | Best-in-class for tabular, you already have expertise | Still needs calibration | Low (for you) | Minutes | Medium (SHAP helps) |
| Feedforward Neural Net | Bridge to PyTorch before sequence models | No sequence awareness | Low–Medium | Minutes | Low |
| LSTM | Naturally models point sequences and momentum | Harder to train, needs more data, slower | Medium–High | Hours | Low (needs SHAP-for-sequences or attention proxies) |
| Transformer (lightweight) | State-of-the-art sequence modeling, attention is semi-interpretable | Data-hungry, easy to overfit at small scale, real risk of being "cargo cult" complexity | High | Hours | Medium (attention weights) |
| Temporal Convolutional Network | Often trains faster and more stably than LSTM/Transformer for medium-length sequences | Less "buzzword" recognition than Transformer | Medium | Moderate | Low–Medium |
| Hidden Markov Model | Theoretically grounded for point-to-game transitions, highly interpretable, fast | Assumes Markov property (may be a simplification) | Medium | Fast | Very high |

**Recommendation:** Present the HMM as your interpretable, theory-grounded model and the LSTM/Transformer as your high-capacity models — then explicitly compare them on calibration and log-loss, not just accuracy. That comparison narrative ("does more complexity actually help here, and by how much?") is exactly the kind of critical thinking admissions committees and interviewers want to see.

---

## Section 9 — Evaluation

- **Rolling / walk-forward validation** — train on past, test on strictly future matches, exactly as in Project 1.
- **Tournament-based validation** — hold out entire tournaments to test generalization beyond just time (e.g., train on hard/clay, test surface generalization on a held-out grass season).
- **Cross-validation** — only for hyperparameter tuning *within* a training window, never across the temporal boundary.
- **Calibration** — reliability diagrams (predicted probability bucket vs. observed win rate) are your single most important non-accuracy chart for this project.
- **Log loss** — primary metric; punishes overconfident wrong predictions, which matters enormously for a "trustworthy probability" product.
- **Brier score** — secondary calibration-sensitive metric, easy to explain to non-technical readers.
- **ROC / precision / recall** — useful but secondary; win-probability is fundamentally a *probability estimation* problem, not a classification-threshold problem.
- **Confidence intervals via bootstrapping** — reuse your Project 1 methodology, apply it to log-loss and Brier score across resampled match sets.
- **Statistical significance testing** — when comparing model A vs. B, use a paired test (e.g., paired bootstrap or Diebold-Mariano style test for forecast comparison) rather than eyeballing metric differences.

**Why accuracy alone is insufficient:** In tennis, favorites win roughly 65-70% of matches, so a model that always predicts "higher-ranked player wins" gets decent accuracy while being useless as a *probability* product. The entire value of this project is in well-calibrated, well-differentiated probabilities — accuracy tells you almost nothing about that.

---

## Section 10 — Deployment

- **FastAPI service** exposing endpoints such as `/predict/match`, `/predict/live` (accepts current match state, returns updated probability), `/simulate`, `/players/{id}/stats`.
- **Docker** — single Dockerfile for the API, a separate one for the dashboard, orchestrated via `docker-compose` for local dev/demo.
- **GitHub Actions** — CI runs lint/tests/type-checks on every PR; CD builds and pushes the Docker image (and optionally deploys) on merge to `main`.
- **Streamlit dashboard** — consumes the FastAPI service over HTTP, never imports model code directly (keeps a clean service boundary, which is a strong engineering signal).
- **Cloud deployment** — start with a single small VM (AWS Lightsail, or Render/Fly.io for lower cost) running docker-compose; treat Kubernetes as a stretch goal only, not a requirement — over-engineering the infra is a common trap that eats weeks for little portfolio benefit.
- **MLflow** — track every experiment; promote the best model to a "Production" alias in the MLflow Model Registry, and have the API load *from the registry*, not a hardcoded file path — this is the detail that makes it look like real MLOps rather than a demo hack.
- **Monitoring/Logging** — structured JSON logs from the API (latency, request volume, prediction distribution) at minimum; Prometheus/Grafana dashboards as a stretch goal.

---

## Section 11 — Dashboard Design

- **Live Win Probability page** — real-time (or replayed historical) point-by-point probability chart for a selected match, updating as points are simulated/replayed.
- **Momentum graph** — rolling point-win rate over the last N points for each player, overlaid on the probability timeline, so users can visually connect "momentum swings" to probability shifts.
- **Probability timeline** — full match history: set boundaries, break points, key moments annotated on the probability curve.
- **Serve statistics panel** — 1st/2nd serve %, ace rate, break points saved/converted, shown per set.
- **Point-by-point simulation** — a "replay" control that steps through a historical match point-by-point, showing how the model's probability evolves alongside the real outcome.
- **What-if scenarios** — interactive sliders/toggles: "what if this break point was saved?", "what if first-serve % was +5%?" — re-runs the simulation engine live and shows the counterfactual probability curve next to the real one.
- **Player comparison page** — side-by-side Elo history, surface performance, serve/return profile radar chart.
- **Tournament dashboard** — bracket view with live/simulated win probabilities per matchup, aggregated into a "who's most likely to win the tournament" view (this requires chaining your match model through the whole draw via simulation).
- **Historical match explorer** — searchable table of past matches with model predictions vs. actual outcomes, filterable by surface, tournament, and player, so users (and interviewers) can audit model performance themselves.

---

## Section 12 — Simulation Engine (Monte Carlo)

Design a simulation engine that samples point outcomes from your calibrated point-level model (HMM or LSTM output) and rolls them forward through tennis's actual scoring rules (game → set → match, including tiebreak logic).

- **Full match simulation** — simulate N (e.g., 10,000) complete matches given starting serve/return probabilities, producing a distribution over match winner, score lines, and duration.
- **Alternative point outcomes** — resample a specific point (e.g., a break point) under a different assumed outcome and re-simulate the rest of the match to quantify its downstream impact — this is your core "what-if" feature.
- **Alternative serve percentages** — perturb a player's serve-win probability by ±X% and re-run simulations to show sensitivity (useful for commentary-style insights: "a 5% improvement in first-serve points won would raise Player A's win probability from 62% to 71%").
- **Alternative break point conversion** — same sensitivity-analysis pattern applied to return-side performance.
- **Outputs to estimate:**
  - Win probability (with bootstrap confidence interval across simulation runs)
  - Expected match duration (games/sets, and optionally wall-clock time using historical point-duration data)
  - Expected number of games/sets
  - Full outcome distribution (not just point estimates) — this is what separates "simulation" from "just another prediction"

This is the component most likely to genuinely impress a technical interviewer, because Monte Carlo simulation engines are rare in student portfolios and demonstrate a different kind of thinking than supervised learning alone.

---

## Section 13 — MLOps

- **MLflow** — experiment tracking (params, metrics, artifacts) and model registry, consistent with Project 1.
- **DVC** — version raw/processed datasets and large model artifacts alongside git, so any commit can reproduce its exact data state.
- **Hydra** — manage the (large) space of configs: which features, which model, which simulation parameters, which deployment target.
- **GitHub Actions** — CI (lint, type-check, unit + integration tests) on every PR; CD (build/push Docker image, optionally deploy) on merge to main.
- **Docker** — reproducible environments for API, dashboard, and (optionally) training.
- **Unit tests** — feature engineering functions (especially leakage-prone ones like rolling stats and Elo), simulation engine correctness (e.g., does it correctly implement tennis scoring rules — write explicit tests for tiebreaks, deuce, etc.).
- **Integration tests** — end-to-end test that hits the FastAPI endpoints and checks response shape/sanity (e.g., probabilities sum to 1, are within [0,1]).
- **Linting/Formatting** — `ruff` + `black` + `isort`, enforced via pre-commit hooks and CI.
- **Pre-commit hooks** — run formatting/linting/basic tests before every commit.
- **Environment management** — `pyproject.toml` with `uv` or `poetry` for reproducible dependency resolution.
- **Logging** — structured logs from both training pipelines and the live API.
- **Monitoring** — at minimum, log prediction distributions over time to catch drift; Prometheus/Grafana as a stretch goal.
- **Model registry** — promote models through `Staging` → `Production` stages in MLflow, with the API always loading the current `Production` alias.

---

## Section 14 — Timeline (16 Weeks, 20–25 hrs/week)

> Assumes start date = today. Adjust week numbers to actual calendar dates as you begin.

**Week 1 — Setup & EDA**
Objectives: repo scaffolding, environment setup, download and explore Sackmann datasets.
Deliverables: working repo skeleton, EDA notebook, data quality report.
Commits: ~8–10. Reading: Sackmann repo docs, *Designing ML Systems* ch. 1–2. Blockers: messy/inconsistent historical data — budget extra cleaning time. Success criteria: can load and describe both match-level and point-by-point datasets cleanly.

**Week 2 — Data Pipeline & Leakage-Safe Feature Engineering (Part 1)**
Objectives: build the walk-forward-safe feature pipeline (Elo, surface Elo, rolling form).
Deliverables: `src/tennis_intel/data/` pipeline, unit tests for leakage safety.
Commits: ~8. Reading: leakage sections of *Designing ML Systems*. Success criteria: unit tests prove no future data leaks into any feature.

**Week 3 — Feature Engineering (Part 2) & Baselines**
Objectives: finish feature set (fatigue, H2H, tournament importance), train Logistic Regression + XGBoost baselines.
Deliverables: MLflow-tracked baseline experiments, first calibration plots.
Commits: ~8. Success criteria: baseline log-loss beats "always predict favorite" heuristic.

**Week 4 — Calibration & Rigorous Evaluation Framework**
Objectives: implement reliability diagrams, bootstrap CIs, walk-forward + tournament-holdout evaluation harness.
Deliverables: `src/tennis_intel/evaluation/` module, reusable across all future models.
Commits: ~6. Reading: calibration chapter of any applied ML text; Niculescu-Mizil & Caruana (2005) on predicting good probabilities. Success criteria: evaluation harness works identically for any new model dropped in.

**Week 5 — Markov Chain / HMM for Point Sequences**
Objectives: implement tennis scoring as a Markov process; fit point-transition probabilities; validate against known analytical win-probability formulas (e.g., classic serve-win-probability-to-game-win-probability formulas from tennis analytics literature).
Deliverables: HMM module + validation notebook comparing to closed-form results.
Commits: ~8. Reading: Klaassen & Magnus tennis probability papers (classic in this space). Blockers: getting the state space right (score states, tiebreaks). Success criteria: HMM-derived game-win-probability matches known closed-form formulas within tolerance.

**Week 6 — PyTorch Fundamentals**
Objectives: complete PyTorch tutorials, reimplement a simple feedforward net on your existing tabular features as a bridge exercise.
Deliverables: `models/ffn_baseline.py`, MLflow-tracked run.
Commits: ~6. Reading: PyTorch 60-min blitz, *Deep Learning with PyTorch* ch. 1–4. Success criteria: comfortable writing training loops, dataloaders, and debugging tensor shapes without hand-holding.

**Week 7 — LSTM on Point Sequences (Part 1)**
Objectives: build point-sequence dataset/dataloader, design LSTM architecture for live win probability.
Deliverables: working training loop, first (likely mediocre) results.
Commits: ~8. Blockers: variable-length sequences, padding/masking. Success criteria: LSTM trains without errors and produces *some* signal above random.

**Week 8 — LSTM (Part 2): Tuning & Calibration**
Objectives: tune architecture/hyperparameters, apply calibration, compare against HMM baseline on the same evaluation harness.
Deliverables: calibrated LSTM, comparison report (HMM vs. LSTM) with significance testing.
Commits: ~6. Success criteria: clear, honest comparison — if LSTM doesn't beat HMM, that's a valid and interesting finding, document it.

**Week 9 — Transformer for Point Sequences**
Objectives: implement a lightweight self-attention encoder over point sequences.
Deliverables: Transformer model + attention visualization notebook.
Commits: ~8. Reading: "Attention Is All You Need" (skim, focus on intuition not full derivation); Illustrated Transformer (Jay Alammar blog). Blockers: overfitting on limited data — use heavy regularization/small model size. Success criteria: Transformer trains stably; attention weights are visualizable and at least plausible.

**Week 10 — Model Comparison, Final Model Selection, SHAP/Attention Explainability**
Objectives: consolidate all models into one comparison report; select production model(s) — likely HMM for live simplicity + LSTM/Transformer for a "research" comparison.
Deliverables: `docs/model_comparison.md`, SHAP report for tabular models, attention visualizations for sequence models.
Commits: ~6. Success criteria: you can explain, in one paragraph each, why each model does or doesn't outperform the others.

**Week 11 — Monte Carlo Simulation Engine (Part 1)**
Objectives: implement core match simulation (game/set/match rollout from point-level probabilities).
Deliverables: `src/tennis_intel/simulation/engine.py` + unit tests for scoring rule correctness.
Commits: ~8. Blockers: tiebreak and deuce edge cases — test exhaustively. Success criteria: simulated match outcome distributions match real-world base rates on held-out matches.

**Week 12 — Simulation Engine (Part 2): What-If Scenarios**
Objectives: add scenario perturbation (alternative break points, serve %) and sensitivity analysis outputs.
Deliverables: what-if API functions + notebook demoing 3–4 concrete scenarios.
Commits: ~6. Success criteria: can answer "what if Player A saved that break point?" with a re-simulated probability curve.

**Week 13 — FastAPI Service**
Objectives: build API endpoints (`/predict/match`, `/predict/live`, `/simulate`, `/players/{id}`), Pydantic schemas, load model from MLflow registry.
Deliverables: working local API with OpenAPI docs.
Commits: ~8. Reading: FastAPI official tutorial (full). Success criteria: can hit every endpoint locally and get sane, schema-validated responses.

**Week 14 — Dashboard**
Objectives: build Streamlit dashboard consuming the API — live probability page, momentum graph, what-if sliders, player comparison.
Deliverables: working local dashboard.
Commits: ~10. Success criteria: a non-technical person can open the dashboard and understand what's happening within 60 seconds.

**Week 15 — Dockerization, CI/CD, Cloud Deployment**
Objectives: Dockerfiles for API and dashboard, docker-compose, GitHub Actions CI/CD, deploy to a small cloud VM or Render/Fly.io.
Deliverables: publicly accessible live demo URL.
Commits: ~8. Blockers: environment/dependency mismatches between local and cloud — budget a full day for this alone. Success criteria: a stranger can visit a URL and use the dashboard with zero setup.

**Week 16 — Documentation, Portfolio Packaging, Polish**
Objectives: write README, architecture diagram, technical report, demo GIF, resume bullets, LinkedIn post (see Section 17).
Deliverables: fully polished public repo + live demo + written technical report.
Commits: ~6. Success criteria: the repo passes a "would I be proud to link this in an application today" test.

*(This is a template — real blockers will shift weeks around. Treat Weeks 15–16 as flexible buffer if earlier deep learning weeks run long, which they often do.)*

---

## Section 15 — Learning Resources

**Books**
- *Deep Learning with PyTorch* (Stevens, Antiga, Viehmann) — the most practical PyTorch reference available, worth reading cover to cover during Weeks 6–9.
- *Designing Machine Learning Systems* (Chip Huyen) — the best single book on the MLOps/production mindset this project requires.
- *Forecasting: Principles and Practice* (Hyndman & Athanasopoulos, free online) — time series fundamentals.
- *Statistical Rethinking* (McElreath) — for Bayesian/probabilistic intuition if you go deeper into the HMM/Bayesian stretch goals.

**Research papers**
- Klaassen & Magnus, "Are Points in Tennis Independent and Identically Distributed?" — foundational tennis probability paper, directly relevant to your HMM.
- Vaswani et al., "Attention Is All You Need" — skim for intuition before Week 9.
- Niculescu-Mizil & Caruana, "Predicting Good Probabilities With Supervised Learning" — directly informs your calibration work.

**GitHub repositories**
- Jeff Sackmann's `tennis_atp`, `tennis_wta`, `tennis_slam_pointbypoint` — your primary data source; also study his own analysis scripts for domain-modeling ideas.
- Any well-starred "win probability model" repos from other sports (NBA/NFL win probability projects) — the *methodology* (not code) transfers directly.

**YouTube / video**
- 3Blue1Brown — "Essence of Linear Algebra" and any videos on attention/transformers, for visual intuition.
- StatQuest (Josh Starmer) — HMMs, calibration, and general stats refreshers explained clearly.
- Jay Alammar's "Illustrated Transformer" (blog + associated talks) — the clearest transformer intuition resource available.

**University lectures**
- MIT 6.041 (Probability) — OCW, free.
- Stanford CS224N or CS230 lecture recordings — for deep learning/sequence modeling depth if you want it.

**Courses**
- FastAPI official tutorial — do the entire thing, it doubles as documentation and course.
- Full Stack Deep Learning (free course materials) — excellent for the deployment/MLOps mindset.

**Communities**
- r/tennis analytics threads, Tennis Abstract's blog/comments section — real practitioner discussion of exactly these problems.
- MLOps Community (Slack/Discord) — useful when you hit deployment blockers.

---

## Section 16 — Stretch Goals

- **Player embeddings** — learn dense vector representations of players (from match/point history) instead of hand-engineered Elo, then visualize via t-SNE/UMAP to see if playing-style clusters emerge.
- **Self-supervised / contrastive learning** — pretrain a point-sequence encoder on a "predict the next point outcome" objective before fine-tuning on win probability, mirroring modern pretraining paradigms.
- **Bayesian deep learning / uncertainty estimation** — use MC Dropout or a Bayesian last layer so the model reports *uncertainty* on its probability estimates, not just a point estimate — genuinely differentiating for a portfolio.
- **Graph Neural Networks** — model the tournament draw as a graph and propagate win probabilities through bracket structure.
- **Reinforcement learning** — frame in-match strategy (e.g., serve placement decisions) as a sequential decision problem; only pursue if you have a genuinely well-justified formulation, since RL is easy to bolt on badly.
- **LLM-generated match reports** — feed simulation/prediction outputs into an LLM to auto-generate natural-language match summaries or "what-if" narratives for the dashboard.
- **Multimodal models** — if any video/ball-tracking public data becomes available, fuse it with the point-sequence model (explicitly frame this as future work if data access is the blocker).
- **Computer vision** — court/ball tracking from broadcast video, only as a far-future extension.
- **Real-time streaming inference** — wire the live model to an actual live-scoring feed (if a legal/public one exists) rather than replayed historical data.
- **Full cloud infra-as-code** — Terraform + Kubernetes deployment, for candidates specifically targeting infra-heavy MLE roles.

---

## Section 17 — Final Portfolio Packaging

**README structure**
1. One-paragraph hook: what the project does and a link to the live demo, right at the top.
2. Architecture diagram (see below).
3. Key results table (model comparison: log-loss, Brier score, calibration, by model type).
4. Quickstart (`docker-compose up`, or link to hosted demo).
5. Project structure overview.
6. Link to full technical report in `docs/`.

**Architecture diagram**
A single clean diagram (draw.io, Excalidraw, or Mermaid in the README) showing: data sources → feature pipeline → MLflow-tracked training → model registry → FastAPI → Streamlit dashboard, plus the CI/CD loop. This single image does more for a recruiter's first impression than any amount of text.

**Screenshots**
3–5 screenshots: the live probability chart, the what-if scenario panel, the tournament bracket dashboard, a calibration reliability diagram.

**Demo GIF**
A 10–15 second GIF of the what-if slider changing the probability curve live — this is the single most compelling asset for a LinkedIn post or resume link preview.

**Documentation**
MkDocs site (deployed via GitHub Pages) covering architecture, methodology, and API reference — separate from the README, which should stay concise.

**Technical report**
A 4–8 page write-up (PDF or long-form markdown) structured like a mini-paper: problem formulation, data, methodology, results (with the model comparison and calibration analysis), limitations, future work. This is the artifact you attach to MS applications or send to a recruiter who wants depth.

**Blog article**
A shorter, more narrative version of the technical report (Medium/personal site/dev.to) — "Building a Live Tennis Win Probability Engine" — written for a general technical audience, good for visibility.

**Presentation**
A 10-slide deck summarizing the project, usable for interviews or a university application portfolio review.

**Resume bullet points** (examples to adapt)
- *Built and deployed a live tennis win-probability engine (FastAPI + PyTorch + Docker) modeling point-level sequential dynamics with LSTM/Transformer architectures, achieving [X]% improvement in log-loss over a gradient-boosted baseline while maintaining calibration within [Y] of ideal.*
- *Designed a Monte Carlo simulation engine enabling counterfactual "what-if" match analysis, deployed via a full CI/CD pipeline (GitHub Actions, Docker, MLflow model registry) to a public cloud endpoint.*

**LinkedIn post**
A short post pairing the demo GIF with 3–4 sentences on what the project does and what you learned, linking to the live demo and repo — timed for right after Week 16 for maximum relevance when applying.

**Overall framing:** the goal across every one of these artifacts is to make it obvious, within seconds, that this is an engineered *system* with rigorous evaluation behind it — not a notebook with a good README wrapped around it.

---

## Final Notes

- **Do not optimize for speed.** This is explicitly a 3–4 month (16-week) flagship project. Optimize for learning depth, engineering quality, reproducibility, and the honesty of your evaluation — a well-documented negative result (e.g., "the Transformer didn't beat the HMM, and here's the calibration analysis showing why") is more impressive to a technical reviewer than inflated claims.
- **Reuse your Project 1 discipline.** You already know how to do temporal validation, bootstrap CIs, and MLflow tracking rigorously — the main new territory here is PyTorch sequence modeling, simulation, and the full deployment loop. Don't relearn what you already know; spend your hours where the actual gap is.
- **Keep the "product" framing throughout.** Every design decision (from feature engineering to dashboard layout) should be justifiable as "this is what a sports analytics team would actually build," not "this is what maximizes buzzword coverage."
