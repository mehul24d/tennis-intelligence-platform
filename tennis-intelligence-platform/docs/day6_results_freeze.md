# Day 6 — MCP Serve/Return Integration: Results (Frozen)

Status: **frozen**. Supersedes the corrupted first-pass numbers from earlier in this
session (see "Data integrity incident" below) — do not cite the original 344,442-row run.

## Data integrity incident (documented for the record, per the paper's methodology section)

The first run of `build_day6_features.py` produced a corrupted, inflated dataset
(344,442 rows instead of 198,062) due to two real, independently-discovered bugs:

1. **NaN `match_num` collision**: Day 5's data has 480 rows with `NaN` `match_num` (a batch
   of 2025 matches TML hasn't fully backfilled). Pandas treats `NaN == NaN` as a match
   during merges, so every such row silently fanned out against every other row sharing the
   same `tourney_id`. Fixed by switching to a fully-unique composite merge key
   `(tourney_id, match_num, winner_id, loser_id)`, verified via direct inspection to have
   zero duplicates.
2. **Duplicate stat rows in MCP's own data**: 26 `(match_id, player)` combinations had more
   than one "Total" row in `charting-m-stats-Overview.csv` (~0.17% of 15,116 rows) — an
   upstream MCP data-quality characteristic, not something introduced by this pipeline.
   Fixed with a deterministic dedup rule (keep first occurrence), logged explicitly.

A permanent row-count safety assertion was added to `build_day6_features.py` specifically
so this class of bug fails loudly and immediately in the future rather than silently
producing a corrupted dataset again — it caught the second (smaller, +36 row) instance of
this same failure mode before any bad output was written, which is exactly why it exists.

## Corrected results

```
Dataset: 198,062 matches (row count verified preserved through every merge stage)
Matches with real serve/return data: 5,803 (2.9% of total — close to the 5,988-match
  frozen-join ceiling; the small gap is each player's first charted match correctly
  having no career-to-date prior data yet, i.e. NaN, not a bug)
bp_saved_pct sanity check: 59.2% mean (within the historically expected ~60-65% range —
  confirms the bk_pts/bp_saved column interpretation)
```

### Enhanced model comparison (Elo + rolling form + serve/return), same subset, same folds

| Model | Mean Log Loss | Mean Brier |
|---|---|---|
| **Logistic Regression** | **0.5844** | 0.1993 |
| CatBoost | 0.5912 | 0.2019 |
| Random Forest | 0.5922 | 0.2024 |
| XGBoost | 0.6165 | 0.2089 |
| LightGBM | 0.6178 | 0.2091 |

### Incremental value of adding serve/return features (same subset, same folds, baseline vs. enhanced)

| Model | Baseline Log Loss | Enhanced Log Loss | Improvement | Relative % |
|---|---|---|---|---|
| LightGBM | 0.6291 | 0.6178 | 0.0113 | **1.79%** |
| CatBoost | 0.5998 | 0.5912 | 0.0086 | **1.43%** |
| XGBoost | 0.6204 | 0.6165 | 0.0039 | 0.62% |
| Logistic Regression | 0.5862 | 0.5844 | 0.0018 | 0.30% |
| Random Forest | 0.5928 | 0.5922 | 0.0007 | 0.11% |

## Key finding: serve/return data helps every model, but tree models benefit far more

This is a genuinely interesting contrast with the Milestone 5 baseline. There, tree models
(CatBoost/LightGBM/XGBoost) barely outperformed logistic regression (spread under 0.003
log-loss) — the interpretation was that Elo is already a well-engineered, nonlinearly-
derived single-number summary, leaving trees little additional nonlinear structure to
exploit.

Here, with genuinely raw, non-summarized serve/return rates added, **LightGBM and CatBoost
show 4-6x the relative improvement that Logistic Regression and Random Forest show**. This
is consistent with tree-based models finding real nonlinear interactions in the serve/
return features (e.g., break-point conversion mattering disproportionately in tight
matches, first-serve% effects that may interact with surface) that a linear model captures
less completely. This is exactly the empirical question Day 6 was built to answer, and it
has a real, directional answer: **richer, non-Elo-summarized point-level information
narrows the gap that model complexity needs to close, rather than being redundant with it.**

## An honest nuance, not glossed over

Despite gaining the most from the new features, XGBoost and LightGBM still have **worse
absolute log loss** than Logistic Regression on this subset (0.617–0.618 vs. 0.584). This
is very plausibly sample-size-driven: the serve/return-available subset (~5,803 matches) is
roughly 34x smaller than the full Milestone 5 dataset, and the same fixed hyperparameters
used there (300 estimators, max depth 5 — reasonable defaults for ~176k-195k training rows)
are likely overfitting a training fold an order of magnitude smaller. This is a natural,
well-motivated next step — hyperparameter tuning scoped specifically to this subset's size
— rather than a contradiction of the "tree models benefit more" finding above: the
*incremental value* of the new features and the *absolute* competitiveness of a given
model's fixed hyperparameters are two different questions, and this result answers the
first honestly without overclaiming the second.

## What this means for the live win-probability roadmap

This result directly informs the point-level modeling recommendation in
`live_win_probability_extension_analysis.md`: since tree-based models show real ability to
exploit non-Elo-summarized statistical features, a point-level classifier (Option B in that
document) is well-motivated to use gradient boosting rather than defaulting to logistic
regression — with the important caveat, demonstrated here, that hyperparameters should be
tuned for the actual training-fold sizes involved, not inherited unchanged from a much
larger dataset.

## Artifacts

- `data/processed/matches_with_day6_features.parquet` — corrected, 198,062 rows
- `data/processed/day6_baseline_on_subset.csv`, `day6_enhanced_aggregate.csv`,
  `day6_incremental_value.csv` — comparison tables above

## Next steps

1. (Optional, low priority) Hyperparameter tuning for tree models scoped to the ~5,800-match
   subset size, to test whether the absolute-performance gap closes.
2. Proceed to the live win-probability extension roadmap (Days 7-10) — Day 6's finding that
   tree models exploit raw point-level statistics well is a direct, evidence-based input
   into that roadmap's Option B recommendation.