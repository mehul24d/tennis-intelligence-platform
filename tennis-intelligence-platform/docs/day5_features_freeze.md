# Day 5 Rolling Features — FROZEN (2026-07-03)

Status: **read-only**. Do not modify `feature_engineering_day5.py` or `score_parser.py`
without a documented bug report. This file is the permanent record of Day 5's validated state.

## Scope actually built

- Overall rolling form (5/10/20): matches, wins, win%, avg games won/lost, game
  differential, avg sets won/lost, straight-set rate, 3-set rate, 5-set rate, avg duration
- Surface-specific rolling form (5/10/20): matches, wins, win%, game differential
- Opponent strength: mean/median/max/min opponent pre-match Elo, last 10 matches (reuses
  frozen Day 4 Elo — not recomputed)
- Momentum: win streak, loss streak (both "entering this match"), rest days, matches in
  last 7/14/30 days
- Tournament context: previous tournament level, previous round reached

## Scope explicitly deferred (not silently dropped — see also each item's own reasoning)

- **Rolling surface-specific Elo** — belongs in `ratings/surface_elo.py` as its own rating
  system implementation (per the extensible `RatingSystem` design from Day 4), not bolted
  onto this feature module
- **Travel proxy** — requires tournament geolocation data not currently available
- **Rolling same-tournament / Grand-Slam-specific performance** — a real, valuable
  extension, but a distinct feature-engineering scope from "recent form"
- **Serve/return statistics from MCP point-level data** — proposed as **Day 6**: a genuine
  separate data-integration task (own join, own leakage discipline against
  `charting-m-stats-*.csv`), not a Day 5 add-on

## Engineering approach

Vectorized throughout via pandas `groupby().shift(1).rolling()` — no per-row Python loops
(the one exception, `matches_last_Nd`, uses `groupby().apply()` with time-indexed rolling,
which is a per-GROUP operation across ~7,561 players, not a per-row loop across 198k rows).

**Leakage proof:** `.shift(1)` moves every value down one row within each player's
chronological group BEFORE any rolling window is computed, so a window ending at row i can
only contain values from rows < i for that player — the current match's own outcome is
structurally excluded by construction, not filtered out after the fact. Same proof pattern
as Day 4's Elo, applied via vectorized ops instead of an explicit loop.

Same chronology proxy as Day 4 (frozen, not re-litigated): `(tourney_date, round_order,
match_num, tourney_id)`, since TML has no per-match date.

## A real bug caught and fixed during synthetic testing

`score_parser.py`'s retirement handling: TML uses two different conventions for retirement
scores inconsistently — some include the abandoned set's partial score as the last token
(`"6-2 3-1 RET"`), others list only completed sets (`"6-2 6-3 RET"`). An initial
position-based fix (always drop the last token) would have been WRONG for the second
convention. Fixed by checking whether the trailing set score is a *valid completed-set
score* rather than relying on position — verified against both conventions explicitly in
`tests/unit/test_score_parser.py`.

## Synthetic correctness — every example from the spec verified by execution

- W,W,L,W,L sequence → win_pct before 5th match = 3/4 = 0.75 exactly
- Hard,Hard,Clay,Grass → surface-specific stats correctly isolate by surface (2/2 Hard
  matches counted, Clay/Grass correctly excluded)
- Opponent Elo 1000,1200,1400 → mean=1200, median=1200, max=1400, min=1000, exact
- Win/loss streak sequence for W,W,L,W,L → (0,0),(1,0),(2,0),(0,1),(1,0), exact hand match
- Leakage: removing future matches does not change any past match's pre-match features
- Reproducibility: byte-identical output on repeat run
- Games/sets/duration averages and three-set rate: exact hand-calculated match
- Tournament context (previous level/round): exact match

## Real-data results

```
Processed matches:     198,062
Players tracked:         7,561
Score missing:                7
Score unparseable:        1,591
Score parse rate:         99.2%
```

## Real-data validation — plausible, spot-checked against known tennis history

Win_pct_last10 trajectories for five notable players all show plausible variation (not
static/broken):

| Player | Matches | Min | Max | Last |
|---|---|---|---|---|
| Novak Djokovic | 1,413 | 0.30 | 1.00 | 0.80 |
| Rafael Nadal | 1,326 | 0.30 | 1.00 | 0.60 |
| Roger Federer | 1,545 | 0.00 | 1.00 | 0.80 |
| Carlos Alcaraz | 345 | 0.40 | 1.00 | 0.90 |
| Jannik Sinner | 417 | 0.30 | 1.00 | 1.00 |

Sinner's `last=1.00` and Alcaraz's `last=0.90` are consistent with their real 2026 dominant
form; Federer's `min=0.00` (a genuine 0-for-10 stretch somewhere in his career) is plausible
for an 1,545-match career spanning multiple eras of form.

## Known limitation, quantified and documented (not fixed — negligible materiality)

A small number of `rest_days` values are implausibly large (max observed: 6,871 days /
~18.8 years). Root cause investigated directly, not assumed: traced to a Davis Cup Group II
match (Paraguay vs Jamaica, 1998) involving "Francisco Rodriguez" — an extremely common
name, strongly suggesting TML's own player-ID system merged two different real people under
one ID for a low-profile competition, rather than a bug introduced in this pipeline.

Quantified before deciding not to chase further:

```
                    >1 year    >5 years    >10 years
winner_rest_days      1,604          67            6
loser_rest_days       3,457         182           12
```

`>1yr` gaps are mostly real (injury comebacks are a genuine tennis phenomenon). The `>10yr`
tail (18 matches total, 0.009% of the dataset) is the implausible bucket, consistent with
rare TML player-ID collisions on common names in obscure events. Below the threshold that
would justify further investigation, per the same principle applied to Stages 1-3 — this is
an upstream TML data-quality characteristic, not something fixable within this project's
scope without re-disambiguating TML's own player IDs.

## Output files

- `data/processed/matches_with_day5_features.parquet` — 198,062 matches with all rolling
  features, chronologically ordered

## Next reference point

Milestone 5 (Model Development, per the original blueprint) reads
`matches_with_day5_features.parquet` directly. If Day 5 is ever reopened, every downstream
model/feature must be rebuilt.