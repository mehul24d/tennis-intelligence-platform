# Elo Rating v1 — FROZEN (2026-07-03)

Status: **read-only**. Do not modify `EloRating`, `compute_ratings`, or `ROUND_ORDER` without
a documented bug report. This file is the permanent record of v1's validated state.

## Configuration

```
Rating system:       standard Elo (src/tennis_intel/ratings/elo.py)
Cold-start rating:    1500.0
K-factor:             32.0 (fixed, applied uniformly — no surface/level/era scaling in v1)
Chronology proxy:     (tourney_date, round_order, match_num, tourney_id)
                       — TML has no per-match date, only tournament start date; round_order
                       encodes standard bracket progression as a documented approximation
Input:                data/processed/matches_with_player_ids.parquet
Output:                data/processed/matches_with_elo.parquet
```

## Results

```
Processed matches:      198,062  (1 dropped — missing winner_id/loser_id, logged)
Players rated:           7,561
Initializations:         7,561
Average rating:          1500.0  (exactly, by construction of zero-sum Elo)
Min rating:              1282.7
Max rating:              2278.0
Largest single update:     31.7
Mean update magnitude:     12.9
```

## Synthetic correctness — all 8 tests passed (verified by execution, not just written)

1. Two new players, A beats B, K=32 → 1500→1516 / 1500→1484 (exact)
2. Repeated wins (A beats B ×3) → strictly monotonic increase
3. Upset (1500 beats 2000) → +30.3 (large update)
4. Expected result (2000 beats 1500) → +1.7 (small update)
5. Reproducibility — byte-identical output on repeat run
6. Chronology robustness — shuffled input produces identical output after internal re-sort
7. Leakage protection — removing future matches does not change any past pre-match Elo
8. (added) Round-ordering — same-`tourney_date` matches sort correctly by bracket round
   (R32 before F), catching the exact bug a naive date-only sort would introduce silently

## Round-label handling (data quality note for the paper's methodology section)

TML's `round` column has inconsistent casing/formatting across eras. Confirmed via direct
inspection (not assumed) and mapped explicitly:

- `R256` (59 matches, all 1968 Roland Garros) — an early Open-era Slam with an unusually
  large draw; ranked before R128.
- `'3rd/4th'` (104 matches, various 1968-era WCT events) — third-place playoff; ranked at
  the same tier as a bronze-medal match, just before the Final.
- `'Fs'` (13 matches, all 1968 WCT round-robin events) — **low-confidence mapping** to
  Final-tier rank. Raw data gives no definition; "Final Stage" of a round-robin format is
  the most plausible reading given era and event type. Volume is negligible (13 / 198,063 =
  0.007%), so even if the guess is wrong, impact on aggregate Elo trajectories is immaterial.
  Documented here explicitly rather than silently assumed correct.

No unrecognized round labels remain as of this freeze (verified: zero warnings on the final
run).

## Known characteristic, not a bug: retired players appear in the "Top 20 by final Elo"

Robin Soderling, Patrick Rafter, Arthur Ashe, Rod Laver, Pete Sampras, Bjorn Borg, Andre
Agassi, and Stefan Edberg all appear in the top 20 despite being retired. This is because
v1's Elo has **no time decay** — `final_ratings` reflects each player's rating as of their
*last recorded match ever*, which for a player who retired near a career peak stays frozen
there permanently. This is technically correct (it genuinely is their last-known rating) but
should not be read as a "current rankings" list. Worth an explicit caveat wherever this table
is used in the paper or dashboard, and a natural candidate for a v2 enhancement (e.g. a
"rating as of date X" query, or a recently-active filter) — not a v1 blocker.

## Real-data sanity checks — all plausible, spot-checked against known tennis history

- **Top 2 (Sinner, Alcaraz) match the actual 2026 ATP landscape** — consistent with a
  dataset extending into 2026.
- **Trajectories for Djokovic/Nadal/Federer/Alcaraz/Sinner** all show plausible peak/decline
  or peak/ascent patterns (e.g. Djokovic max=2463, Federer max=2392, both far above their
  "last" values — consistent with aging-decline; Alcaraz and Sinner both still near their
  max as of their last match — consistent with being active peak-era players).
- **Elo difference distribution** (mean=161.7, median=129.3, max=1001.4) — the max is
  internally consistent with min=1282.7/max=2278.0 (a ~995-point spread exists in the
  ratings pool, so a near-1000-point single-match gap, e.g. a top seed vs. a qualifier in an
  early round, is plausible, not a bug).
- **Calibration bucket counts** show the expected shape (most matches cluster at moderate
  confidence, fewer at extreme confidence) — a full calibration curve (predicted probability
  vs. observed win rate) is deferred to the Evaluation milestone, as this aggregate table
  isn't a proper calibration check (see code comment in build_elo.py).

## Output files

- `data/processed/matches_with_elo.parquet` — 198,062 matches with pre/post-match Elo,
  delta, expected win probability, and k-factor used, in chronological order

## Design decisions baked into v1 (for the paper's methodology section)

- Player IDs consumed directly from `matches_with_player_ids.parquet` — zero string-based
  player lookups anywhere in the Elo pipeline.
- `RatingSystem` is an abstract interface (`src/tennis_intel/ratings/base.py`); `EloRating`
  is one implementation. Glicko-2, surface-specific Elo, and time-decayed variants can be
  added later as new implementations without touching `processor.py`'s chronological loop.
- Explicit tie-breaking via `(tourney_date, round_order, match_num, tourney_id)` — no
  reliance on pandas' incidental row order.
- Cold-start rating (1500.0) defined once in `RatingSystem.__init__`, never silently
  overridden elsewhere.

## Next reference point

Rolling stats, surface form, and fatigue features (Day 5) read `matches_with_elo.parquet`
and join further context from `players.parquet`. If Elo v1 is ever reopened, every feature
built on top of it must be rebuilt.