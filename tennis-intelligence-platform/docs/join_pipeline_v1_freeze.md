# Join Pipeline v1 — FROZEN (2026-07-03)

Status: **read-only**. Do not modify `join_tml_mcp.py` join logic without a documented bug
report. This file is the permanent record of v1's validated state.

## Results (ATP/men's — the only gender currently joinable; see WTA note below)

```
TML matches (total pool):       198,063
MCP matches (total pool):       7,566

Coverage:                       79.1%  (5,988 / 7,566)
Duplicates:                     0
Ambiguous, unresolved:          6
Resolved via nearest-date:      353 (Stage 3) + 3 (Stage 4) = 356
Resolved via round-relaxation:  40

Unmatched:                      1,578
```

## Unmatched breakdown (qualitative, from manual review of the sample)

Two distinct patterns, both benign for v1 scope:

1. **Very recent matches (Apr–May 2026)** — likely a lag between TML-Database's latest
   season file and MCP's most recent charting additions, not a name-matching failure.
2. **Pre-Open-era / historical events (1960s–1980s)** — obscure or defunct tournaments
   (Davis Cup ties, "WITC Hilton Head", "Forest Hills") where TML and MCP likely use
   different tournament-naming conventions for events neither source prioritizes heavily.

Neither pattern is expected to materially affect a modern ATP win-probability model.

## The 6 remaining ambiguous cases

Logged (not arbitrarily matched) via the `ambiguous_unresolved` strategy tag in the join
log — reproducible by re-running `pipelines/diagnose_join_issues.py`. Explicitly excluded
from the joined dataset rather than guessed. Below the threshold that would justify further
investigation before moving on, per the "don't block the pipeline for the last 0.01%" rule —
revisit only if a downstream analysis specifically needs one of these matches, or if a
similar pattern turns out to be more widespread than 6 cases once point-level features are
built (worth a re-check after Milestone 5-7, not before).

## Known limitation: WTA

TML-Database contains ATP match-level data only. WTA join is structurally impossible until
a live WTA match-level source is found (see `data/README.md`). The pipeline correctly
reports 0% WTA coverage rather than failing silently — this is expected, not a bug.

## Design decisions baked into v1 (for future reference / the paper's methodology section)

- Join key: `(tournament_norm, round_norm, player_pair)` — deliberately excludes date as a
  hard key, since TML's `tourney_date` is tournament START date, not match date.
- Player pairs matched as unordered sets (frozenset), since MCP's Player 1/2 order isn't
  guaranteed to be winner-first.
- Multi-candidate ties broken by nearest-date (not boolean date-band) with a strict
  no-tie requirement — exact-distance ties are left unresolved rather than guessed.
- Consumed-index tracking across Stage 3 and Stage 4 prevents one TML row from being
  double-matched by two different MCP rows (fixed 2026-07-03, see commit history).

## Output files

- `data/processed/joined_matches_m.parquet` (5,988 rows)
- `data/processed/joined_matches_w.parquet` (0 rows — expected, see WTA note)

## Next reference point

Player ID canonicalization (Day 3) builds on top of this frozen output. If the join
pipeline is ever reopened, player IDs and everything downstream must be rebuilt.