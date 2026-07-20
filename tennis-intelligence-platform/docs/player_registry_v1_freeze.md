# Player Registry v1 — FROZEN (2026-07-03)

Status: **read-only**. Do not modify the resolution logic in `canonical_players.py` /
`player_aliases.py` without a documented bug report. This file is the permanent record of
v1's validated state.

## Results

```
Canonical players (registry size):  7,561
Unique MCP player strings tested:   1,003

Resolved:                           913  (91.0%)
  - join_derived (ground truth):    675
  - full_name_match:                201
  - loose_match (hyphen/apos.):      26
  - initials_match (unambiguous):    11

Unresolved:                          90  (9.0%)
```

Registry integrity: no duplicate IDs, no missing names/keys, referential integrity confirmed
(every non-null winner_id/loser_id in 198,063 TML matches exists in the registry; the one
flagged case during testing was a NaN id, not a real orphan).

## Root cause of the 90 unresolved names — confirmed, not assumed

Spot-checked directly: "Michael Zheng" does not appear anywhere in TML-Database under any
spelling (only an unrelated "Yu Zheng" does). Combined with the pattern in the unresolved
sample (young/low-ranked players: Alexandr Binda, Kaylan Bigun, Cooper Kose, etc.), this
confirms a **data-coverage gap**, not a matching-logic bug: TML-Database appears to include
tour-level main-draw matches only, while MCP's volunteer-charted matches include some
qualifying/challenger/futures-tier players who never appear in TML at all.

This is structurally different from the 6 unresolved join cases (which were TML/MCP
disagreeing about the SAME real match) — here, the player genuinely may not exist in TML's
data at any spelling. No further alias engineering can fix a name that refers to a player
outside the source's coverage.

One further known artifact: `"R"` in the unresolved sample is a malformed/truncated raw name
in MCP's source data, not a real player — worth excluding explicitly if it recurs as noise
downstream, but not worth a special-case rule for one row.

## Decision: accept as documented margin

Per the same principle applied to the join pipeline: do not block feature development for a
data-coverage gap that is (a) explained by a confirmed root cause, (b) logged and
reproducible, and (c) unlikely to disproportionately affect a modern tour-level ATP win-
probability model, since the affected players are largely obscure/low-ranked and appear in
few charted matches each.

## Output files

- `data/processed/players.parquet` — 7,561 canonical players
- `data/processed/matches_with_player_ids.parquet` — 198,063 TML matches with native IDs,
  referential integrity verified

## Design decisions baked into v1 (for the paper's methodology section)

- Player IDs are NOT invented — TML's existing alphanumeric IDs (winner_id/loser_id,
  ATP_Database.csv `id`) are adopted directly as the canonical ID scheme.
- Three-tier resolution for names without a native ID (i.e. MCP's Player 1/2 columns):
  1. Join-derived (free, ground truth) — from the frozen TML<->MCP match join
  2. Full-name match (accent/case/punctuation normalized, comma-reordered)
  3. Loose match (hyphen->space, apostrophe removed) — for compound-surname spelling variants
  4. Initials match (first-initial + surname) — ONLY if the key maps to a single registry
     entry; ambiguous initials collisions are left unresolved, never guessed
- Every resolution (and non-resolution) is logged with its strategy for audit.

## Next reference point

Elo computation (Day 4) reads `matches_with_player_ids.parquet` and joins point-level
features against `players.parquet` via `player_id` — never against raw name strings again.
If the registry is ever reopened, Elo and everything downstream must be rebuilt.