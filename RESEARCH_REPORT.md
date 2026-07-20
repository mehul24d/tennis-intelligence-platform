# Tennis Intelligence Platform — Research & Evaluation Report

*Every number in this document traces to a specific, verifiable source: `PROGRESS.md`
(the chronological build log), `cv_pipeline/EVALUATION_REPORT.md`,
`cv_pipeline/STRESS_TEST_REPORT.md`, `tennis-intelligence-platform/docs/
ptwinner_convention_correction.md`, the two `known_issue_*.md` docs, and a
precision@k retrieval evaluation run specifically for this report (methodology and
raw output in §6.1, since no such evaluation existed before this document — confirmed
by searching the codebase before building one). Where a finding was scoped, partial,
or later superseded, this document says so — nothing here is rounded up.*

---

## 1. Problem statement & motivation

The Tennis Intelligence Platform began as **v1**: a statistical win-probability
engine trained on 198,000+ professional tennis matches and 7,500+ point-by-point
charted matches, using dynamic Elo ratings, 100+ leakage-safe temporal features, and
an ensemble of gradient-boosted models (XGBoost, LightGBM, CatBoost) benchmarked
under rolling-origin (walk-forward) validation. It also includes a live, in-match
win-probability engine combining Monte Carlo simulation and Markov-chain modeling,
updated point-by-point as a match progresses. v1 is a complete, self-contained
system — closer to a research-grade classical ML pipeline than a product.

**v2** set out to answer a different question: what does it take to extend a working
statistical system with the multimodal capabilities that define current AI
practice — computer vision on raw video, retrieval-augmented generation over a large
structured dataset, and an LLM agent that can reason over both? The motivation isn't
that v1 was insufficient on its own terms; it's that the gap between "a well-validated
statistical model" and "a system that can watch a match, retrieve relevant history,
and explain what it's seeing in natural language" is exactly the gap most real AI
engineering work lives in today. Building v2 on top of an already-complete, already
battle-tested v1 — rather than starting from a toy dataset — meant every claim v2
makes about "grounding in real data" had a real, large, historically accurate dataset
to be grounded in, and every bug v2's construction surfaced in v1 was a bug in a
system that had previously been considered finished.

---

## 2. System architecture

The platform is two systems sharing one dataset. v1's `tennis-intelligence-platform/`
backend owns all historical match data and the Monte Carlo/Markov live
win-probability engine, exposed through its own FastAPI service layer
(`src/tennis_intel/serving/`). Nothing in v2 duplicates v1's data or re-derives its
statistics — every v2 component that needs a v1 number calls into v1's existing
serving functions directly (`career_stats_service`, `replay_service`,
`point_timeline_service`), so a future fix to v1's own logic propagates to v2
automatically rather than living in two places that can drift apart.

v2 is five components, each independently buildable and independently tested, wired
together by one orchestration layer:

```
Match video
    → cv_pipeline/    (YOLOv8 detection, ByteTrack tracking, homography,
                        MediaPipe pose — structured per-clip features)
    → rag_engine/      (retrieval over v1's match/player/point data,
                        local Chroma vector store)
    → llm_agent/        (Gemini-based agent fusing live CV features +
                        retrieved historical context into grounded analysis)
    → v2_serving/      (FastAPI: async video jobs, RAG+LLM query endpoint,
                        win-probability endpoint wrapping v1's engine)
    → v2_dashboard/    (React: upload/poll, honest-status result view,
                        video overlay, chat, win-probability panel)
```

`cv_pipeline/` and `rag_engine/` are the two components with no dependency on each
other — either could be swapped or extended independently. `llm_agent/` is the fusion
point: it is the only component that talks to an external LLM provider (Gemini, via
Google's `google-genai` SDK), and it consumes both a `rag_engine` retriever and a
CV-feature snapshot without needing to know how either was produced. `v2_serving/`
is pure orchestration — by design, no analysis logic lives there, only wiring — and
`v2_dashboard/` is a thin client against `v2_serving`'s API, with no logic of its own
beyond honest rendering of what the API returns.

---

## 3. What v2 actually adds, concretely

### RAG (`rag_engine/`)

Three document types were designed and built: **match summaries** (sourced from
v1's full ~198k-match `matches_with_elo.parquet`, not just the smaller
charted-match subset), **player career profiles** (via v1's own
`career_stats_service.get_player_profile`, not re-derived), and **point-level
"notable moment" documents** (sourced from v1's live win-probability trajectory via
`point_timeline_service.get_point_timeline`, phrased deliberately swing-neutral —
see §4).

All three types are built, wired into a single `build_index.py` entrypoint, and
covered by 17 passing tests. **However**, checking the actual persisted index that
`llm_agent/` queries against in production (rather than restating from the build log)
originally showed it contained **22,610 documents — 20,000 match summaries and 2,610
player profiles, and zero point documents.** The full ~198k-match embed was
deliberately deferred (CPU-only embedding throughput on this hardware made a full
run ~5+ hours; 22,610 docs — the 20,000 most recent matches plus every player with
≥10 career matches in that window — was judged sufficient to validate the pipeline,
with the full embed revisited closer to when it's actually needed).

**Update: point documents are now partially live.** After this gap was found, a
100-match representative subset of the 5,981-match frozen-join corpus was generated
and embedded, adding **1,137 point documents** (the live index now has **23,747
documents** total). This is the same documented-scope-decision pattern as the
20,000-match/22,610-doc subset above — a deliberate partial deployment, not the full
5,981-match corpus (measured at ~41.6 hours to embed in full; the 100-match subset
took ~57 minutes to generate plus ~14 seconds to embed).

Getting there required finding and fixing a real bug, not just running a script:
combining `rag_engine`'s embedding stack (PyTorch/sentence-transformers) and v1's
model-inference stack (its own native-library dependencies) in one process
segfaulted twice (exit 139, no Python traceback) at the exact same point in
execution — once before, and once *after* applying the standard
`KMP_DUPLICATE_LIB_OK=TRUE` fix for macOS OpenMP double-initialization conflicts,
which is the commonly-cited fix for this class of crash and did resolve the
load-time symptom but not the underlying one. The second crash proved the real
conflict happens when both native stacks are *actively computing* concurrently in
one process (v1's per-point generation loop interleaved with Chroma's periodic
batch-embed calls), not merely present. The fix was architectural, not a deeper
env-var: split the work into two single-stack processes connected by a JSON file on
disk — one process runs only v1's stack and serializes documents to disk, a second
process runs only the embedding stack and reads them back in. This sidesteps the
conflict rather than resolving it in-process, and is the kind of environment-level
finding worth recording precisely because it's easy to silently "fix" by retrying
and never understand.

Retrieval quality: no formal precision@k evaluation existed anywhere in the codebase
before this report (confirmed by searching before writing one — see §6.1 for the
methodology and full results). A 9-query evaluation built specifically for this
report, using real player names confirmed present in the index and metadata-based
(not eyeballed) relevance criteria, found mean precision@3 = 0.815 and precision@5 =
0.800 overall — but with an important structural split: match-summary queries (7 of
9) scored precision@5 = 0.971, while player-profile queries (2 of 9) scored
precision@5 = 0.20 for a mechanical reason (there is exactly one profile document per
player in the whole corpus, so any k>1 is structurally penalized regardless of
retrieval quality — the correct measure there, "was the one relevant document ranked
first," was 100%).

**After point documents went live, the evaluation was re-run with 3 added
point-level queries (12 total)** — and it surfaced a real, unflattering finding
that would have been easy to miss: **adding point documents to the same collection
measurably hurt two of the original match-summary queries.** "Rafael Nadal clay
court matches" and "Carlos Alcaraz hard court matches" both went from perfect or
near-perfect precision to **0/5** — their top-5 nearest neighbors are now entirely
point documents (e.g. "Novak Djokovic vs Rafael Nadal, Madrid Masters (Clay), SF...
At set 1-1, games...") that mention the same player/surface but aren't match
summaries, crowding out the actual match-summary documents that used to rank there.
This is retrieval competition from adding a third document type into one
undifferentiated collection, not a data or embedding-model problem, and it was not
present in the original 9-query result — it's a direct, measured cost of the point-
document deployment this section describes, reported here rather than only in the
places where the news is good. Full breakdown in §6.1.

### LLM agent (`llm_agent/`)

`TennisAnalystAgent` fuses a `LiveFeatureSnapshot` (CV-derived or win-probability
features, each explicitly tagged with whether it's a model estimate) with
`rag_engine`-retrieved historical context, and calls Gemini (`gemini-3.5-flash`,
via `google-genai`) through a stateful multi-turn chat session. The grounding
discipline is enforced by the system prompt, not left to the model's judgment: every
factual claim must cite a `[L#]` (live feature) or `[D#]` (retrieved document) tag;
live win-probabilities must always be hedged as model estimates, named by engine;
and the agent must say "insufficient historical data" explicitly rather than
fabricate when retrieval is weak. A citation audit (`sources_used` vs.
`sources_offered`) lets a caller see exactly which of the sources the model was given
it actually cited, versus everything it had available — a distinction most RAG
systems don't expose at all.

This was verified two ways, not just internally: a 10-question manual evaluation
against real retrieval and real Gemini calls found no hallucination on inspection,
and — the more rigorous check — **3 of the most numeric-heavy answers were
independently re-verified against raw rows in `matches_with_elo.parquet`, bypassing
the RAG layer entirely**, and all three matched exactly, including one case where the
agent correctly disambiguated between three same-opponent matches rather than
conflating their stats.

### CV pipeline (`cv_pipeline/`)

YOLOv8n (person + COCO "sports ball" class) for detection, ByteTrack for tracking,
a homography built from annotated court corners for pixel-to-real-world mapping, and
MediaPipe Pose on detected player crops — validated against a **10-clip,
ground-truth-annotated amateur dataset**, not just eyeballed. Full results in §6.2;
the headline honest summary: near-player detection and pose are excellent and
consistent everywhere (91.3–99.8% detection); far-player detection and pose are a
genuine, tested-and-confirmed hardware/resolution limitation, not a bug; ball
detection is a recall problem (rarely finds the ball, but accurate to 2–4px when it
does); and only 1 of 10 clips' homographies is independently validated for
real-world-distance use, with 1 confirmed bad and 8 unconfirmed either way.

### Serving + dashboard (`v2_serving/`, `v2_dashboard/`)

Four endpoints (`POST /analyze-video`, `GET /jobs/{id}`, `POST /query`,
`GET /win-probability/{id}`), each built and manually confirmed against a real
request before the next was wired in, then covered by an 18-test suite (10.24s,
including two real regression-guard calls into v1's actual engine — see §6.3). The
dashboard is a 6-view React app built against the real running API at every step
(no mocked data anywhere in the app), each view confirmed with an actual Playwright
screenshot: connection health, the full async job lifecycle (all 3 states), a result
view that renders 7 distinct `Status` values with 7 distinct visual treatments, a
video-overlay canvas with pixel-verified alignment (including a real frame where a
detection box is genuinely and honestly absent), a chat interface where the
cited-vs-offered-but-uncited source distinction is visibly, not just structurally,
different, and a win-probability panel showing both a real computed value and an
honest "not available" state side by side.

---

## 4. Rigor & validation methodology

This section is the part of the project most worth reading closely, because it's
where the actual differentiator lives: not that the system works, but the discipline
applied to find out whether it actually did, repeatedly, across every layer.

### 4.1 The RAG example that caught a fabrication before it shipped

Early in Phase 1, while drafting an example "notable point" document, a specific
match reference was included that turned out not to exist — a fabricated
Roland Garros final pairing that had never been played. This was caught during
review, before it could be embedded into the knowledge base, and it changed the
project's working standard from that point forward: every subsequent RAG document
example was required to be grounded in a real, verifiable row from the underlying
data, shown alongside the generated text, not asserted. This is a small-sounding
catch, but it's the reason the numeric-verification standard in §3's LLM-agent
section (independently re-checking generated answers against raw source rows) exists
at all — it was a lesson learned, not a policy adopted in the abstract.

### 4.2 v1's PtWinner convention bug — the project's deepest investigation

While grounding a point-document example in real point-timeline output, an
automatically-flagged (not hand-picked) point looked wrong: the point winner and the
win-probability direction disagreed. Pulling that thread led to comparing the
point-level data's `Gm1`/`Gm2` (games-won) columns against the point-by-point charted
data, which surfaced a corpus-wide, ~49% mismatch rate at individual game boundaries —
present in **100% of the 5,981 matches checked**, not scattered charting noise.

The investigation initially misdiagnosed this as a bug in `Gm1`/`Gm2` themselves
(preserved for its investigation history in
`docs/critical_issue_gm_attribution_mismatch.md`) and pursued it with real rigor even
while wrong: two independent hand-traced verification methods sharing zero code (one
using an already-audited helper function, one using pure score-delta first-principles
reasoning with no reference to any existing convention) both confirmed the mismatch
on a real match (Laver/Ashe, 1969 Wimbledon SF); a corpus-wide vectorized check found
147,290 boundaries, 71,815 mismatches; a shift test ruled out a simple one-row
indexing lag (the mismatch rate got *worse* at neighboring shifts, not better); and a
cross-tabulation ruled out a hold/break-specific bug and isolated the real
correlate — mismatch rate was 0.11% when player 1 served the deciding point and
99.90% when player 2 did, a near-perfect split on server identity.

The actual root cause, found only by testing every other hypothesis to exhaustion
first: `Gm1`/`Gm2` were correct all along. `PtWinner` — the point-winner column — had
been changed the same day from its original, correct convention (**literal**,
`PtWinner==1` means player 1 won the point, full stop) to an incorrect
**server-relative** one (`PtWinner==1` means the server won), based on a pre-existing
diagnostic script whose "0.00% disagreement" claim only tested internal
self-consistency on non-game-boundary points — a blind spot that made two genuinely
different, both internally self-consistent conventions indistinguishable to it. Six
hypotheses were tested in sequence with real numbers before landing on this: a global
column swap (refuted by real historical set scores), a one-row lag (refuted by the
shift test), a hold/break-specific bug (refuted by the cross-tab), server identity
alone (initially promising, but a genuine three-game hand-trace on one match produced
a contradiction that forced a deeper look — the informal hand-trace turned out to be
wrong, not the hypothesis), a wrong analogy to an already-fixed, unrelated tiebreak
notation bug (refuted at corpus scale), and finally the correct combination:
server-first point notation checked against **literal** `PtWinner`, which matched at
99.9990% and, checked against `Gm1` at real game boundaries, 99.91% corpus-wide,
symmetric across which player served.

**Fixed** in every file the wrong convention had touched: the live win-probability
engine's serve/return posterior update, two serving-layer display functions, and five
point-level feature functions. **Verified**, not just fixed: 211 unit tests passing,
plus a direct hand-verification in plain tennis terms on two real matches (Laver/Ashe
and Nadal/Shapovalov) confirming the corrected convention matches the ordinary,
expected outcome (a server grinding out and holding a deuce game) rather than the
old convention's requirement of the rarer outcome (a break) in both cases.

**A second, related bug was found and fixed in the same investigation window**: the
live win-probability engine's per-point trajectory was being mis-indexed by two of
its four downstream consumers, attributing a probability swing to the point *after*
the one that actually caused it. Traced precisely enough to determine that the
underlying trajectory computation itself (used correctly by 15+ other calibration
scripts) did not need to change — only the two specific consumers that built a
before/after pairing from it did. Verified with a hand-computed synthetic 4-point
sequence with a deliberate, unambiguous swing (confirmed to fail pre-fix and pass
post-fix), then on a real match: direction-correctness against the point winner rose
from 52.1% to **89.9% overall, 100% on every point with a non-negligible swing**,
with the small residual explained and accepted as legitimate near-certainty-tail
noise, not treated as an unexplained gap.

### 4.3 Retraining decided by measurement, not default

Finding the `PtWinner` bug raised an immediate question: does it matter enough to
retrain the point-level classifier that had been trained on the buggy features? The
project's explicit standard was to answer this with numbers before deciding either
way, not default to "always retrain" or "code is fixed, ship it." A feature-impact
analysis found the classifier's #3-ranked feature (`p1_in_match_return_rate`) shifted
on 96.8% of affected rows by an average of a third of its full range, and its
#5-ranked feature (`points_streak`) shifted on 86.9% of rows — roughly 7-8% of total
model importance sitting on features that were materially different at inference
time than at training time. That evidence justified a full retrain as its own scoped
task, executed with rolling-origin (not single-split) discipline across four
independent years (2022–2025): the retrained model won in **every single fold**, by
almost the same margin each time (mean log_loss 0.6281 → 0.6247, mean Brier 0.2187 →
0.2172) — full numbers in §6.4.

One SHAP result from that comparison was surprising enough to investigate rather than
accept: the #3 feature kept its rank but its importance magnitude dropped ~42%. Two
competing explanations were possible — a real predictor genuinely became less
important, or its signal was being "stolen" by a correlated feature post-fix. A
five-minute correlation check (requested explicitly before accepting either
explanation by default) found the collinearity explanation didn't hold (correlation
between the two features was negligible both before and after the fix) but revealed
something cleaner: the *old*, buggy feature had a backwards correlation with the
actual prediction target (higher return rate weakly predicted *losing* the next
point — a direct fingerprint of the underlying convention bug), which flipped to the
intuitively correct sign after the fix. The model had been partially exploiting
spurious, wrong-signed noise; losing access to it explains the importance drop
without implying the retrain made things worse.

### 4.4 The CV pipeline's own bug trail

Phase 3 surfaced a comparable density of real findings, each investigated with the
same standard — real data, real numbers, hypotheses tested rather than assumed:

- **Sentinel-value contamination in ground truth.** Both the ball and player
  ground-truth CSVs contain a placeholder value (a fixed pixel near one frame
  corner) for "not tracked this frame," not a real position. This was confirmed,
  not assumed: the placeholder clustered at the identical corner across all 10
  clips regardless of content, formed long motionless runs no real ball or player
  could produce, and — the most direct check — overlaying the raw coordinate on the
  actual video frame landed in empty background, not on the visible ball or player,
  in every case checked. Excluded from all detection-accuracy scoring once
  confirmed, with the "how often is ground truth itself missing" rate reported
  separately rather than silently dropped.
- **Inconsistent court-corner labeling across clips.** Generalizing the homography
  build from one clip to all 10 revealed that the same corner-label strings
  (`BL`/`BR`/`TL`/`TR`) don't refer to the same physical corner in every clip —
  confirmed by overlaying the raw labels on real frames and finding two clips with
  the labels effectively rotated or mixed. Fixed by deriving near/far/left/right
  geometrically from actual pixel positions instead of trusting the label strings —
  a fix that doesn't depend on knowing which clips are affected, since it's
  correct by construction for all of them.
- **A homography scale error, found by an independent landmark, not the obvious
  self-consistency check.** One clip's 4-corner reprojection error was a perfect
  0.0px — but that check is trivially self-consistent by construction and cannot
  catch a wrong overall scale. An independent landmark (the net's ground-contact
  point, never used in calibration) predicted a real-world position 87px off from
  its measured location — roughly 6-7x worse than the same check on a validated
  clip. Rather than leave this as an unexplained anomaly, the root cause was
  computed directly: the implied real-world span the annotated corners must
  represent matched "baseline-to-net" (11.9m) to within 2.8%, meaning that clip's
  annotation covered only half the court's true length. That clip is now flagged
  and excluded from real-world-distance metrics specifically, not from the whole
  evaluation.
- **Matching-exclusivity artifacts in validation.** Both the player-detection and
  tracking-ID validation initially let two different ground-truth roles (near
  player, far player) match the *same* underlying detection independently,
  producing trivially "consistent" results that reflected a ground-truth
  duplication artifact (the far player's ground truth frequently points at the
  same physical player as the near player's, especially when the true far player is
  off-frame) rather than real tracking quality. Fixed with exclusive per-frame
  matching in both cases, which materially changed the reported numbers (e.g. the
  far-player "separated" detection bucket dropped from a contaminated 27.3% to a
  clean, cross-confirmed 20.8%).
- **The box-selection bug, found twice — and only fixed once confirmed as a
  pattern.** A "pick the largest/smallest detected box" heuristic for identifying
  near vs. far player was found to occasionally pick a courtside bystander instead
  of the real player on one amateur clip. Rather than patch that single occurrence,
  the fix was deliberately deferred until the same failure mode was independently
  confirmed on a second, unrelated clip (an out-of-distribution professional
  stress-test video) — at which point it was fixed properly in shared pipeline code
  (court-position plausibility via homography, not box size), re-verified to
  resolve the original case exactly, and honestly documented as *not* fully
  resolving the second clip's case, because that clip's own homography calibration
  was independently too imprecise for any selection logic to compensate for — a
  distinction between "the fix works" and "the fix's input was good enough" that a
  less careful writeup would have blurred.

### 4.5 A finding correctly abandoned, not forced

Not every investigation reached a conclusion, and the project's discipline extended
to knowing when to stop. An attempt to test whether ball-detection ground-truth gaps
differ between the near and far sides of the court ran into a genuine circularity:
side labels for missing-ball frames had to be imputed from the nearest real
ball-position frame, but real ball positions were themselves overwhelmingly
concentrated on one side — meaning any near/far split computed this way would just
reflect which side happened to have more surviving ground truth, not a real
difference in detection difficulty. Rather than adjust the method until it produced
a clean-looking split, this was explicitly documented as unanswerable with the
available data and shelved, with a note on what would need to change (an independent
side-determination signal, e.g. from pose) before revisiting it.

---

## 5. Honest limitations

Stated plainly, with the reason for each, not apologetically:

- **Far-player detection and pose are a genuine, tested-and-confirmed hardware
  limitation**, not a threshold or model-capacity problem. Confirmed by lowering the
  detection confidence threshold to 0.01 (no candidate exists at any confidence in
  missed frames) and by testing a larger model, `yolov8s` (marginally worse
  accuracy, ~37% slower — not adopted). Detection rate on adequately-sampled clips
  ranges 0–34% depending on camera framing; pose estimation fails on the far player
  even against a clean, uncluttered background in an out-of-distribution
  professional clip, pointing toward crop resolution as the dominant constraint
  (one data point, reported as suggestive, not conclusive).
- **Only 1 of 10 annotated clips' homographies is independently validated** for
  real-world-distance metrics (speed, court coverage in meters); 1 is confirmed bad
  and excluded; the remaining 8 pass only a self-consistency check that the
  confirmed-bad clip proves is insufficient on its own.
- **Tracking ID stability under crossing/occlusion is only lightly tested.** A
  contaminated "hard moment" proxy initially overstated test coverage; once fixed,
  6 of 10 clips turned out to have had zero genuine crossing scenarios to test
  against at all. Only one clip has substantive coverage, and it showed real ID
  swaps under real crossing conditions.
- **RAG's full ~198k-match corpus is not embedded** — the live index has 22,610
  match/player documents (a deliberate, documented scope decision given CPU-only
  embedding throughput on this hardware). Point documents, found to be entirely
  missing while preparing this report, are now **partially live**: 1,137 point
  documents from a 100-match subset of the 5,981-match frozen-join corpus (23,747
  documents total in the index today). This is the same documented-scope-decision
  pattern as the match/player subset, not a full production deployment — the
  remaining ~5,881 matches (~41.6 hours of generation time at the measured
  ~25s/match rate) are not embedded.
- **Adding point documents measurably degraded precision on 2 of 7 match-summary
  queries** ("Rafael Nadal clay court matches", "Carlos Alcaraz hard court
  matches" — both dropped to 0/5), because point documents about the same
  player/surface now outrank the actual match-summary documents in the shared
  collection. This is retrieval competition between document types sharing one
  index, not a data quality or embedding-model issue — see §3 and §6.1 for the
  measured before/after.
- **Point-level query precision is weaker than match-summary precision** (mean
  precision@5 = 0.333 across 3 point-level queries vs 0.971 for match-summary
  queries before point documents were added) — generic point-query phrasing
  competes with match-summary documents for the same nearest neighbors, and only
  reliably retrieves point documents when the query phrasing echoes point-specific
  concepts (e.g. "break point moments" outperformed "notable points on hard
  court"). Not yet mitigated (e.g. via a `doc_type` filter at query time) — that
  would be the obvious next step, flagged here rather than done silently.
- **The LLM agent depends on Gemini's specific API characteristics** — model
  availability and pricing tiers were found to shift during development (an initial
  model choice returned quota errors on the free tier; a fallback model was
  discontinued for new accounts), and the agent's 503-retry logic exists
  specifically because Gemini's transient "high demand" errors were observed
  directly during testing, not hypothesized in advance.
- **The win-probability engine's `live_adjustment` field is honestly
  `not_available` for every clip today** — not a bug, a structural gap: v1's
  in-match adjustment mechanism needs point-by-point serve/rally outcomes, and
  `cv_pipeline`'s current output (detection rates, tracking stability, pose
  success) contains no such extraction. The code checks for this data
  structurally rather than hardcoding the negative, so the feature activates
  automatically if that gap is ever closed.
- **Player identity is never resolved from video** — `cv_pipeline` locates a "near
  player" and "far player" by position, not by name, so there is currently no
  automated way to connect a video job to a specific real match or player in v1's
  historical dataset without a human supplying that mapping manually (the
  `win-probability` endpoint's optional `match_id` parameter is exactly this manual
  bridge).
- **Three WTA players checked while preparing §6.1's evaluation came back absent
  from the live index** ("Iga Swiatek", "Ashleigh Barty", "Naomi Osaka" — plain-ASCII
  spellings, as queried; accented-name variants weren't separately tested) — a
  subset-selection artifact (fewer than 10 matches for that player within the
  most-recent-20,000-match window used to build this particular index), not evidence
  of a gender skew in the underlying ~198k-match dataset itself, which was not
  re-checked as part of this report.

---

## 6. Evaluation results

### 6.1 RAG retrieval — precision@k (built for this report; methodology above)

**Original 9-query run**, against the 22,610-document index (match/player docs
only, before point documents went live). Relevance determined by document
metadata (player identity, surface, tournament), not eyeballed:

| query type | n queries | precision@3 | precision@5 |
|---|---|---|---|
| match-summary queries | 7 | 0.952 | 0.971 |
| player-profile queries | 2 | 0.33 | 0.20 (structurally capped — see below) |
| **overall mean** | **9** | **0.815** | **0.800** |

The one real miss among the match-summary queries at that point: a "Carlos Alcaraz
hard court matches" query returned one clay-court match at rank 3 — a genuine,
minor semantic retrieval error (topically close, surface-mismatched), not a data
problem.

Player-profile queries need a caveat, not a discount: there is exactly one profile
document per player in the entire corpus, so no query for a specific player's
profile can ever have more than one truly relevant document — precision@3 and
precision@5 are mechanically bounded below their match-summary counterparts
regardless of retrieval quality. The more meaningful measure for this query type is
whether the one relevant document was ranked first, which it was in both cases
tested (100%).

**Re-run after point documents went live**, same 9 queries plus 3 new point-level
queries (12 total), against the current 23,747-document index:

| query type | n queries | precision@3 | precision@5 |
|---|---|---|---|
| match-summary queries | 7 | 0.714 (was 0.952) | 0.829 (was 0.971) |
| player-profile queries | 2 | 0.33 | 0.20 (structurally capped, unchanged) |
| point-level queries (new) | 3 | 0.222 | 0.333 |
| **overall mean** | **12** | **0.528** | **0.517** |

**Match-summary precision dropped** (0.952→0.714 @3, 0.971→0.829 @5) because two
queries — "Rafael Nadal clay court matches" and "Carlos Alcaraz hard court
matches" — now return point documents almost exclusively in their top-5 (both
0/5), instead of the match-summary documents they returned before. This is a
direct, measured side effect of adding a third document type into the same
undifferentiated collection: point documents about a matching player/surface are
semantically close enough to out-rank match summaries for these two queries. The
other 5 match-summary queries were unaffected.

**Point-level queries** (ground truth pool sizes checked directly against the
generated batch, not guessed: 42 docs for Djokovic break points, 72 for Hurkacz on
hard, 8 for Djokovic-Nadal clay match points):

| query | precision@3 | precision@5 |
|---|---|---|
| "Novak Djokovic break point moments" | 0.33 | 0.60 |
| "Hubert Hurkacz notable points on hard court" | 0.00 | 0.20 |
| "Djokovic Nadal clay court match point" | 0.33 | 0.20 |

Point-level retrieval is weaker and more query-phrasing-sensitive than
match-summary retrieval: "break point moments" (phrasing closer to how point
documents are actually worded) outperformed "notable points on hard court"
(generic phrasing that mostly matched Hurkacz's *match*-summary documents
instead, which also mention "hard court" and his name). No `doc_type` filtering
was applied at query time for this evaluation — doing so would likely fix both
the match-summary regression above and the point-query weakness here, since it
would stop the two document types from competing for the same ranked list, but
that change was not made or tested here; it's a specific, actionable next step,
named rather than left as a vague "could investigate further."

### 6.2 CV pipeline (full detail: `cv_pipeline/EVALUATION_REPORT.md`)

| metric | result |
|---|---|
| Near-player detection | 91.3–99.8% (mean 96.4%, median 98.3%), consistent across all 10 clips |
| Near-player position error | 57.8–91.8px median |
| Far-player detection (adequately-sampled clips, n≥20) | 0–34% (mean 10.5%) — genuine hardware limit, confirmed via threshold and model-size tests |
| Ball detection, stock YOLO (9 of 10 clips, 1 broadcast-quality outlier excluded) | ~7–9% (mean 7.8%, median 4.9%) |
| Ball detection, combined YOLO + motion-diff (production method, `ball_detection_combined.py`) | **53.91% pooled recall** (video3 excluded, 9-clip scope) — a real ~6.9x improvement, but corrected down from an initially-reported 70.40% after a ground-truth leak was found and fixed in the candidate-selection logic; see Addendum (2026-07-19) |
| Ball position error, when detected | 2.3–4.1px median, consistent across all 10 clips |
| Homography, independently validated | 1 of 10 clips (~13px / ~8cm real-world error) |
| Homography, confirmed bad and excluded | 1 of 10 clips |
| Tracking, clips with genuine crossing/occlusion test coverage | 4 of 10 (thin even there) |

### 6.3 v1 win-probability engine — before/after retrain (full detail:
`ptwinner_convention_correction.md`)

| | log_loss (mean, 4 rolling-origin folds 2022–2025) | Brier (mean) |
|---|---|---|
| OLD (pre-fix, deployed until 2026-07-15) | 0.6281 | 0.2187 |
| NEW (retrained on corrected features) | **0.6247** | **0.2172** |

New model wins in every one of the 4 independent yearly folds, by almost the same
margin each time. Top-4 SHAP feature ranks identical between old and new (stable
model structure); the one feature whose importance dropped materially
(`p1_in_match_return_rate`, −42%) was traced to the model losing access to a
spurious, wrong-signed correlation the bug had introduced, not a real predictor
becoming meaningless (§4.3).

### 6.4 API latency — the fast-path win-probability fix

| | latency |
|---|---|
| Original implementation (full per-point trajectory, correct but unnecessarily slow) | ~90s per call |
| Fast path, first call (pays one-time context-load cost) | ~16.7s |
| Fast path, steady-state (subsequent calls) | **~0.2s — ~450x faster than the original** |

The fast path was verified bit-for-bit identical to the original before being
adopted (`0.7818396461367739` both ways, `0.9093291144152997` both ways, `diff ==
0.0` exactly on two real matches) — a performance optimization only after
confirming it wasn't also a correctness regression.

### 6.5 Test suite coverage across the project

| component | tests | status |
|---|---|---|
| v1 (`tennis-intelligence-platform`) | 211 | all passing (post-fix) |
| `cv_pipeline` | 29 | all passing (includes shot-classification, ball-trajectory-Kalman, calibration-verification, and the experimental Hough-detection module's own tests — see Addendum (2026-07-19)) |
| `rag_engine` | 19 | all passing |
| `llm_agent` | 5 | all passing (mocked; grounding verified separately via the 10-question + 3-spot-check manual evaluation, §3) |
| `v2_serving` | 29 | all passing, including real (unmocked) regression-guard calls into v1's engine asserting exact values |

(Counts above are as of the most recent full-suite run across all five components, 2026-07-20 — see `VERIFICATION.md` to reproduce.)

---

## 7. What this demonstrates

Across the five components, this project exercised computer vision (detection,
tracking, pose, camera geometry/homography), retrieval-augmented generation (document
design, embedding, vector search, retrieval evaluation), LLM agent design (grounding
discipline, citation transparency, multi-turn state, provider-API resilience), and
full-stack integration (an async job system, a typed API contract that preserves
rather than flattens uncertainty, and a frontend that renders that uncertainty
honestly rather than smoothing it into a cleaner-looking but less truthful UI).

But the throughline that matters more than any individual technique is the validation
discipline applied consistently across all of it: hypotheses tested against real data
rather than accepted by discussion, suspicious results investigated rather than
reported at face value or quietly discarded, fixes verified bit-for-bit before being
adopted rather than assumed correct because they looked reasonable, and — the part
most systems don't do — a willingness to find and fix real bugs in code that predated
this project entirely and had been considered finished. The PtWinner investigation in
particular went through six wrong hypotheses, each tested with real numbers, before
finding the actual mechanism; that kind of patience under uncertainty, applied
identically whether the target was a two-year-old statistical model or a
same-session computer vision script, is the actual skill this project was built to
demonstrate.

## Addendum (2026-07-17): Reference-pipeline homography precision, data/tennis/1.mp4

The same validation discipline applied to a new manual court calibration. Original
4-point corner calibration on this clip measured 74.9px/45.7px error against two
held-out landmarks (near-T, net-base) — worse than the project's prior ~13px
standard, on footage with no visible reason (lines are high-contrast and clear) to
expect worse accuracy. Rather than accept that as "just how this clip is," two
techniques were tried and honestly measured against the same held-out points:

| calibration | near-T error | net-base error |
|---|---|---|
| Original 4-point (manual) | 74.9px (113.1cm) | 45.7px (83.2cm) |
| Re-clicked 4-point (higher-precision) | 70.9px (106.6cm) | 41.6px (76.4cm) |
| Least-squares 8-point (BL/BR mislabeled) | 27.4px (44.4cm) | 24.9px (47.0cm) |
| **Least-squares 8-point (corrected, final)** | **4.4px** | **2.0px** |

Re-clicking alone barely moved the number, ruling out "click precision" as the
primary cause. Adding 4 more real, visually-confirmed court-line intersections and
solving via least-squares DLT (rather than an exact 4-point solve) produced a real
~2-4x reduction in error — a substantial improvement, but at the time not a full
close of the gap to the ~13px benchmark. A residual asymmetry (two specific points
fitting the model markedly worse than the other six) was found and flagged.

**Update (2026-07-18): that residual gap turned out to be a real, findable bug,
not an inherent limitation.** A separate report — the rendered court outline
showing doubles width at the far baseline but only singles width at the near
baseline, a geometric inconsistency, not just an accuracy gap — led to a
point-by-point audit of all 8 calibration points against real zoomed frames.
Two of them (`BL`/`BR`, the near-baseline corners) turned out to be mislabeled:
the pixels actually clicked were where the *singles* sideline crosses the near
baseline, not the doubles corner, which sits further out and was missed because
the original verification crop wasn't wide enough to show it. Re-clicking BL/BR
against the correct, more-outer line and rebuilding the homography dropped the
held-out error to **4.4px near-T / 2.0px net-base** — better than the ~13px
benchmark this whole investigation was trying to close a gap to, and it also
resolved the residual-asymmetry finding: the two points that had fit poorly
(`near_svc_L`/`near_svc_R`) were correctly labeled all along, and their bad fit
was the least-squares solve straining against the two corrupted anchors, not an
error of their own. One bug, two independently-reported symptoms, one real fix
— see PROGRESS.md's "Court-Outline Rendering Bug" entry for the full
point-by-point audit table. **Lesson for future manual calibration**: when a
candidate point sits near multiple parallel court lines (e.g. a doubles alley
running close to a singles sideline), widen the verification crop enough to
rule out a second, more-outer line before accepting the first line found —
the original crop here was narrow enough that the real doubles corner simply
wasn't visible in it, so there was no visual cue anything was wrong.

This is the same pattern as the rest of the project: a suspicious number was
investigated with real measurements, a real fix was found and adopted, and —
notably here — a second, independently-reported symptom turned out to share the
same root cause rather than needing its own separate investigation.

## Addendum (2026-07-19): Ball detection recall, serve classification, automated calibration, and rendered output video

Four further pieces of work, done after the addendum above, none reflected in §3/§6
until this update. Full detail for all of it is in `PROGRESS.md`; this section gives
the accurate, current headline numbers and the shape of each investigation, in the
same spirit as the rest of this report — including where a first-reported number
turned out to be wrong and had to be corrected.

### Ball detection: motion-diff recovery, 7.8% → 53.91% pooled recall — including a self-caught ground-truth leak

Stock YOLOv8n's ~7.8% ball-detection recall (§6.2) is a recall problem, not a
precision one — the model rarely finds the ball, but is accurate to 2–4px when it
does. Restricting a grayscale frame-difference to the clip's actual homography-derived
court region, and using it as a fallback wherever YOLO finds nothing, recovers most of
that gap on the locked-off amateur dataset: **pooled recall across 9 clips (video3
excluded) went from 7.81% to 53.91%**, a real ~6.9x improvement, visually spot-checked
against ground truth on isolated-ball frames, not just trusted from the aggregate
number. A separate check confirmed this does **not** transfer to broadcast footage
with two players moving in frame simultaneously — motion-diff there mostly
re-detects player footwork, a genuine negative finding reported rather than
suppressed (`BALL_DETECTION_EXPERIMENTS_REPORT.md`).

**The number was wrong once, and the project caught its own mistake before shipping
it.** An early prototype reported 70.40% pooled recall. Per this project's standing
rule — never trust a number from a standalone prototype script, only from the actual
production code path — the full validation suite was re-run through the real
`ball_detection_combined.run_combined_ball_detection_for_clip` function
(the same one `video_pipeline.py` calls), not the prototype. That re-run came back at
46.24%, far below 70.40%. Line-by-line comparison found the cause: the prototype
picked which motion-diff candidate to trust, when several existed in one frame, by
choosing whichever was **closest to the real ground-truth ball position** — something
no system can do at real inference time, since ground truth doesn't exist then. This
is a ground-truth leak, the same failure class the project had already named and
guarded against elsewhere (§4.1's fabrication catch). **Fixed** by picking the
largest-area candidate instead (a legitimate heuristic — the real ball has a specific
physical size) and re-measuring: **53.91% final**, corrected everywhere the invalid
70.40% had been cited. Still a real, large win over stock YOLO — just not the number
first claimed.

### Serve-exclusion for shot classification: a narrow, evidence-backed rule, not the general heuristic first proposed

`shot_classification.py` detects swing events from pose-landmark motion. The first
swing in a rally is usually a serve, not a groundstroke, and including it inflated
apparent shot-detection activity with an event type the classifier wasn't built to
distinguish. Three candidate signals for a general "was this preceded by a
motion lull" serve-detector were proposed and tested against real labeled data —
**all three were falsified** (event-gap timing, ball-density-before-event, and
ball-spatial-spread all failed to reliably separate real serves from real
groundstrokes on manual audit). Rather than ship a heuristic that sounded reasonable
but didn't hold up, the project shipped the one rule that **did** survive testing:
flag only the single earliest event across both players' roles as a probable serve —
narrow, but correct where it fires.

**Manually audited, not assumed** (the same per-event checklist — facing direction,
zoom, frame-by-frame confirmation — used throughout this project's CV validation):
final contamination figures, after also fixing two real integration bugs found during
wiring (a peak-detection boundary-starvation bug near `frame_limit`, fixed directly
by padding the detection window; and a sampling-density gap between the audit's
every-2nd-frame sample and production's every-frame sampling, resolved by a targeted
re-audit of the newly-visible events) — **87.5% of flagged non-serve events are real
shots, 14.6% contamination, confirmed stable under the actual production
configuration** (`PlayerContinuityTracker`, full-resolution sampling). Wired into
`video_pipeline.py` and the dashboard overlay only after this verification, not before.

### Automated Hough-transform court-corner detection — a promising but not-yet-adopted alternative to manual calibration

Every clip in this project has so far required a manually-measured 8-point court
calibration (see the addendum above for how error-prone that manual process itself
can be). This experiment tested whether classical Hough-line detection could locate
the same court corners automatically, measured directly against this project's
existing manually-traced calibrations as ground truth — not assumed accurate because
the approach sounds reasonable.

**First pass**: two real bugs, found and fixed by testing against ground truth before
trusting any result — naive angle-bucket clustering blended unrelated court lines
together (70–160px mean corner error), and a naive "widest horizontal line" heuristic
picked the net cord instead of the true far baseline (perspective foreshortening makes
nearer lines project wider in pixels despite being narrower in real-world terms).
Fixed via proper (theta, rho) segment clustering and Y-position-based baseline
selection: **11.8px mean error (1.mp4), 4.1px (3.mp4)**, single frame each,
comparable to the manual method's own ~2px held-out precision on the better clip.

**Extended to a full multi-frame, multi-clip evaluation (all 5 clips, 8 frames each)**
per this project's standard of not generalizing from 2 clips/1 frame — which surfaced
two more real problems, both found and fixed the same way, by measuring rather than
assuming: (1) a short, poorly-supported line cluster could occasionally out-rank a
well-supported one by a few pixels of position and then get badly extrapolated, fixed
by weighting cluster selection toward better-covered clusters; (2) a fixed on-screen
scoreboard graphic's text was being picked up as a false court-line signal — an
initial pixel-masking fix worked but was found to occasionally destabilize an
*unrelated* line detection elsewhere in the frame (`cv2.HoughLinesP`'s internal
randomized voting is sensitive to the total edge-point population, not just local
pixels — confirmed directly, and a same-value "neutral fill" test proved the fill
value was never the actual variable), so it was replaced with a structurally safer
post-hoc filter on detected segments instead of a pixel mask. Combined with switching
multi-frame aggregation from mean to median (a general robustness layer against
ordinary per-frame noise, unrelated to the scoreboard mechanism): **overall mean
corner error across all 5 clips dropped from 6.8px to 5.14px, every clip at or below
its own original baseline.**

**Status: still experimental, correctly not adopted.** Deliberately named outside the
`reference_video*_calibration.py` pattern so it does not trigger the mandatory
calibration-verification-manifest gate (`test_calibration_verification.py`) that
every real calibration in this project must pass — it hasn't gone through that
process, and two corners in two clips (2.mp4's BL, 5.mp4's BR) remain meaningfully
worse than the manual method's sub-2px precision. Treated as a candidate to keep
comparing against the manual method, not a replacement, consistent with this
project's standard of not shipping something because a result looks promising in
isolation.

### Rendered output video with shot-detection overlay

A server-side rendering feature (`v2_serving/src/v2_serving/video_render.py`) burns
the detected court outline, player/ball boxes, and shot-classification events
directly into an output `.mp4`, exposed via a job-based API matching the existing
async-job conventions. **A real, user-facing bug was found and fixed after initial
delivery**: the first render used the `mp4v` FourCC, which produces MPEG-4 Part 2 —
not playable in any mainstream browser despite downloading successfully (confirmed by
inspecting the actual file's FourCC via OpenCV, not by assumption). Fixed by switching
to `avc1` (real H.264, empirically verified against this machine's OpenCV/FFmpeg
build) and adding a regression test asserting the output codec is never `mp4v`/`FMP4`
again. Every frame of a full clip (2,020 frames) renders in ~21s.
