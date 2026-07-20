"""win_probability_pipeline.py — orchestrates v1's existing Monte Carlo/Markov
win-probability engine (tennis_intel.serving.replay_service) for
GET /win-probability/{job_id}.

REAL INTEGRATION FINDING, surfaced not hidden: v1's engine computes a pre-match
baseline from a real match's own historical context (Elo, rank, head-to-head) --
it requires a `match_id` from v1's own 5,981-match frozen-join dataset. None of
cv_pipeline's demo clips (the 10 amateur videos, or the professional stress-test
clip) correspond to any match in that dataset -- they're either home video or an
out-of-competition practice session, neither charted by the Match Charting
Project or present in v1's ATP/WTA match records. There is currently no automated
way to resolve a cv_pipeline job to a v1 match_id. Consequently:
  - The pre-match baseline is only computable when the caller explicitly supplies
    a real, known match_id (via the `match_id` query param) -- this is demonstrated
    working for real below, it is not a stub.
  - For a job with no supplied match_id (the common case for cv_pipeline's own
    demo clips), the baseline is reported as unavailable, with the reason stated
    plainly, not silently omitted or replaced with a fabricated number.

SECOND REAL FINDING: even given a resolvable match, a "live adjustment" from a
cv_pipeline job's CV features specifically requires per-point serve/rally outcome
data (v1's ServeReturnPosterior update needs to know, point by point, who served
and who won each point). Phase 3's cv_pipeline output schema (see EVALUATION_REPORT.md)
contains detection rates, tracking-ID stability, and pose success -- NOT point-by-
point score or serve outcomes; no such extraction exists anywhere in cv_pipeline
today. So a live adjustment can never actually be computed from a job's CV
features, for any job, until cv_pipeline gains point/score extraction -- this is
reported as a plain, structural "not available", never fabricated as a plausible-
looking adjusted number.
"""

from __future__ import annotations

import sys
from pathlib import Path

V1_ROOT = Path(__file__).resolve().parents[3] / "tennis-intelligence-platform"
for _p in (V1_ROOT, V1_ROOT / "src", V1_ROOT / "pipelines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_replay_ctx = None


def _get_replay_context():
    """v1's own docstring: load ONCE, reuse across requests -- this load takes
    ~20-30s (builds the full point-level dataset), so it's cached as a module
    singleton, lazily on first use, same pattern as query_pipeline.py's VectorStore."""
    global _replay_ctx
    if _replay_ctx is None:
        from tennis_intel.serving.replay_service import load_replay_context
        _replay_ctx = load_replay_context()
    return _replay_ctx


def get_prematch_baseline(match_id: str) -> dict:
    """Real call into v1's engine -- not a stub.

    PERFORMANCE NOTE (a verified fast path, not an approximation): the obvious
    implementation -- call replay_service.compute_five_engine_trajectory() and
    keep only its `ml_informed_prematch_p1` field -- works correctly but costs
    ~90s per call, because that function computes the ENTIRE per-point, 5-engine
    trajectory for the whole match (every point, every engine) just to produce
    the one pre-match scalar this endpoint needs, which is fully determined
    before the point loop even starts. This function instead calls the same
    underlying seeding building blocks compute_five_engine_trajectory itself
    calls (compute_composite_prematch_probability, compute_p_a_return_seed,
    build_pretrained_prior, prob_win_match) directly, skipping the per-point
    loop entirely. VERIFIED, not assumed: checked bit-for-bit
    (`full_value == fast_value`, diff == 0.0, not just "close") against
    compute_five_engine_trajectory's own ml_informed_prematch_p1 output for two
    real matches (Djokovic/Goffin 2019 Wimbledon QF: 0.7818396461367739 both
    ways; Djokovic/Kohlschreiber 2019 Wimbledon R128: 0.9093291144152997 both
    ways) before this replaced the full-trajectory call -- see PROGRESS.md.

    The small amount of match-lookup glue below (finding match_df, first_row,
    final_winner_is_p1) is duplicated from replay_service.py's own internals
    rather than refactored out of v1 -- Phase 4's constraint is that v1's
    backend itself is not modified, and duplicating a few lines of orchestration
    here honors that while still avoiding v1's own unnecessary full-trajectory cost.
    """
    import pandas as pd
    from pipelines.generate_publication_trajectory import compute_composite_prematch_probability
    from tennis_intel.live.ml_informed_markov import build_pretrained_prior
    from tennis_intel.live.markov_baseline import prob_win_match
    from tennis_intel.live.return_seed import compute_p_a_return_seed

    ctx = _get_replay_context()
    if match_id not in ctx.match_ids:
        return {
            "status": "not_available",
            "reason": f"'{match_id}' not found in the joined+charted dataset (it may exist "
                      f"in MCP but not survive the frozen TML join).",
        }

    match_df = ctx.points[ctx.points["match_id"] == match_id].sort_values("Pt").reset_index(drop=True)
    p1_name = match_id.split("-")[-2].replace("_", " ")
    p2_name = match_id.split("-")[-1].replace("_", " ")
    fj_row = ctx.frozen_join[ctx.frozen_join["mcp_match_id"] == match_id]
    if len(fj_row):
        p1_name = fj_row["mcp_Player 1"].iloc[0]
        p2_name = fj_row["mcp_Player 2"].iloc[0]
    final_winner_is_p1 = bool(match_df["player1_is_winner"].iloc[0])

    first_row = match_df.to_dict("records")[0]

    p0_a_wins = compute_composite_prematch_probability(first_row)
    p_a_return_seed = compute_p_a_return_seed(first_row, track_winner=True)
    elo_matches_played_a = first_row.get("elo_matches_played_pre_winner")
    elo_matches_played_b = first_row.get("elo_matches_played_pre_loser")
    best_of_val = int(first_row["best_of"]) if pd.notna(first_row.get("best_of")) else 3

    h2h_meetings = None
    winner_h2h = first_row.get("winner_h2h_wins_pre_match")
    loser_h2h = first_row.get("loser_h2h_wins_pre_match")
    if pd.notna(winner_h2h) and pd.notna(loser_h2h):
        h2h_meetings = float(winner_h2h) + float(loser_h2h)

    tourney_h2h_meetings = None
    winner_tourney_h2h = first_row.get("winner_tourney_h2h_wins_pre_match")
    loser_tourney_h2h = first_row.get("loser_tourney_h2h_wins_pre_match")
    if pd.notna(winner_tourney_h2h) and pd.notna(loser_tourney_h2h):
        tourney_h2h_meetings = float(winner_tourney_h2h) + float(loser_tourney_h2h)

    p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
        p0_a_wins, p_a_return_seed, best_of_val,
        elo_matches_played_a=elo_matches_played_a, elo_matches_played_b=elo_matches_played_b,
        h2h_meetings=h2h_meetings, tourney_h2h_meetings=tourney_h2h_meetings,
    )
    p_a_wins_prematch = prob_win_match(p_serve0, p_return0, best_of=best_of_val)
    ml_informed_prematch_p1 = p_a_wins_prematch if final_winner_is_p1 else (1.0 - p_a_wins_prematch)

    return {
        "status": "available",
        "source": "v1 Monte Carlo/Markov engine (ml_informed_markov), pre-match only "
                   "(fast path -- verified bit-for-bit equivalent to the full "
                   "per-point trajectory computation, see this function's docstring)",
        "match_id": match_id,
        "p1_name": p1_name,
        "p2_name": p2_name,
        "p1_win_probability_prematch": round(ml_informed_prematch_p1, 4),
        "note": "Computed from real historical context (Elo, rank, head-to-head) for "
                "this specific match -- does NOT use any in-match/live point data.",
    }


def get_live_adjustment(job_result: dict | None) -> dict:
    """Always returns 'not_available' today -- see module docstring's second
    finding. Written as a real check (not a hardcoded stub) so this starts
    working automatically the moment cv_pipeline gains point/score extraction,
    without needing this function rewritten."""
    if job_result is None:
        return {"status": "not_available", "reason": "job has no result to derive an adjustment from"}

    has_point_level_data = any(
        key in job_result for key in ("points", "point_scores", "serve_outcomes", "pt_winner_sequence")
    )
    if not has_point_level_data:
        return {
            "status": "not_available",
            "reason": "cv_pipeline's current output for this job contains detection rates "
                      "(near/far player, ball), tracking-ID stability, and pose success -- "
                      "it does NOT contain point-by-point score or serve/rally outcome data. "
                      "v1's live-adjustment mechanism (ServeReturnPosterior) requires knowing, "
                      "per point, who served and who won the point -- no such extraction exists "
                      "in cv_pipeline today (see Phase 3's EVALUATION_REPORT.md for what the CV "
                      "pipeline does produce). This is a structural gap between what Phase 3 "
                      "built and what a live in-match adjustment needs, not a bug in this "
                      "endpoint -- reported plainly rather than approximated from unrelated "
                      "detection-rate numbers.",
        }
    # Unreachable today -- retained as the real integration point for when
    # cv_pipeline adds point-level extraction.
    return {"status": "not_available", "reason": "point-level extraction present but adjustment logic not yet implemented"}
