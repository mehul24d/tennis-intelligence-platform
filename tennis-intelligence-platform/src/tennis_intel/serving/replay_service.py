"""
replay_service.py — the core service layer behind the future FastAPI endpoints for
match replay and the publication probability chart. Extracts pipelines/replay_match.py's
per-match computation into clean, reusable, JSON-serializable functions, rather than
duplicating any of that already-built, already-bug-fixed logic.

REUSES, DOES NOT REIMPLEMENT: every prediction call here (markov_p_player1,
ml_p_player1, ml_informed_markov_p_player1, ml_informed_markov_p_player1_unsmoothed,
hybrid_predict), every seeding step (compute_composite_prematch_probability,
compute_p_a_return_seed, build_pretrained_prior), and the set-boundary detection
(detect_set_boundaries) are the EXACT SAME already-validated functions
pipelines/replay_match.py calls — this module only reorganizes the orchestration around
them into something importable and JSON-friendly, matching this project's standing
discipline of never re-deriving already-tested logic.

The heavy, one-time setup (loading the classifier, building the full point-level
dataset for ALL matches) is separated into load_replay_context(), meant to be called
ONCE at API server startup and reused across every request — rebuilding the full
point-level dataset on every single request would be far too slow for a live API.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "pipelines"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.ml_informed_markov import ServeReturnPosterior, build_pretrained_prior
from tennis_intel.live.hybrid_engine import hybrid_predict
from tennis_intel.live.markov_baseline import prob_win_match
from tennis_intel.viz.trajectory_events import detect_set_boundaries

from pipelines.replay_match import (
    markov_p_player1, ml_p_player1, ml_informed_markov_p_player1,
    ml_informed_markov_p_player1_unsmoothed, find_match,
    PROCESSED, POINT_FILES, ROLLOUT_MODEL_NAME,
)
from pipelines.generate_publication_trajectory import (
    compute_composite_prematch_probability, compute_ml_pre_match_probability,
)


@dataclass
class ReplayContext:
    """Everything the replay computation needs, loaded ONCE and reused across every
    request — the classifier, feature columns, and the full point-level dataset for
    every match in the frozen-join corpus. day6 is also kept here (not just
    frozen_join) so tennis_intel.serving.match_list_service can build its own
    enriched match table without re-reading the same parquet file a second time."""
    model: object
    feature_cols: list[str]
    frozen_join: pd.DataFrame
    day6: pd.DataFrame
    points: pd.DataFrame
    match_ids: set[str] = field(default_factory=set)


def load_replay_context() -> ReplayContext:
    """
    Loads the trained classifier and builds the full point-level dataset ONCE — call
    this exactly once, at API server startup (e.g. in a FastAPI lifespan/startup
    event), and pass the resulting ReplayContext into every replay_match_by_id() call.
    This is the same one-time cost pipelines/replay_match.py pays at the top of its
    own main(), just factored out so a live server doesn't repeat it per request.

    RETRAINED 2026-07-15 on features computed under the corrected, literal PtWinner
    convention (see docs/ptwinner_convention_correction.md's "Retrain results" section
    for the full before/after comparison — rolling-origin log_loss improved from
    0.6281 to 0.6247, Brier from 0.2187 to 0.2172, consistently across all four
    2022-2025 folds; top-4 SHAP features unchanged in rank). The prior (pre-retrain)
    classifier is preserved at day9_point_classifiers_PRE_PTWINNER_FIX.joblib.
    """
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload[ROLLOUT_MODEL_NAME], payload["feature_cols"]

    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]

    return ReplayContext(
        model=model, feature_cols=feature_cols, frozen_join=frozen_join, day6=day6,
        points=points, match_ids=set(points["match_id"].unique()),
    )


def list_available_match_ids(ctx: ReplayContext) -> list[str]:
    """All match_ids that can actually be replayed — i.e. survive the frozen TML/MCP
    join, matching exactly what pipelines/replay_match.py itself checks before
    attempting a replay."""
    return sorted(ctx.match_ids)


def search_match_ids(ctx: ReplayContext, search_terms: list[str]) -> list[str]:
    """Thin wrapper around replay_match.py's own find_match logic, but returning ALL
    matches (not raising on multiple matches, since an API caller should get a list to
    choose from, not a CLI-style error)."""
    ids = ctx.points["match_id"].unique()
    return sorted(m for m in ids if all(t.lower() in m.lower() for t in search_terms))


def compute_five_engine_trajectory(ctx: ReplayContext, match_id: str) -> dict:
    """
    Shared core computation used by replay_match_by_id, get_match_summary,
    get_model_agreement, and get_point_timeline — factored out here specifically to
    avoid a THIRD independent copy of the same seeding + five-engine per-point loop
    (match_summary_service.py already had its own separate copy before this refactor;
    a third copy for model-agreement/point-timeline would have meant three places
    that could silently drift out of sync with each other over time).

    Raises ValueError if match_id isn't in the frozen-join corpus.

    Returns a dict with: match_df (the raw, full point-level dataframe — includes
    is_break_point/Svr/PtWinner/etc., NOT just the slimmed engine-probability columns
    the old replay_match_by_id output exposed), final_winner_is_p1, p1_name, p2_name,
    ml_informed_prematch_p1, and five parallel lists (markov_p1, ml_p1,
    ml_informed_p1, ml_informed_unsmoothed_p1, hybrid_p1) — one entry per point, in
    the same order as match_df's own rows.
    """
    if match_id not in ctx.match_ids:
        raise ValueError(
            f"'{match_id}' not found in the joined+charted dataset (it may exist in "
            f"MCP but not survive the frozen TML join)."
        )

    match_df = ctx.points[ctx.points["match_id"] == match_id].sort_values("Pt").reset_index(drop=True)
    p1_name = match_id.split("-")[-2].replace("_", " ")
    p2_name = match_id.split("-")[-1].replace("_", " ")
    fj_row = ctx.frozen_join[ctx.frozen_join["mcp_match_id"] == match_id]
    tournament, tourney_date, final_score = None, None, None
    if len(fj_row):
        p1_name = fj_row["mcp_Player 1"].iloc[0]
        p2_name = fj_row["mcp_Player 2"].iloc[0]
        # Tournament/date/score aren't in frozen_join itself — look them up in day6
        # via the SAME (tourney_id, match_num, winner_id, loser_id) composite key
        # already used elsewhere in this project (career_stats_service.py's
        # get_full_match_list, build_point_dataset.py's own join) to link an MCP
        # match back to its TML match-level row.
        tml_tourney_id = fj_row["tml_tourney_id"].iloc[0]
        tml_match_num = fj_row["tml_match_num"].iloc[0]
        tml_winner_id = fj_row["tml_winner_id"].iloc[0]
        tml_loser_id = fj_row["tml_loser_id"].iloc[0]
        day6_row = ctx.day6[
            (ctx.day6["tourney_id"] == tml_tourney_id)
            & (ctx.day6["match_num"] == tml_match_num)
            & (ctx.day6["winner_id"] == tml_winner_id)
            & (ctx.day6["loser_id"] == tml_loser_id)
        ]
        if len(day6_row):
            tournament = day6_row["tourney_name"].iloc[0]
            date_val = day6_row["tourney_date"].iloc[0]
            tourney_date = date_val.isoformat() if pd.notna(date_val) else None
            score_val = day6_row["score"].iloc[0]
            final_score = score_val if pd.notna(score_val) else None
    final_winner_is_p1 = bool(match_df["player1_is_winner"].iloc[0])

    records = match_df.to_dict("records")
    first_row = records[0]

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
    posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)

    p_a_wins_prematch = prob_win_match(p_serve0, p_return0, best_of=best_of_val)
    ml_informed_prematch_p1 = p_a_wins_prematch if final_winner_is_p1 else (1.0 - p_a_wins_prematch)

    markov_p1, ml_p1, ml_informed_p1, ml_informed_unsmoothed_p1, hybrid_p1 = [], [], [], [], []

    for i, row in enumerate(records):
        p_markov = markov_p_player1(row)
        p_ml_mc = ml_p_player1(row, ctx.model, ctx.feature_cols, rng_seed=i)
        p_ml_informed, posterior = ml_informed_markov_p_player1(
            row, ctx.model, ctx.feature_cols, posterior)
        p_ml_informed_unsmoothed = ml_informed_markov_p_player1_unsmoothed(
            row, ctx.model, ctx.feature_cols)
        p_hybrid = hybrid_predict(markov_p=p_markov, ml_mc_p=p_ml_mc)

        markov_p1.append(p_markov)
        ml_p1.append(p_ml_mc)
        ml_informed_p1.append(p_ml_informed)
        ml_informed_unsmoothed_p1.append(p_ml_informed_unsmoothed)
        hybrid_p1.append(p_hybrid)

    return {
        "match_df": match_df, "records": records,
        "final_winner_is_p1": final_winner_is_p1,
        "p1_name": p1_name, "p2_name": p2_name,
        "tournament": tournament, "tourney_date": tourney_date, "final_score": final_score,
        "ml_informed_prematch_p1": ml_informed_prematch_p1,
        "markov_p1": markov_p1, "ml_p1": ml_p1,
        "ml_informed_p1": ml_informed_p1,
        "ml_informed_unsmoothed_p1": ml_informed_unsmoothed_p1,
        "hybrid_p1": hybrid_p1,
    }


def replay_match_by_id(ctx: ReplayContext, match_id: str) -> dict:
    """
    Runs the full, exact same five-engine replay computation as
    pipelines/replay_match.py's main(), for ONE match, returning a JSON-serializable
    dict instead of printing to stdout / writing a CSV+PNG.

    Every prediction and seeding step is a DIRECT call to the same, already-validated
    functions replay_match.py itself calls, via compute_five_engine_trajectory above
    (shared with match_summary_service.py, model_agreement_service.py, and
    point_timeline_service.py — see that function's own docstring for why this was
    factored out rather than left duplicated across four places).
    """
    computed = compute_five_engine_trajectory(ctx, match_id)
    records = computed["records"]
    pts = [int(row["Pt"]) for row in records]
    set1_vals = [int(row["Set1"]) for row in records]
    set2_vals = [int(row["Set2"]) for row in records]
    gm1_vals = [int(row["Gm1"]) for row in records]
    gm2_vals = [int(row["Gm2"]) for row in records]

    boundary_lookup_df = pd.DataFrame({
        "point_index": pts, "Set1": set1_vals, "Set2": set2_vals,
        "Gm1": gm1_vals, "Gm2": gm2_vals,
    })
    boundaries = detect_set_boundaries(boundary_lookup_df)

    return {
        "match_id": match_id,
        "player1": {"name": computed["p1_name"]}, "player2": {"name": computed["p2_name"]},
        "winner": computed["p1_name"] if computed["final_winner_is_p1"] else computed["p2_name"],
        "n_points": len(records),
        "tournament": computed["tournament"], "date": computed["tourney_date"],
        "final_score": computed["final_score"],
        "prematch": {
            "markov": None, "ml_mc": None, "ml_informed_unsmoothed": None,
            "ml_informed_smoothed": round(computed["ml_informed_prematch_p1"], 6),
            "hybrid": None,
        },
        "points": [
            {
                "point_index": pts[i],
                "set1": set1_vals[i], "set2": set2_vals[i],
                "gm1": gm1_vals[i], "gm2": gm2_vals[i],
                "markov_p1": round(computed["markov_p1"][i], 6),
                "ml_mc_p1": round(computed["ml_p1"][i], 6),
                "ml_informed_unsmoothed_p1": round(computed["ml_informed_unsmoothed_p1"][i], 6),
                "ml_informed_smoothed_p1": round(computed["ml_informed_p1"][i], 6),
                "hybrid_p1": round(computed["hybrid_p1"][i], 6),
            }
            for i in range(len(records))
        ],
        "set_boundaries": [
            {
                "set_number": b.set_number, "point_index": b.point_index,
                "score": b.score_str, "winner_is_p1": b.winner_is_p1,
            }
            for b in boundaries
        ],
    }