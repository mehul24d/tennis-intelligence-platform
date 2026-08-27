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
from urllib.parse import quote, unquote

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "pipelines"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _rss_mb() -> float:
    """Current process resident memory in MB — used only in startup diagnostic
    prints. resource.ru_maxrss is bytes on macOS but KB on Linux (where this actually
    deploys), so this isn't hardcoded to one or the other."""
    import platform
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024 / 1024 if platform.system() == "Darwin" else raw / 1024

from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.ml_informed_markov import ServeReturnPosterior, build_pretrained_prior
from tennis_intel.live.hybrid_engine import hybrid_predict
from tennis_intel.live.markov_baseline import prob_win_match
from tennis_intel.viz.trajectory_events import detect_set_boundaries

from pipelines.replay_match import (
    markov_p_player1, ml_p_player1, ml_informed_markov_p_player1,
    ml_informed_markov_p_player1_unsmoothed, find_match,
    PROCESSED, ROLLOUT_MODEL_NAME,
)
from pipelines.generate_publication_trajectory import (
    compute_composite_prematch_probability, compute_ml_pre_match_probability,
)

# Per-match point-level data, pre-built ONCE (see pipelines/build_points_cache.py) and
# partitioned by match_id on disk — see the "memory footprint" note on
# load_replay_context below for why the live server reads one match's ~175 rows on
# demand here rather than holding all ~1M rows for all ~6000 matches in RAM at once.
POINTS_CACHE_DIR = PROCESSED / "points_by_match"


@dataclass
class ReplayContext:
    """Everything the replay computation needs, loaded ONCE and reused across every
    request — the classifier and feature columns. day6 is also kept here (not just
    frozen_join) so tennis_intel.serving.match_list_service can build its own
    enriched match table without re-reading the same parquet file a second time.
    Per-match point-level data is NOT held here — see POINTS_CACHE_DIR — only the set
    of match_ids that exist is kept, for existence/search checks."""
    model: object
    feature_cols: list[str]
    frozen_join: pd.DataFrame
    day6: pd.DataFrame
    match_ids: set[str] = field(default_factory=set)


def _load_match_points(match_id: str) -> pd.DataFrame:
    """Reads ONE match's point-level rows from the partitioned cache on demand — reads
    that match's specific partition file directly (its path is fully determined by
    match_id) rather than a filtered whole-dataset read. That matters beyond just
    speed: pyarrow's whole-dataset read resolves ONE schema across all ~5981
    partitions, and a minority of them (see build_trajectory_cache.py) have extra
    precomputed-trajectory columns the rest don't — a dataset-wide read silently drops
    those columns as "not in the common schema" even when filtered down to a match
    that has them (measured: the cache-hit short-circuit below never triggered until
    this was fixed to read the single file directly). match_id itself isn't stored in
    the partition file (it's encoded only in the directory name, per how
    build_points_cache.py wrote it), so it's added back here.
    """
    match_dir = POINTS_CACHE_DIR / f"match_id={quote(match_id, safe='')}"
    files = sorted(match_dir.glob("*.parquet"))
    # Debug (temporary — tracking down a parquet-read failure seen only on the actual
    # deployment, not reproducible locally): report exactly what was found and each
    # candidate file's size before attempting to read it, since the failure gives no
    # indication of which file or why.
    if not files:
        raise RuntimeError(
            f"No parquet files found in {match_dir} (exists={match_dir.exists()}, "
            f"dir contents: {list(match_dir.iterdir()) if match_dir.exists() else 'N/A'})"
        )
    sizes = [(f, f.stat().st_size) for f in files]
    try:
        df = pd.read_parquet(files[0])
    except Exception as e:
        raise RuntimeError(f"Failed reading {files[0]} (sizes found: {sizes}): {e}") from e
    df["match_id"] = match_id
    return df


def _shrink_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts a DataFrame's dtypes purely to reduce memory footprint, for a frame
    that's already done all its precision-sensitive work (feeding a model, feeding
    arithmetic that compounds over many steps) and is kept around only to be read back
    out for display/aggregation. float64 -> float32 (~7 significant digits, invisible
    at any percentage/rating display precision); object columns with meaningfully
    fewer unique values than rows -> category (repeated strings like tourney names,
    surfaces, rounds stored once instead of per-row). Mutates columns of the passed-in
    frame in place (one column at a time, each old array freed as soon as replaced)
    rather than copying the whole frame first, which would transiently double its
    memory right when the goal is to shrink it — the caller is expected to discard its
    own reference to the pre-shrink frame (`day6 = _shrink_for_display(day6)`).
    """
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "object":
            n = len(df)
            if n and df[col].nunique(dropna=True) < 0.5 * n:
                df[col] = df[col].astype("category")
    return df


def load_full_day6() -> pd.DataFrame:
    """
    Loads day6, independent of ReplayContext.day6 (which only holds the 7 columns
    compute_five_engine_trajectory's tourney lookup needs), for
    career_stats_service's rankings/profile/match-explorer endpoints — called lazily,
    on first request to one of those routes (see
    api/routers/match_list.py's get_career_stats_context/get_match_list_context),
    NOT at server startup.

    MEMORY FOOTPRINT (2026-08): this used to load all 294 columns (via
    _read_parquet_shrunk's column-chunked read) — ~235MB even fully downcast, which
    reproducibly OOM-killed the live Render deployment the moment this lazy load was
    actually triggered (a full app boot was already using most of the 512MB free-tier
    ceiling before this ever ran; the chunked read bounds TRANSIENT overhead during
    loading, but the final 235MB frame still has to coexist with everything already
    resident, and that alone was enough). But match_list_service.py and
    career_stats_service.py (this function's only two callers) between them read only
    17 of those 294 columns — verified via a full read of both files and everything
    they call. So this loads exactly that column list directly (a plain
    pd.read_parquet(columns=[...]), no chunking needed at this size) rather than the
    full table. Adding a new field to either service that needs another day6 column
    means adding it here too — pd.read_parquet(columns=...) raises a clear KeyError
    immediately if a service tries to read a column that wasn't requested here.
    """
    columns = [
        "tourney_id", "match_num", "winner_id", "loser_id",
        "winner_name", "loser_name", "tourney_date", "tourney_name",
        "surface", "round", "tourney_level", "score", "best_of",
        "elo_pre_match_winner", "elo_pre_match_loser",
        "elo_surface_pre_match_winner", "elo_surface_pre_match_loser",
    ]
    return _shrink_for_display(
        pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet", columns=columns)
    )


def load_replay_context() -> ReplayContext:
    """
    Loads the trained classifier ONCE — call this exactly once, at API server startup
    (e.g. in a FastAPI lifespan/startup event), and pass the resulting ReplayContext
    into every replay_match_by_id() call.

    MEMORY FOOTPRINT (2026-08): this used to also build the full point-level dataset
    for ALL ~6000 matches (via build_point_dataset) and hold it in ctx.points for the
    life of the process — measured at ~590MB resident, on top of day6 and the rest,
    enough to OOM-kill a 512MB deployment. But per-request, compute_five_engine_trajectory
    only ever needs ONE match's ~175 points at a time. So the full dataset is now built
    exactly once, OFFLINE (see pipelines/build_points_cache.py — same build_point_dataset
    call, same output, just persisted instead of held in RAM), partitioned by match_id
    on disk at POINTS_CACHE_DIR. The live server reads only the requested match's
    partition per request (_load_match_points) instead of holding the whole corpus —
    verified via a before/after diff of replay_match_by_id output across 5 real matches
    spanning 1969-2023 that this changes no computed probability, since the underlying
    build_point_dataset computation itself is byte-for-byte unchanged, only where its
    output lives.

    RETRAINED 2026-07-15 on features computed under the corrected, literal PtWinner
    convention (see docs/ptwinner_convention_correction.md's "Retrain results" section
    for the full before/after comparison — rolling-origin log_loss improved from
    0.6281 to 0.6247, Brier from 0.2187 to 0.2172, consistently across all four
    2022-2025 folds; top-4 SHAP features unchanged in rank). The prior (pre-retrain)
    classifier is preserved at day9_point_classifiers_PRE_PTWINNER_FIX.joblib.
    """
    # Timed + flushed: deploying to a CPU/disk-throttled free-tier instance made
    # startup take far longer than the ~2s measured locally, with no visibility into
    # which step was slow — this pins it down precisely in the deploy log instead of
    # guessing from macOS timings again. Also prints real RSS at each step: local
    # macOS measurements have already been wrong twice for estimating Render's actual
    # Linux container behavior (once too high via phys-footprint, once apparently too
    # low given how much slower every step ran there) — actual numbers from the
    # deployment itself are worth more than any more local guessing.
    import os
    import time as _time
    _t0 = _time.monotonic()

    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload[ROLLOUT_MODEL_NAME], payload["feature_cols"]
    print(f"[startup] model loaded: {_time.monotonic() - _t0:.1f}s, {_rss_mb():.0f}MB", flush=True)

    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    print(f"[startup] frozen_join loaded: {_time.monotonic() - _t0:.1f}s, {_rss_mb():.0f}MB", flush=True)

    # Memory trim (2026-08, real numbers from the deployment itself): day6 is 294
    # columns x 198k ATP matches (646MB at full precision, ~235MB even after full
    # downcasting) — but compute_five_engine_trajectory's tourney-name/date/score
    # lookup below only ever reads 7 of those columns (the 4 join keys plus
    # tourney_name/tourney_date/score). The other 287 exist purely for
    # career_stats_service's rankings/profile/match-explorer endpoints, which are
    # ALREADY lazily loaded on first request to those specific routes (see
    # api/routers/match_list.py's get_career_stats_context/get_match_list_context) —
    # but were still reading the SAME eagerly-loaded, full-294-column ctx.day6 rather
    # than loading their own. Measured on the real deployment: loading all 294
    # columns here, even fully downcast, pushed a full app boot (FastAPI + all
    # routers/schemas add their own baseline, measured ~591MB total locally when
    # reproduced by booting api.main:app directly rather than testing
    # load_replay_context() in isolation) well past the 512MB free-tier ceiling. So
    # ctx.day6 now holds ONLY those 7 columns (a few hundred KB, no chunking needed);
    # get_career_stats_context/get_match_list_context now load a fresh, full,
    # independently-shrunk day6 (_load_full_day6 below) on first request to those
    # routes instead of reusing this minimal one — deferring their memory cost to
    # first actual use of those secondary features rather than paying it for every
    # server boot regardless of whether anyone visits rankings/profile at all.
    day6 = pd.read_parquet(
        PROCESSED / "matches_with_day6_features.parquet",
        columns=["tourney_id", "match_num", "winner_id", "loser_id",
                 "tourney_name", "tourney_date", "score"],
    )
    print(f"[startup] day6 (minimal, 7 cols) loaded: {_time.monotonic() - _t0:.1f}s, "
          f"{_rss_mb():.0f}MB", flush=True)

    # Cheap: just the partition directory names, not a single row of point data read.
    # os.listdir (name only, no per-entry stat) rather than Path.iterdir()+is_dir()
    # (a stat syscall per entry) — cheap locally, but ~6000 stat calls could matter on
    # a slower/network-backed disk.
    match_ids = {unquote(name.split("=", 1)[1]) for name in os.listdir(POINTS_CACHE_DIR)
                 if name.startswith("match_id=")}
    print(f"[startup] match_ids listed ({len(match_ids)}): {_time.monotonic() - _t0:.1f}s, "
          f"{_rss_mb():.0f}MB", flush=True)

    return ReplayContext(
        model=model, feature_cols=feature_cols, frozen_join=frozen_join, day6=day6,
        match_ids=match_ids,
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
    return sorted(m for m in ctx.match_ids if all(t.lower() in m.lower() for t in search_terms))


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

    match_df = _load_match_points(match_id).sort_values("Pt").reset_index(drop=True)
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

    # Precomputed-trajectory cache (see pipelines/build_trajectory_cache.py): the
    # per-point loop below is dominated by ml_p_player1's Monte Carlo simulation
    # (~300ms/point measured), too slow for a live free-tier request across a full
    # match — but fully deterministic (ml_p_player1 is seeded by point index), so for
    # a curated set of matches (Grand Slam finals) the five probability lists and
    # ml_informed_prematch_p1 are computed once offline and stored as extra columns
    # on that match's cached partition. If present, skip straight to the same return
    # shape the live computation below would produce — same values, just already done.
    if "markov_p1" in match_df.columns:
        records = match_df.to_dict("records")
        return {
            "match_df": match_df, "records": records,
            "final_winner_is_p1": final_winner_is_p1,
            "p1_name": p1_name, "p2_name": p2_name,
            "tournament": tournament, "tourney_date": tourney_date, "final_score": final_score,
            "ml_informed_prematch_p1": float(match_df["ml_informed_prematch_p1_cached"].iloc[0]),
            "markov_p1": match_df["markov_p1"].tolist(),
            "ml_p1": match_df["ml_p1"].tolist(),
            "ml_informed_p1": match_df["ml_informed_p1"].tolist(),
            "ml_informed_unsmoothed_p1": match_df["ml_informed_unsmoothed_p1"].tolist(),
            "hybrid_p1": match_df["hybrid_p1"].tolist(),
        }

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