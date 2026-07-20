"""
replay_match.py — runs both the Markov and ML+MC engines point-by-point on ONE specified
charted match, producing a full win-probability trajectory (point, game, and set markers)
for both players. Built for exactly this use case: "show me Sinner vs Alcaraz, 2025 Roland
Garros final, point by point."

Reuses the exact same engines, feature construction, and dynamic rollout validated in
Days 8-11 — this is not a new model, just a single-match lens on the already-frozen
evaluation pipeline (evaluate_live_engines_v2.py).

Usage:
    python pipelines/replay_match.py --match-id "20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz"
    python pipelines/replay_match.py --search "Sinner" "Alcaraz" "Roland_Garros"
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.live_win_probability import MatchState, prob_a_wins_match_from_state
from tennis_intel.live.match_state_conversion import row_to_match_state
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.monte_carlo_engine import batch_simulate_dynamic
from tennis_intel.live.markov_baseline import prob_win_match
from tennis_intel.live.ml_informed_markov import (
    ml_informed_markov_predict, ml_informed_point_probabilities, ServeReturnPosterior,
    build_pretrained_prior,
)
from tennis_intel.live.hybrid_engine import hybrid_predict
from tennis_intel.viz.trajectory_events import detect_set_boundaries
from generate_publication_trajectory import (
    compute_ml_pre_match_probability, compute_composite_prematch_probability,
)

# --- Self-diagnostic: prints EXACTLY what file this process is actually importing, and
# whether it has the tiebreak fixes, at the moment this script runs. This exists because a
# separate diagnostic command can resolve tennis_intel differently (different sys.path,
# a stale editable/regular pip install, PYTHONPATH set differently) than this script does
# when run directly — the only way to be certain what code THIS run is using is to check
# from inside this exact process. -----------------------------------------------------
import tennis_intel.live.live_win_probability as _lwp_module
import inspect as _inspect
print("=" * 70)
print("IMPORT DIAGNOSTIC (this run):")
print(f"  live_win_probability imported from: {_lwp_module.__file__}")
_lwp_src = _inspect.getsource(_lwp_module)
print(f"  Has extended-deuce fix:  {'extended_deuce_prob' in _lwp_src}")
print(f"  Has OLD flat-constant bug: {'p_a_win_deuce_phase' in _lwp_src}")
import tennis_intel.live.build_point_dataset as _bpd_module
print(f"  build_point_dataset imported from: {_bpd_module.__file__}")
print("=" * 70)
# --- End self-diagnostic ---------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_MCP = PROJECT_ROOT / "data" / "raw" / "tennis_MatchChartingProject"
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "docs" / "trajectories"

# Display-only EMA smoothing factor for the ML-Informed Markov (smoothed) chart line.
# See the plotting block below for the full explanation: this affects ONLY what gets
# rendered in the chart, never the CSV, any evaluation, or the underlying mechanism,
# which the early-match calibration backtest confirmed is already earning its accuracy.
# Lower alpha = more smoothing (slower to react, calmer-looking line); higher alpha =
# closer to the real, unsmoothed-for-display series. 0.3 is a reasonable starting point
# — chosen for visual calmness only, not tuned against any calibration metric, since
# tuning this against accuracy would be a category error (it's cosmetic, not a model
# parameter).
EMA_DISPLAY_ALPHA = 0.3

POINT_FILES = [
    RAW_MCP / "charting-m-points-to-2009.csv",
    RAW_MCP / "charting-m-points-2010s.csv",
    RAW_MCP / "charting-m-points-2020s.csv",
]

N_SIMULATIONS = 300  # can afford more here — only one match, not 150
ROLLOUT_MODEL_NAME = "gradient_boosting"
# Feature schema centralized (external audit, 2026-07, Code Review finding #6): see
# src/tennis_intel/live/feature_schema.py for the single source of truth.
from tennis_intel.live.feature_schema import PREMATCH_FEATURE_NAMES as STATIC_FEATURE_NAMES


def _row_to_match_state(row: dict) -> MatchState:
    """Thin alias to the single canonical implementation — kept under this file's
    original name so every existing call site below (markov_p_player1, ml_p_player1,
    etc.) continues to work unchanged. See match_state_conversion.py for the full
    implementation and both bug-fix histories this centralization consolidates."""
    return row_to_match_state(row)


def markov_p_player1(row: dict) -> float:
    """P(Player 1 wins).

    BUG FIX (2026-07, paired with the _row_to_match_state fix above): that function's "A"
    now consistently means "the winner" (matching batch_simulate_dynamic's own internal
    assumption, needed to fix ml_p_player1 below). This function's ps/pr must now be
    realigned to the SAME entity (the winner), not Player 1 directly — then the final
    result inverted back to P(Player 1 wins). Previously ps/pr were Player-1-oriented while
    the state was ALSO Player-1-oriented (consistent, correct at the time); leaving ps/pr
    Player-1-oriented after the state became winner-oriented would have silently mismatched
    the two, breaking a previously-correct calculation — caught before shipping by tracing
    through both functions together rather than fixing one in isolation.

    Also includes the earlier p_return fix: p_return must be 1 - the OPPONENT's real
    serve-win rate, not the tracked player's own generic return statistic — see
    evaluate_live_engines_v2.py's markov_p_winner for the full explanation.

    BUG FIX #2 (external review, 2026-07, found via the Sinner-Alcaraz investigation):
    this function had its OWN separate copy of the ps/pr construction, missed when
    evaluate_live_engines_v2.py's markov_p_winner was fixed for the same underlying bug —
    both ps and pr were using first_serve_win_pct_career directly as if it were each
    player's TRUE overall serve-win rate, systematically understating both. Fixed the
    same way: ps from the new combined_serve_win_pct_career column, pr via the corrected,
    opponent-conditioned compute_p_a_return_seed. See return_seed.py's module docstring
    for the full derivation, including a documented near-miss worth reading before
    touching this kind of construction again."""
    state = _row_to_match_state(row)  # state's "A" = the winner
    p1_is_winner = bool(row["player1_is_winner"])
    ps = row.get("winner_combined_serve_win_pct_career")
    if ps is None or pd.isna(ps):
        ps = row.get("winner_first_serve_win_pct_career")  # known-inferior fallback
    ps = 0.65 if (ps is None or pd.isna(ps)) else float(ps)
    pr = compute_p_a_return_seed(row, track_winner=True)
    p_winner_wins = prob_a_wins_match_from_state(state, ps, pr)
    return p_winner_wins if p1_is_winner else (1.0 - p_winner_wins)


def ml_p_player1(row: dict, model, feature_cols: list, rng_seed: int) -> float:
    """P(Player 1 wins).

    BUG FIX (2026-07): batch_simulate_dynamic's OWN internal assumption is that "A" (from
    the initial_state tuple) means the tracked winner (see that function's own docstring).
    _row_to_match_state now constructs its state to match this ("A" = the winner), so this
    function's raw return value is P(the winner wins) — must invert to get P(Player 1 wins)
    whenever Player 1 is not the real winner. Previously the state incorrectly represented
    "A = Player 1" regardless of who won, causing batch_simulate_dynamic to silently
    compute server_is_winner = server_is_a as "is Player 1 serving" and treat that as "is
    the winner serving" — an inverted feature fed to the classifier on every single point
    whenever Player 1 lost the real match. This was the root cause of a visible discontinuity
    between the (correctly-computed) pre-match point and the first replayed point on the
    publication trajectory chart."""
    state = _row_to_match_state(row)  # state's "A" = the winner
    p1_is_winner = bool(row["player1_is_winner"])
    static = {c: row.get(c, np.nan) for c in STATIC_FEATURE_NAMES if c in feature_cols}
    seed_mom = {"p1_momentum_last10": row.get("p1_momentum_last10"),
                "p1_momentum_last20": row.get("p1_momentum_last20")}

    def predict_fn(fm):
        return model.predict_proba(fm)[:, 1]

    p_winner_wins = batch_simulate_dynamic(
        (state.a_sets, state.b_sets, state.a_games, state.b_games,
         state.a_points, state.b_points, state.server_is_a, state.is_tiebreak),
        static, feature_cols, predict_fn, best_of=state.best_of,
        player1_is_winner=p1_is_winner,
        seed_momentum=seed_mom, n_simulations=N_SIMULATIONS, rng=random.Random(rng_seed),
    )
    return p_winner_wins if p1_is_winner else (1.0 - p_winner_wins)


def ml_informed_markov_p_player1(
    row: dict, model, feature_cols: list, posterior,
) -> tuple[float, "ServeReturnPosterior"]:
    """P(Player 1 wins), via the ML-informed Markov engine (context-aware, Bayesian-
    smoothed, sensitivity-weighted point probability fed into the validated recursion —
    see src/tennis_intel/live/ml_informed_markov.py). Mirrors the exact same invert-if-
    needed pattern as markov_p_player1/ml_p_player1 above: _row_to_match_state's "A" means
    the winner, so the raw result is P(the winner wins) and must be inverted to P(Player 1
    wins) whenever Player 1 is not the real winner.

    STATEFUL: takes and returns the running ServeReturnPosterior, since the whole point of
    the Beta-Binomial smoothing is that it accumulates real observed evidence ACROSS the
    match — the caller must thread this through its point-by-point loop, not recreate it
    fresh each call."""
    state = _row_to_match_state(row)
    p1_is_winner = bool(row["player1_is_winner"])
    p_winner_wins, updated_posterior = ml_informed_markov_predict(
        state, row, model, feature_cols, posterior
    )
    p_player1 = p_winner_wins if p1_is_winner else (1.0 - p_winner_wins)
    return p_player1, updated_posterior


def ml_informed_markov_p_player1_unsmoothed(row: dict, model, feature_cols: list) -> float:
    """P(Player 1 wins), via the ML-informed Markov engine WITHOUT Bayesian/sensitivity-
    aware smoothing — the raw classifier point-prediction fed directly into the recursion.
    Kept as a separate function (not just prior_strength=0 in the smoothed version) since
    it's a genuinely different mode: no posterior, no blend, no state to thread across
    points. This is the version that achieved log_loss=0.1815 in evaluate_ml_informed_markov.py
    before smoothing was added; plotted alongside the smoothed version for direct
    visual comparison of the accuracy-vs-calibration tradeoff found in that evaluation."""
    state = _row_to_match_state(row)
    p1_is_winner = bool(row["player1_is_winner"])
    p_a_serve, p_a_return = ml_informed_point_probabilities(row, model, feature_cols)
    p_a_serve = float(np.clip(p_a_serve, 0.01, 0.99))
    p_a_return = float(np.clip(p_a_return, 0.01, 0.99))
    p_winner_wins = prob_a_wins_match_from_state(state, p_a_serve, p_a_return)
    return p_winner_wins if p1_is_winner else (1.0 - p_winner_wins)


def find_match(points: pd.DataFrame, search_terms: list[str]) -> str:
    ids = points["match_id"].unique()
    matches = [m for m in ids if all(t.lower() in m.lower() for t in search_terms)]
    if not matches:
        raise SystemExit(f"No match found containing all of: {search_terms}")
    if len(matches) > 1:
        print("Multiple matches found — pick one with --match-id:")
        for m in sorted(matches):
            print(f"  {m}")
        raise SystemExit(1)
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", type=str, default=None)
    parser.add_argument("--search", nargs="*", default=None,
                        help='e.g. --search Sinner Alcaraz Roland_Garros')
    args = parser.parse_args()

    logger.info("Loading model and building point dataset (one-time cost)...")
    payload = joblib.load(PROCESSED / "day9_point_classifiers.joblib")
    model, feature_cols = payload[ROLLOUT_MODEL_NAME], payload["feature_cols"]

    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]

    if args.match_id:
        match_id = args.match_id
        if match_id not in set(points["match_id"]):
            raise SystemExit(f"'{match_id}' not found in the joined+charted dataset "
                             f"(it may exist in MCP but not survive the frozen TML join).")
    else:
        if not args.search:
            raise SystemExit("Provide --match-id or --search TERM [TERM ...]")
        match_id = find_match(points, args.search)

    logger.info("Replaying match: %s", match_id)
    match_df = points[points["match_id"] == match_id].sort_values("Pt").reset_index(drop=True)
    p1_name = match_id.split("-")[-2].replace("_", " ")
    p2_name = match_id.split("-")[-1].replace("_", " ")
    # MCP match_id ends in ...-Player1-Player2; verify against frozen join for certainty
    fj_row = frozen_join[frozen_join["mcp_match_id"] == match_id]
    if len(fj_row):
        p1_name = fj_row["mcp_Player 1"].iloc[0]
        p2_name = fj_row["mcp_Player 2"].iloc[0]
    logger.info("%s (Player 1) vs %s (Player 2), %d points", p1_name, p2_name, len(match_df))
    final_winner_is_p1 = bool(match_df["player1_is_winner"].iloc[0])

    records = match_df.to_dict("records")
    markov_p1, ml_p1, ml_informed_p1, ml_informed_unsmoothed_p1, hybrid_p1 = [], [], [], [], []
    pts, set_markers, game_markers = [], [], []
    prev_games_total = -1

    # Seed the Beta-Binomial posterior via the CORRECTED construction. Two fixes now,
    # applied in sequence as each was diagnosed:
    #   1. (2026-07, first pass) Stopped seeding p_serve0 directly from a raw career
    #      point-rate, which bypassed every richer pre-match feature (surface Elo, H2H,
    #      tournament form) — replaced with an inversion of a feature-rich pre-match
    #      probability via build_pretrained_prior.
    #   2. (2026-07, this fix) That feature-rich probability was, until now,
    #      compute_ml_pre_match_probability — a 200-trial Monte Carlo rollout of an
    #      entire simulated match from a blank slate. Diagnosed directly on this real
    #      match: real features (Elo, surface Elo, H2H, tournament H2H) were confirmed
    #      correctly loaded and correctly signed, but the rollout was STABLE (std=0.025
    #      across 10 random seeds — reproducible, not noisy) while being SYSTEMATICALLY
    #      overconfident (~0.87 for Alcaraz vs. a real-world ~55-60% expectation),
    #      because simulating an entire match compounds the point classifier's per-point
    #      sharpness multiplicatively into an even more extreme match-level number. Now
    #      uses compute_composite_prematch_probability — a single deterministic
    #      inference from the already-trained, already-validated XGBoost pre-match model,
    #      with no simulation and no compounding — as the actual historically-grounded
    #      baseline this project's objective calls for.
    #   3. Sets n0 (prior effective sample size) from elo_matches_played_pre_winner — a
    #      real, already-computed confidence signal — instead of one fixed constant for
    #      every match regardless of how much real history backs the estimate.
    first_row = records[0]
    p0_a_wins = compute_composite_prematch_probability(first_row)
    logger.info("Composite pre-match P(actual winner wins) = %.4f (direct XGBoost "
               "inference, no simulation) — compare against the rollout-based estimate, "
               "P(actual winner wins)=%.4f, to see the gap this fix closes",
               p0_a_wins, compute_ml_pre_match_probability(first_row, model, feature_cols))

    # BUG FIX (external review, 2026-07, following the Sinner-Alcaraz "does the model
    # neglect its pre-match prior" investigation): this block previously reconstructed
    # p_a_return_seed as 1 - opponent's first_serve_win_pct_career, which systematically
    # understates the opponent's TRUE overall serve rate (it ignores second-serve points
    # entirely) and therefore overstates how weak the returner's seed should be. Confirmed
    # numerically: this produced p_a_return_seed=0.26 for this exact match, which combined
    # with the recursion's high early-match sensitivity to p_a_return, distorted the very
    # first prediction of the match. Fixed by using the tracked player's OWN, directly-
    # measured, already-leakage-safe return_pts_won_pct_career via return_seed.py, with
    # the old opponent-inversion approach kept only as a documented, known-inferior
    # fallback. See that module's docstring for the full derivation and numeric evidence.
    p_a_return_seed = compute_p_a_return_seed(first_row, track_winner=True)

    elo_matches_played_a = first_row.get("elo_matches_played_pre_winner")
    elo_matches_played_b = first_row.get("elo_matches_played_pre_loser")
    best_of_val = int(first_row["best_of"]) if pd.notna(first_row.get("best_of")) else 3

    # Composite n0 upgrade (external audit, 2026-07, Architecture Review finding C):
    # matchup-specific H2H depth, not just career match count. Total prior meetings
    # between THESE TWO players is symmetric — the same value regardless of which
    # player's own H2H-win count you look at, so summing both winner_/loser_-labeled
    # counts gives the real total meeting count, not double-counting a shared history.
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

    # GENUINE pre-match value for ML-Informed Markov (smoothed), computed BEFORE any
    # point is played — this is a real zero-information estimate, not a proxy from the
    # first charted point. Unlike the other four engines here (Markov's ps/pr, ML+MC,
    # unsmoothed, hybrid), which all require at least one point's row context to compute
    # anything in this script, the smoothed engine's own seeding already produces exactly
    # this quantity as a side effect — reused directly, not recomputed.
    p_a_wins_prematch = prob_win_match(p_serve0, p_return0, best_of=best_of_val)
    ml_informed_prematch_p1 = p_a_wins_prematch if final_winner_is_p1 else (1.0 - p_a_wins_prematch)

    winner_name = p1_name if final_winner_is_p1 else p2_name
    logger.info("Seeded CORRECTED posterior for %s (the real match winner, tracked "
               "internally as 'A'): p0_a_wins=%.4f (feature-rich pre-match estimate) -> "
               "inverted p_serve0=%.4f (n0=%.1f), p_return0=%.4f (n0=%.1f). This is "
               "intentionally winner-relative, NOT Player-1-relative — the chart below "
               "converts back to P(%s wins) for display.",
               winner_name, p0_a_wins, p_serve0, n0_serve, p_return0, n0_return, p1_name)

    t0 = time.time()
    for i, row in enumerate(records):
        p_markov = markov_p_player1(row)
        p_ml_mc = ml_p_player1(row, model, feature_cols, rng_seed=i)
        p_ml_informed, posterior = ml_informed_markov_p_player1(row, model, feature_cols, posterior)
        p_ml_informed_unsmoothed = ml_informed_markov_p_player1_unsmoothed(row, model, feature_cols)
        p_hybrid = hybrid_predict(markov_p=p_markov, ml_mc_p=p_ml_mc)

        markov_p1.append(p_markov)
        ml_p1.append(p_ml_mc)
        ml_informed_p1.append(p_ml_informed)
        ml_informed_unsmoothed_p1.append(p_ml_informed_unsmoothed)
        hybrid_p1.append(p_hybrid)
        pts.append(row["Pt"])
        games_total = int(row["Gm1"]) + int(row["Gm2"])
        game_markers.append(games_total != prev_games_total)
        prev_games_total = games_total
        if (i + 1) % 50 == 0:
            logger.info("  %d / %d points (%.1fs elapsed)", i + 1, len(records), time.time() - t0)

    logger.info("Done in %.1fs", time.time() - t0)

    print(f"\n{p1_name} vs {p2_name}")
    print(f"Actual result: {p1_name if final_winner_is_p1 else p2_name} won")
    print(f"Final Markov P({p1_name} wins): {markov_p1[-1]:.4f}")
    print(f"Final ML+MC P({p1_name} wins): {ml_p1[-1]:.4f}")
    print(f"Final ML-Informed Markov (smoothed) P({p1_name} wins): {ml_informed_p1[-1]:.4f}")
    print(f"Final ML-Informed Markov (unsmoothed) P({p1_name} wins): {ml_informed_unsmoothed_p1[-1]:.4f}")
    print(f"Final Hybrid (Markov/ML+MC fixed-weight, DEPRECATED — see hybrid_engine.py) "
          f"P({p1_name} wins): {hybrid_p1[-1]:.4f}")

    # Checkpoint summary: all five engines' P(p1_name wins) before the match and after
    # each completed set — reuses detect_set_boundaries (already built and unit-tested in
    # trajectory_events.py for the publication chart) rather than re-deriving set-boundary
    # logic here.
    #
    # PRE-MATCH ROW, FIXED (external review, 2026-07): previously used the first charted
    # point's value as a "pt 1 proxy" for ALL FIVE engines, including ML-Informed Markov —
    # but that engine has a genuine, already-computed, zero-information pre-match estimate
    # (ml_informed_prematch_p1, from prob_win_match(p_serve0, p_return0, ...), computed at
    # the seeding stage above). The other four engines (Markov, ML+MC, unsmoothed, hybrid)
    # have no equivalent zero-point computation path in this script — each requires at
    # least one real row's context — so they now show a dash rather than a misleading
    # proxy value, instead of silently treating "value at point 1" as "pre-match."
    #
    # "AFTER SET N" CHECKPOINT: uses b.point_index directly, unmodified. Traced this
    # carefully after an earlier, INCORRECT attempt to "fix" an off-by-one here (external
    # review, 2026-07) — worth documenting the error precisely since it's a genuine trap:
    # this project's established convention (confirmed in ml_informed_markov_predict) is
    # that a row's Set1/Set2/Gm1/Gm2 describe the score BEFORE that row's own point is
    # played, and the prediction returned for a row uses that row's OWN score state as
    # "current." detect_set_boundaries returns point_index for the row where Set1+Set2
    # has already incremented — meaning that row's OWN state genuinely represents
    # "entering the first point of the next set," which is precisely "after the previous
    # set ended, before any point of the new set is played." Using idx directly is
    # therefore already correct; stepping back to idx-1 would point one point too early
    # (the completed set's own last point), understating the checkpoint.
    boundary_lookup_df = pd.DataFrame({
        "point_index": pts, "Set1": match_df["Set1"].values, "Set2": match_df["Set2"].values,
        "Gm1": match_df["Gm1"].values, "Gm2": match_df["Gm2"].values,
    })
    boundaries = detect_set_boundaries(boundary_lookup_df)
    engines = [
        ("Markov", markov_p1), ("ML+MC", ml_p1),
        ("ML-Informed Markov (unsmoothed)", ml_informed_unsmoothed_p1),
        ("ML-Informed Markov (smoothed)", ml_informed_p1),
        ("Hybrid", hybrid_p1),
    ]
    # Which engines have a genuine pre-match computation in THIS script — only the
    # smoothed engine does; everything else gets a dash on the pre-match row.
    prematch_values = {
        "Markov": None, "ML+MC": None,
        "ML-Informed Markov (unsmoothed)": None,
        "ML-Informed Markov (smoothed)": ml_informed_prematch_p1,
        "Hybrid": None,
    }

    print(f"\n=== Checkpoint summary: P({p1_name} wins) ===")
    header = f"{'Checkpoint':<28}" + "".join(f"{name:>26}" for name, _ in engines)
    print(header)
    row0 = "  ".join(
        f"{prematch_values[name]:>24.4f}" if prematch_values[name] is not None else f"{'—':>24}"
        for name, _ in engines
    )
    print(f"{'Before match (genuine pre-match)':<28}{row0}")
    for b in boundaries:
        # b.point_index is 1-indexed to match `pts`; find its position in the pts list.
        # No further adjustment — see the note above for why idx (not idx-1) is correct.
        try:
            idx = pts.index(b.point_index)
        except ValueError:
            idx = min(b.point_index - 1, len(pts) - 1)
        row = "  ".join(f"{vals[idx]:>24.4f}" for _, vals in engines)
        print(f"{'After Set ' + str(b.set_number) + ' (' + b.score_str + ')':<28}{row}")

    # Save the full point-by-point table
    out_csv = OUT_DIR / f"replay_{match_id}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "point_index": pts, "markov_p1": markov_p1, "ml_mc_p1": ml_p1,
        "ml_informed_markov_p1": ml_informed_p1,
        "ml_informed_markov_unsmoothed_p1": ml_informed_unsmoothed_p1,
        "hybrid_p1": hybrid_p1,
        "set1": match_df["Set1"], "set2": match_df["Set2"],
        "gm1": match_df["Gm1"], "gm2": match_df["Gm2"],
    }).to_csv(out_csv, index=False)
    print(f"\nFull point-by-point table saved to {out_csv}")

    # DISPLAY-ONLY EMA (2026-07, following the early-match calibration backtest that
    # confirmed the smoothed engine's early-match volatility is EARNED — it beats pure
    # Markov's LogLoss/Brier in every points-played-so-far bin (0-10, 10-25, 25-50)
    # tested): the underlying mechanism is validated and must NOT be touched. This EMA
    # exists purely to make the CHART calmer to look at — it is computed here, in the
    # plotting block, from ml_informed_p1 (the real, already-computed, already-saved
    # values) and used ONLY for the line drawn below. It never feeds back into the CSV
    # (already written above, unaffected), any evaluation script, the posterior, or the
    # recursion. Do not import this smoothed series anywhere outside this plotting call.
    ml_informed_p1_display = pd.Series(ml_informed_p1).ewm(alpha=EMA_DISPLAY_ALPHA, adjust=False).mean().tolist()

    # Plot — all five engines this project has built and validated, per-engine styling
    # chosen for readability: Markov (solid, most saturated), ML+MC (dashed), ML-informed
    # variants (solid green family, smoothed vs. unsmoothed distinguished by alpha/dash),
    # hybrid (dotted, since it's a simple post-hoc combination of two lines already shown).
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(pts, markov_p1, label=f"Markov: P({p1_name} wins)", color="#1f77b4", lw=1.6)
    ax.plot(pts, ml_p1, label=f"ML+MC: P({p1_name} wins)", color="#d62728", lw=1.4,
           ls="--", alpha=0.9)
    ax.plot(pts, ml_informed_unsmoothed_p1,
           label=f"ML-Informed Markov (unsmoothed): P({p1_name} wins)",
           color="#2ca02c", lw=1.4, alpha=0.9)
    ax.plot(pts, ml_informed_p1_display,
           label=f"ML-Informed Markov (Bayesian-smoothed, display-EMA "
                 f"alpha={EMA_DISPLAY_ALPHA}): P({p1_name} wins)",
           color="#17becf", lw=1.6, alpha=0.95)
    ax.plot(pts, hybrid_p1,
           label=f"Hybrid (fixed-weight Markov/ML+MC, DEPRECATED — underperforms both "
                 f"inputs): P({p1_name} wins)",
           color="#9467bd", lw=1.2, ls=":", alpha=0.85)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.axhline(1.0 if final_winner_is_p1 else 0.0, color="black", lw=1.0, ls="--",
               label=f"Actual outcome ({p1_name if final_winner_is_p1 else p2_name} won)")
    for x, is_new_game in zip(pts, game_markers):
        if is_new_game:
            ax.axvline(x, color="grey", lw=0.3, alpha=0.3)
    ax.set_xlabel("Point index")
    ax.set_ylabel(f"P({p1_name} wins match)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{p1_name} vs {p2_name} — {match_id.split('-')[2].replace('_',' ')}", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png = OUT_DIR / f"replay_{match_id}.png"
    fig.savefig(out_png)
    print(f"Trajectory plot saved to {out_png}")


if __name__ == "__main__":
    main()