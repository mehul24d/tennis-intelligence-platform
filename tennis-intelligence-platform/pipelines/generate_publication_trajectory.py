"""
generate_publication_trajectory.py — orchestrates the full pipeline for one real match:
computes genuine, INDEPENDENT pre-match probabilities for both engines (per requirement 1
and the follow-up correction below), assembles the trajectory (prepend pre-match, append
deterministic outcome), and renders the publication-quality figure.

PRE-MATCH FEATURE AUDIT (required before trusting the synthetic point-0 vector below).
Every column in Day 9's POINT_FEATURE_COLS (pipelines/build_day9_point_model.py), audited
against whether it is genuinely known before the first point is served:

  Category 1 — Always available pre-match (13 features, used with REAL values at point 0):
    elo_pre_match_winner, elo_pre_match_loser, winner_win_pct_last10, loser_win_pct_last10,
    winner_surface_win_pct_last10, loser_surface_win_pct_last10,
    winner_first_serve_in_pct_career, loser_first_serve_in_pct_career,
    winner_first_serve_win_pct_career, loser_first_serve_win_pct_career,
    winner_bp_saved_pct_career, loser_bp_saved_pct_career, server_is_winner
    (server_is_winner is knowable pre-match: it only requires knowing who serves the
    opening game, which is known match metadata, not a live observation.)

  Category 2 — Available after the first point but not before: NONE. This model's feature
    set has no such intermediate category — every feature is either static pre-match
    context or a live situational/momentum signal undefined before any point is played.

  Category 3 — Only available during the match (9 features, given neutral defaults at
    point 0): is_tiebreak_game=False (the opening game is never a tiebreak — a correct
    fact, not a fallback), is_break_point=False, is_set_point=False, is_match_point=False,
    is_second_serve_point=False, p1_momentum_last10=0.5, p2_momentum_last10=0.5,
    p1_momentum_last20=0.5, p2_momentum_last20=0.5 (0.5 = no information yet, i.e. neither
    player has momentum, the honest neutral prior for a fraction with no observations).

  NOTE: an earlier draft of this script computed the ML pre-match probability by reusing
  the analytical Markov formula, on the reasoning that ML's live-situational features are
  undefined pre-match. That reasoning wrongly conflated "some features are undefined
  pre-match" with "no distinct ML pre-match estimate is worth computing" — the 13
  Category-1 features above ARE fully defined pre-match and were being discarded for no
  good reason. Fixed: the ML pre-match probability is now a genuine, independent prediction
  from the trained classifier on this real feature vector, and is expected to differ from
  the Markov value, since the two engines answer different questions (a closed-form
  function of two point-win probabilities, vs. a learned function of the full feature set).

Usage:
    python pipelines/generate_publication_trajectory.py \\
        --match-id "20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.markov_baseline import prob_win_match
from tennis_intel.viz.trajectory_generation import build_trajectory
from tennis_intel.viz.trajectory_plot import plot_trajectory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_MCP = PROJECT_ROOT / "data" / "raw" / "tennis_MatchChartingProject"
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "docs" / "trajectories"

POINT_FILES = [
    RAW_MCP / "charting-m-points-to-2009.csv",
    RAW_MCP / "charting-m-points-2010s.csv",
    RAW_MCP / "charting-m-points-2020s.csv",
]

# Category 3 neutral defaults — see the audit in this file's docstring for justification
# of every single default value chosen here.
LIVE_FEATURE_DEFAULTS = {
    "is_tiebreak_game": False,
    "is_break_point": False,
    "is_set_point": False,
    "is_match_point": False,
    "is_second_serve_point": False,
    "p1_momentum_last10": 0.5,
    "p2_momentum_last10": 0.5,
    "p1_momentum_last20": 0.5,
    "p2_momentum_last20": 0.5,
}
# Category 1 — real pre-match values are read directly from the match row for these
# Feature schema centralized (external audit, 2026-07, Code Review finding #6): see
# src/tennis_intel/live/feature_schema.py for the single source of truth.
from tennis_intel.live.feature_schema import PREMATCH_FEATURE_NAMES
from tennis_intel.live.return_seed import compute_p_a_return_seed


def compute_markov_pre_match_probability(row: dict) -> float:
    """
    Markov engine's pre-match P(player 1 wins).

    BUG FIX (found via manual inspection of an implausible 0.995 pre-match probability
    for a genuine Sinner-Alcaraz coin-flip final): prob_win_match's p_return parameter is
    documented as "A's probability of winning a point on B's serve, i.e. 1 - B's own
    serve-win probability" (see markov_baseline.py / live_win_probability.py docstrings).
    The original implementation instead used player A's own generic
    "return_pts_won_pct_career" statistic — A's return performance averaged across A's
    ENTIRE career against a mix of past opponents, not against this specific opponent's
    real serve ability. Since the two numbers can differ substantially (in this match,
    Sinner's own return-stat implied a hypothetical opponent serve-win-rate of 59.3%, while
    Alcaraz's REAL serve-win-rate is 72.6%), this silently pitted the player against a much
    weaker hypothetical opponent than the real one, inflating confidence enormously.

    Corrected: p_return = 1 - (the ACTUAL opponent's own career first-serve-win rate),
    using both players' real, independently-known stats rather than only one player's.
    """
    p1_is_winner = bool(row["player1_is_winner"])
    best_of = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3

    ps_key = "winner_first_serve_win_pct_career" if p1_is_winner else "loser_first_serve_win_pct_career"
    ps = row.get(ps_key, 0.65)
    ps = 0.65 if pd.isna(ps) else float(ps)
    # BUG FIX (external review, 2026-07): see return_seed.py's module docstring. This
    # function's "A" is Player 1 (not the tracked winner), so track_winner is passed as
    # p1_is_winner directly, matching compute_p_a_return_seed's own convention.
    pr = compute_p_a_return_seed(row, track_winner=p1_is_winner)

    return prob_win_match(ps, pr, best_of=best_of)


def compute_composite_prematch_probability(row: dict) -> float:
    """
    P(the tracked winner, "A", wins) via a DIRECT, DETERMINISTIC single inference from the
    already-trained, already-validated XGBoost pre-match model (build_xgboost_prematch_model.py)
    — no simulation, no random seed, no compounding of point-level predictions across an
    entire simulated match.

    REPLACES compute_ml_pre_match_probability as the source of the pre-match baseline
    (found via direct external critique, 2026-07): that function ran a full 200-trial
    Monte Carlo rollout of an entire simulated match from a blank 0-0-0 state, using the
    Day 9 POINT classifier. Diagnosed on the real 2025 Roland Garros final: verified the
    real pre-match features (Elo, surface Elo, H2H, tournament H2H) were all correctly
    loaded and correctly signed favoring Alcaraz — ruling out a data-loading bug — but
    found the rollout consistently produced ~0.87 for Alcaraz (std=0.025 across 10 random
    seeds — LOW variance, i.e. reproducible, not noisy) against a real-world betting-market
    expectation of roughly 55-60%. The rollout is stable but SYSTEMATICALLY overconfident:
    the point classifier's per-point predictions are more extreme than the true skill gap
    warrants (the same is_second_serve_point-driven sharpness diagnosed earlier this
    project), and simulating an entire match forces those extreme point probabilities to
    compound multiplicatively into an even more extreme match-level estimate. This is
    exactly the "prior is broken, no amount of correct in-match updating can fix it"
    failure mode identified directly: the fix is not more smoothing downstream, it's a
    genuinely different, low-variance MATCH-level estimator upstream.

    This function is the "composite pre-match win probability... trained/fit offline on
    historical matches" originally specified as the correct source for P0 — already built
    and validated (log_loss=0.6235 vs naive 0.6931 on a held-out 2022+ test set) but never
    previously wired in as the actual source of the pre-match baseline.
    """
    import joblib as _joblib
    from build_xgboost_prematch_model import PREMATCH_FEATURE_PAIRS

    model_path = PROCESSED / "xgboost_prematch_model.joblib"
    if not model_path.exists():
        raise SystemExit(
            f"'{model_path}' not found. Run this first:\n"
            f"  python pipelines/build_xgboost_prematch_model.py\n"
            f"then re-run this script — it needs that model to compute the pre-match baseline."
        )
    payload = _joblib.load(model_path)
    prematch_model, feature_cols = payload["model"], payload["feature_cols"]

    p1_is_winner = bool(row["player1_is_winner"])
    diffs = {}
    for winner_col, loser_col, name in PREMATCH_FEATURE_PAIRS:
        winner_val = row.get(winner_col)
        loser_val = row.get(loser_col)
        if winner_val is None or loser_val is None or pd.isna(winner_val) or pd.isna(loser_val):
            diffs[f"{name}_diff"] = np.nan
            continue
        # "A" tracks the WINNER throughout this project's convention — diff is always
        # winner_val - loser_val here, i.e. this is P(A=winner wins), matching every other
        # pre-match function's own "A=winner" convention (compute_markov_pre_match_probability,
        # the original compute_ml_pre_match_probability) for direct comparability.
        diffs[f"{name}_diff"] = float(winner_val) - float(loser_val)

    X = pd.DataFrame([diffs])[feature_cols]
    p_a_wins = float(prematch_model.predict_proba(X)[0, 1])
    return p_a_wins


def compute_ml_pre_match_probability(row: dict, model, feature_cols: list) -> float:
    """
    ML engine's pre-match P(server wins point) at a synthetic 'point 0' state, then
    converted to P(player 1 wins MATCH) by treating this as the server-win-rate input to
    the SAME rollout used everywhere else in this project (batch_simulate_dynamic) — i.e.
    the ML engine's pre-match belief is obtained by actually rolling the trained model
    forward from a fresh 0-0-0 state with real pre-match features, not just reading a
    single point-probability. This keeps the ML pre-match number consistent in kind with
    the ML in-match numbers on the same chart (both are match-win probabilities from the
    Monte Carlo engine), rather than mixing a point-probability with match-probabilities.
    """
    from tennis_intel.live.monte_carlo_engine import batch_simulate_dynamic
    import random

    p1_is_winner = bool(row["player1_is_winner"])
    static_features = {}
    for name in PREMATCH_FEATURE_NAMES:
        if name in feature_cols:
            val = row.get(name)
            static_features[name] = float(val) if pd.notna(val) else np.nan
    # server_is_winner at point 0: determined by who serves the opening game, known from
    # match metadata (Svr on the first charted point), genuinely pre-match information.
    if "server_is_winner" in feature_cols:
        first_server_is_p1 = (row.get("Svr", 1) == 1)
        static_features["server_is_winner"] = (
            first_server_is_p1 if p1_is_winner else not first_server_is_p1
        )

    best_of = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3
    # A tracks the WINNER (not "Player 1") throughout batch_simulate_dynamic — this is the
    # convention the function's server_is_winner=server_is_a assumption depends on, verified
    # by hand for both possible cases (comment corrected 2026-07, was previously misleading).
    first_server_is_a = (row.get("Svr", 1) == 1) == p1_is_winner

    def predict_fn(fm):
        return model.predict_proba(fm)[:, 1]

    p_a_wins = batch_simulate_dynamic(
        (0, 0, 0, 0, 0, 0, first_server_is_a, False),
        static_features, feature_cols, predict_fn, best_of=best_of,
        player1_is_winner=p1_is_winner,
        seed_momentum={"p1_momentum_last10": 0.5, "p1_momentum_last20": 0.5},
        n_simulations=200, rng=random.Random(0),
    )
    return p_a_wins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    logger.info("Loading data...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]

    match_df = points[points["match_id"] == args.match_id].sort_values("Pt").reset_index(drop=True)
    if len(match_df) == 0:
        raise SystemExit(f"Match '{args.match_id}' not found.")

    fj_row = frozen_join[frozen_join["mcp_match_id"] == args.match_id].iloc[0]
    p1_name, p2_name = fj_row["mcp_Player 1"], fj_row["mcp_Player 2"]
    surface = fj_row.get("mcp_Surface", None)
    tournament = fj_row.get("mcp_Tournament", None)
    best_of = int(match_df["best_of"].iloc[0]) if pd.notna(match_df["best_of"].iloc[0]) else 3
    winner_is_p1 = bool(match_df["player1_is_winner"].iloc[0])
    final_score = fj_row.get("tml_score", None)

    logger.info("Loading trained classifier...")
    payload = joblib.load(PROCESSED / "day9_point_classifiers.joblib")
    model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

    logger.info("Computing INDEPENDENT pre-match probabilities for both engines...")
    row0 = match_df.iloc[0].to_dict()
    p1_is_winner = bool(row0["player1_is_winner"])
    pre_markov = compute_markov_pre_match_probability(row0)

    # BUG FIX (found via a real 2025 Roland Garros final where every new contextual
    # feature — surface Elo, H2H, tournament-specific H2H and form — correctly favored
    # Alcaraz, yet the chart showed 0.86 favoring Sinner): compute_ml_pre_match_probability
    # returns P(A wins), where A tracks the ACTUAL MATCH WINNER by construction (see that
    # function's docstring) — NOT P(Player 1 wins). Assigning its raw output directly as
    # pre_match_ml_p1 is only correct when Player 1 happens to BE the real winner; when
    # Player 1 is the real loser (player1_is_winner=False, as in this match — Sinner lost),
    # the value must be inverted (1 - p) to correctly represent Player 1's probability.
    # Markov's pre-match function does NOT have this issue — it already reasons in
    # Player-1-relative terms throughout, never in "the actual winner" terms.
    pre_ml_for_winner = compute_ml_pre_match_probability(row0, model, feature_cols)
    pre_ml = pre_ml_for_winner if p1_is_winner else (1.0 - pre_ml_for_winner)

    logger.info("Markov pre-match P(%s wins) = %.3f", p1_name, pre_markov)
    logger.info("ML pre-match     P(%s wins) = %.3f  (independent estimate, not borrowed "
               "from Markov; inverted from P(actual winner wins)=%.3f since player1_is_winner=%s)",
               p1_name, pre_ml, pre_ml_for_winner, p1_is_winner)

    # ML-Informed Markov's pre-match value: this project's PRIMARY engine per its stated
    # objective (a historically-grounded pre-match baseline, updated coherently by real
    # in-match evidence).
    #
    # BUG FIX (found via direct external critique, 2026-07): this previously inverted
    # pre_ml_for_winner (the ML+MC rollout's own pre-match estimate) — but that rollout
    # was diagnosed as SYSTEMATICALLY overconfident as a pre-match estimator specifically
    # (stable across random seeds, std=0.025, but converging on ~0.87 for a match a real
    # betting market would price closer to 55-60%): simulating an entire match compounds
    # the point classifier's per-point sharpness multiplicatively into an even more
    # extreme match-level number. Now uses compute_composite_prematch_probability — a
    # DIRECT, single-inference call to the already-trained, already-validated XGBoost
    # pre-match model (build_xgboost_prematch_model.py), with no simulation and no
    # compounding — as the actual "historically-grounded pre-match baseline" the
    # project's own objective calls for. This is the fix for the root cause your reviewer
    # named directly: the prior itself was broken, and no amount of correct Beta-Binomial
    # updating downstream could have fixed a wrong starting point.
    #
    # NOTE: pre_ml (the ML+MC line's OWN pre-match dot) intentionally still uses
    # pre_ml_for_winner (the rollout estimate), NOT this new one — the ML+MC line's
    # every OTHER point also comes from that same rollout mechanism, so mixing sources at
    # just the pre-match point would recreate the exact point-0-to-point-1 seam bug fixed
    # earlier this project. Only ML-Informed Markov's prior changes here.
    from tennis_intel.live.ml_informed_markov import build_pretrained_prior
    from tennis_intel.live.markov_baseline import prob_win_match

    p0_a_wins_composite = compute_composite_prematch_probability(row0)
    logger.info("Composite pre-match P(actual winner wins) = %.3f  (direct XGBoost "
               "inference, no simulation — the corrected source for ML-Informed Markov's "
               "prior; compare against the rollout-based estimate, P(actual winner "
               "wins)=%.3f, to see the gap this fix closes)",
               p0_a_wins_composite, pre_ml_for_winner)

    # BUG FIX (external review, 2026-07): see return_seed.py's module docstring.
    p_a_return_seed = compute_p_a_return_seed(row0, track_winner=True)
    elo_matches_played_a = row0.get("elo_matches_played_pre_winner")
    elo_matches_played_b = row0.get("elo_matches_played_pre_loser")
    best_of_val = int(row0["best_of"]) if pd.notna(row0.get("best_of")) else best_of

    # Composite n0 upgrade (external audit, 2026-07, Architecture Review finding C):
    # matchup-specific H2H depth, not just career match count.
    h2h_meetings = None
    winner_h2h = row0.get("winner_h2h_wins_pre_match")
    loser_h2h = row0.get("loser_h2h_wins_pre_match")
    if pd.notna(winner_h2h) and pd.notna(loser_h2h):
        h2h_meetings = float(winner_h2h) + float(loser_h2h)

    tourney_h2h_meetings = None
    winner_tourney_h2h = row0.get("winner_tourney_h2h_wins_pre_match")
    loser_tourney_h2h = row0.get("loser_tourney_h2h_wins_pre_match")
    if pd.notna(winner_tourney_h2h) and pd.notna(loser_tourney_h2h):
        tourney_h2h_meetings = float(winner_tourney_h2h) + float(loser_tourney_h2h)

    p_serve0, _, p_return0, _ = build_pretrained_prior(
        p0_a_wins_composite, p_a_return_seed, best_of_val,
        elo_matches_played_a=elo_matches_played_a, elo_matches_played_b=elo_matches_played_b,
        h2h_meetings=h2h_meetings, tourney_h2h_meetings=tourney_h2h_meetings,
    )
    pre_ml_informed_for_winner = prob_win_match(p_serve0, p_return0, best_of=best_of_val)
    pre_ml_informed = pre_ml_informed_for_winner if p1_is_winner else (1.0 - pre_ml_informed_for_winner)
    logger.info("ML-Informed Markov pre-match P(%s wins) = %.3f", p1_name, pre_ml_informed)

    # Load the per-point predictions from replay_match.py's saved CSV — the two scripts
    # are connected via this file, not via in-memory columns (match_df from
    # build_point_dataset never has markov_pred/ml_pred; those only exist after a replay
    # or evaluation run has actually computed them and written them to disk).
    replay_csv = OUT_DIR / f"replay_{args.match_id}.csv"
    if not replay_csv.exists():
        raise SystemExit(
            f"'{replay_csv}' not found. Run this first:\n"
            f'  python pipelines/replay_match.py --match-id "{args.match_id}"\n'
            f"then re-run this script — it reads that file's saved predictions."
        )
    logger.info("Loading per-point predictions from %s", replay_csv)
    replay_df = pd.read_csv(replay_csv)
    if len(replay_df) != len(match_df):
        raise SystemExit(
            f"Row count mismatch: replay CSV has {len(replay_df)} points, "
            f"but the freshly-built point dataset has {len(match_df)}. The replay was "
            f"likely run before a data/parser fix was applied — re-run replay_match.py "
            f"for this match to regenerate it with the current code."
        )
    match_df["markov_pred"] = replay_df["markov_p1"].values
    match_df["ml_pred"] = replay_df["ml_mc_p1"].values
    # ml_informed_markov_p1 was added to replay_match.py's CSV in an earlier update; older
    # CSVs won't have it — degrade gracefully rather than crash, matching this project's
    # established pattern for optional columns elsewhere.
    #
    # NOTE (external audit, 2026-07, Architecture Review finding A — "fixed-weight hybrid
    # path still reachable"): this script previously ALSO computed and loaded a Hybrid
    # pre-match value and per-point column here, but plot_trajectory.py was already
    # simplified to draw ONLY the ML-Informed Markov engine — meaning that computation was
    # pure dead weight, never actually rendered. Removed here as part of reducing this
    # known-inferior engine's footprint in the codebase (see hybrid_engine.py and
    # evaluate_hybrid_engine.py for the measured finding and the reasoning for keeping it
    # available, clearly labeled, in replay_match.py's explicit multi-engine comparison
    # rather than deleting it outright).
    if "ml_informed_markov_p1" in replay_df.columns:
        match_df["ml_informed_markov_p1"] = replay_df["ml_informed_markov_p1"].values
    else:
        logger.warning("replay CSV has no ml_informed_markov_p1 column (likely generated "
                       "before that engine was added) — re-run replay_match.py to include "
                       "it.")

    traj = build_trajectory(
        match_df,
        pre_match_markov_p1=pre_markov, pre_match_ml_p1=pre_ml,
        p1_name=p1_name, p2_name=p2_name, winner_is_p1=winner_is_p1,
        surface=surface, tournament=tournament, best_of=best_of, final_score=final_score,
        pre_match_ml_informed_p1=pre_ml_informed,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"publication_{args.match_id}.png"
    out_path = plot_trajectory(traj, match_df, OUT_DIR / out_name)
    logger.info("Saved publication-quality trajectory to %s", out_path)


if __name__ == "__main__":
    main()