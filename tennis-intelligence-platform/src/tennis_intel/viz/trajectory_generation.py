"""
trajectory_generation.py — assembles the full win-probability trajectory: pre-match
probability (point 0) -> every charted point -> the deterministic final outcome. Kept
separate from plotting per requirement 10, so evaluation/generation logic never has to
change when styling changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MatchTrajectory:
    point_index: list[int]      # 0 = pre-match, 1..N = charted points, N+1 = final outcome
    markov_p1: list[float]
    ml_mc_p1: list[float]
    p1_name: str
    p2_name: str
    winner_is_p1: bool
    surface: str | None = None
    tournament: str | None = None
    best_of: int = 3
    final_score: str | None = None
    # ML-Informed Markov (corrected Elo/H2H-inverted prior + Bayesian smoothing) — added
    # alongside the two original engines rather than replacing them, per this project's
    # objective of tracking a historically-grounded pre-match baseline updated coherently
    # by real in-match evidence. Optional (defaults to None) so any existing caller that
    # only supplies Markov/ML+MC data continues to work unchanged.
    ml_informed_p1: list[float] | None = None
    # Hybrid (fixed-weight blend of Markov and ML+MC's own predictions, per point) — added
    # alongside the other three so the polished chart can show how the blend compares
    # against the two engines it's derived from. Optional for the same backward-
    # compatibility reason as ml_informed_p1 above.
    hybrid_p1: list[float] | None = None


def build_trajectory(
    match_df: pd.DataFrame,
    pre_match_markov_p1: float,
    pre_match_ml_p1: float,
    p1_name: str,
    p2_name: str,
    winner_is_p1: bool,
    surface: str | None = None,
    tournament: str | None = None,
    best_of: int = 3,
    final_score: str | None = None,
    pre_match_ml_informed_p1: float | None = None,
    pre_match_hybrid_p1: float | None = None,
) -> MatchTrajectory:
    """
    match_df must already contain one row per charted point with columns 'markov_pred'
    (or 'markov_p1') and 'ml_pred' (or 'ml_mc_p1') as P(player 1 wins) at that point, sorted
    by point order. Pre-match probabilities must be computed from PRE-MATCH information
    ONLY (Elo, rolling form, serve/return strength) — computing them is the caller's
    responsibility (this function only assembles the trajectory, per requirement 10's
    separation of concerns); passing in a mid-match value here would silently violate
    requirement 1's "pre-match only" constraint with no way for this function to detect it.

    ML-INFORMED MARKOV (optional): if match_df has an 'ml_informed_markov_p1' (or
    'ml_informed_pred') column AND pre_match_ml_informed_p1 is provided, that engine's
    trajectory is assembled the same way as the other two.

    HYBRID (optional): same pattern, keyed on an 'hybrid_p1' column and
    pre_match_hybrid_p1. If either engine's required inputs are missing, that field is
    left as None on the returned MatchTrajectory — the caller (and plot_trajectory) must
    handle that gracefully, not assume it is always present.
    """
    markov_col = "markov_p1" if "markov_p1" in match_df.columns else "markov_pred"
    ml_col = "ml_mc_p1" if "ml_mc_p1" in match_df.columns else "ml_pred"
    ml_informed_col = None
    for candidate in ("ml_informed_markov_p1", "ml_informed_pred"):
        if candidate in match_df.columns:
            ml_informed_col = candidate
            break
    hybrid_col = "hybrid_p1" if "hybrid_p1" in match_df.columns else None

    n = len(match_df)
    point_index = [0] + list(range(1, n + 1)) + [n + 1]
    markov_p1 = [pre_match_markov_p1] + match_df[markov_col].tolist() + [1.0 if winner_is_p1 else 0.0]
    ml_mc_p1 = [pre_match_ml_p1] + match_df[ml_col].tolist() + [1.0 if winner_is_p1 else 0.0]

    ml_informed_p1 = None
    if ml_informed_col is not None and pre_match_ml_informed_p1 is not None:
        ml_informed_p1 = (
            [pre_match_ml_informed_p1] + match_df[ml_informed_col].tolist()
            + [1.0 if winner_is_p1 else 0.0]
        )

    hybrid_p1 = None
    if hybrid_col is not None and pre_match_hybrid_p1 is not None:
        hybrid_p1 = (
            [pre_match_hybrid_p1] + match_df[hybrid_col].tolist()
            + [1.0 if winner_is_p1 else 0.0]
        )

    return MatchTrajectory(
        point_index=point_index, markov_p1=markov_p1, ml_mc_p1=ml_mc_p1,
        p1_name=p1_name, p2_name=p2_name, winner_is_p1=winner_is_p1,
        surface=surface, tournament=tournament, best_of=best_of, final_score=final_score,
        ml_informed_p1=ml_informed_p1, hybrid_p1=hybrid_p1,
    )