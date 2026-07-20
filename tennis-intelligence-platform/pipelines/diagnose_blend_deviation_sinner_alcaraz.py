"""
diagnose_blend_deviation_sinner_alcaraz.py — tests, against the actual, already-flagged
Sinner-Alcaraz anomaly (cyan starting near 0.30-0.44 when Markov's prior-implied start was
~0.72), whether the pre-match prior's near-erasure comes from the posterior (slow-decaying,
already confirmed correct in isolation) or from the blend layer (sensitivity_aware_blend)
pulling the final output toward the raw, instantaneous classifier prediction before the
evidence floor has meaningfully decayed.

Also logs is_second_serve_point per point, to rule out (or confirm) the simplest possible
explanation first: an unusual early concentration of second-serve points dragging raw_clf
(and, via the blend, blend_serve) down, independent of any seeding or blend-weighting bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_live_engines_v2 import POINT_FILES, PROCESSED
from tennis_intel.live.match_state_conversion import row_to_match_state
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.ml_informed_markov import (
    build_pretrained_prior, ServeReturnPosterior, ml_informed_point_probabilities,
    recursion_sensitivity, sensitivity_aware_blend,
)
from generate_publication_trajectory import compute_composite_prematch_probability

MATCH_ID = "20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz"
N_POINTS = 70


def main() -> None:
    print("Loading trained classifier...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

    print("Building point dataset...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]
    match = points[points["match_id"] == MATCH_ID].sort_values("Pt").reset_index(drop=True)

    if len(match) == 0:
        raise SystemExit(f"No points found for match_id={MATCH_ID!r} — check the ID is "
                          f"correct and this match exists in the loaded point files.")

    first_row = match.iloc[0].to_dict()
    p0_a_wins = compute_composite_prematch_probability(first_row)
    # BUG FIX (external review, 2026-07): see return_seed.py's module docstring.
    p_a_return_seed = compute_p_a_return_seed(first_row, track_winner=True)
    elo_a = first_row.get("elo_matches_played_pre_winner")
    elo_b = first_row.get("elo_matches_played_pre_loser")
    h2h = None
    if pd.notna(first_row.get("winner_h2h_wins_pre_match")) and pd.notna(first_row.get("loser_h2h_wins_pre_match")):
        h2h = float(first_row["winner_h2h_wins_pre_match"]) + float(first_row["loser_h2h_wins_pre_match"])
    best_of_val = int(first_row["best_of"]) if pd.notna(first_row.get("best_of")) else 5

    p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
        p0_a_wins, p_a_return_seed, best_of_val,
        elo_matches_played_a=elo_a, elo_matches_played_b=elo_b, h2h_meetings=h2h,
    )
    print(f"\nSeeded: p0_a_wins (XGBoost pre-match) = {p0_a_wins:.4f}")
    print(f"        p_serve0 (inverted point-level prior) = {p_serve0:.4f}")
    print(f"        n0_serve = {n0_serve:.2f}, n0_return = {n0_return:.2f}")
    print(f"(Compare p_serve0/the resulting Markov-implied match probability against the")
    print(f" ~0.72 Markov prior-implied start flagged in the anomaly — if p_serve0 itself")
    print(f" is already far from what would produce ~0.72, the bug is upstream of the")
    print(f" blend entirely, in this seeding/inversion step.)\n")

    posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)

    print(f"{'Pt':>4} {'2ndSrv':>7} {'raw_clf':>8} {'posterior_mean':>14} {'sens':>7} "
          f"{'weight_evid':>11} {'blend_serve':>11} {'blend_deviation':>15}")

    for i in range(min(N_POINTS, len(match))):
        row = match.iloc[i].to_dict()
        state = row_to_match_state(row)
        p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(row, model, feature_cols)

        sens_serve = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "serve")
        pts_obs_serve = posterior.points_observed_serve()
        weight_evid = 200.0 / (200.0 + pts_obs_serve)  # matches the current default
                                                        # evidence_floor_reference_points
        blend_serve = sensitivity_aware_blend(p_a_serve_raw, posterior.mean_serve(), sens_serve,
                                              points_observed=pts_obs_serve)
        blend_deviation = blend_serve - posterior.mean_serve()

        is_2nd = bool(row.get("is_second_serve_point"))
        print(f"{i+1:>4} {str(is_2nd):>7} {p_a_serve_raw:>8.4f} {posterior.mean_serve():>14.4f} "
              f"{sens_serve:>7.2f} {weight_evid:>11.4f} {blend_serve:>11.4f} "
              f"{blend_deviation:>+15.4f}")

        pt_winner = row.get("PtWinner")
        if pd.notna(pt_winner):
            p1_is_winner = bool(row.get("player1_is_winner", True))
            a_won = (int(pt_winner) == 1) if p1_is_winner else (int(pt_winner) == 2)
            if state.server_is_a:
                posterior = posterior.update_serve(a_won)
            else:
                posterior = posterior.update_return(a_won)

    n_2nd_serve = sum(1 for i in range(min(N_POINTS, len(match)))
                      if bool(match.iloc[i].get("is_second_serve_point")))
    print(f"\nSecond-serve points in this {min(N_POINTS, len(match))}-point window: "
          f"{n_2nd_serve} ({100*n_2nd_serve/min(N_POINTS, len(match)):.1f}%)")
    print("(Compare against this classifier's overall base rate for is_second_serve_point")
    print(" — if this window's concentration is unusually high relative to a typical ~35-40%")
    print(" second-serve rate, that alone could explain an early low/volatile raw_clf run,")
    print(" independent of any blend-weighting or seeding issue.)")

    print("\nWhat to check:")
    print("1. Does posterior_mean start near the Markov-implied ~0.72 and decay slowly")
    print("   (as already confirmed in isolation), or does it start noticeably lower")
    print("   than that — which would reopen the seeding/inversion question?")
    print("2. Is blend_deviation large and consistently NEGATIVE early on (before")
    print("   weight_evid has meaningfully decayed)? That would directly confirm the")
    print("   blend layer, not the posterior, is responsible for cyan's anomalous start.")
    print("3. Does the is_second_serve_point column show an unusual early concentration")
    print("   of True values, coinciding with the lowest raw_clf/blend_serve readings?")
    print("   If so, this may be a simpler, more mundane explanation than a structural")
    print("   blend-weighting bug.")


if __name__ == "__main__":
    main()