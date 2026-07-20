"""
model_agreement_service.py — the service layer behind the Model Agreement Panel,
computing per-point cross-engine statistics (highest/lowest/average/std-dev
probability, max disagreement, which engine is most/least confident, which is
changing fastest) plus a match-wide summary of how often engines disagree by more
than 5%/10%/20%.

REUSES tennis_intel.serving.replay_service.compute_five_engine_trajectory — same
shared computation as replay_match_by_id and get_match_summary, not a fifth
independent copy of the seeding + per-point loop.
"""

from __future__ import annotations

import numpy as np

from tennis_intel.serving.replay_service import ReplayContext, compute_five_engine_trajectory

ENGINE_DISPLAY_NAMES = {
    "markov_p1": "Analytical Markov",
    "ml_p1": "Machine Learning + Monte Carlo",
    "ml_informed_unsmoothed_p1": "ML-Informed Markov (Unsmoothed)",
    "ml_informed_p1": "ML-Informed Markov (Smoothed)",
    "hybrid_p1": "Hybrid Engine",
}


def get_model_agreement(ctx: ReplayContext, match_id: str) -> dict:
    """
    Computes the Model Agreement Panel data for one match. Raises ValueError if
    match_id isn't in the frozen-join corpus.
    """
    computed = compute_five_engine_trajectory(ctx, match_id)
    records = computed["records"]
    n_points = len(records)

    # engine_matrix[i] = [markov, ml_mc, unsmoothed, smoothed, hybrid] at point i
    engine_keys = ["markov_p1", "ml_p1", "ml_informed_unsmoothed_p1", "ml_informed_p1", "hybrid_p1"]
    engine_matrix = np.array([computed[k] for k in engine_keys]).T  # shape (n_points, 5)

    per_point = []
    disagreement_5pct = disagreement_10pct = disagreement_20pct = 0
    prev_row = None

    for i in range(n_points):
        row_vals = engine_matrix[i]
        highest_idx = int(np.argmax(row_vals))
        lowest_idx = int(np.argmin(row_vals))
        max_disagreement = float(row_vals[highest_idx] - row_vals[lowest_idx])

        # "Most confident" = furthest from 0.5 (most decisive); "least confident" =
        # closest to 0.5 (most uncertain) — a genuinely different notion from
        # highest/lowest RAW probability, worth keeping distinct.
        distances_from_half = np.abs(row_vals - 0.5)
        most_confident_idx = int(np.argmax(distances_from_half))
        least_confident_idx = int(np.argmin(distances_from_half))

        changing_fastest_idx = None
        if prev_row is not None:
            deltas = np.abs(row_vals - prev_row)
            changing_fastest_idx = int(np.argmax(deltas))

        if max_disagreement >= 0.20:
            disagreement_20pct += 1
        if max_disagreement >= 0.10:
            disagreement_10pct += 1
        if max_disagreement >= 0.05:
            disagreement_5pct += 1

        per_point.append({
            "point_index": int(records[i]["Pt"]),
            "highest_probability": round(float(row_vals[highest_idx]), 6),
            "highest_probability_engine": ENGINE_DISPLAY_NAMES[engine_keys[highest_idx]],
            "lowest_probability": round(float(row_vals[lowest_idx]), 6),
            "lowest_probability_engine": ENGINE_DISPLAY_NAMES[engine_keys[lowest_idx]],
            "average_probability": round(float(row_vals.mean()), 6),
            "std_dev": round(float(row_vals.std()), 6),
            "max_disagreement": round(max_disagreement, 6),
            "most_confident_engine": ENGINE_DISPLAY_NAMES[engine_keys[most_confident_idx]],
            "least_confident_engine": ENGINE_DISPLAY_NAMES[engine_keys[least_confident_idx]],
            "changing_fastest_engine": (
                ENGINE_DISPLAY_NAMES[engine_keys[changing_fastest_idx]]
                if changing_fastest_idx is not None else None
            ),
        })
        prev_row = row_vals

    return {
        "match_id": match_id,
        "n_points": n_points,
        "points": per_point,
        "disagreement_summary": {
            "points_disagreeing_over_5pct": disagreement_5pct,
            "points_disagreeing_over_10pct": disagreement_10pct,
            "points_disagreeing_over_20pct": disagreement_20pct,
        },
    }