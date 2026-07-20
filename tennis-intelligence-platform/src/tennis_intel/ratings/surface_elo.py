"""
surface_elo.py — surface-specific Elo (hard/clay/grass), reusing the exact same
chronological, leakage-safe processing loop in processor.py — one independent pass per
surface, not a new loop.

DESIGN DECISION, stated explicitly: each surface's ratings are seeded independently at
1500 (the same cold-start policy as overall Elo), NOT initialized from a player's overall
Elo. This is deliberate: mixing the two would entangle overall and surface-specific skill
in a way that undermines the whole point of having both as separate signals (see the
project's own Elo-redesign design note). It also matches standard practice in published
tennis rating systems (Sackmann, FiveThirtyEight) that maintain independent per-surface
ladders. The tradeoff — a player's grass rating starts uninformatively at 1500 even if we
already know a lot about them from hard/clay — is a real limitation of this simple
approach, not an oversight; a hierarchical/Bayesian shrinkage model would be the correct
fix and is intentionally out of scope for this pass (documented as a future extension).

LEAKAGE: each surface subset is chronologically sorted and processed via compute_ratings'
existing, already-tested sequential loop — leakage safety is inherited directly from that
function, not re-implemented here. Matches on OTHER surfaces are simply invisible to a
given surface's rating stream — a player's clay rating is influenced only by their own
past clay matches, never by a hard-court result sandwiched between two clay matches in
real chronological time.
"""

from __future__ import annotations

import logging

import pandas as pd

from tennis_intel.ratings.base import RatingSystem
from tennis_intel.ratings.processor import compute_ratings

logger = logging.getLogger(__name__)

STANDARD_SURFACES = ["Hard", "Clay", "Grass"]


def compute_surface_ratings(
    matches: pd.DataFrame,
    rating_system_factory,  # callable() -> RatingSystem, called once per surface (fresh instance)
    surface_col: str = "surface",
    winner_id_col: str = "winner_id",
    loser_id_col: str = "loser_id",
    k: float | None = None,
    k_fn=None,
    retirement_col: str | None = None,
    walkover_col: str | None = None,
    surfaces: list[str] | None = None,
) -> pd.DataFrame:
    """
    Computes independent Elo ratings per surface and merges the results back onto the
    original match index, with columns named elo_surface_pre_match_winner/loser,
    elo_surface_post_match_winner/loser, elo_surface_matches_played_pre_winner/loser.

    Matches whose surface is not in `surfaces` (default: Hard/Clay/Grass) are left with
    NaN in these columns — e.g. "Carpet", a discontinued surface with very few matches in
    the dataset, deliberately not given its own ladder here.

    rating_system_factory must return a FRESH RatingSystem instance each call (not a
    shared one) — each surface needs its own independent internal state, and reusing one
    instance across calls would be a subtle bug this factory pattern exists to prevent.
    """
    surfaces = surfaces or STANDARD_SURFACES
    matches = matches.copy()
    matches["_orig_row_id"] = range(len(matches))

    result_frames = []
    for surface in surfaces:
        subset = matches[matches[surface_col] == surface].copy()
        if subset.empty:
            logger.warning("No matches found for surface '%s' — skipping.", surface)
            continue
        logger.info("Computing surface Elo for %s: %d matches", surface, len(subset))

        result = compute_ratings(
            subset, rating_system_factory(), winner_id_col=winner_id_col,
            loser_id_col=loser_id_col, k=k, k_fn=k_fn,
            retirement_col=retirement_col, walkover_col=walkover_col,
        )
        aug = result.augmented[[
            "_orig_row_id", "elo_pre_match_winner", "elo_pre_match_loser",
            "elo_post_match_winner", "elo_post_match_loser",
            "elo_matches_played_pre_winner", "elo_matches_played_pre_loser",
        ]].rename(columns={
            "elo_pre_match_winner": "elo_surface_pre_match_winner",
            "elo_pre_match_loser": "elo_surface_pre_match_loser",
            "elo_post_match_winner": "elo_surface_post_match_winner",
            "elo_post_match_loser": "elo_surface_post_match_loser",
            "elo_matches_played_pre_winner": "elo_surface_matches_played_pre_winner",
            "elo_matches_played_pre_loser": "elo_surface_matches_played_pre_loser",
        })
        result_frames.append(aug)

    if not result_frames:
        raise ValueError(f"No matches found for any of the requested surfaces: {surfaces}")

    all_surface_results = pd.concat(result_frames, ignore_index=True)
    merged = matches.merge(all_surface_results, on="_orig_row_id", how="left")
    return merged.drop(columns=["_orig_row_id"])