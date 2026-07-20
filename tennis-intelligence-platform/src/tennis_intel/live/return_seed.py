"""
return_seed.py — the single, canonical computation of p_a_return_seed (the tracked
player A's pre-match return-point-win-rate seed, used to construct the Bayesian prior for
the ML-Informed Markov engine and, via markov_p_winner, pure Markov's own seeding too).

HISTORY, IMPORTANT (a real near-miss, documented so it doesn't recur): this module's
FIRST version fixed the wrong half of the problem. It replaced the old
"1 - opponent's first_serve_win_pct_career" construction with the tracked player's OWN
"return_pts_won_pct_career" directly — but markov_p_winner's own docstring already
documented, from an EARLIER bug fix, that using a player's own generic return average was
deliberately abandoned: it "silently ignores the actual loser's real serve ability,"
since a career return average is computed against a MIX of past opponents of varying
serve strength, not specifically against the current opponent. The first version of this
module would have reintroduced that exact, already-identified-and-rejected regression.

THE REAL, SINGLE ROOT CAUSE (confirmed on real data via the Sinner-Alcaraz investigation):
first_serve_win_pct_career is, by construction, ONLY the win rate on points where the
FIRST serve went in — not the opponent's TRUE overall serve-points-won rate across both
first and second serves (which are won at a meaningfully lower rate). Using
"1 - opponent's first_serve_win_pct_career" as an opponent-conditioned return-rate proxy
systematically UNDERSTATES the opponent's true serve strength, and therefore OVERSTATES
how weak the returner's seed should be. This is the SAME underlying gap responsible for
pure Markov's own serve-side bug (markov_p_winner's ps also used first_serve_win_pct_career
directly as if it were the overall serve rate) — one missing column
(combined_serve_win_pct_career, built in build_point_dataset.py), not two independent bugs.

THE FIX: keep the original opponent-conditioned SHAPE (1 - opponent's serve rate), fix
only the WEIGHTING — use the opponent's properly first+second-serve-weighted
combined_serve_win_pct_career instead of first_serve_win_pct_career alone.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_RETURN_SEED = 0.35


def compute_p_a_return_seed(row: dict, track_winner: bool = True) -> float:
    """
    Returns A's (the tracked player's) pre-match return-point-win-rate seed, as
    1 - the OPPONENT's true, properly-weighted overall serve-win rate — preserving
    opponent-conditioning, while fixing the first/second-serve weighting bug documented
    above.

    track_winner: True (the default) means A is the real match winner; pass False if the
    caller's own convention tracks the loser instead.

    Preference order:
      1. 1 - opponent's combined_serve_win_pct_career (see build_point_dataset.py).
      2. Fallback: 1 - opponent's first_serve_win_pct_career (surface then career) — the
         OLD, known-inferior construction, kept only when the combined-rate column is
         unavailable.
      3. Fallback: DEFAULT_RETURN_SEED.
    """
    opponent_combined_key = "loser_combined_serve_win_pct_career" if track_winner else "winner_combined_serve_win_pct_career"
    opponent_combined = row.get(opponent_combined_key)
    if opponent_combined is not None and pd.notna(opponent_combined):
        return 1.0 - float(opponent_combined)

    opponent_surface_key = "loser_first_serve_win_pct_surface_career" if track_winner else "winner_first_serve_win_pct_surface_career"
    opponent_career_key = "loser_first_serve_win_pct_career" if track_winner else "winner_first_serve_win_pct_career"

    opponent_serve_surface = row.get(opponent_surface_key)
    opponent_serve_career = row.get(opponent_career_key)
    if opponent_serve_surface is not None and pd.notna(opponent_serve_surface):
        return 1.0 - float(opponent_serve_surface)
    if opponent_serve_career is not None and pd.notna(opponent_serve_career):
        return 1.0 - float(opponent_serve_career)

    return DEFAULT_RETURN_SEED