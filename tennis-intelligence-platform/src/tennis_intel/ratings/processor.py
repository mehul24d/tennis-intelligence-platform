"""
processor.py — chronological match-by-match rating processing, generic across any
RatingSystem implementation (Elo today; Glicko-2, surface-specific, or decayed variants
later reuse this same loop unchanged).

CRITICAL DESIGN NOTE ON CHRONOLOGY:
TML-Database does not provide an actual per-match date — only `tourney_date` (the
TOURNAMENT'S START date, shared by every match in that event) plus `round` and `match_num`.
True chronological order within a tournament is therefore approximated as:

    (tourney_date, round_order, match_num)

where round_order encodes the standard bracket progression (R128 before R64 before ... before
F). This is a documented proxy, not exact match-by-match timestamps — it is precise enough
for Elo (which only needs "did this match happen before or after that one", not exact
timing), but should NOT be treated as a source of exact match dates elsewhere in the project.

LEAKAGE PROTECTION:
Ratings are updated via a single sequential pass in chronological order. Every match's
`elo_pre_match_*` values are read from the ratings dict BEFORE that match's outcome is
applied. There is no possibility of a later match influencing an earlier match's pre-match
rating, by construction of the loop — this is verified explicitly in
tests/unit/test_elo.py::TestLeakageProtection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from tennis_intel.ratings.base import RatingSystem

logger = logging.getLogger(__name__)

# Standard ATP bracket progression. RR (round robin) is treated as occurring before any
# knockout round, since it's used only in group-stage formats (e.g. ATP/WTA Finals) that have
# no R128/R64/etc in the same event. R256 (round of 256, seen in a handful of huge early
# Open-era Slam draws, e.g. 1968 Roland Garros) is placed before R128. BR / 3RD/4TH (bronze
# medal / 3rd-place playoff — both used across different eras/formats for the same concept)
# are placed just before the final. "Fs" (13 matches, all 1968 WCT round-robin events) is a
# LOW-CONFIDENCE mapping to "Final" — raw data gives no definition, but "Final Stage" of a
# round-robin format is the most plausible reading given the era and event type; volume is
# negligible (13 / 198,063 = 0.007%) so even if wrong, impact on aggregate Elo is immaterial.
# Unknown round labels beyond these are logged once (not per-row) and sorted last, after F.
ROUND_ORDER: dict[str, int] = {
    "RR": 0,
    "R256": 0,
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "BR": 7,
    "3RD/4TH": 7,
    "F": 8,
    "FS": 8,  # low-confidence, see note above
}
_UNKNOWN_ROUND_RANK = 99


@dataclass
class EloProcessingResult:
    augmented: pd.DataFrame  # input matches + elo columns, sorted chronologically
    final_ratings: dict[str, float]
    diagnostics: dict[str, float | int]


def _round_rank(round_series: pd.Series) -> pd.Series:
    cleaned = round_series.astype(str).str.strip().str.upper()
    unknown = set(cleaned.unique()) - set(ROUND_ORDER.keys())
    if unknown:
        logger.warning(
            "Unrecognized round label(s) %s — sorted after 'F' as a fallback. "
            "Add to ROUND_ORDER if this is a real, common round code.", sorted(unknown)
        )
    return cleaned.map(ROUND_ORDER).fillna(_UNKNOWN_ROUND_RANK).astype(int)


def default_dynamic_k(matches_played: int, base_k: float = 32.0,
                       provisional_k: float = 40.0, provisional_threshold: int = 30) -> float:
    """
    Standard FIDE-style dynamic K-factor: elevated K while a player's rating is still
    'provisional' (few rated matches, high uncertainty), stepping down to the standard K
    once enough matches have been played to trust the rating more. matches_played is the
    count BEFORE this match (leakage-safe by construction — see compute_ratings).
    """
    return provisional_k if matches_played < provisional_threshold else base_k


def compute_ratings(
    matches: pd.DataFrame,
    rating_system: RatingSystem,
    winner_id_col: str = "winner_id",
    loser_id_col: str = "loser_id",
    k: float | None = None,
    k_fn=None,
    retirement_col: str | None = None,
    retirement_k_multiplier: float = 0.5,
    walkover_col: str | None = None,
) -> EloProcessingResult:
    """
    Processes matches in strict chronological order, computing pre/post-match ratings for
    every match via a single sequential pass. Returns the augmented dataframe SORTED
    CHRONOLOGICALLY (not in the input's original order) — this is deliberate, since
    chronological order is the natural order for any downstream time-series feature (rolling
    stats, fatigue, etc.) built on top of this output.

    k_fn: optional callable(matches_played_before_this_match: int) -> float, for a dynamic
    K-factor (e.g. default_dynamic_k above). When provided, OVERRIDES the fixed `k` for
    every match — each player's own K is computed from THEIR OWN prior match count, so a
    provisional player facing an established one can have a different K than their
    opponent for the same match. When k_fn is None, behavior is identical to before this
    parameter existed (fixed k throughout) — this keeps every existing test passing
    unchanged, since none of them pass k_fn.

    retirement_col: optional column name (bool-like) flagging matches that ended in a
    retirement — such matches get their K multiplied by retirement_k_multiplier (default
    0.5), since the outcome may reflect injury rather than a genuine skill gap. When None,
    no matches are treated specially (identical to prior behavior).

    walkover_col: optional column name (bool-like) flagging true walkovers (no match was
    actually played) — such matches are EXCLUDED from the rating update entirely (ratings
    pass through unchanged), rather than counted as a real result. When None, no matches
    are excluded (identical to prior behavior).
    """
    df = matches.copy()

    missing_ids = df[winner_id_col].isna().sum() + df[loser_id_col].isna().sum()
    if missing_ids:
        logger.warning(
            "%d row(s) have a missing winner_id/loser_id and will be dropped before rating "
            "processing (cannot rate a match without knowing who played).", missing_ids
        )
        df = df.dropna(subset=[winner_id_col, loser_id_col])

    df["_round_rank"] = _round_rank(df["round"])
    sort_cols = ["tourney_date", "_round_rank", "match_num", "tourney_id"]
    missing_sort_cols = [c for c in sort_cols if c not in df.columns]
    if missing_sort_cols:
        raise ValueError(f"Missing required column(s) for chronological sort: {missing_sort_cols}")

    # kind="mergesort" for a stable sort — combined with a fully-specified tie-break tuple
    # (tourney_date, round, match_num, tourney_id should be jointly unique per row), this
    # makes the resulting order deterministic regardless of the input's original row order.
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    ratings: dict[str, float] = {}
    matches_played: dict[str, int] = {}  # for k_fn and the confidence signal
    cold_starts = 0
    update_magnitudes: list[float] = []
    walkovers_skipped = 0
    retirements_discounted = 0

    pre_w, pre_l, post_w, post_l, deltas, exp_probs, k_used = [], [], [], [], [], [], []
    matches_played_pre_w, matches_played_pre_l = [], []
    k_value = rating_system.default_k if k is None else k

    for row in df.itertuples(index=False):
        winner_id = getattr(row, winner_id_col)
        loser_id = getattr(row, loser_id_col)

        if winner_id not in ratings:
            ratings[winner_id] = rating_system.initial_rating
            matches_played[winner_id] = 0
            cold_starts += 1
        if loser_id not in ratings:
            ratings[loser_id] = rating_system.initial_rating
            matches_played[loser_id] = 0
            cold_starts += 1

        rw, rl = ratings[winner_id], ratings[loser_id]
        mp_w_pre, mp_l_pre = matches_played[winner_id], matches_played[loser_id]
        expected_w = rating_system.expected_score(rw, rl)

        is_walkover = bool(getattr(row, walkover_col)) if walkover_col else False

        if is_walkover:
            # No match was actually played — ratings pass through unchanged. Still record
            # pre-match values and expected probability for completeness/debugging, but
            # post == pre and delta == 0, and this match does NOT count toward either
            # player's matches_played (consistent with "no evidence about skill gathered").
            new_w, new_l = rw, rl
            delta = 0.0
            walkovers_skipped += 1
        else:
            this_k = k_fn(mp_w_pre) if k_fn is not None else k_value
            # Retirement discount applies the SAME reduced K to both players' updates for
            # this match (the outcome is less informative for BOTH directions, not just
            # the winner's gain) — implemented by scaling k_fn's/k's output before calling
            # update_ratings, so RatingSystem implementations need no retirement-awareness
            # of their own.
            is_retirement = bool(getattr(row, retirement_col)) if retirement_col else False
            if is_retirement:
                this_k = this_k * retirement_k_multiplier
                retirements_discounted += 1
            new_w, new_l = rating_system.update_ratings(rw, rl, k=this_k)
            delta = new_w - rw
            matches_played[winner_id] = mp_w_pre + 1
            matches_played[loser_id] = mp_l_pre + 1

        pre_w.append(rw)
        pre_l.append(rl)
        post_w.append(new_w)
        post_l.append(new_l)
        deltas.append(delta)
        exp_probs.append(expected_w)
        k_used.append(0.0 if is_walkover else (k_fn(mp_w_pre) if k_fn is not None else k_value))
        matches_played_pre_w.append(mp_w_pre)
        matches_played_pre_l.append(mp_l_pre)
        update_magnitudes.append(abs(delta))

        ratings[winner_id] = new_w
        ratings[loser_id] = new_l

    df["elo_pre_match_winner"] = pre_w
    df["elo_pre_match_loser"] = pre_l
    df["elo_post_match_winner"] = post_w
    df["elo_post_match_loser"] = post_l
    df["elo_delta"] = deltas
    df["expected_win_prob"] = exp_probs
    df["k_factor_used"] = k_used
    df["elo_matches_played_pre_winner"] = matches_played_pre_w
    df["elo_matches_played_pre_loser"] = matches_played_pre_l
    df = df.drop(columns=["_round_rank"])

    final_values = list(ratings.values())
    diagnostics = {
        "processed_matches": len(df),
        "players_rated": len(ratings),
        "initializations": cold_starts,
        "walkovers_skipped": walkovers_skipped,
        "retirements_discounted": retirements_discounted,
        "average_rating": sum(final_values) / len(final_values) if final_values else float("nan"),
        "min_rating": min(final_values) if final_values else float("nan"),
        "max_rating": max(final_values) if final_values else float("nan"),
        "largest_single_update": max(update_magnitudes) if update_magnitudes else float("nan"),
        "mean_update_magnitude": (
            sum(update_magnitudes) / len(update_magnitudes) if update_magnitudes else float("nan")
        ),
    }

    return EloProcessingResult(augmented=df, final_ratings=ratings, diagnostics=diagnostics)