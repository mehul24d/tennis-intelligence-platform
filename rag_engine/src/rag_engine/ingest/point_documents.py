"""point_documents.py — turns notable in-match points (large live win-probability
swings) into retrievable RagDocuments, sourced from v1's point_timeline_service.

REUSES, DOES NOT REIMPLEMENT: every probability/swing value here comes straight from
tennis_intel.serving.point_timeline_service.get_point_timeline — the same, now-fixed
service backing the v1 Point Timeline table (see
tennis-intelligence-platform/docs/known_issue_ml_informed_markov_pre_point_state.md
for the pre-point/post-point indexing fix this relies on).

SWING-NEUTRAL PHRASING, DELIBERATE: see
tennis-intelligence-platform/docs/known_issue_after_point_swing_includes_next_point_context.md
— a swing measured "around" point i is NOT purely a consequence of point i's own
outcome; a large share of it can come from point i+1's own context (e.g. whether it's
a second-serve point), unrelated to who won point i. Text generated here therefore
never asserts that the point's outcome CAUSED the swing (no "X's break-point save
swung the match" framing) — it states the swing as an observed fact anchored to the
point's score/context, and leaves causal interpretation to the downstream LLM agent
(Phase 2), which can reason about it with more context than a templated string can.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from rag_engine import _v1_path  # noqa: F401 — sets up sys.path for the imports below
from rag_engine.ingest.types import RagDocument

from tennis_intel.serving.replay_service import ReplayContext, load_replay_context
from tennis_intel.serving.point_timeline_service import get_point_timeline

# Only points whose |swing| is at least this are "notable" enough to index — matches
# the threshold already used for the Point Timeline table's own default filtering.
MIN_SWING = 0.10


def _point_type_clause(point: dict) -> str:
    """Objective facts about the point itself (break/set/match/tiebreak point) — safe
    to state factually, unlike the swing's cause, since these flags describe the point
    being played, not an inference about why the probability moved."""
    flags = []
    if point.get("is_match_point"):
        flags.append("match point")
    elif point.get("is_set_point"):
        flags.append("set point")
    elif point.get("is_break_point"):
        flags.append("break point")
    if point.get("is_tiebreak_point"):
        flags.append("tiebreak")
    return f" ({', '.join(flags)})" if flags else ""


def _point_text(
    point: dict, p1_name: str, p2_name: str, tournament: str | None,
    round_: str | None, surface: str | None, date_str: str | None,
) -> str:
    tourney_clause = f"{tournament} ({surface}), {round_}" if tournament else "this match"
    date_clause = f" on {date_str}" if date_str else ""
    point_type = _point_type_clause(point)

    sentence = (
        f"{p1_name} vs {p2_name}, {tourney_clause}{date_clause}. At set "
        f"{point['set1']}-{point['set2']}, games {point['gm1']}-{point['gm2']}, "
        f"score {point['score_before']} ({point['server']} serving){point_type}: "
        f"point won by {point['winner']}."
    )

    # Swing stated as a SEPARATE, explicitly-labeled fact about the OVERALL MATCH
    # probability, not folded into the same sentence as "point won by X" — juxtaposing
    # them there implies causality even without causal words (see module docstring:
    # a large share of a swing can come from the FOLLOWING point's own context, e.g.
    # whether it's a second serve, unrelated to who won THIS point). "moved" (not
    # "rose"/"fell") avoids even a directional causal framing.
    before, after = point["probability_before_p1"], point["probability_after_p1"]
    sentence += (
        f" Separately: {p1_name}'s overall match win probability moved from "
        f"{before:.1%} to {after:.1%} (a {point['probability_swing']:.1%} swing) around "
        f"this stage of the match."
    )

    # When the swing direction doesn't match "the point winner's side went up" (exactly
    # the pattern traced in known_issue_after_point_swing_includes_next_point_context.md
    # -- e.g. player 2 wins the point shown above, but player 1's probability rises),
    # make that explicit rather than leaving a reader (or the downstream LLM agent) to
    # infer a false causal link from the juxtaposition alone.
    p1_won = point["winner"] == p1_name
    probability_moved_for_p1 = after > before
    direction_matches_winner = p1_won == probability_moved_for_p1
    if not direction_matches_winner:
        sentence += (
            " This movement is not attributable solely to this point's outcome -- it "
            "likely also reflects broader context around this stage of the match "
            "(e.g. the difficulty of the next point to be served)."
        )
    return sentence


def _point_metadata(
    point: dict, match_id: str, p1_name: str, p2_name: str,
    surface: str | None, tourney_level: str | None, round_: str | None,
    date_str: str | None, year: int,
) -> dict:
    return {
        "doc_type": "notable_point",
        "match_id": match_id,
        "point_index": int(point["point_index"]),
        "player1": p1_name,
        "player2": p2_name,
        "winner": str(point["winner"]),
        "server": str(point["server"]),
        "surface": surface or "",
        "tourney_level": tourney_level or "",
        "round": round_ or "",
        "date": date_str or "",
        "year": year,
        "swing": float(point["probability_swing"]),
        "is_break_point": bool(point.get("is_break_point", False)),
        "is_set_point": bool(point.get("is_set_point", False)),
        "is_match_point": bool(point.get("is_match_point", False)),
        "is_tiebreak_point": bool(point.get("is_tiebreak_point", False)),
    }


def build_point_documents(
    ctx: ReplayContext | None = None, match_limit: int | None = None,
    min_swing: float = MIN_SWING,
) -> Iterator[RagDocument]:
    """Yields one RagDocument per notable point (|swing| >= min_swing) across matches
    in the frozen-join corpus. `ctx` can be passed in (already loaded) to avoid paying
    the load cost twice when building multiple document types in one process; if
    omitted, loads it here. `match_limit` caps the number of MATCHES scanned (not
    points), for fast iteration.
    """
    if ctx is None:
        ctx = load_replay_context()

    match_ids = list(ctx.match_ids)
    if match_limit is not None:
        match_ids = match_ids[:match_limit]

    frozen_join = ctx.frozen_join
    day6 = ctx.day6

    for match_id in match_ids:
        timeline = get_point_timeline(ctx, match_id, min_swing=min_swing)
        if timeline["n_points_returned"] == 0:
            continue

        fj_row = frozen_join[frozen_join["mcp_match_id"] == match_id]
        p1_name = fj_row["mcp_Player 1"].iloc[0] if len(fj_row) else match_id.split("-")[-2]
        p2_name = fj_row["mcp_Player 2"].iloc[0] if len(fj_row) else match_id.split("-")[-1]

        surface = tourney_level = round_ = date_str = None
        year = -1
        if len(fj_row):
            tml_tourney_id = fj_row["tml_tourney_id"].iloc[0]
            tml_match_num = fj_row["tml_match_num"].iloc[0]
            tml_winner_id = fj_row["tml_winner_id"].iloc[0]
            tml_loser_id = fj_row["tml_loser_id"].iloc[0]
            day6_row = day6[
                (day6["tourney_id"] == tml_tourney_id)
                & (day6["match_num"] == tml_match_num)
                & (day6["winner_id"] == tml_winner_id)
                & (day6["loser_id"] == tml_loser_id)
            ]
            if len(day6_row):
                tournament = day6_row["tourney_name"].iloc[0]
                surface = day6_row["surface"].iloc[0]
                tourney_level = day6_row["tourney_level"].iloc[0]
                round_ = day6_row["round"].iloc[0]
                date_val = day6_row["tourney_date"].iloc[0]
                if date_val is not None:
                    date_str = date_val.date().isoformat()
                    year = date_val.year
            else:
                tournament = None
        else:
            tournament = None

        for point in timeline["points"]:
            yield RagDocument(
                doc_id=f"point:{match_id}:{point['point_index']}",
                text=_point_text(point, p1_name, p2_name, tournament, round_, surface, date_str),
                metadata=_point_metadata(
                    point, match_id, p1_name, p2_name, surface, tourney_level, round_,
                    date_str, year,
                ),
            )
