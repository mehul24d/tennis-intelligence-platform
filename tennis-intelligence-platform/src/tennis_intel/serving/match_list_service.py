"""
match_list_service.py — the service layer behind the Match Explorer table endpoint.

Every column referenced here is CONFIRMED to exist via
pipelines/diagnose_frozen_join_schema.py's real output (run against the actual
frozen_join and day6 parquet files) — not guessed, per this project's standing
discipline after two real bugs this session came from assuming a column existed
without checking (the surface second-serve merge-list gap, the
combined_serve_win_pct_career KeyError).

frozen_join and day6 are joined on (tml_tourney_id, tml_match_num, tml_winner_id,
tml_loser_id) — the SAME merge key already established and validated in
pipelines/build_day6_features.py, reused here rather than re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MatchListContext:
    """The merged, enriched match-level table this service's functions operate on —
    built once (reuses the same frozen_join/day6 parquet files the replay context
    already loads) and reused across every request."""
    matches: pd.DataFrame


def load_match_list_context(frozen_join: pd.DataFrame, day6: pd.DataFrame) -> MatchListContext:
    """
    Builds the enriched, one-row-per-match table backing the Match Explorer page —
    tournament, year, surface, round, winner/loser, final score, duration, tournament
    level, best-of, and pre-match Elo for both players.

    Pass the SAME frozen_join/day6 dataframes already loaded for the replay context
    (tennis_intel.serving.replay_service.load_replay_context) — no need to reload from
    disk a second time if the caller already has them in memory.
    """
    merge_key = ["tml_tourney_id", "tml_match_num", "tml_winner_id", "tml_loser_id"]
    day6_renamed = day6.rename(columns={
        "tourney_id": "tml_tourney_id", "match_num": "tml_match_num",
        "winner_id": "tml_winner_id", "loser_id": "tml_loser_id",
    })
    merged = frozen_join.merge(
        day6_renamed[merge_key + [
            "elo_pre_match_winner", "elo_pre_match_loser",
            "elo_surface_pre_match_winner", "elo_surface_pre_match_loser",
        ]],
        on=merge_key, how="left",
    )

    if len(merged) != len(frozen_join):
        raise AssertionError(
            f"Row count changed during merge: {len(frozen_join):,} -> {len(merged):,}. "
            f"This indicates a non-unique merge key (fan-out bug) — do not trust this "
            f"output. See this project's own established discipline on this exact "
            f"failure mode (build_day6_features.py's row-count safety net)."
        )

    return MatchListContext(matches=merged)


def _row_to_match_summary(row: pd.Series) -> dict:
    """One row of the Match Explorer table. Field names deliberately match the
    spec's own column list (Tournament, Year, Surface, Round, Winner, Loser, Final
    Score, Duration, Tournament Level, Best of, Winner Elo, Loser Elo, Pre-match
    favourite) — see api/schemas/match_list.py for the corresponding Pydantic model."""
    winner_elo = row.get("elo_pre_match_winner")
    loser_elo = row.get("elo_pre_match_loser")
    prematch_favourite = None
    if pd.notna(winner_elo) and pd.notna(loser_elo):
        # Uses tml_winner_name/tml_loser_name (unambiguous) rather than mcp_Player 1/2,
        # since MCP's Player 1/2 slots do NOT consistently map to winner/loser across
        # matches (confirmed throughout this project — see
        # match_state_conversion.py's own extensive commentary on this exact point).
        prematch_favourite = row["tml_winner_name"] if winner_elo >= loser_elo else row["tml_loser_name"]

    return {
        "match_id": row["mcp_match_id"],
        "tournament": row["mcp_Tournament"],
        "year": int(row["tml_tourney_date"].year) if pd.notna(row["tml_tourney_date"]) else None,
        "surface": row["mcp_Surface"],
        "round": row["mcp_Round"],
        "winner": row["tml_winner_name"],
        "loser": row["tml_loser_name"],
        "final_score": row.get("tml_score"),
        "duration_minutes": (
            int(row["tml_minutes"]) if pd.notna(row.get("tml_minutes")) else None
        ),
        "tournament_level": row.get("tml_tourney_level"),
        "best_of": int(row["tml_best_of"]) if pd.notna(row.get("tml_best_of")) else None,
        "winner_elo": round(float(winner_elo), 1) if pd.notna(winner_elo) else None,
        "loser_elo": round(float(loser_elo), 1) if pd.notna(loser_elo) else None,
        "prematch_favourite": prematch_favourite,
    }


def get_match_list(
    ctx: MatchListContext,
    player: str | None = None,
    tournament: str | None = None,
    year: int | None = None,
    surface: str | None = None,
    round_: str | None = None,
    tourney_level: str | None = None,
    best_of: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """
    Filterable, paginated match list for the Match Explorer table.

    player: substring match against EITHER mcp_Player 1 or mcp_Player 2 (case-
    insensitive), so a search for "Sinner" finds every match Sinner played in,
    regardless of which slot he occupies for that specific match.
    tourney_level: TML's own coding, confirmed directly against real data (not
    assumed): '250', '500', 'M' (Masters), 'G' (Grand Slam) — an earlier version of
    this docstring incorrectly claimed 250/500 were collapsed into a single code 'A'
    and therefore unfilterable separately; that was wrong, caught by inspecting a
    real API response rather than the raw schema dump alone. Both ATP500 and ATP250
    ARE filterable directly via this parameter, exactly as the Match Explorer spec
    asks for — no gap here after all.
    """
    df = ctx.matches

    if player:
        p = player.lower()
        mask = (
            df["mcp_Player 1"].str.lower().str.contains(p, na=False)
            | df["mcp_Player 2"].str.lower().str.contains(p, na=False)
        )
        df = df[mask]
    if tournament:
        df = df[df["mcp_Tournament"].str.lower().str.contains(tournament.lower(), na=False)]
    if year is not None:
        df = df[df["tml_tourney_date"].dt.year == year]
    if surface:
        df = df[df["mcp_Surface"].str.lower() == surface.lower()]
    if round_:
        df = df[df["mcp_Round"].str.lower() == round_.lower()]
    if tourney_level:
        df = df[df["tml_tourney_level"] == tourney_level]
    if best_of is not None:
        df = df[df["tml_best_of"] == best_of]

    total = len(df)
    page = df.iloc[offset:offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "matches": [_row_to_match_summary(row) for _, row in page.iterrows()],
    }