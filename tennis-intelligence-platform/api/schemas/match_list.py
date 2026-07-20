"""
schemas/match_list.py — Pydantic response models for the Match Explorer endpoint.

Field names match tennis_intel.serving.match_list_service's own return shape exactly
(see that module's docstring on why "prematch_favourite" compares Elo via
tml_winner_name/tml_loser_name, not MCP's own Player 1/2 slots).
"""

from __future__ import annotations

from pydantic import BaseModel


class MatchSummary(BaseModel):
    match_id: str
    tournament: str
    year: int | None
    surface: str
    round: str
    winner: str
    loser: str
    final_score: str | None
    duration_minutes: int | None
    tournament_level: str | None
    best_of: int | None
    winner_elo: float | None
    loser_elo: float | None
    prematch_favourite: str | None


class MatchListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    matches: list[MatchSummary]