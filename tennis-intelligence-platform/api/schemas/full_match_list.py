"""
schemas/full_match_list.py — Pydantic response models for the FULL-corpus Match
Explorer endpoint. Genuinely separate from schemas/match_list.py, which backs the
existing, MCP-limited endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel


class FullMatchSummary(BaseModel):
    match_id: str
    has_replay_data: bool
    tournament: str
    year: int | None
    surface: str
    round: str
    winner: str
    loser: str
    final_score: str | None
    tournament_level: str | None
    best_of: int | None
    winner_elo: float | None
    loser_elo: float | None


class FullMatchListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    matches: list[FullMatchSummary]