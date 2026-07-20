"""
schemas/rankings.py — Pydantic response models for the Rankings dashboard.
"""

from __future__ import annotations

from pydantic import BaseModel


class CurrentEloEntry(BaseModel):
    rank: int
    player_id: str
    player_name: str
    elo: float


class PeakEloEntry(BaseModel):
    rank: int
    player_id: str
    player_name: str
    peak_elo: float
    date_achieved: str | None


class SurfaceEloEntry(BaseModel):
    rank: int
    player_id: str
    player_name: str
    surface_elo: float


class PeakSurfaceEloEntry(BaseModel):
    rank: int
    player_id: str
    player_name: str
    peak_surface_elo: float
    date_achieved: str | None


class UpsetEntry(BaseModel):
    rank: int
    match_id: str
    date: str | None
    tournament: str
    round: str
    winner_name: str
    winner_elo: float
    loser_name: str
    loser_elo: float
    elo_gap: float