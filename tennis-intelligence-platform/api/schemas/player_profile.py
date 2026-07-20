"""
schemas/player_profile.py — Pydantic response models for the Player Profile page.
"""

from __future__ import annotations

from pydantic import BaseModel


class PlayerSearchResult(BaseModel):
    player_id: str
    player_name: str


class SurfaceStat(BaseModel):
    matches: int
    wins: int
    win_pct: float | None


class GrandSlamStats(BaseModel):
    matches: int
    wins: int
    win_pct: float | None


class RecentFormEntry(BaseModel):
    match_id: str
    date: str | None
    opponent: str
    won: bool
    surface: str
    tournament: str


class EloTimelinePoint(BaseModel):
    match_id: str
    date: str | None
    elo: float | None
    surface_elo: float | None


class PlayerProfileResponse(BaseModel):
    player_id: str
    player_name: str
    current_elo: float | None
    peak_elo: float | None
    career_matches: int
    career_wins: int
    career_losses: int
    career_win_pct: float | None
    surface_stats: dict[str, SurfaceStat]
    grand_slam_stats: GrandSlamStats
    recent_form: list[RecentFormEntry]
    elo_timeline: list[EloTimelinePoint]


class HeadToHeadMatch(BaseModel):
    match_id: str
    date: str | None
    tournament: str
    surface: str
    winner_id: str
    score: str | None


class HeadToHeadResponse(BaseModel):
    player_id_a: str
    player_id_b: str
    a_wins: int
    b_wins: int
    matches: list[HeadToHeadMatch]