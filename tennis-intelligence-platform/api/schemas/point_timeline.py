"""
schemas/point_timeline.py — Pydantic response model for the Point Timeline table.
"""

from __future__ import annotations

from pydantic import BaseModel


class PointTimelineEntry(BaseModel):
    point_index: int
    server: str
    receiver: str
    winner: str
    score_before: str | None
    set1: int
    set2: int
    gm1: int
    gm2: int
    probability_before_p1: float
    probability_after_p1: float
    probability_swing: float
    is_break_point: bool
    is_set_point: bool
    is_match_point: bool
    is_tiebreak_point: bool
    is_largest_swing: bool


class PointTimelineResponse(BaseModel):
    match_id: str
    n_points_total: int
    n_points_returned: int
    points: list[PointTimelineEntry]