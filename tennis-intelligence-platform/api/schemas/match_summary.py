"""
schemas/match_summary.py — Pydantic response model for the Match Summary cards
endpoint. Field names match tennis_intel.serving.match_summary_service's own return
shape exactly.
"""

from __future__ import annotations

from pydantic import BaseModel


class ProbabilitySwing(BaseModel):
    point_index: int
    probability_before: float
    probability_after: float
    swing: float


class LargestComeback(BaseModel):
    lowest_win_probability: float
    comeback_margin: float
    point_index_of_low: int


class StreakByPlayer(BaseModel):
    player1: int
    player2: int


class BreakPoints(BaseModel):
    player1_created: int
    player1_converted: int
    player2_created: int
    player2_converted: int


class MatchSummaryResponse(BaseModel):
    match_id: str
    largest_probability_swing: ProbabilitySwing
    largest_comeback: LargestComeback
    longest_winning_streak_points: StreakByPlayer
    longest_service_hold_points: int
    break_points: BreakPoints
    total_winners: int | None
    total_unforced_errors: int | None
    serve_percentage: float | None