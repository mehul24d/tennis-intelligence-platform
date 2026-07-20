"""
schemas/match.py — Pydantic response models for match-related endpoints.

Field names and shapes here are DELIBERATELY kept identical to what
tennis_intel.serving.replay_service.replay_match_by_id() already returns as a plain
dict — these models exist to give FastAPI request validation and OpenAPI docs, not to
introduce a second, divergent shape the frontend would need to reconcile.
"""

from __future__ import annotations

from pydantic import BaseModel


class PlayerRef(BaseModel):
    name: str


class PrematchProbabilities(BaseModel):
    """One or more of these fields is null when that engine has no genuine,
    zero-information pre-match computation available in this pipeline (see
    replay_match.py's own extensive comment on this exact point) — the frontend
    should render a dash, not a fabricated 50%, for any null field here."""
    markov: float | None = None
    ml_mc: float | None = None
    ml_informed_unsmoothed: float | None = None
    ml_informed_smoothed: float | None = None
    hybrid: float | None = None


class PointPrediction(BaseModel):
    point_index: int
    set1: int
    set2: int
    gm1: int
    gm2: int
    markov_p1: float
    ml_mc_p1: float
    ml_informed_unsmoothed_p1: float
    ml_informed_smoothed_p1: float
    hybrid_p1: float


class SetBoundary(BaseModel):
    set_number: int
    point_index: int
    score: str
    winner_is_p1: bool


class MatchReplayResponse(BaseModel):
    match_id: str
    player1: PlayerRef
    player2: PlayerRef
    winner: str
    n_points: int
    tournament: str | None = None
    date: str | None = None
    final_score: str | None = None
    prematch: PrematchProbabilities
    points: list[PointPrediction]
    set_boundaries: list[SetBoundary]


class MatchSearchResponse(BaseModel):
    match_ids: list[str]