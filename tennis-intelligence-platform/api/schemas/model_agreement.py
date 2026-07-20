"""
schemas/model_agreement.py — Pydantic response model for the Model Agreement Panel.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelAgreementPoint(BaseModel):
    point_index: int
    highest_probability: float
    highest_probability_engine: str
    lowest_probability: float
    lowest_probability_engine: str
    average_probability: float
    std_dev: float
    max_disagreement: float
    most_confident_engine: str
    least_confident_engine: str
    changing_fastest_engine: str | None


class DisagreementSummary(BaseModel):
    points_disagreeing_over_5pct: int
    points_disagreeing_over_10pct: int
    points_disagreeing_over_20pct: int


class ModelAgreementResponse(BaseModel):
    match_id: str
    n_points: int
    points: list[ModelAgreementPoint]
    disagreement_summary: DisagreementSummary