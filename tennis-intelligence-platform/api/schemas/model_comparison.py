"""
schemas/model_comparison.py — Pydantic response model for the Model Comparison page.
Field names match tennis_intel.serving.model_comparison_service's own JSON export
shape exactly.
"""

from __future__ import annotations

from pydantic import BaseModel


class EngineStats(BaseModel):
    display_name: str
    n_points: int
    log_loss: float
    brier: float
    ece: float


class ModelComparisonResponse(BaseModel):
    n_matches: int
    n_points: int
    holdout_year: int
    is_full_holdout: bool
    engines: dict[str, EngineStats]