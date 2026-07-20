"""
schemas/research_dashboard.py — Pydantic response model for the Research Dashboard.
Field names match tennis_intel.serving.research_dashboard_service's own JSON export
shape exactly.
"""

from __future__ import annotations

from pydantic import BaseModel


class MetricWithCI(BaseModel):
    point_estimate: float
    ci_lower: float
    ci_upper: float


class ReliabilityPoint(BaseModel):
    bin_index: int
    n: int
    mean_predicted: float
    observed_win_rate: float
    calibration_gap: float


class PredictionHistogram(BaseModel):
    bin_edges: list[float]
    counts: list[int]


class EngineResearchStats(BaseModel):
    display_name: str
    n_points: int
    log_loss: MetricWithCI
    brier: MetricWithCI
    ece: float
    sharpness: float
    reliability_diagram: list[ReliabilityPoint]
    prediction_histogram: PredictionHistogram


class ResearchDashboardResponse(BaseModel):
    n_matches: int
    n_points: int
    holdout_year: int
    is_full_holdout: bool
    n_calibration_bins: int
    n_bootstrap: int
    engines: dict[str, EngineResearchStats]