"""
routers/research_dashboard.py — the Research Dashboard endpoint.

Serves a PRECOMPUTED export — see tennis_intel.serving.research_dashboard_service's
own docstring for why this is never computed live.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.research_dashboard import ResearchDashboardResponse
from tennis_intel.serving.research_dashboard_service import load_research_dashboard

router = APIRouter(prefix="/api/research-dashboard", tags=["research-dashboard"])


@router.get("", response_model=ResearchDashboardResponse)
def get_research_dashboard() -> ResearchDashboardResponse:
    """
    Reliability diagrams, bootstrap-CI LogLoss/Brier, ECE, sharpness, and prediction
    distributions for all five engines on the holdout set — see
    pipelines/export_research_dashboard.py for how this is generated (run offline).
    """
    try:
        result = load_research_dashboard()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ResearchDashboardResponse(**result)