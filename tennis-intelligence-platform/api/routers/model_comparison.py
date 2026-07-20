"""
routers/model_comparison.py — the Model Comparison page endpoint.

Serves a PRECOMPUTED export (see tennis_intel.serving.model_comparison_service's own
docstring for why this is never computed live) — a 404 with a clear message if the
export hasn't been generated yet, not a bare 500.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.model_comparison import ModelComparisonResponse
from tennis_intel.serving.model_comparison_service import load_model_comparison

router = APIRouter(prefix="/api/model-comparison", tags=["model-comparison"])


@router.get("", response_model=ModelComparisonResponse)
def get_model_comparison() -> ModelComparisonResponse:
    """
    LogLoss/Brier/ECE for all five prediction engines on the holdout set — see
    pipelines/export_model_comparison.py for how this is generated (run offline,
    not on every request).
    """
    try:
        result = load_model_comparison()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ModelComparisonResponse(**result)