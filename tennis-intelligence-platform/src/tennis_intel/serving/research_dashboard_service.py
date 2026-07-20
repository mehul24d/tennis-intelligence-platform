"""
research_dashboard_service.py — serves the precomputed research dashboard export
produced by pipelines/export_research_dashboard.py. Same reasoning as
model_comparison_service.py: never recomputed live.
"""

from __future__ import annotations

import json
from pathlib import Path

EXPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "research_dashboard_export.json"


def load_research_dashboard() -> dict:
    """Reads the precomputed research dashboard export. Raises FileNotFoundError
    with a clear, actionable message if the export hasn't been generated yet."""
    if not EXPORT_PATH.exists():
        raise FileNotFoundError(
            f"Research dashboard export not found at {EXPORT_PATH}. Run "
            f"'python pipelines/export_research_dashboard.py' (add --full for the "
            f"complete holdout set once a quick run looks sane) before this endpoint "
            f"can serve anything."
        )
    return json.loads(EXPORT_PATH.read_text())