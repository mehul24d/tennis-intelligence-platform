"""
model_comparison_service.py — serves the precomputed model-comparison stats produced
by pipelines/export_model_comparison.py.

DELIBERATELY does NOT recompute anything live — a full-holdout run takes ~9+ minutes
and involves per-point Monte Carlo simulation for ML+MC, making it completely
unsuitable for a live HTTP request. Run export_model_comparison.py OFFLINE (ideally
with --full, once a smoke-test run confirms it looks sane) whenever the underlying
model or holdout set changes, then this service just reads the resulting JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

EXPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "model_comparison_export.json"


def load_model_comparison() -> dict:
    """
    Reads the precomputed model-comparison export. Raises FileNotFoundError with a
    clear, actionable message (not a bare stack trace) if the export hasn't been
    generated yet — a genuinely common first-run state for a fresh checkout of this
    project, not an error condition worth hiding behind a generic 500.
    """
    if not EXPORT_PATH.exists():
        raise FileNotFoundError(
            f"Model comparison export not found at {EXPORT_PATH}. Run "
            f"'python pipelines/export_model_comparison.py' (add --full for the "
            f"complete holdout set once a quick run looks sane) before this endpoint "
            f"can serve anything."
        )
    return json.loads(EXPORT_PATH.read_text())