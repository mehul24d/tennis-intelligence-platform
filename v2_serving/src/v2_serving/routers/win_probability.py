"""routers/win_probability.py — GET /win-probability/{job_id}: v1's pre-match
baseline (real call, requires a known match_id -- see win_probability_pipeline.py)
always shown alongside any live adjustment derivable from the job's CV features."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from v2_serving.job_store import job_store
from v2_serving.models import WinProbabilityResponse
from v2_serving.win_probability_pipeline import get_live_adjustment, get_prematch_baseline

router = APIRouter(tags=["win-probability"])


@router.get("/win-probability/{job_id}", response_model=WinProbabilityResponse)
def win_probability(job_id: str, match_id: str | None = None) -> WinProbabilityResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job with id {job_id}")

    if match_id is not None:
        try:
            prematch = get_prematch_baseline(match_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"v1 engine call failed: {type(exc).__name__}: {exc}")
    else:
        prematch = {
            "status": "not_available",
            "reason": f"No known v1 historical match_id corresponds to job '{job_id}''s clip "
                      f"('{job.video_path}') -- v1's engine requires a real match_id from its "
                      f"own 5,981-match frozen-join dataset (Elo, rank, head-to-head) to compute "
                      f"a pre-match baseline. This clip is not part of that dataset. Pass "
                      f"?match_id=<a real v1 match_id> to compute a real baseline for a known match.",
        }

    live_adjustment = get_live_adjustment(job.result if job.status == "complete" else None)

    return WinProbabilityResponse(job_id=job_id, prematch_baseline=prematch, live_adjustment=live_adjustment)
