"""schema.py — the structured per-clip output schema (step 8), combining detection,
tracking, homography, and pose results from steps 3-7 into one coherent record.

CORE DESIGN RULE: every measurable field is paired with an explicit `Status` enum
value, not just a bare number-or-null. A field being `None`/absent must never be the
only signal for WHY a value is missing -- "genuinely zero," "not detected," "not
attempted," "excluded due to a known data-quality issue," and "sample too small to
trust" are different situations with different downstream implications, and this
schema keeps them distinguishable rather than collapsing them all into `null`. Anyone
(including future us) reading this JSON without having read PROGRESS.md should be
able to tell these apart from the schema alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    MEASURED = "measured"  # a real value was computed from real ground truth
    NOT_DETECTED = "not_detected"  # detection/pose was attempted, found nothing
    NOT_ATTEMPTED = "not_attempted"  # never even tried (e.g. no box existed to run pose on)
    SENTINEL_EXCLUDED = "sentinel_excluded"  # ground truth existed but was a known placeholder, not a real position
    INSUFFICIENT_SAMPLE = "insufficient_sample"  # a rate WAS computed but n is too small to trust it (see min_n)
    EXCLUDED_KNOWN_ISSUE = "excluded_known_issue"  # deliberately excluded due to a documented, resolved-cause data problem
    UNVALIDATED = "unvalidated"  # internally self-consistent but never checked against an independent ground truth
    NOT_APPLICABLE = "not_applicable"  # the concept doesn't apply to this clip/field at all


@dataclass(frozen=True)
class RateMetric:
    """A detection/match rate with its sample size and status -- never report a rate
    without n, and never let a rate stand alone without a status explaining its
    trustworthiness.

    `method` names WHICH detection approach actually produced this rate --
    added when ball_detection_combined.py's validated combined method (fine-
    tuned YOLO + frequency-based artifact filter + motion-diff) was wired in
    alongside the original stock-YOLO path. Defaults to "stock_yolo" so every
    pre-existing caller keeps working unchanged. This is what lets a consumer
    (e.g. the dashboard) distinguish "improved method, validated at 53.91%
    pooled recall on the amateur dataset" from "best-effort stock YOLO,
    known ~7.8% baseline recall" for the SAME field, rather than silently
    reporting a number with no indication of which regime produced it."""
    status: Status
    rate: float | None = None
    n: int | None = None
    median_error_px: float | None = None
    note: str | None = None
    method: str = "stock_yolo"
    MIN_TRUSTED_N: int = 20  # below this, status should be INSUFFICIENT_SAMPLE even if a rate was computed

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value, "rate": self.rate, "n": self.n,
            "median_error_px": self.median_error_px, "note": self.note, "method": self.method,
        }


@dataclass(frozen=True)
class HomographyReport:
    geometric_sanity_status: Status  # reprojection-error / near-far-span checks (cheap, self-consistent)
    real_world_scale_status: Status  # independent landmark validation (expensive, not done for every clip)
    real_world_distance_metrics_usable: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometric_sanity_status": self.geometric_sanity_status.value,
            "real_world_scale_status": self.real_world_scale_status.value,
            "real_world_distance_metrics_usable": self.real_world_distance_metrics_usable,
            "note": self.note,
        }


@dataclass(frozen=True)
class TrackingReport:
    player_r_n_segments: int
    player_l_n_segments: int
    n_id_swaps: int
    hard_moment_frame_count: int
    hard_moment_coverage_status: Status  # NOT_APPLICABLE-ish concept: ADEQUATE vs NONE, encoded via note
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_r_n_segments": self.player_r_n_segments,
            "player_l_n_segments": self.player_l_n_segments,
            "n_id_swaps": self.n_id_swaps,
            "hard_moment_frame_count": self.hard_moment_frame_count,
            "hard_moment_coverage_status": self.hard_moment_coverage_status.value,
            "note": self.note,
        }


@dataclass(frozen=True)
class PoseReport:
    """Pose has NO ground truth at all (see step 7) -- this is qualitative/spot-check
    status only, never a rate or error metric. sampled=False means this clip wasn't
    one of the 6 hand-picked spot-check cases in step 7, which is itself meaningful
    (NOT_APPLICABLE, not silently absent)."""
    near_player_status: Status
    far_player_status: Status
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "near_player_status": self.near_player_status.value,
            "far_player_status": self.far_player_status.value,
            "note": self.note,
        }


@dataclass(frozen=True)
class ClipReport:
    clip: str
    n_frames: int
    fps: float

    homography: HomographyReport
    player_r_detection: RateMetric
    player_l_detection_separated: RateMetric
    player_l_detection_ambiguous: RateMetric
    ball_detection: RateMetric
    tracking: TrackingReport
    pose: PoseReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip, "n_frames": self.n_frames, "fps": self.fps,
            "homography": self.homography.to_dict(),
            "player_r_detection": self.player_r_detection.to_dict(),
            "player_l_detection_separated": self.player_l_detection_separated.to_dict(),
            "player_l_detection_ambiguous": self.player_l_detection_ambiguous.to_dict(),
            "ball_detection": self.ball_detection.to_dict(),
            "tracking": self.tracking.to_dict(),
            "pose": self.pose.to_dict(),
        }
