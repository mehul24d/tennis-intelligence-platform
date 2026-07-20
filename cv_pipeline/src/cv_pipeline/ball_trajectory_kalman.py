"""ball_trajectory_kalman.py -- physically-grounded gap filling on top of
condition B (fine-tuned-YOLO-only, motion-diff disabled) ball detections.

CONTEXT (see PROGRESS.md's "Ball Detection: Coverage vs. Real Accuracy Under 3
Conditions" and "Interpolation-Integration Discrepancy" entries): condition C
(a bare per-axis quadratic polyfit through 2-4 neighboring confirmed points,
`ball_detection_experiments.py`'s `quad_interp`) was tested and REJECTED --
worse effective coverage AND worse accuracy than B alone, with interpolated
samples missing the real ball in 4/5 clips. This module is deliberately NOT
that: it is a genuine Kalman filter with physically-motivated state, not a
locally-refit polynomial.

WHY NOT A FULL 3D BALLISTIC MODEL: the project's `CourtHomography` is a
planar, ground-plane-only (Z=0) mapping -- it has no camera height, tilt, or
focal length, so it cannot project a true 3D point (X, Y, Z>0) to pixel space.
Recovering that would require full camera calibration (e.g. cv2.solvePnP),
which is fundamentally underdetermined from a SINGLE planar homography (focal
length trades off against camera height/distance with no second constraint to
break the tie -- picking one would be an unstated guess, not a measurement).
PROGRESS.md's Phase 5 entry also already found monocular real-height/bounce
recovery unreliable on this exact kind of footage ("racket-contact false
positives indistinguishable from real bounces"). This module avoids both
problems by never trying to recover a metric height in meters.

THE MODEL (confirmed with the user before implementation, all three
decisions below were explicit sign-offs, not silent defaults):

1. LATERAL motion: a genuine constant-velocity Kalman filter in real-world
   ground-plane meters (X, Y), via the existing `homography.pixel_to_world`.
   Physically correct in this projection: gravity has no horizontal
   component. Reuses the clip's existing calibration, no new camera work.
2. FLIGHT ARC: a separate constant-acceleration ("nearly-constant
   acceleration" / white-noise-jerk) Kalman filter on the PIXEL-SPACE
   VERTICAL RESIDUAL -- `detected_pixel_y - world_to_pixel(X, Y).y` for the
   SAME real detection -- rather than on a recovered metric height. This
   sidesteps the ill-posed height inversion above while still capturing a
   real gravity arc: a ball's true height above the court shows up almost
   entirely as this residual (a ground-plane point at the same (X, Y) would
   project to a different pixel-y than an airborne one), and a real flight
   arc traces a smooth, consistently-curved (not S-shaped) residual.
   IMPORTANT HONESTY NOTE: the acceleration STATE is estimated online from
   real detections (heavily smoothed via a small process-noise "nearly
   constant" prior), not hardcoded to a literal g value converted to pixel
   units -- that conversion needs the same missing camera calibration as
   point 1. "Gravity-consistent" here means the model structurally expects
   one smoothly-persisting curvature per flight segment (matching how
   gravity actually behaves), not that the true 9.8 m/s^2 is baked in.
   Per explicit sign-off, GRAVITY-ONLY: no separate drag term (drag is
   velocity-dependent, would need an EKF plus assumed ball mass/cross-
   section/drag-coefficient/air-density constants for a plausibly small
   effect over most gaps here -- see the module docstring's gap-length
   table in the design conversation). Any drag-shaped model mismatch is
   absorbed into this filter's process noise instead.
3. SHOT-BOUNDARY HANDLING: constant-velocity/constant-acceleration
   assumptions break at a bounce or racket contact (velocity changes
   discontinuously, not smoothly). Per explicit sign-off, this is handled by
   RESETTING the filter's velocity/acceleration state whenever a real
   detection strongly disagrees with the running filter's prediction
   (large Mahalanobis-gated innovation, or a sign-reversal between the
   filter's current velocity and the raw implied velocity from the last two
   real detections) -- the new real detection is trusted over the filter's
   own momentum, rather than trying to explicitly detect contacts (which
   would repeat Phase 5's already-abandoned unreliable heuristic).

GAP FILLING: for a gap strictly between two real detections that fall in the
SAME segment (no reset occurred at the far end), both filters are run
FORWARD from the earlier detection AND BACKWARD (time-reversed transition,
same-magnitude process noise -- noise accumulates positively regardless of
time direction) from the later one, then fused via inverse-covariance
weighting at each intermediate frame (a standard two-filter/RTS-style
smoother fusion) -- legitimate here because this is OFFLINE post-processing
of an already-fully-decoded clip, not a live/online system, so using
"future" real detections to constrain a fill is not cheating, unlike the
ground-truth leak this project already found and fixed elsewhere (that leak
used GROUND TRUTH, which a real system never has; this uses the clip's own
OTHER real detections, which a real system does have once the clip is fully
processed).

Gaps that are NOT filled (left as genuine gaps, with a `skip_reason`,
matching this project's "an honest gap beats a wrong marker" precedent from
the motion-diff decision):
- Before the first real detection or after the last (unbounded/
  extrapolation-only -- far less trustworthy, not attempted).
- Gaps where a reset occurred at the bounding real detection on either side
  (a shot-boundary likely fell inside the gap; forward/backward fusion
  across a real discontinuity would be invalid).
- Gaps where the fused confidence (see `fill_confidence_px`) exceeds
  `max_fill_confidence_px` -- PROVISIONAL default, anchored loosely on
  `ball_detection.py`'s existing MAX_BALL_MATCH_DISTANCE_PX=100.0 convention
  for "plausibly the same ball" but not validated for this purpose; intended
  to be tuned against the manual visual audit this phase's spec requires,
  not treated as final. `raw_kalman_center`/`raw_kalman_confidence_px` are
  still reported on skipped-for-confidence frames so the audit/evaluation
  script can explore other thresholds without re-running the filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .ball_detection_combined import CombinedBallDetectionResult
from .homography import CourtHomography

# ---- tunable constants -- explicitly flagged as provisional, not validated ----

LATERAL_PROCESS_NOISE_Q = 1e-3  # m^2, white-noise-acceleration model scale
LATERAL_MEASUREMENT_NOISE_R = 0.05**2  # m^2, ~5cm assumed detection/homography noise
RESIDUAL_PROCESS_NOISE_Q = 2e-2  # px^2, white-noise-jerk model scale
RESIDUAL_MEASUREMENT_NOISE_R = 3.0**2  # px^2, assumed ball-box localization noise
INIT_POSITION_VARIANCE = 1.0  # large uninformative prior on (re)initialization

RESET_CHI2_THRESHOLD = 9.21  # 99% critical value, chi-square, 2 DOF (lateral innovation)
MAX_FILL_CONFIDENCE_PX = 100.0  # see module docstring -- provisional, audit-tunable


def _cv_F(dt: float) -> np.ndarray:
    return np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def _cv_Q(dt: float, q: float) -> np.ndarray:
    # Discretized white-noise-acceleration model, closed form (valid for any dt,
    # not just unit steps -- see module docstring's n-step-jump note).
    dt = abs(dt)
    return q * np.array([
        [dt**4 / 4, 0, dt**3 / 2, 0],
        [0, dt**4 / 4, 0, dt**3 / 2],
        [dt**3 / 2, 0, dt**2, 0],
        [0, dt**3 / 2, 0, dt**2],
    ])


def _ca_F(dt: float) -> np.ndarray:
    return np.array([
        [1, dt, dt**2 / 2],
        [0, 1, dt],
        [0, 0, 1],
    ], dtype=float)


def _ca_Q(dt: float, q: float) -> np.ndarray:
    # Discretized white-noise-jerk model, closed form. `dt` magnitude only --
    # noise accumulates positively regardless of forward/backward direction
    # (see module docstring); using a signed dt here would produce a
    # non-PSD matrix, which is physically wrong, not just a sign quirk.
    dt = abs(dt)
    return q * np.array([
        [dt**5 / 20, dt**4 / 8, dt**3 / 6],
        [dt**4 / 8, dt**3 / 3, dt**2 / 2],
        [dt**3 / 6, dt**2 / 2, dt],
    ])


class _LinearKalman:
    """Shared predict/update machinery for both the 4-state lateral CV filter
    and the 3-state vertical-residual CA filter -- only F/Q/H/R differ."""

    def __init__(self, F_fn, Q_fn, H: np.ndarray, q: float, r: np.ndarray):
        self._F_fn = F_fn
        self._Q_fn = Q_fn
        self.H = H
        self._q = q
        self.R = r
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    def init(self, x0: np.ndarray, p0: float = INIT_POSITION_VARIANCE) -> None:
        self.x = np.array(x0, dtype=float)
        self.P = p0 * np.eye(len(x0))

    def predict_n(self, n: float) -> None:
        """Jumps forward (n>0) or backward (n<0) by |n| frames in one step --
        exact for these polynomial-kinematic models (F(dt1) F(dt2) = F(dt1+dt2)),
        not an approximation, so this is equivalent to n unit predict() calls
        without their floating-point accumulation."""
        F = self._F_fn(n)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q_fn(n, self._q)

    def innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.atleast_1d(np.array(z, dtype=float))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def update(self, z: np.ndarray) -> None:
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P

    def copy(self) -> "_LinearKalman":
        c = _LinearKalman(self._F_fn, self._Q_fn, self.H, self._q, self.R)
        c.x, c.P = self.x.copy(), self.P.copy()
        return c


def _new_lateral_kf() -> _LinearKalman:
    return _LinearKalman(
        _cv_F, _cv_Q, H=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float),
        q=LATERAL_PROCESS_NOISE_Q, r=LATERAL_MEASUREMENT_NOISE_R * np.eye(2),
    )


def _new_residual_kf() -> _LinearKalman:
    return _LinearKalman(
        _ca_F, _ca_Q, H=np.array([[1, 0, 0]], dtype=float),
        q=RESIDUAL_PROCESS_NOISE_Q, r=np.array([[RESIDUAL_MEASUREMENT_NOISE_R]]),
    )


def _world_to_pixel_jacobian(homography: CourtHomography, x: float, y: float, eps: float = 0.01) -> np.ndarray:
    p0 = np.array(homography.world_to_pixel(x, y))
    px = np.array(homography.world_to_pixel(x + eps, y))
    py = np.array(homography.world_to_pixel(x, y + eps))
    return np.column_stack([(px - p0) / eps, (py - p0) / eps])  # 2x2, columns = d(pixel)/dX, d(pixel)/dY


@dataclass(frozen=True)
class TrajectoryFrameResult:
    frame_index: int
    center: tuple[float, float] | None
    source: Literal["fine_tuned_yolo", "kalman_filled", "none"]
    segment_id: int | None  # continuous-flight-segment id; None for unfilled "none" frames
    fill_confidence_px: float | None = None  # only set when source == "kalman_filled"
    skip_reason: str | None = None  # only set for "none" frames that had a gap candidate
    raw_kalman_center: tuple[float, float] | None = None  # diagnostic: what the filter
    raw_kalman_confidence_px: float | None = None  # would have filled, even if skipped


@dataclass
class _RealDetectionState:
    frame_index: int
    world_xy: tuple[float, float]
    pixel_xy: tuple[float, float]
    lateral: _LinearKalman  # post-update state at this detection
    residual: _LinearKalman
    segment_id: int
    reset_here: bool  # True if a reset was triggered arriving at this detection


def _fit_real_detection_states(
    real: list[CombinedBallDetectionResult], homography: CourtHomography,
) -> list[_RealDetectionState]:
    """Forward pass over real detections only: builds up the lateral + residual
    Kalman state at each one, applying innovation-gated resets. This is the
    only strictly-forward part of the design -- gap filling itself (below)
    also uses a backward pass, which is legitimate for offline post-
    processing (see module docstring)."""
    states: list[_RealDetectionState] = []
    lateral: _LinearKalman | None = None
    residual: _LinearKalman | None = None
    segment_id = 0
    prev_frame = None
    prev_world = None

    for det in real:
        cx, cy = det.center  # type: ignore[misc]
        world_xy = homography.pixel_to_world(cx, cy)
        ground_pixel_y = homography.world_to_pixel(*world_xy)[1]
        residual_z = cy - ground_pixel_y

        reset_here = False
        if lateral is None:
            lateral = _new_lateral_kf()
            lateral.init(np.array([world_xy[0], world_xy[1], 0.0, 0.0]))
            residual = _new_residual_kf()
            residual.init(np.array([residual_z, 0.0, 0.0]))
        else:
            n = det.frame_index - prev_frame
            cand_lateral = lateral.copy()
            cand_lateral.predict_n(n)
            y, S = cand_lateral.innovation(np.array(world_xy))
            d2 = float(y @ np.linalg.inv(S) @ y)

            implied_vel = ((world_xy[0] - prev_world[0]) / n, (world_xy[1] - prev_world[1]) / n)
            prior_vel = (cand_lateral.x[2], cand_lateral.x[3])
            direction_reversed = (implied_vel[0] * prior_vel[0] + implied_vel[1] * prior_vel[1]) < 0

            if d2 > RESET_CHI2_THRESHOLD or direction_reversed:
                reset_here = True
                segment_id += 1
                lateral = _new_lateral_kf()
                lateral.init(np.array([world_xy[0], world_xy[1], implied_vel[0], implied_vel[1]]))
                residual = _new_residual_kf()
                residual.init(np.array([residual_z, 0.0, 0.0]))
            else:
                lateral = cand_lateral
                lateral.update(np.array(world_xy))
                residual.predict_n(n)
                residual.update(np.array([residual_z]))

        states.append(_RealDetectionState(
            frame_index=det.frame_index, world_xy=world_xy, pixel_xy=(cx, cy),
            lateral=lateral.copy(), residual=residual.copy(),
            segment_id=segment_id, reset_here=reset_here,
        ))
        prev_frame = det.frame_index
        prev_world = world_xy

    return states


def _fill_gap(
    before: _RealDetectionState, after: _RealDetectionState, homography: CourtHomography,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Forward+backward fuse both filters across the gap strictly between
    `before` and `after` (same segment, caller's responsibility to check).
    Returns, per gap frame: (frame_index, world_xy_fused, world_cov_fused,
    residual_x_fused, residual_cov_fused)."""
    out = []
    span = after.frame_index - before.frame_index
    fwd_lat, fwd_res = before.lateral.copy(), before.residual.copy()
    bwd_lat, bwd_res = after.lateral.copy(), after.residual.copy()
    fwd_states: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for offset in range(1, span):
        fwd_lat.predict_n(1)
        fwd_res.predict_n(1)
        fwd_states[before.frame_index + offset] = (fwd_lat.x.copy(), fwd_lat.P.copy(), fwd_res.x.copy(), fwd_res.P.copy())

    bwd_states: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for offset in range(1, span):
        bwd_lat.predict_n(-1)
        bwd_res.predict_n(-1)
        bwd_states[after.frame_index - offset] = (bwd_lat.x.copy(), bwd_lat.P.copy(), bwd_res.x.copy(), bwd_res.P.copy())

    for frame_idx in range(before.frame_index + 1, after.frame_index):
        fx, fP, frx, frP = fwd_states[frame_idx]
        bx, bP, brx, brP = bwd_states[frame_idx]

        # inverse-covariance-weighted fusion of two independent estimates of
        # the same quantity (standard two-filter smoother fusion) -- position/
        # velocity jointly for the lateral filter, position/vel/accel jointly
        # for the residual filter.
        fP_inv, bP_inv = np.linalg.inv(fP), np.linalg.inv(bP)
        P_fused = np.linalg.inv(fP_inv + bP_inv)
        x_fused = P_fused @ (fP_inv @ fx + bP_inv @ bx)

        frP_inv, brP_inv = np.linalg.inv(frP), np.linalg.inv(brP)
        rP_fused = np.linalg.inv(frP_inv + brP_inv)
        rx_fused = rP_fused @ (frP_inv @ frx + brP_inv @ brx)

        out.append((frame_idx, x_fused, P_fused, rx_fused, rP_fused))

    return out


def fit_trajectory_for_clip(
    detections: list[CombinedBallDetectionResult], homography: CourtHomography,
    max_fill_confidence_px: float = MAX_FILL_CONFIDENCE_PX,
) -> list[TrajectoryFrameResult]:
    """Main entry point. `detections` is exactly condition B's output (real
    detections have source=='fine_tuned_yolo'; everything else is a gap
    candidate) -- see module docstring for the full design."""
    real = [d for d in detections if d.source == "fine_tuned_yolo" and d.center is not None]
    real_states = _fit_real_detection_states(real, homography)
    states_by_frame = {s.frame_index: s for s in real_states}

    out: list[TrajectoryFrameResult] = []
    for i, det in enumerate(detections):
        if det.frame_index in states_by_frame:
            s = states_by_frame[det.frame_index]
            out.append(TrajectoryFrameResult(
                frame_index=det.frame_index, center=s.pixel_xy, source="fine_tuned_yolo",
                segment_id=s.segment_id,
            ))
            continue
        out.append(TrajectoryFrameResult(
            frame_index=det.frame_index, center=None, source="none", segment_id=None,
        ))

    # Walk consecutive real detections, filling gaps in-place on `out`.
    out_by_frame = {r.frame_index: idx for idx, r in enumerate(out)}
    for prev_state, next_state in zip(real_states, real_states[1:]):
        span = next_state.frame_index - prev_state.frame_index
        if span <= 1:
            continue  # no gap
        if next_state.reset_here:
            for fi in range(prev_state.frame_index + 1, next_state.frame_index):
                idx = out_by_frame[fi]
                out[idx] = TrajectoryFrameResult(
                    frame_index=fi, center=None, source="none", segment_id=None,
                    skip_reason="segment_boundary_in_gap",
                )
            continue

        filled = _fill_gap(prev_state, next_state, homography)
        for frame_idx, x_fused, P_fused, rx_fused, rP_fused in filled:
            X, Y = float(x_fused[0]), float(x_fused[1])
            ground_px = homography.world_to_pixel(X, Y)
            pixel_x, pixel_y = ground_px[0], ground_px[1] + float(rx_fused[0])

            J = _world_to_pixel_jacobian(homography, X, Y)
            lateral_pixel_cov = J @ P_fused[:2, :2] @ J.T
            confidence_px = float(np.sqrt(max(lateral_pixel_cov[0, 0], 0) + max(lateral_pixel_cov[1, 1], 0) + max(rP_fused[0, 0], 0)))

            idx = out_by_frame[frame_idx]
            if confidence_px <= max_fill_confidence_px:
                out[idx] = TrajectoryFrameResult(
                    frame_index=frame_idx, center=(pixel_x, pixel_y), source="kalman_filled",
                    segment_id=prev_state.segment_id, fill_confidence_px=confidence_px,
                )
            else:
                out[idx] = TrajectoryFrameResult(
                    frame_index=frame_idx, center=None, source="none", segment_id=None,
                    skip_reason="low_confidence",
                    raw_kalman_center=(pixel_x, pixel_y), raw_kalman_confidence_px=confidence_px,
                )

    # Frames before the first real detection or after the last real detection
    # are left as "none" with an explicit reason -- extrapolation-only fills
    # were never attempted (see module docstring).
    if real_states:
        first_f, last_f = real_states[0].frame_index, real_states[-1].frame_index
        for fi, idx in out_by_frame.items():
            if (fi < first_f or fi > last_f) and out[idx].source == "none" and out[idx].skip_reason is None:
                out[idx] = TrajectoryFrameResult(
                    frame_index=fi, center=None, source="none", segment_id=None,
                    skip_reason="clip_edge_unbounded",
                )

    return out
