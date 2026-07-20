"""homography.py — pixel <-> real-world court coordinate mapping, built from the 4
annotated doubles-court corners (BL/BR near baseline, TL/TR far baseline). Real-world
coordinates are in meters, origin at the near-baseline left doubles sideline corner,
x = across the court (0..10.97m doubles width), y = down the court length (0..23.77m).

COURT DIMENSIONS (ITF standard, doubles): length 23.77m (78ft), doubles width 10.97m
(36ft), singles width 8.23m (27ft) -- singles sidelines sit (10.97-8.23)/2 = 1.37m
inset from each doubles sideline. Net is at the court's midline, y=11.885m -- but its
PIXEL position is NOT a single constant y (the frame is a perspective projection, not
an orthographic one): the net's real-world line y=11.885 spans x in [0, 10.97], and
each point on that line projects to a different pixel (x,y) via the homography, so the
net appears as a (very slightly) non-horizontal, non-constant-y line in pixel space.
net_pixel_polyline() below returns that projected curve rather than a single y value.
"""

from __future__ import annotations

import cv2
import numpy as np

DOUBLES_WIDTH_M = 10.97
SINGLES_WIDTH_M = 8.23
COURT_LENGTH_M = 23.77
SINGLES_INSET_M = (DOUBLES_WIDTH_M - SINGLES_WIDTH_M) / 2  # 1.37m
NET_Y_M = COURT_LENGTH_M / 2  # 11.885m from either baseline


class CourtHomography:
    """Built from one clip's 4 court corners. `court_corners` must have keys
    BL/BR/TR/TL (pixel coordinates), matching cv_pipeline.annotations.FrameAnnotation
    -- BUT those label STRINGS are not trusted for near/far/left/right assignment.

    CONFIRMED (2026-07-15, by overlaying the raw labels on real frames for video7 and
    video9): the BL/BR/TL/TR label naming convention is NOT consistent across clips.
    video1 uses B=near baseline (larger pixel-y, closer to camera), T=far baseline --
    but video7 has this flipped (its "BL"/"BR" sit at the NET, its "TL"/"TR" sit at the
    near baseline), and video9 mixes it further (its "BL"/"TL" are both at the net, its
    "BR"/"TR" are both at the near baseline -- L/R doesn't even consistently pair with
    B/T the same way). Trusting the label strings would silently build a wrong,
    rotated/mirrored homography for at least 2 of the 10 clips. Instead, corners are
    reassigned geometrically below: the two points with the LARGEST pixel-y are near
    baseline (closer to camera), the two with smallest pixel-y are far baseline; within
    each pair, smaller pixel-x is left, larger is right.
    """

    def __init__(self, court_corners: dict[str, tuple[float, float]]):
        points = list(court_corners.values())
        by_y = sorted(points, key=lambda p: p[1])
        far_pair = sorted(by_y[:2], key=lambda p: p[0])  # smallest 2 y = far baseline
        near_pair = sorted(by_y[2:], key=lambda p: p[0])  # largest 2 y = near baseline
        near_left, near_right = near_pair
        far_left, far_right = far_pair

        # Real-world corners: near-left, near-right, far-right, far-left.
        real_world = np.array(
            [
                [0.0, 0.0],
                [DOUBLES_WIDTH_M, 0.0],
                [DOUBLES_WIDTH_M, COURT_LENGTH_M],
                [0.0, COURT_LENGTH_M],
            ],
            dtype=np.float32,
        )
        pixel = np.array([near_left, near_right, far_right, far_left], dtype=np.float32)
        self._pixel_to_world_H, _ = cv2.findHomography(pixel, real_world)
        self._world_to_pixel_H, _ = cv2.findHomography(real_world, pixel)
        self._pixel_quad = pixel  # near_left, near_right, far_right, far_left, in this order

    @classmethod
    def from_point_correspondences(
        cls, world_pixel_pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> "CourtHomography":
        """Builds a homography from >=6 arbitrary (world_xy, pixel_xy) court-line-
        intersection correspondences via ordinary least-squares DLT
        (cv2.findHomography(..., method=0)), instead of the exact 4-corner solve
        used by __init__. Added 2026-07-17 for data/tennis/1.mp4's homography-
        precision investigation: an 8-point least-squares fit (4 outer corners +
        near-service-line corners L/R + baseline-center + far-T) measured
        17.7px/16.8px held-out-landmark error, vs 74.9px/45.7px for the original
        exact 4-corner calibration -- see PROGRESS.md's "Homography Precision
        Improvement" writeup for the full before/after table and the flagged
        BR/near_svc_R residual-asymmetry open question.

        Bypasses __init__'s near/far/left/right pixel-label reassignment (that
        logic only makes sense for exactly 4 raw corners) -- the 4 outer doubles
        corners' pixel positions are instead recovered by projecting the known
        real-world corners through the fitted world->pixel matrix, so
        court_polygon_pixels()/net_pixel_polyline() etc. keep working unchanged.
        """
        obj = cls.__new__(cls)
        world_pts = np.array([p[0] for p in world_pixel_pairs], dtype=np.float32)
        pixel_pts = np.array([p[1] for p in world_pixel_pairs], dtype=np.float32)
        obj._pixel_to_world_H, _ = cv2.findHomography(pixel_pts, world_pts, method=0)
        obj._world_to_pixel_H, _ = cv2.findHomography(world_pts, pixel_pts, method=0)

        real_world_corners = np.array(
            [[0.0, 0.0], [DOUBLES_WIDTH_M, 0.0],
             [DOUBLES_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
            dtype=np.float32,
        )
        pixel_corners = cv2.perspectiveTransform(
            real_world_corners.reshape(-1, 1, 2), obj._world_to_pixel_H,
        ).reshape(-1, 2)
        obj._pixel_quad = pixel_corners
        return obj

    def court_polygon_pixels(self, dilate: float = 1.0) -> np.ndarray:
        """The 4 court corners in pixel space, correctly ordered (via the same
        geometric near/far/left/right reassignment used in __init__, not the raw
        label strings) as an int32 polygon suitable for cv2.fillPoly. `dilate`
        scales the quad outward from its centroid (e.g. 1.3 to include shots
        landing just past the lines) -- used by ball_detection_combined's
        court-region motion mask, so that mask-building logic doesn't have to
        re-derive corner ordering itself."""
        poly = self._pixel_quad
        if dilate != 1.0:
            center = poly.mean(axis=0)
            poly = center + (poly - center) * dilate
        return poly.astype(np.int32)

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._pixel_to_world_H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._world_to_pixel_H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def net_pixel_polyline(self, n_points: int = 21) -> list[tuple[float, float]]:
        """The net's true pixel-space curve: the real-world line y=NET_Y_M, x in
        [0, DOUBLES_WIDTH_M], projected point-by-point through the homography. Not a
        single constant pixel-y -- see module docstring."""
        xs = np.linspace(0.0, DOUBLES_WIDTH_M, n_points)
        return [self.world_to_pixel(x, NET_Y_M) for x in xs]

    def net_pixel_y_at_x(self, pixel_x: float) -> float:
        """Interpolates the net's pixel-y at a given pixel-x, by projecting the net
        polyline and linearly interpolating -- this is what near/far-side ball
        classification should compare a ball's pixel-y against, not a flat constant."""
        polyline = self.net_pixel_polyline()
        xs = np.array([p[0] for p in polyline])
        ys = np.array([p[1] for p in polyline])
        order = np.argsort(xs)
        return float(np.interp(pixel_x, xs[order], ys[order]))

    def singles_corners_pixels(self) -> dict[str, tuple[float, float]]:
        """The 4 singles-court corners (doubles corners inset by SINGLES_INSET_M
        on each side), projected through this same homography -- for overlays
        that need to draw both the doubles and singles court outlines
        simultaneously. Not re-clicked/re-calibrated: derived from the existing
        doubles calibration via tennis's fixed standard proportions, both
        centered on the same center line."""
        return {
            "BL": self.world_to_pixel(SINGLES_INSET_M, 0.0),
            "BR": self.world_to_pixel(DOUBLES_WIDTH_M - SINGLES_INSET_M, 0.0),
            "TR": self.world_to_pixel(DOUBLES_WIDTH_M - SINGLES_INSET_M, COURT_LENGTH_M),
            "TL": self.world_to_pixel(SINGLES_INSET_M, COURT_LENGTH_M),
        }

    def singles_sideline_pixel_xs_at_y(self, world_y: float) -> tuple[float, float]:
        """Pixel-x of the two singles sidelines at a given real-world court depth
        (world_y) -- used as an independent visual-validation check (see
        verify_homography.py): these lines are visible in the source video and were
        NOT used to calibrate the homography, so comparing their predicted pixel
        position against where they actually appear is a genuine test, not a
        trivially-circular one."""
        left = self.world_to_pixel(SINGLES_INSET_M, world_y)
        right = self.world_to_pixel(DOUBLES_WIDTH_M - SINGLES_INSET_M, world_y)
        return left[0], right[0]
