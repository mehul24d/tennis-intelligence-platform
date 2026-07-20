"""player_selection.py — selects which detected person-box is "the near player" /
"the far player" using COURT-POSITION PLAUSIBILITY (via homography), not raw
bounding-box size.

WHY THIS EXISTS: pure size-based selection ("largest box = near player, smallest
box = far player") was used ad hoc in early pose spot-check scripts and confirmed
WRONG on two independent clips -- the amateur dataset's `video9` (picked a sideline
bystander over the real far player, nearly identical box size) and the professional
stress-test clip (picked a courtside official standing near the net post, again
similar size to the real far player). Box size alone cannot distinguish "a small
box because this person is far from camera" from "a small box because this person
is simply standing far to the side, off the court" -- both look the same in raw
pixel area. Court position doesn't have that ambiguity: a real player's box
bottom-center should project, via the homography, to somewhere ON or very near the
court; a bystander's generally won't.

This does NOT require a real-world-scale-validated homography -- but it DOES need
to be robust to a specific, now-confirmed failure mode: at least 2 of the 10
amateur clips (video7, confirmed; video9, found while building this fix -- see
below) have court-corner annotations that only span the near baseline to the NET,
not the full baseline-to-baseline court, which roughly DOUBLES the effective
scale error along the court-length (Y) axis specifically (see homography.py and
PROGRESS.md's video7 writeup). The X axis (court WIDTH) is not affected by that
truncation and remains scale-correct regardless. Consequently:
  - X-axis plausibility uses a real, fairly tight meter-based margin -- this is
    the check that actually rejects bystanders/officials/furniture (confirmed:
    both known bad-selection cases, video9's sideline bystander and the Phase 3
    stress-test clip's courtside official, had wildly implausible X, not just Y).
  - Y-axis is used only for RELATIVE ordering (near = smallest projected Y among
    X-plausible boxes, far = largest) -- never as an absolute meter bound, since
    Y is exactly the axis known to be unreliable in scale for some clips. A very
    loose Y sanity multiplier still guards against a fully degenerate homography
    producing a nonsensical position, without rejecting the ~2x overrun a
    half-court-only truncation produces.

KNOWN RESIDUAL LIMITATION (found while validating this fix, not swept under the
rug): this selection is only as good as the homography it's given. On the amateur
dataset's clips (real, geometrically-consistent annotated court corners), this
correctly fixed both known bad-selection cases end to end. On the Phase 3
stress-test clip (a single rough, MANUALLY-estimated homography from eyeballing one
frame -- see STRESS_TEST_REPORT.md), a courtside bystander happened to project to a
world-X position genuinely inside the assumed court width under that rough
calibration -- not a margin-tuning problem, a homography-precision problem.
Deliberately NOT overfitting the margin to force that one ambiguous frame to pass;
doing so risks rejecting real off-court lunges elsewhere. Takeaway: this fix
meaningfully improves selection reliability given a reasonable homography, but does
not fully compensate for a poor one.
"""

from __future__ import annotations

from dataclasses import dataclass

from cv_pipeline.homography import COURT_LENGTH_M, CourtHomography, DOUBLES_WIDTH_M

# X (court width): real, fairly tight meter margin -- not affected by the
# near-baseline-to-net truncation issue, so this can and should be strict enough
# to actually reject off-court bystanders. 2.5m (a bit less than a quarter of the
# 10.97m doubles width) is tight enough to reject a row of courtside seating just
# outside the sideline while still allowing a real lunge/slide off the court.
X_PLAUSIBILITY_MARGIN_M = 2.5

# Y (court length): NOT used as an absolute bound (see module docstring) -- only
# this loose multiplier guards against a fully-degenerate homography. 3x the
# assumed court length comfortably accommodates the known ~2x half-court-scale
# error while still rejecting truly nonsensical projections.
Y_SANITY_MULTIPLIER = 3.0


# Max plausible per-frame world-space displacement for temporal continuity
# (select_players_sequence_with_continuity), in meters. Generous relative to
# real sprint speed over a single frame interval at this project's typical
# fps range -- chosen to almost never trigger a track reset for a genuine
# player, while still being tight enough to refuse jumping onto a
# stationary/differently-positioned bystander. See that function's docstring
# for the bug this exists to fix.
MAX_TEMPORAL_JUMP_M = 3.0


@dataclass(frozen=True)
class PlayerSelection:
    near_box: tuple[float, float, float, float] | None
    far_box: tuple[float, float, float, float] | None
    near_world_y: float | None
    far_world_y: float | None
    n_plausible_boxes: int
    n_total_boxes: int
    note: str


def _bottom_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2


def _is_plausible(world_x: float, world_y: float) -> bool:
    x_ok = -X_PLAUSIBILITY_MARGIN_M <= world_x <= DOUBLES_WIDTH_M + X_PLAUSIBILITY_MARGIN_M
    y_ok = -Y_SANITY_MULTIPLIER * COURT_LENGTH_M <= world_y <= (1 + Y_SANITY_MULTIPLIER) * COURT_LENGTH_M
    return x_ok and y_ok


def select_players_by_court_position(
    boxes: list[tuple[float, float, float, float]], homography: CourtHomography,
    y_upper_bound_m: float | None = None,
) -> PlayerSelection:
    """Projects each box's bottom-center through the homography and keeps only
    boxes that land plausibly on/near the court. Among those, the smallest
    world_y is "near" (closer to the camera-side baseline, y=0) and the largest
    world_y is "far" (closer to the far baseline). If 0 or 1 plausible boxes are
    found, returns what's available rather than guessing -- callers must check
    n_plausible_boxes / the note before assuming both near_box and far_box are
    populated.

    `y_upper_bound_m` (opt-in; default None leaves behavior unchanged for every
    existing caller): an ABSOLUTE upper bound on projected world_y, for use ONLY
    with a full-court homography whose scale is trustworthy. Added 2026-07-16
    after the Miami reference-video build (data/tennis/1.mp4) found the
    "largest world_y wins far" rule deterministically selecting a staff member
    standing at the BACK WALL over the real far player in 96.6% of frames: she
    is centered behind the court (passes the tight X check) and projects to
    world_y ~31-32m, always outranking the real player's <=28m. Measured on
    that clip's full 2,020 frames: the two clusters are cleanly bimodal -- real
    far player 23-28m, back-wall staff 31-33m, with the 28-31m band essentially
    empty (9 of ~4,880 far-half detections) -- so a bound of ~29m (court length
    23.77m + ~5m for deep return positions) separates them with real margin on
    both sides. This deliberately CANNOT be the default: the loose
    Y_SANITY_MULTIPLIER exists precisely because truncated
    (near-baseline-to-net) calibrations like video7/video9's inflate world_y by
    ~2x, and an absolute bound would wrongly reject the real far player there.
    Pass it only when the homography's scale has been independently validated
    (the video1 standard)."""
    if not boxes:
        return PlayerSelection(
            near_box=None, far_box=None, near_world_y=None, far_world_y=None,
            n_plausible_boxes=0, n_total_boxes=0, note="no boxes given",
        )

    plausible = []
    for box in boxes:
        bx, by = _bottom_center(box)
        wx, wy = homography.pixel_to_world(bx, by)
        if _is_plausible(wx, wy) and (y_upper_bound_m is None or wy <= y_upper_bound_m):
            plausible.append((box, wy))

    n_total = len(boxes)
    n_plausible = len(plausible)

    if n_plausible == 0:
        return PlayerSelection(
            near_box=None, far_box=None, near_world_y=None, far_world_y=None,
            n_plausible_boxes=0, n_total_boxes=n_total,
            note=f"none of the {n_total} detected boxes projected plausibly onto the court "
                 f"-- likely no real player box was detected this frame, or the homography "
                 f"is badly wrong for this clip",
        )

    plausible.sort(key=lambda pair: pair[1])  # sort by world_y ascending
    near_box, near_y = plausible[0]
    far_box, far_y = plausible[-1]

    if n_plausible == 1:
        note = ("only 1 of the detected boxes projected plausibly onto the court -- "
                "returning it as BOTH near and far candidate is wrong, so far_box is "
                "left as this same box only if it's genuinely ambiguous; caller should "
                "treat this as 'only one player position available this frame'")
        return PlayerSelection(
            near_box=near_box, far_box=None, near_world_y=near_y, far_world_y=None,
            n_plausible_boxes=1, n_total_boxes=n_total, note=note,
        )

    note = (f"{n_plausible}/{n_total} boxes plausible; near/far chosen by court-position "
            f"(world_y), not box size -- rejected {n_total - n_plausible} box(es) as "
            f"off-court (likely bystanders/staff)")
    return PlayerSelection(
        near_box=near_box, far_box=far_box, near_world_y=near_y, far_world_y=far_y,
        n_plausible_boxes=n_plausible, n_total_boxes=n_total, note=note,
    )


class PlayerContinuityTracker:
    """Stateful, one-frame-at-a-time version of the temporal-continuity logic in
    select_players_sequence_with_continuity -- for streaming callers (e.g.
    video_pipeline.py's per-frame inference loop) that can't buffer the whole
    clip's boxes into one list up front. Construct once per clip, call
    .select(boxes) once per frame in order; internal state (previous near/far
    world position) carries across calls automatically. See
    select_players_sequence_with_continuity's docstring for why this exists and
    what bug it fixes -- the logic here is identical, just incrementalized."""

    def __init__(self, homography: CourtHomography, y_upper_bound_m: float | None = None):
        self._homography = homography
        self._y_upper_bound_m = y_upper_bound_m
        self._prev_near_world: tuple[float, float] | None = None
        self._prev_far_world: tuple[float, float] | None = None

    def select(self, boxes: list[tuple[float, float, float, float]]) -> PlayerSelection:
        if not boxes:
            return PlayerSelection(
                near_box=None, far_box=None, near_world_y=None, far_world_y=None,
                n_plausible_boxes=0, n_total_boxes=0, note="no boxes given",
            )

        plausible = []
        for box in boxes:
            bx, by = _bottom_center(box)
            wx, wy = self._homography.pixel_to_world(bx, by)
            if _is_plausible(wx, wy) and (self._y_upper_bound_m is None or wy <= self._y_upper_bound_m):
                plausible.append((box, wx, wy))

        n_total, n_plausible = len(boxes), len(plausible)
        if n_plausible == 0:
            return PlayerSelection(
                near_box=None, far_box=None, near_world_y=None, far_world_y=None,
                n_plausible_boxes=0, n_total_boxes=n_total,
                note="none of the detected boxes projected plausibly onto the court this frame",
            )

        def _closest_to(target, candidates):
            dists = [((wx - target[0]) ** 2 + (wy - target[1]) ** 2) ** 0.5 for _, wx, wy in candidates]
            i = min(range(len(candidates)), key=lambda k: dists[k])
            return i, dists[i]

        if self._prev_near_world is not None:
            i, d = _closest_to(self._prev_near_world, plausible)
            near_box, nwx, nwy = plausible[i] if d <= MAX_TEMPORAL_JUMP_M else min(plausible, key=lambda p: p[2])
        else:
            near_box, nwx, nwy = min(plausible, key=lambda p: p[2])
        self._prev_near_world = (nwx, nwy)

        remaining = [p for p in plausible if p[0] is not near_box]
        far_box = far_wy = None
        if remaining:
            if self._prev_far_world is not None:
                i, d = _closest_to(self._prev_far_world, remaining)
                far_box, fwx, fwy = remaining[i] if d <= MAX_TEMPORAL_JUMP_M else max(remaining, key=lambda p: p[2])
            else:
                far_box, fwx, fwy = max(remaining, key=lambda p: p[2])
            self._prev_far_world = (fwx, fwy)
            far_wy = fwy

        return PlayerSelection(
            near_box=near_box, far_box=far_box, near_world_y=nwy, far_world_y=far_wy,
            n_plausible_boxes=n_plausible, n_total_boxes=n_total,
            note=f"{n_plausible}/{n_total} boxes plausible; near/far chosen by temporal "
                 f"continuity to the previous frame where available, court-position "
                 f"(world_y) otherwise",
        )


def select_players_sequence_with_continuity(
    frames_boxes: list[list[tuple[float, float, float, float]]], homography: CourtHomography,
    y_upper_bound_m: float | None = None,
) -> list[PlayerSelection]:
    """Sequence version of select_players_by_court_position that adds temporal
    continuity as a SECOND selection signal, tried only after the pure
    court-position rule ('smallest/largest plausible world_y') is confirmed
    insufficient on its own.

    BUG FOUND (2026-07-19, data/tennis/1.mp4): a front-row spectator/
    photographer standing just behind the near-baseline barrier was
    selected as "near player" in dozens of frames. Root-caused: this
    person's box consistently projects to world_x~9.8-9.95m (near the
    right doubles sideline, within the existing tight X margin) and
    world_y~-3.7m (behind the near baseline, within the loose Y sanity
    bound) -- and 'near' is chosen as the SMALLEST plausible world_y, so a
    stationary bystander sitting further behind the baseline than either
    real player outranks both of them every frame they're visible.

    TRIED AND REJECTED FIRST, per project discipline (cheapest fix
    first): tightening the world_y bound to something like
    'inside the court polygon plus a small (~2m) margin', instead of
    fixing this with temporal continuity. Measured directly against real
    data before adopting anything: real near-player positions during play
    legitimately reach world_y as low as -3.0m (players stretching for a
    short return, confirmed via smoothly-varying frame-to-frame world
    coordinates, i.e. real motion, not noise) -- overlapping enough with
    the spectator's -3.7m that no single Y-margin value (nor a
    court-rectangle-distance metric, also tested) cleanly separates the
    two without also rejecting genuine deep-return frames. The two
    populations are NOT geometrically separable by position alone in this
    clip -- confirmed by directly inspecting the real world_y distribution
    of validated near-player selections, not assumed.

    FIX: temporal continuity. The spectator is stationary (or
    near-stationary); a real player is continuously moving and never
    teleports between frames. For each frame after the first, the
    candidate closest (in world-space Euclidean distance) to the PREVIOUS
    frame's selected position wins that role, as long as that distance is
    within MAX_TEMPORAL_JUMP_M -- otherwise falls back to the pure
    court-position rule (smallest/largest world_y) for that frame only,
    exactly as before. The first frame of a sequence has no previous
    position, so it always uses the pure rule. Verified end to end on this
    clip: the spectator is selected in 0 of 2020 frames after this fix
    (vs. 66+ before), all 7 of Phase 2's manually spot-checked frames are
    unaffected, near-player coverage stays 100%, far-player coverage stays
    96.9% (unchanged from the y_upper_bound_m-only fix), and the temporal
    fallback (track distance > MAX_TEMPORAL_JUMP_M) never actually
    triggered across the whole clip -- see PROGRESS.md for the full
    before/after evidence including visual spot-checks.

    y_upper_bound_m behaves exactly as in select_players_by_court_position
    (see that function's docstring) -- this is a SEPARATE, independent fix
    for a different (near-side, not far-side) failure mode; both are
    typically needed together, as they are for data/tennis/1.mp4.

    Thin wrapper over PlayerContinuityTracker -- see that class for streaming
    (one-frame-at-a-time) callers that can't buffer the whole clip up front."""
    tracker = PlayerContinuityTracker(homography, y_upper_bound_m=y_upper_bound_m)
    return [tracker.select(boxes) for boxes in frames_boxes]
