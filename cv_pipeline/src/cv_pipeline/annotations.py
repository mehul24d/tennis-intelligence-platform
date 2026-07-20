"""annotations.py — loads and joins one clip's three ground-truth CSVs
(court/ball/player) back together by frame, and applies the confirmed
"ball not visible" sentinel rule.

SENTINEL RULE (confirmed 2026-07-15 by cross-clip statistics + visual inspection of
actual video frames, see PROGRESS.md's Phase 3 section for the full writeup): ball rows
within BALL_SENTINEL_RADIUS_PX of BALL_SENTINEL_CORNER are a placeholder the annotation
tool emits when no ball was detected/tracked in that frame, not a real position.
Evidence: (1) 54-77% of frames in EVERY one of the 10 clips cluster at this exact same
corner, regardless of what's happening in each clip; (2) within a clip these rows form
long unbroken runs (up to 101 consecutive frames = ~1.7s motionless, which a ball in
play never does); (3) directly overlaying the raw coordinate on the source video frame
lands in empty background/sky, not on the visible ball, whereas non-corner rows land
exactly on the visible ball.

COURT SPARSE-SAMPLING: court.csv has only ~7 rows per clip (every 100 frames), not one
per frame -- confirmed the values are static (locked camera, +/-1-2px annotation
jitter), so this module holds the most recent court row forward for any frame without
its own row, rather than treating it as missing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BALL_SENTINEL_CORNER = (1920.0, 0.0)
BALL_SENTINEL_RADIUS_PX = 25.0

DEFAULT_ANNOTATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "cv_annotated" / "annotations"
)
DEFAULT_VIDEOS_DIR = Path(__file__).resolve().parents[3] / "data" / "cv_annotated" / "videos"


@dataclass(frozen=True)
class FrameAnnotation:
    frame_index: int
    court_corners: dict[str, tuple[float, float]] | None  # keys: BL, BR, TR, TL
    player_r: tuple[float, float] | None  # None if this frame is a player-position sentinel
    player_l: tuple[float, float] | None
    player_r_is_sentinel: bool
    player_l_is_sentinel: bool
    ball: tuple[float, float] | None  # None if this frame is a sentinel / no ground truth
    ball_is_sentinel: bool  # True if a ball row existed but was excluded by the sentinel rule
    ball_row_missing: bool  # True if there was no ball row at all for this frame (e.g. tail gap)


# Discovered while debugging step-4 player-detection results (2026-07-15): player_r/
# player_l occasionally sit at the exact same top-right-corner region as the ball
# sentinel (e.g. video1 frame_005: player_l=(1916,4)) -- a "far player not
# tracked/visible" placeholder, not a real position, using the same corner as the ball
# sentinel. Same radius reused since it's visually the same annotation-tool convention.
PLAYER_SENTINEL_CORNER = BALL_SENTINEL_CORNER
PLAYER_SENTINEL_RADIUS_PX = 30.0


def _frame_str_to_index(frame_str: str) -> int:
    return int(frame_str.split("_")[1])


def _is_ball_sentinel(x: float, y: float) -> bool:
    cx, cy = BALL_SENTINEL_CORNER
    return abs(x - cx) < BALL_SENTINEL_RADIUS_PX and abs(y - cy) < BALL_SENTINEL_RADIUS_PX


def _is_player_sentinel(x: float, y: float) -> bool:
    cx, cy = PLAYER_SENTINEL_CORNER
    return abs(x - cx) < PLAYER_SENTINEL_RADIUS_PX and abs(y - cy) < PLAYER_SENTINEL_RADIUS_PX


def load_clip_annotations(
    clip_name: str, annotations_dir: Path = DEFAULT_ANNOTATIONS_DIR
) -> dict[int, FrameAnnotation]:
    """Loads and joins {clip_name}_{court,ball,player}.csv by frame index. Returns a
    dict keyed by integer frame index (0-based, matching the video's own frame
    numbering) so callers can look up any frame directly, including ones with no ball
    row at all (ball_row_missing=True) or a held-forward court row."""
    court_df = pd.read_csv(annotations_dir / f"{clip_name}_court.csv")
    ball_df = pd.read_csv(annotations_dir / f"{clip_name}_ball.csv")
    player_df = pd.read_csv(annotations_dir / f"{clip_name}_player.csv")

    # Column NAME casing (BL_x vs bl_x) and column ORDER both vary across clips
    # (confirmed by inspecting all 10 court CSVs' headers directly) -- normalize to
    # lowercase and select by name below, never by position.
    court_df.columns = [c if c == "frame" else c.lower() for c in court_df.columns]

    court_df["frame_idx"] = court_df["frame"].map(_frame_str_to_index)
    ball_df["frame_idx"] = ball_df["frame"].map(_frame_str_to_index)
    player_df["frame_idx"] = player_df["frame"].map(_frame_str_to_index)

    court_by_frame = court_df.set_index("frame_idx").sort_index()
    ball_by_frame = ball_df.set_index("frame_idx")
    player_by_frame = player_df.set_index("frame_idx")

    max_frame = int(player_df["frame_idx"].max())
    out: dict[int, FrameAnnotation] = {}

    for frame_idx in range(max_frame + 1):
        # Court: hold the most recent sampled row at or before this frame.
        court_corners = None
        prior_court_rows = court_by_frame[court_by_frame.index <= frame_idx]
        if len(prior_court_rows):
            row = prior_court_rows.iloc[-1]
            court_corners = {
                "BL": (row["bl_x"], row["bl_y"]), "BR": (row["br_x"], row["br_y"]),
                "TR": (row["tr_x"], row["tr_y"]), "TL": (row["tl_x"], row["tl_y"]),
            }

        player_r = player_l = None
        player_r_is_sentinel = player_l_is_sentinel = False
        if frame_idx in player_by_frame.index:
            prow = player_by_frame.loc[frame_idx]
            rx, ry = float(prow["player_r_x"]), float(prow["player_r_y"])
            lx, ly = float(prow["player_l_x"]), float(prow["player_l_y"])
            if _is_player_sentinel(rx, ry):
                player_r_is_sentinel = True
            else:
                player_r = (rx, ry)
            if _is_player_sentinel(lx, ly):
                player_l_is_sentinel = True
            else:
                player_l = (lx, ly)

        ball = None
        ball_is_sentinel = False
        ball_row_missing = frame_idx not in ball_by_frame.index
        if not ball_row_missing:
            brow = ball_by_frame.loc[frame_idx]
            bx, by = float(brow["ball_x"]), float(brow["ball_y"])
            if _is_ball_sentinel(bx, by):
                ball_is_sentinel = True
            else:
                ball = (bx, by)

        out[frame_idx] = FrameAnnotation(
            frame_index=frame_idx, court_corners=court_corners,
            player_r=player_r, player_l=player_l,
            player_r_is_sentinel=player_r_is_sentinel, player_l_is_sentinel=player_l_is_sentinel,
            ball=ball, ball_is_sentinel=ball_is_sentinel, ball_row_missing=ball_row_missing,
        )

    return out
