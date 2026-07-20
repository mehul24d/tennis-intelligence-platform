"""pose_estimation.py — MediaPipe Pose (Tasks API, pose_landmarker_lite) run on
YOLOv8-detected player bounding-box crops.

NO GROUND TRUTH EXISTS for pose (the dataset's annotations are court/ball/player-point
only) -- this module and its validation script are explicitly a VISUAL SPOT-CHECK
tool, not a quantitative accuracy measurement. Anything downstream must not cite a
pose "accuracy" or "error rate" from this step -- there is nothing to compute one
against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_lite.task"

# Crop padding around a YOLO box before running pose -- a tight crop right at the box
# edge often clips extended limbs (a racket-arm swing, a serve toss) that pose
# estimation needs a bit of margin to pick up.
CROP_PADDING_FRAC = 0.15


@dataclass(frozen=True)
class PoseResult:
    box: tuple[float, float, float, float]  # the YOLO box this pose was run on, in original-frame coords
    landmarks: list[tuple[float, float, float]] | None  # (x, y, visibility) in ORIGINAL frame coords, or None if no pose found


def _pad_box(box, frame_shape, pad_frac: float = CROP_PADDING_FRAC):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = w * pad_frac, h * pad_frac
    H, W = frame_shape[:2]
    return (
        max(0, int(x1 - px)), max(0, int(y1 - py)),
        min(W, int(x2 + px)), min(H, int(y2 + py)),
    )


def run_pose_on_box(landmarker, frame, box) -> PoseResult:
    """Crops `frame` to `box` (padded), runs MediaPipe Pose on the crop, and maps
    resulting landmarks back to ORIGINAL frame pixel coordinates -- callers should
    never have to think about the crop's own coordinate system."""
    import mediapipe as mp
    import numpy as np

    cx1, cy1, cx2, cy2 = _pad_box(box, frame.shape)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return PoseResult(box=box, landmarks=None)

    crop_rgb = crop[:, :, ::-1]  # BGR (cv2) -> RGB (mediapipe)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_rgb))
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return PoseResult(box=box, landmarks=None)

    crop_h, crop_w = crop.shape[:2]
    landmarks = [
        (cx1 + lm.x * crop_w, cy1 + lm.y * crop_h, lm.visibility)
        for lm in result.pose_landmarks[0]
    ]
    return PoseResult(box=box, landmarks=landmarks)


def make_landmarker():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,  # one crop = one expected person
    )
    return vision.PoseLandmarker.create_from_options(options)
