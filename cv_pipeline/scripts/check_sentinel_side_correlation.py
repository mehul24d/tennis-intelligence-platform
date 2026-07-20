"""check_sentinel_side_correlation.py — tests whether ball-annotation sentinel
(missing-ball) rows correlate with which side of the court the ball is on.

REVISION (2026-07-15): the first version of this script used a flat pixel-y midpoint
between the near and far baseline corners as a stand-in for "the net's pixel position"
-- this produced an implausible 99-near-frame vs 6709-far-frame split (1.4%/98.6%),
which was the tell that the heuristic was wrong: it doesn't account for perspective
(the far half of the court occupies a much smaller pixel band than the near half, so a
flat midpoint sits too close to the camera). Fixed by building the real
pixel<->court-coordinate homography (cv_pipeline.homography.CourtHomography, validated
independently against the baseline center hash mark to within ~13px/~8cm -- see
verify_homography.py) and projecting the net's TRUE real-world line (y=11.885m,
x in [0, 10.97m]) through it, which is NOT a constant pixel-y -- it's interpolated
per-x via net_pixel_y_at_x().

Side classification for a real ball row: 'near' if ball_pixel_y > net_pixel_y_at_x(ball_pixel_x)
(larger pixel-y = closer to camera), else 'far'. Sentinel/missing rows still have no
ball position to classify directly, so side is imputed from the temporally nearest
real-ball frame -- same limitation as before: assumes the ball's side doesn't flip in
the handful of frames between real detections, which can be wrong right at a
side-crossing frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_pipeline.annotations import load_clip_annotations
from cv_pipeline.homography import CourtHomography

CLIPS = [f"video{i}" for i in range(1, 11)]


def build_homography_for_clip(annotations: dict) -> CourtHomography | None:
    for ann in annotations.values():
        if ann.court_corners:
            return CourtHomography(ann.court_corners)
    return None


def label_sides(annotations: dict, homography: CourtHomography) -> dict[int, str]:
    """Returns frame_index -> 'near'/'far' for every frame, real ball frames labeled
    directly against the homography-projected net line, sentinel/missing frames
    imputed from the nearest real-ball frame."""
    frame_indices = sorted(annotations.keys())
    direct_side: dict[int, str] = {}
    for idx in frame_indices:
        ann = annotations[idx]
        if ann.ball is not None:
            bx, by = ann.ball
            net_y_here = homography.net_pixel_y_at_x(bx)
            direct_side[idx] = "near" if by > net_y_here else "far"

    if not direct_side:
        return {}

    real_idxs = np.array(sorted(direct_side.keys()))
    out: dict[int, str] = {}
    for idx in frame_indices:
        if idx in direct_side:
            out[idx] = direct_side[idx]
        else:
            nearest = real_idxs[np.argmin(np.abs(real_idxs - idx))]
            out[idx] = direct_side[int(nearest)]
    return out


def main():
    rows = []
    for clip in CLIPS:
        annotations = load_clip_annotations(clip)
        homography = build_homography_for_clip(annotations)
        if homography is None:
            print(f"{clip}: no court row found, skipping")
            continue
        sides = label_sides(annotations, homography)

        for idx, ann in annotations.items():
            side = sides.get(idx)
            if side is None:
                continue
            is_sentinel = ann.ball is None  # covers both sentinel-excluded and row-missing
            rows.append({"clip": clip, "frame": idx, "side": side, "is_sentinel": is_sentinel})

    df = pd.DataFrame(rows)

    # --- Sanity check FIRST, per the explicit instruction: does the near/far split
    # itself look like a plausible real rally distribution now? ---
    print("=== Frame-count split by side (sanity check on the classification itself) ===")
    print(f"{'clip':<10} {'near_n':>8} {'far_n':>8} {'near_%':>8}")
    for clip in CLIPS:
        sub = df[df["clip"] == clip]
        if not len(sub):
            continue
        n_near = (sub["side"] == "near").sum()
        n_far = (sub["side"] == "far").sum()
        total = n_near + n_far
        print(f"{clip:<10} {n_near:>8} {n_far:>8} {n_near/total:>7.1%}")

    total_near = (df["side"] == "near").sum()
    total_far = (df["side"] == "far").sum()
    total = total_near + total_far
    print(f"\nAGGREGATE: near={total_near} ({total_near/total:.1%})  far={total_far} ({total_far/total:.1%})")

    if total_near / total < 0.15 or total_near / total > 0.85:
        print("\n*** STILL LOOKS IMPLAUSIBLE (expect both sides to get meaningful playing")
        print("*** time in a real rally) -- do NOT trust the sentinel-rate comparison below")
        print("*** until this split itself looks reasonable.")
        return

    # --- Only proceed to the sentinel-rate comparison if the split above looks sane ---
    print("\n=== Sentinel rate by side ===")
    print(f"{'clip':<10} {'near_rate':>12} {'near_n':>8} {'far_rate':>12} {'far_n':>8}")
    for clip in CLIPS:
        sub = df[df["clip"] == clip]
        if not len(sub):
            continue
        near = sub[sub["side"] == "near"]
        far = sub[sub["side"] == "far"]
        near_rate = near["is_sentinel"].mean() if len(near) else float("nan")
        far_rate = far["is_sentinel"].mean() if len(far) else float("nan")
        print(f"{clip:<10} {near_rate:>11.1%} {len(near):>8} {far_rate:>11.1%} {len(far):>8}")

    near_all = df[df["side"] == "near"]
    far_all = df[df["side"] == "far"]
    near_rate_all = near_all["is_sentinel"].mean()
    far_rate_all = far_all["is_sentinel"].mean()
    print(f"\nAGGREGATE across all 10 clips:")
    print(f"  near-side sentinel rate: {near_rate_all:.1%}  (n={len(near_all)})")
    print(f"  far-side  sentinel rate: {far_rate_all:.1%}  (n={len(far_all)})")
    print(f"  difference (far - near): {far_rate_all - near_rate_all:+.1%}")


if __name__ == "__main__":
    main()
