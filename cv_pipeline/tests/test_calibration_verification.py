"""test_calibration_verification.py — the mandatory gate for any new court
homography calibration, added 2026-07-19.

WHY THIS EXISTS: data/tennis/2.mp4's BL corner was mismeasured by ~49px, and
the calibration was accepted anyway because its held-out-landmark error
(10.1px) looked reasonable -- the least-squares fit absorbed most of the one
bad point's error into the other 7. The bug was only caught later, by
inspecting the rendered overlay on a frame nobody had checked before. This
test makes that inspection step mandatory and non-skippable: it fails, by
design, for any reference_video*_calibration.py module that does NOT have a
complete, checked-in verification manifest recording a real human sign-off
that all 4 doubles corners were visually confirmed against the real court
lines on at least 3 frames spanning the clip (start/middle/end).

This test cannot verify the human actually looked -- that's not something
code can check. What it CAN and does enforce is that the paperwork exists,
is complete (3+ frames, all 4 corners, a non-empty note), and stays in sync
with which calibration modules actually exist in the codebase, so a future
clip's calibration can't be silently merged without going through
render_verification_frames and filling in a real manifest first.
"""

from __future__ import annotations

import pkgutil

import pytest

import cv_pipeline
from cv_pipeline.calibration_verification import manifest_path_for, CalibrationVerificationManifest


def _discover_calibration_modules() -> list[str]:
    """Every cv_pipeline module matching reference_video*_calibration -- found
    by walking the package, not hardcoded, so a new clip's calibration module
    is automatically picked up by this test without editing this file.
    cv_pipeline is a namespace package (no __init__.py), so __path__ is used
    directly rather than __file__ (which is None for namespace packages)."""
    return sorted(
        name for _, name, _ in pkgutil.iter_modules(list(cv_pipeline.__path__))
        if name.startswith("reference_video") and name.endswith("_calibration")
    )


CALIBRATION_MODULES = _discover_calibration_modules()


@pytest.mark.parametrize("module_name", CALIBRATION_MODULES)
def test_calibration_has_verification_manifest(module_name: str):
    # video1 -> "video1", video2 -> "video2", ...
    clip_stem = module_name.removeprefix("reference_").removesuffix("_calibration")
    manifest_file = manifest_path_for(clip_stem)

    assert manifest_file.exists(), (
        f"{module_name} has no calibration-verification manifest at "
        f"{manifest_file}. Before this calibration can be trusted: run "
        f"cv_pipeline.calibration_verification.render_verification_frames on "
        f"start/middle/end frames of the clip, VISUALLY confirm all 4 doubles "
        f"corners land on the real court lines in every frame, then write a "
        f"CalibrationVerificationManifest recording that check and save it to "
        f"{manifest_file}. This is mandatory, not optional -- see this test "
        f"module's docstring and PROGRESS.md's data/tennis/2.mp4 BL-corner entry "
        f"for why."
    )

    manifest = CalibrationVerificationManifest.load(manifest_file)
    assert manifest.calibration_module.endswith(module_name), (
        f"manifest at {manifest_file} is for {manifest.calibration_module}, "
        f"not {module_name} -- stale or mismatched manifest"
    )

    ok, reason = manifest.is_complete()
    assert ok, f"{module_name}'s verification manifest is incomplete: {reason}"


def test_at_least_one_calibration_module_exists():
    # A guard against this whole test file silently collecting zero tests
    # (e.g. if the discovery glob ever breaks) and passing trivially.
    assert CALIBRATION_MODULES, "no reference_video*_calibration modules found -- discovery may be broken"
