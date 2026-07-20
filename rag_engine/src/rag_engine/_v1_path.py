"""_v1_path.py — makes tennis-intelligence-platform's `src/` importable from
rag_engine, mirroring the exact sys.path pattern replay_service.py already uses to
import across the v1/v2 boundary (PROJECT_ROOT / "src"), so v2 modules reuse v1's
serving-layer functions instead of re-deriving stats from raw parquet files.

Also adds the tennis-intelligence-platform ROOT itself (not just src/) to sys.path:
replay_service.py does `from pipelines.replay_match import ...` (a dotted import
requiring pipelines/'s PARENT directory on the path), which it has only ever worked
via the CALLER's cwd happening to be tennis-intelligence-platform (Python implicitly
puts '' on sys.path for that case) — not via its own sys.path.insert(PROJECT_ROOT /
"pipelines"), which adds the wrong directory for that import. Since rag_engine's own
scripts don't run with that cwd, `pipelines.replay_match` fails to import without this
explicit addition — a caller-cwd-dependent fragility in v1, not something to "fix" in
v1 itself here, just worked around on this side of the v1/v2 boundary."""

from __future__ import annotations

import sys
from pathlib import Path

_V1_ROOT = Path(__file__).resolve().parents[3] / "tennis-intelligence-platform"
_V1_SRC = _V1_ROOT / "src"

for _path in (_V1_ROOT, _V1_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
