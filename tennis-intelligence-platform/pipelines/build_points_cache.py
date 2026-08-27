"""
build_points_cache.py — one-time (rerun only when the underlying features change)
offline build of the full point-level dataset, persisted partitioned by match_id
instead of held in memory.

WHY THIS EXISTS: tennis_intel.serving.replay_service.load_replay_context() used to
call build_point_dataset() itself at server startup and hold the ~590MB result in RAM
for the life of the process, even though any single request only ever needs ONE
match's ~175 points (compute_five_engine_trajectory filters by match_id). That's fine
locally, but it OOM-kills a memory-constrained deployment. This script runs the EXACT
SAME build_point_dataset() call — no logic duplicated or reimplemented — and writes
the result to data/processed/points_by_match/, a Hive-partitioned parquet directory
(match_id=<id>/*.parquet). The live server then reads only the requested match's
partition per request (see replay_service._load_match_points), via pyarrow partition
pruning, instead of ever materializing the full corpus.

Run with:
    python pipelines/build_points_cache.py

Rerun this whenever POINT_FILES, frozen_join, or day6's features change (i.e.
whenever pipelines/replay_match.py's own inputs would produce a different point
dataset) — the cache is a persisted snapshot, not something that regenerates itself.
"""

from __future__ import annotations

import shutil

import pandas as pd

from tennis_intel.live.build_point_dataset import build_point_dataset
from pipelines.replay_match import PROCESSED, POINT_FILES

OUT_DIR = PROCESSED / "points_by_match"


def main() -> None:
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")

    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["player1_is_winner"] = (points["Svr"] == 1) == points["server_is_winner"]

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    # match_id as plain string for partitioning — category dtype isn't preserved
    # through a partitioned write regardless, and pyarrow partitions on the literal
    # value either way.
    points["match_id"] = points["match_id"].astype(str)
    points.to_parquet(OUT_DIR, partition_cols=["match_id"], index=False)

    print(f"Wrote {len(points)} points across {points['match_id'].nunique()} match "
          f"partitions to {OUT_DIR}")


if __name__ == "__main__":
    main()
