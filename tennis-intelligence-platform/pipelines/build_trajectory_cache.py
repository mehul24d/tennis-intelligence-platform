"""
build_trajectory_cache.py — one-time offline precompute of the five-engine
probability trajectory for a curated subset of matches (Grand Slam finals),
persisted as extra columns on their existing points_by_match cache partitions.

WHY THIS EXISTS: compute_five_engine_trajectory's per-point loop is dominated by
ml_p_player1's Monte Carlo simulation — measured at ~300ms/point locally, meaning a
single ~200-point match takes roughly a minute even on a full-speed machine, and far
longer under a free-tier deployment's throttled CPU (long enough to make a live
request hang and, worse, starve the single worker process of time to answer health
checks, causing the whole service to be killed and restarted). But the computation is
fully deterministic — ml_p_player1 is seeded by point index, not wall-clock or global
RNG state — so for any match, the same result comes out every time. That means it can
be computed once, offline, and served as an instant lookup instead of recomputed per
request.

Precomputing the FULL ~6000-match corpus this way would take ~11.5 hours even
parallelized across 8 cores (measured: ~317ms/point x ~1.04M total points), so this
script targets a curated, demo-appropriate subset instead: every Grand Slam final in
the charted corpus (183 matches as of 2026-08) — the matches most likely to actually
be viewed. Other matches remain searchable and replayable, just via the slower live
computation path (compute_five_engine_trajectory's normal fallback, unchanged).

Reuses compute_five_engine_trajectory exactly as the live API calls it — no logic
duplicated or reimplemented, just run once per match instead of per request, with the
result persisted instead of returned.

Run with:
    python pipelines/build_trajectory_cache.py

Rerun whenever the model, features, or point-cache data change (i.e. whenever
pipelines/build_points_cache.py would need rerunning too) — like that script, this is
a persisted snapshot, not something that regenerates itself.
"""

from __future__ import annotations

import multiprocessing as mp
import time

from tennis_intel.serving.replay_service import (
    load_replay_context, compute_five_engine_trajectory, POINTS_CACHE_DIR,
)

MAJOR_NAMES = ["Australian_Open", "Roland_Garros", "French_Open", "Wimbledon", "US_Open"]

_ctx = None


def _init_worker() -> None:
    global _ctx
    _ctx = load_replay_context()


def _precompute_one(match_id: str) -> str:
    result = compute_five_engine_trajectory(_ctx, match_id)
    match_df = result["match_df"].copy()
    match_df["markov_p1"] = result["markov_p1"]
    match_df["ml_p1"] = result["ml_p1"]
    match_df["ml_informed_p1"] = result["ml_informed_p1"]
    match_df["ml_informed_unsmoothed_p1"] = result["ml_informed_unsmoothed_p1"]
    match_df["hybrid_p1"] = result["hybrid_p1"]
    match_df["ml_informed_prematch_p1_cached"] = result["ml_informed_prematch_p1"]

    match_dir = POINTS_CACHE_DIR / f"match_id={match_id}"
    for existing in match_dir.glob("*.parquet"):
        existing.unlink()
    match_df.drop(columns=["match_id"], errors="ignore").to_parquet(
        match_dir / "part.parquet", index=False)
    return match_id


def select_grand_slam_finals(match_ids: set[str]) -> list[str]:
    return sorted(
        mid for mid in match_ids
        if mid.split("-")[-3] == "F" and any(major in mid for major in MAJOR_NAMES)
    )


def main() -> None:
    ctx = load_replay_context()
    targets = select_grand_slam_finals(ctx.match_ids)
    print(f"Precomputing {len(targets)} Grand Slam finals across {mp.cpu_count()} processes...")

    t0 = time.monotonic()
    with mp.Pool(processes=mp.cpu_count(), initializer=_init_worker) as pool:
        for i, mid in enumerate(pool.imap_unordered(_precompute_one, targets), 1):
            elapsed = time.monotonic() - t0
            print(f"[{i}/{len(targets)}] {elapsed:.0f}s elapsed: {mid}", flush=True)

    print(f"Done in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
