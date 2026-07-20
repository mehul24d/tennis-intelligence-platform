"""
profile_ml_informed_markov.py — profiles ml_informed_markov_predict on a real subset of
matches to determine whether recursion_sensitivity (suspected, never measured) is actually
the runtime bottleneck, per the prescribed roadmap: profile before optimizing.

Uses cProfile (stdlib, no extra dependency) over a real subset of the 150-match evaluation
sample, then reports both the top functions by cumulative time and a targeted breakdown of
time spent specifically inside recursion_sensitivity vs. everything else in
ml_informed_markov_predict, so the suspicion from code review can be confirmed or refuted
with real numbers rather than structural reasoning alone.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_live_engines_v2 import (
    tracked_player_is_winner, _row_to_match_state, HOLDOUT_YEAR, N_MATCHES, RANDOM_STATE,
    POINT_FILES, PROCESSED,
)
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.return_seed import compute_p_a_return_seed
from tennis_intel.live.ml_informed_markov import (
    ml_informed_markov_predict, ServeReturnPosterior, build_pretrained_prior,
    recursion_sensitivity, ml_informed_point_probabilities,
)
from generate_publication_trajectory import compute_composite_prematch_probability

# Profile a SUBSET, not the full 150 matches — enough points for a stable profile without
# an excessively long profiled run (cProfile itself adds real overhead per call).
N_PROFILE_MATCHES = 20


def main() -> None:
    print("Loading trained classifier...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

    print("Building point dataset...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["match_year"] = points["match_id"].str[:4].astype(int)
    test_points = points[points["match_year"] >= HOLDOUT_YEAR].copy()
    test_points["player1_is_winner"] = (test_points["Svr"] == 1) == test_points["server_is_winner"]

    match_ids = np.sort(test_points["match_id"].unique())
    n_use = min(N_MATCHES, len(match_ids))
    selected_all = np.random.RandomState(RANDOM_STATE).choice(match_ids, size=n_use, replace=False)
    selected = selected_all[:N_PROFILE_MATCHES]
    eval_df = test_points[test_points["match_id"].isin(selected)].copy()
    eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
    print(f"Profiling {len(selected)} matches, {len(eval_df)} points\n")

    def run_all_points():
        current_match_id, posterior = None, None
        for row in eval_df.to_dict("records"):
            if row["match_id"] != current_match_id:
                current_match_id = row["match_id"]
                p0_a_wins = compute_composite_prematch_probability(row)
                # BUG FIX (external review, 2026-07): see return_seed.py's module docstring.
                p_a_return_seed = compute_p_a_return_seed(row, track_winner=True)
                elo_a = row.get("elo_matches_played_pre_winner")
                elo_b = row.get("elo_matches_played_pre_loser")
                best_of_val = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3
                p_serve0, n0s, p_return0, n0r = build_pretrained_prior(
                    p0_a_wins, p_a_return_seed, best_of_val,
                    elo_matches_played_a=elo_a, elo_matches_played_b=elo_b,
                )
                posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0s, p_return0, n0r)

            state = _row_to_match_state(row)
            _, posterior = ml_informed_markov_predict(state, row, model, feature_cols, posterior)

    t0 = time.perf_counter()
    run_all_points()
    wall_time = time.perf_counter() - t0
    print(f"Wall-clock time for {len(eval_df)} points: {wall_time:.2f}s "
          f"({len(eval_df)/wall_time:.1f} points/sec)\n")

    profiler = cProfile.Profile()
    profiler.enable()
    run_all_points()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)
    print("=== Top 20 functions by cumulative time ===")
    print(stream.getvalue())

    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.print_stats("recursion_sensitivity")
    print("=== recursion_sensitivity specifically ===")
    print(stream2.getvalue())

    stream3 = io.StringIO()
    stats3 = pstats.Stats(profiler, stream=stream3)
    stats3.print_stats("prob_a_wins_match_from_state")
    print("=== prob_a_wins_match_from_state (the recursion itself) specifically ===")
    print(stream3.getvalue())

    stream4 = io.StringIO()
    stats4 = pstats.Stats(profiler, stream=stream4)
    stats4.print_stats("predict_proba")
    print("=== classifier predict_proba calls specifically ===")
    print(stream4.getvalue())

    print("\nWhat to look for: compare recursion_sensitivity's OWN 'tottime' (time spent in")
    print("that function's own code, excluding sub-calls) against predict_proba's tottime")
    print("and against the overall wall-clock time above. If predict_proba dominates, the")
    print("classifier inference itself (not the sensitivity/recursion math) is the real")
    print("bottleneck — a genuinely different optimization target than what code review")
    print("alone suggested.")


if __name__ == "__main__":
    main()