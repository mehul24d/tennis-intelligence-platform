"""
evaluate_full_match_calibration.py — generalizes evaluate_early_deficit_calibration.py's
single-bucket check into a full phase/leverage-segmented reliability table across all
three engines, per the concrete next-step request following the early-deficit backtest's
result (which showed the smoothed engine was NOT overconfident in that bucket relative to
the alternatives — it had the best log loss and Brier of the three).

BUCKET DEFINITIONS (reusing existing, already-validated row-level flags —
is_break_point/is_set_point/is_match_point/is_tiebreak_game — rather than reimplementing
point-type detection):
  - Tiebreak (is_tiebreak_game)
  - Match point (is_match_point, non-tiebreak)
  - Set point (is_set_point, non-tiebreak, non-match-point)
  - Break point (is_break_point, none of the above)
  - Early-set deficit / lead / even (games 0-3 total in current set, non-BP/SP/MP/TB)
  - Mid-set deficit / lead / even (games 4-8, same exclusions)
  - Deciding set, any of the above, tracked as an additional cross-cutting flag alongside
    phase rather than a separate bucket, since "deciding set + break point" is
    meaningfully different from "deciding set, routine point" and collapsing them would
    hide that.

Deficit/lead/even is always relative to the TRACKED player (the real match winner or
loser, matching this project's "A" convention throughout) — reuses the exact
is_early_set_break_deficit-style games comparison already tested in
evaluate_early_deficit_calibration.py, generalized to any game-count range and to
lead/even in addition to deficit.

RUNTIME NOTE: this is a stateful, sequential per-point Bayesian update for the smoothed
engine — not vectorizable. Defaults to the SAME 150-match sample used throughout this
project's other evaluations as a fast smoke test; pass --full to run on the complete
holdout set (5,981 matches), which will take substantially longer (expect on the order of
20-30+ minutes depending on hardware, roughly 40x this script's 150-match runtime) and
is recommended only after confirming the smoke-test run looks sane.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_live_engines_v2 import (
    tracked_player_is_winner, markov_p_winner, HOLDOUT_YEAR, N_MATCHES, RANDOM_STATE,
    POINT_FILES, PROCESSED,
)
from tennis_intel.live.match_state_conversion import row_to_match_state
from tennis_intel.live.build_point_dataset import build_point_dataset
from tennis_intel.live.ml_informed_markov import (
    ml_informed_markov_predict, ml_informed_point_probabilities, ServeReturnPosterior,
    build_pretrained_prior,
)
from tennis_intel.live.live_win_probability import prob_a_wins_match_from_state
from tennis_intel.evaluation.metrics import compute_log_loss, compute_brier_score
from generate_publication_trajectory import compute_composite_prematch_probability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EARLY_SET_MAX_GAMES = 3
MID_SET_MAX_GAMES = 8


def classify_point(row: dict, track_winner: bool) -> tuple[str, bool]:
    """Returns (phase_bucket_name, is_deciding_set). Point-type flags (BP/SP/MP/TB) take
    precedence over game-count phase, since a break point at 2-2 is a fundamentally
    different situation from a routine point at 2-2 — collapsing them would hide exactly
    the kind of phase-dependent miscalibration this table is meant to detect."""
    if bool(row.get("is_tiebreak_game")):
        phase = "tiebreak"
    elif bool(row.get("is_match_point")):
        phase = "match_point"
    elif bool(row.get("is_set_point")):
        phase = "set_point"
    elif bool(row.get("is_break_point")):
        phase = "break_point"
    else:
        p1_is_winner = bool(row["player1_is_winner"])
        a_is_p1 = p1_is_winner if track_winner else (not p1_is_winner)
        a_games = row["Gm1"] if a_is_p1 else row["Gm2"]
        b_games = row["Gm2"] if a_is_p1 else row["Gm1"]
        total_games = a_games + b_games
        if total_games <= EARLY_SET_MAX_GAMES:
            set_stage = "early_set"
        elif total_games <= MID_SET_MAX_GAMES:
            set_stage = "mid_set"
        else:
            set_stage = "late_set"

        if a_games > b_games:
            leverage = "lead"
        elif a_games < b_games:
            leverage = "deficit"
        else:
            leverage = "even"
        phase = f"{set_stage}_{leverage}"

    best_of = int(row.get("best_of", 3)) if pd.notna(row.get("best_of")) else 3
    sets_needed = (best_of // 2) + 1
    is_deciding = (row.get("Set1", 0) == row.get("Set2", 0) == sets_needed - 1)

    return phase, bool(is_deciding)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run on the full holdout set (~5,981 matches) instead of "
                             "the 150-match smoke-test sample. Substantially slower.")
    args = parser.parse_args()

    logger.info("Loading trained classifier...")
    payload = joblib.load(str(PROCESSED / "day9_point_classifiers.joblib"))
    model, feature_cols = payload["gradient_boosting"], payload["feature_cols"]

    logger.info("Building point dataset...")
    frozen_join = pd.read_parquet(PROCESSED / "joined_matches_m.parquet")
    day6 = pd.read_parquet(PROCESSED / "matches_with_day6_features.parquet")
    points = build_point_dataset(POINT_FILES, frozen_join, day6)
    points["match_year"] = points["match_id"].str[:4].astype(int)
    test_points = points[points["match_year"] >= HOLDOUT_YEAR].copy()
    test_points["player1_is_winner"] = (test_points["Svr"] == 1) == test_points["server_is_winner"]

    match_ids = np.sort(test_points["match_id"].unique())
    if args.full:
        selected = match_ids
        logger.info("Running FULL holdout set: %d matches. This will take a while.", len(selected))
    else:
        n_use = min(N_MATCHES, len(match_ids))
        selected = np.random.RandomState(RANDOM_STATE).choice(match_ids, size=n_use, replace=False)
        logger.info("Running smoke-test sample: %d matches. Pass --full for the complete "
                   "holdout set once this looks sane.", n_use)

    eval_df = test_points[test_points["match_id"].isin(selected)].copy()
    eval_df = eval_df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
    logger.info("Evaluating %d matches, %d points total", len(selected), len(eval_df))

    records = []
    current_match_id, posterior = None, None

    for idx, row in enumerate(eval_df.to_dict("records")):
        if idx % 5000 == 0 and idx > 0:
            logger.info("  %d / %d points (%.1f%%)", idx, len(eval_df), 100 * idx / len(eval_df))

        if row["match_id"] != current_match_id:
            current_match_id = row["match_id"]
            p0_a_wins = compute_composite_prematch_probability(row)
            loser_serve_surface = row.get("loser_first_serve_win_pct_surface_career")
            loser_serve_career = row.get("loser_first_serve_win_pct_career")
            if loser_serve_surface is not None and pd.notna(loser_serve_surface):
                opponent_serve = float(loser_serve_surface)
            elif loser_serve_career is not None and pd.notna(loser_serve_career):
                opponent_serve = float(loser_serve_career)
            else:
                opponent_serve = 0.65
            p_a_return_seed = 1.0 - opponent_serve
            elo_a = row.get("elo_matches_played_pre_winner")
            elo_b = row.get("elo_matches_played_pre_loser")
            h2h = None
            if pd.notna(row.get("winner_h2h_wins_pre_match")) and pd.notna(row.get("loser_h2h_wins_pre_match")):
                h2h = float(row["winner_h2h_wins_pre_match"]) + float(row["loser_h2h_wins_pre_match"])
            best_of_val = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3

            p_serve0, n0_serve, p_return0, n0_return = build_pretrained_prior(
                p0_a_wins, p_a_return_seed, best_of_val,
                elo_matches_played_a=elo_a, elo_matches_played_b=elo_b, h2h_meetings=h2h,
            )
            posterior = ServeReturnPosterior.from_pretrained_prior(p_serve0, n0_serve, p_return0, n0_return)

        track_winner = tracked_player_is_winner(row["match_id"])
        state = row_to_match_state(row)

        p_markov_winner = markov_p_winner(row)
        p_smoothed_winner, posterior = ml_informed_markov_predict(state, row, model, feature_cols, posterior)

        p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(row, model, feature_cols)
        p_a_serve_raw_c = float(np.clip(p_a_serve_raw, 0.01, 0.99))
        p_a_return_raw_c = float(np.clip(p_a_return_raw, 0.01, 0.99))
        p_unsmoothed_winner = prob_a_wins_match_from_state(state, p_a_serve_raw_c, p_a_return_raw_c)

        phase, is_deciding = classify_point(row, track_winner)

        target = 1.0 if track_winner else 0.0
        markov_p = p_markov_winner if track_winner else (1.0 - p_markov_winner)
        smoothed_p = p_smoothed_winner if track_winner else (1.0 - p_smoothed_winner)
        unsmoothed_p = p_unsmoothed_winner if track_winner else (1.0 - p_unsmoothed_winner)

        records.append({
            "phase": phase, "is_deciding": is_deciding, "target": target,
            "markov_p": markov_p, "smoothed_p": smoothed_p, "unsmoothed_p": unsmoothed_p,
        })

    df = pd.DataFrame(records)

    def _report_for_engine(engine_col: str, engine_name: str) -> pd.DataFrame:
        rows = []
        for phase, group in df.groupby("phase"):
            y, p = group["target"].values, group[engine_col].values
            rows.append({
                "phase": phase, "n": len(group),
                "mean_predicted": p.mean(), "observed_win_rate": y.mean(),
                "calibration_gap": y.mean() - p.mean(),
                "log_loss": compute_log_loss(y, p) if len(group) >= 5 else np.nan,
                "brier": compute_brier_score(y, p) if len(group) >= 5 else np.nan,
            })
        result = pd.DataFrame(rows).sort_values("phase").reset_index(drop=True)
        print(f"\n=== {engine_name} — full phase-segmented reliability table ===")
        print(result.to_string(index=False))
        gaps = result["calibration_gap"].abs()
        worst_idx = gaps.idxmax()
        print(f"\n{engine_name} consistency summary: "
              f"mean |gap|={gaps.mean():.4f}, max |gap|={gaps.max():.4f} "
              f"(phase: {result.loc[worst_idx, 'phase']}, n={result.loc[worst_idx, 'n']})")
        return result

    for col, name in [("markov_p", "Pure Markov"), ("smoothed_p", "ML-Informed (smoothed)"),
                       ("unsmoothed_p", "ML-Informed (unsmoothed)")]:
        _report_for_engine(col, name)

    print(f"\n=== Deciding-set cross-cut (all phases combined) ===")
    for col, name in [("markov_p", "Pure Markov"), ("smoothed_p", "ML-Informed (smoothed)"),
                       ("unsmoothed_p", "ML-Informed (unsmoothed)")]:
        for is_dec, label in [(True, "deciding set"), (False, "non-deciding set")]:
            sub = df[df["is_deciding"] == is_dec]
            if len(sub) < 5:
                continue
            y, p = sub["target"].values, sub[col].values
            print(f"{name:<26} {label:<16} n={len(sub):>6} "
                  f"log_loss={compute_log_loss(y, p):.4f} brier={compute_brier_score(y, p):.4f}")


if __name__ == "__main__":
    main()