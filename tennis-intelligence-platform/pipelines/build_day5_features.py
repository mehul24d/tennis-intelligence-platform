"""
build_day5_features.py — pipeline entrypoint for Day 5: leakage-safe rolling performance
features, built on top of the FROZEN Day 4 Elo pipeline.

Usage (from project root, with .venv activated):
    python pipelines/build_day5_features.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tennis_intel.features.feature_engineering_day5 import compute_day5_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ELO_MATCHES_PATH = PROCESSED_DIR / "matches_with_elo.parquet"
PLAYERS_PATH = PROCESSED_DIR / "players.parquet"
OUTPUT_PATH = PROCESSED_DIR / "matches_with_day5_features.parquet"

NOTABLE_PLAYERS = ["Novak Djokovic", "Rafael Nadal", "Roger Federer", "Carlos Alcaraz", "Jannik Sinner"]


def print_diagnostics(diagnostics: dict) -> None:
    print("=== Day 5 Feature Engineering Diagnostics ===")
    print(f"Processed matches:     {diagnostics['processed_matches']:,}")
    print(f"Players tracked:       {diagnostics['players_tracked']:,}")
    print(f"Score missing:         {diagnostics['score_missing']:,}")
    print(f"Score unparseable:     {diagnostics['score_unparseable']:,}")
    print(f"Score parse rate:      {diagnostics['score_parse_rate']:.1%}")


def real_data_validation(augmented: pd.DataFrame, players: pd.DataFrame) -> None:
    print("\n=== Real-Data Validation ===")

    key_cols = [
        "winner_win_pct_last10", "winner_surface_win_pct_last10",
        "winner_opponent_elo_mean_last10", "winner_rest_days",
        "winner_win_streak_entering_match",
    ]
    print("\nNull counts / distribution for key features:")
    for col in key_cols:
        if col not in augmented.columns:
            continue
        s = augmented[col]
        print(f"  {col:35s} null={s.isna().sum():>7,} ({s.isna().mean():.1%})  "
              f"min={s.min():.2f}  median={s.median():.2f}  mean={s.mean():.2f}  max={s.max():.2f}")

    id_to_name = dict(zip(players["player_id"], players["canonical_name"]))
    name_to_id = {v: k for k, v in id_to_name.items()}

    print(f"\nSpot-check: chronology for notable players (win_pct_last10 should evolve, not be static):")
    for name in NOTABLE_PLAYERS:
        pid = name_to_id.get(name)
        if pid is None:
            print(f"  {name}: not found in registry")
            continue
        as_w = augmented[augmented["winner_id"] == pid][["tourney_date", "winner_win_pct_last10"]].rename(
            columns={"winner_win_pct_last10": "win_pct_last10"})
        as_l = augmented[augmented["loser_id"] == pid][["tourney_date", "loser_win_pct_last10"]].rename(
            columns={"loser_win_pct_last10": "win_pct_last10"})
        traj = pd.concat([as_w, as_l]).sort_values("tourney_date")
        if traj.empty:
            print(f"  {name}: no matches found")
            continue
        print(f"  {name}: {len(traj)} matches, win_pct_last10 first={traj['win_pct_last10'].iloc[0]:.2f}, "
              f"min={traj['win_pct_last10'].min():.2f}, max={traj['win_pct_last10'].max():.2f}, "
              f"last={traj['win_pct_last10'].iloc[-1]:.2f}")


def main() -> None:
    if not ELO_MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"{ELO_MATCHES_PATH} not found — run pipelines/build_elo.py first "
            "(Day 4 is frozen, but its output is a required input here)."
        )
    matches = pd.read_parquet(ELO_MATCHES_PATH)
    players = pd.read_parquet(PLAYERS_PATH)
    logger.info("Loaded %d matches, %d players", len(matches), len(players))

    result = compute_day5_features(matches)

    print_diagnostics(result.diagnostics)
    real_data_validation(result.augmented, players)

    result.augmented.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(result.augmented):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()