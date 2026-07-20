"""
build_elo.py — pipeline entrypoint for Week 2, Day 4 (v2 revision): leakage-safe Elo,
extended per the project's Elo-redesign design note with dynamic K-factor, retirement/
walkover-aware updates, a match-count confidence signal, and independent surface-specific
ratings (Hard/Clay/Grass) — while deliberately AVOIDING margin-of-victory weighting,
multi-dimensional serve/return Elo, and other extensions judged redundant with this
project's existing rolling-stats/serve-return feature set (see the design note for the
full bucket-by-bucket justification of what was and wasn't included).

Reads matches_with_player_ids.parquet (native TML IDs, no string lookups) and players.parquet
(for the real-data sanity checks at the end), computes chronological Elo via the generic
processor + EloRating implementation, prints diagnostics, and writes matches_with_elo.parquet.

RETIREMENT/WALKOVER DETECTION: TML's `score` column encodes these as substrings ("RET",
"W/O", "DEF" for defaulted matches) rather than a dedicated boolean column — detected here
via a simple substring check, logged so the detected count can be sanity-checked against
known tour statistics (retirements are historically a low-single-digit percentage of tour
matches).

Usage (from project root, with .venv activated):
    python pipelines/build_elo.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tennis_intel.ratings.elo import EloRating
from tennis_intel.ratings.processor import compute_ratings, default_dynamic_k
from tennis_intel.ratings.surface_elo import compute_surface_ratings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MATCHES_PATH = PROCESSED_DIR / "matches_with_player_ids.parquet"
PLAYERS_PATH = PROCESSED_DIR / "players.parquet"
OUTPUT_PATH = PROCESSED_DIR / "matches_with_elo.parquet"


def detect_retirement_and_walkover(matches: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """
    Adds is_retirement and is_walkover boolean columns from TML's score-string convention.
    A walkover ("W/O") means no match was played at all; a retirement ("RET") or defaulted
    match ("DEF") means a match was played but ended early — treated differently downstream
    (walkovers excluded from rating updates entirely, retirements get a discounted update).
    """
    df = matches.copy()
    score_str = df[score_col].astype(str).str.upper()
    df["is_walkover"] = score_str.str.contains("W/O", na=False)
    df["is_retirement"] = (
        score_str.str.contains("RET", na=False) | score_str.str.contains("DEF", na=False)
    ) & ~df["is_walkover"]
    n_wo, n_ret = df["is_walkover"].sum(), df["is_retirement"].sum()
    logger.info("Detected %d walkover(s) (%.2f%%) and %d retirement(s) (%.2f%%) out of %d matches",
                n_wo, 100 * n_wo / len(df), n_ret, 100 * n_ret / len(df), len(df))
    return df


def print_diagnostics(diagnostics: dict) -> None:
    print("=== Elo Processing Diagnostics ===")
    print(f"Processed matches:       {diagnostics['processed_matches']:,}")
    print(f"Players rated:           {diagnostics['players_rated']:,}")
    print(f"Initializations:         {diagnostics['initializations']:,}")
    print(f"Walkovers skipped:       {diagnostics.get('walkovers_skipped', 0):,}")
    print(f"Retirements discounted:  {diagnostics.get('retirements_discounted', 0):,}")
    print(f"Average rating:          {diagnostics['average_rating']:.1f}")
    print(f"Min rating:              {diagnostics['min_rating']:.1f}")
    print(f"Max rating:              {diagnostics['max_rating']:.1f}")
    print(f"Largest single update:   {diagnostics['largest_single_update']:.2f}")
    print(f"Mean update magnitude:   {diagnostics['mean_update_magnitude']:.2f}")


def real_data_sanity_checks(augmented: pd.DataFrame, players: pd.DataFrame, final_ratings: dict) -> None:
    print("\n=== Real-Data Sanity Checks ===")

    id_to_name = dict(zip(players["player_id"], players["canonical_name"]))
    final_df = pd.DataFrame(
        [{"player_id": pid, "final_elo": rating} for pid, rating in final_ratings.items()]
    )
    final_df["name"] = final_df["player_id"].map(id_to_name)
    top20 = final_df.sort_values("final_elo", ascending=False).head(20)
    print("\nTop 20 by final overall Elo:")
    print(top20[["name", "final_elo"]].to_string(index=False))

    print("\nElo trajectory (first, min, max, last) for notable players, if found:")
    notable = ["Novak Djokovic", "Rafael Nadal", "Roger Federer", "Carlos Alcaraz", "Jannik Sinner"]
    name_to_id = {v: k for k, v in id_to_name.items()}
    for name in notable:
        pid = name_to_id.get(name)
        if pid is None:
            print(f"  {name}: not found in registry")
            continue
        as_winner = augmented[augmented["winner_id"] == pid][["tourney_date", "elo_post_match_winner"]].rename(
            columns={"elo_post_match_winner": "elo"})
        as_loser = augmented[augmented["loser_id"] == pid][["tourney_date", "elo_post_match_loser"]].rename(
            columns={"elo_post_match_loser": "elo"})
        trajectory = pd.concat([as_winner, as_loser]).sort_values("tourney_date")
        if trajectory.empty:
            print(f"  {name}: no matches found")
            continue
        print(f"  {name}: first={trajectory['elo'].iloc[0]:.0f}, "
              f"min={trajectory['elo'].min():.0f}, max={trajectory['elo'].max():.0f}, "
              f"last={trajectory['elo'].iloc[-1]:.0f} ({len(trajectory)} matches)")

    elo_diff = (augmented["elo_pre_match_winner"] - augmented["elo_pre_match_loser"]).abs()
    print(f"\nElo difference distribution (|winner_pre - loser_pre|):")
    print(f"  mean={elo_diff.mean():.1f}, median={elo_diff.median():.1f}, "
          f"p90={elo_diff.quantile(0.9):.1f}, max={elo_diff.max():.1f}")

    print(f"\nCalibration check (expected_win_prob bucket -> observed win rate, should be close):")
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    augmented["_prob_bucket"] = pd.cut(augmented["expected_win_prob"], bins=bins, include_lowest=True)
    calibration = augmented.groupby("_prob_bucket", observed=True).size().rename("n_matches")
    print(calibration.to_string())
    print("  (note: winner always has actual outcome=1 by construction of the dataset — "
          "true calibration requires evaluating on a held-out symmetric point-level task, "
          "not this aggregate; this is a rough sanity signal only.)")

    # Surface Elo sanity: per-surface final rating for the same notable players, so a real
    # clay specialist should visibly show a higher final Clay Elo than final Hard/Grass Elo
    # (and vice versa for hard/grass specialists) — a much sharper sanity check than the
    # aggregate coverage counts alone, since it validates the SURFACE-SPECIFIC SIGNAL itself
    # against known real-world player profiles, not just "did the pipeline run".
    if "elo_surface_pre_match_winner" in augmented.columns:
        print("\nSurface Elo coverage:")
        for surface in ["Hard", "Clay", "Grass"]:
            n = (augmented["surface"] == surface).sum()
            n_rated = augmented[augmented["surface"] == surface]["elo_surface_pre_match_winner"].notna().sum()
            print(f"  {surface}: {n:,} matches, {n_rated:,} with surface Elo computed")

        print("\nPeak surface Elo for notable players (Hard / Clay / Grass), if found:")
        print("(Sanity check: known clay specialists should show Clay > Hard/Grass here, and")
        print(" vice versa for hard/grass specialists. Peak rather than final rating is used")
        print(" deliberately — final rating is skewed by end-of-career decline for players")
        print(" whose last matches came well past their surface prime, which would understate")
        print(" how dominant they actually were at their best.)")
        for name in notable:
            pid = name_to_id.get(name)
            if pid is None:
                continue
            surface_peaks = {}
            for surface in ["Hard", "Clay", "Grass"]:
                surf_matches = augmented[augmented["surface"] == surface]
                as_winner = surf_matches[surf_matches["winner_id"] == pid][
                    ["tourney_date", "elo_surface_post_match_winner"]
                ].rename(columns={"elo_surface_post_match_winner": "elo"})
                as_loser = surf_matches[surf_matches["loser_id"] == pid][
                    ["tourney_date", "elo_surface_post_match_loser"]
                ].rename(columns={"elo_surface_post_match_loser": "elo"})
                traj = pd.concat([as_winner, as_loser]).sort_values("tourney_date")
                traj = traj.dropna(subset=["elo"])
                surface_peaks[surface] = (
                    f"{traj['elo'].max():.0f} (n={len(traj)})" if not traj.empty else "no matches"
                )
            print(f"  {name}: Hard={surface_peaks['Hard']}, "
                  f"Clay={surface_peaks['Clay']}, Grass={surface_peaks['Grass']}")


def main() -> None:
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"{MATCHES_PATH} not found — run pipelines/build_player_registry.py first."
        )
    if not PLAYERS_PATH.exists():
        raise FileNotFoundError(
            f"{PLAYERS_PATH} not found — run pipelines/build_player_registry.py first."
        )

    matches = pd.read_parquet(MATCHES_PATH)
    players = pd.read_parquet(PLAYERS_PATH)
    logger.info("Loaded %d matches, %d players", len(matches), len(players))

    matches = detect_retirement_and_walkover(matches)

    logger.info("Computing overall Elo (dynamic K, retirement/walkover-aware)...")
    result = compute_ratings(
        matches, EloRating(),
        k_fn=lambda mp: default_dynamic_k(mp),
        retirement_col="is_retirement",
        walkover_col="is_walkover",
    )
    print_diagnostics(result.diagnostics)

    logger.info("Computing surface-specific Elo (Hard/Clay/Grass, independent ladders)...")
    with_surface = compute_surface_ratings(
        result.augmented, lambda: EloRating(),
        k_fn=lambda mp: default_dynamic_k(mp),
        retirement_col="is_retirement",
        walkover_col="is_walkover",
    )

    real_data_sanity_checks(with_surface, players, result.final_ratings)

    output = with_surface.drop(columns=["_prob_bucket"], errors="ignore")
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(output):,} rows to {OUTPUT_PATH}")
    print(f"New columns vs. the original Day 4 freeze: elo_matches_played_pre_winner/loser, "
          f"elo_surface_pre_match_winner/loser, elo_surface_post_match_winner/loser, "
          f"elo_surface_matches_played_pre_winner/loser, is_retirement, is_walkover")


if __name__ == "__main__":
    main()