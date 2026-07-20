"""
build_day6_features.py — pipeline entrypoint for Day 6: MCP point-level serve/return
features, merged back onto the frozen Day 5 dataset.

Usage (from project root, with .venv activated):
    python pipelines/build_day6_features.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tennis_intel.features.serve_return_features import (
    load_and_prepare_stats, attach_player_ids_and_chronology, compute_rolling_serve_return_features,
    load_and_prepare_key_points_stats, compute_elo_trend_features,
)
from tennis_intel.features.surface_serve_return_features import (
    attach_surface, compute_rolling_surface_serve_return_features, SURFACE_RATE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_MCP_DIR = PROJECT_ROOT / "data" / "raw" / "tennis_MatchChartingProject"

DAY5_PATH = PROCESSED_DIR / "matches_with_day5_features.parquet"
FROZEN_JOIN_PATH = PROCESSED_DIR / "joined_matches_m.parquet"
STATS_PATH = RAW_MCP_DIR / "charting-m-stats-Overview.csv"
KEY_POINTS_SERVE_PATH = RAW_MCP_DIR / "charting-m-stats-KeyPointsServe.csv"
KEY_POINTS_RETURN_PATH = RAW_MCP_DIR / "charting-m-stats-KeyPointsReturn.csv"
OUTPUT_PATH = PROCESSED_DIR / "matches_with_day6_features.parquet"

RATE_COLS = ["first_serve_in_pct", "first_serve_win_pct", "second_serve_win_pct",
             "ace_rate", "df_rate", "bp_saved_pct", "return_pts_won_pct", "bp_converted_pct"]


def sanity_check_bp_saved_interpretation(stats: pd.DataFrame) -> None:
    """Checks the bk_pts/bp_saved column-meaning assumption against known tennis history
    (tour-average break-point-save rate is historically ~60-65%) rather than trusting it
    silently — see module docstring in serve_return_features.py."""
    mean_bp_saved = stats["bp_saved_pct"].mean()
    print(f"\nSanity check — mean bp_saved_pct across all charted matches: {mean_bp_saved:.1%}")
    if 0.55 <= mean_bp_saved <= 0.70:
        print("✅ Within the historically expected ~60-65% range — bk_pts/bp_saved "
              "interpretation (break points faced/saved on serve) is confirmed plausible.")
    else:
        print("⚠️  OUTSIDE the historically expected range — the bk_pts/bp_saved column "
              "interpretation may be wrong. Do not trust downstream bp_saved_pct/"
              "bp_converted_pct features until this is investigated.")


def main() -> None:
    for path in [DAY5_PATH, FROZEN_JOIN_PATH, STATS_PATH, KEY_POINTS_SERVE_PATH, KEY_POINTS_RETURN_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")

    day5 = pd.read_parquet(DAY5_PATH)
    frozen_join = pd.read_parquet(FROZEN_JOIN_PATH)
    logger.info("Loaded %d Day5 matches, %d frozen-join rows", len(day5), len(frozen_join))

    stats = load_and_prepare_stats(STATS_PATH)
    logger.info("Loaded %d per-match stats rows (Total only)", len(stats))
    sanity_check_bp_saved_interpretation(stats)

    stats_with_ids = attach_player_ids_and_chronology(stats, frozen_join)
    rolling = compute_rolling_serve_return_features(stats_with_ids)

    # NEW (surface-conditioned career stats, 2026-07): built following the exact same
    # non-destructive pattern as the surface Elo extension. Motivated by direct evidence
    # that surface-blind career stats are weak (37.6% of match winners had a LOWER career
    # first-serve-win% than the loser) and confirmed via permutation importance that the
    # analogous surface Elo fix is genuinely used by the trained classifier and outperforms
    # both the surface-blind Elo and the existing surface_win_pct_last10 rolling feature.
    stats_with_surface = attach_surface(stats_with_ids, frozen_join)
    surface_rolling = compute_rolling_surface_serve_return_features(stats_with_surface)
    surface_feature_cols = [c for c in surface_rolling.columns if "_surface_" in c]
    logger.info("Computed %d surface-conditioned feature columns", len(surface_feature_cols))

    # NEW (break-point-specific serve/return career rates, 2026-07): built following the
    # exact same non-destructive pattern as the surface Elo/surface-serve extensions.
    # Motivated by the deciding-set investigation (a real, structural log-loss gap that
    # survived conditioning on points-remaining, a graded fatigue proxy, and a binary
    # deciding-set flag) — a player's REAL historical break-point-serving/-returning rate,
    # separate from their routine-point rate, is a genuinely different signal from
    # anything currently in the schema. Confirmed via direct inspection of real sample
    # rows (not assumed) that KeyPointsServe's row=='BP' and KeyPointsReturn's
    # row=='BPO' are the correct break-point-specific rows, distinct from game-point
    # ('GP'/'GPF'), deuce ('Deuce'/'DeuceR'), and match-wide rollup ('STotal'/'RTotal')
    # rows. Leakage safety is inherited from reusing the SAME shift(1)-then-expanding()
    # rolling function already proven for every other _career feature here — see
    # load_and_prepare_key_points_stats's own docstring for the full derivation.
    #
    # RETURN-SIDE ONLY (2026-07 follow-up): bp_serve_win_pct was built, tested via
    # permutation importance, and REMOVED after direct verification that it is
    # mathematically identical to the already-existing bp_saved_pct_career (r=1.0000
    # across 347,093 held-out points — a break point is "saved" iff the server wins that
    # point, so both ratios are computed over the same numerator/denominator, just
    # sourced from different MCP files). No longer computed at all, even at this
    # intermediate stage, since keeping a confirmed-duplicate column around serves no
    # purpose. bp_return_win_pct does NOT share this property (measured r=0.45-0.51 with
    # the analogous existing feature — related but genuinely distinct) and showed real
    # permutation importance, so it alone is kept.
    bp_stats = load_and_prepare_key_points_stats(KEY_POINTS_SERVE_PATH, KEY_POINTS_RETURN_PATH)
    logger.info("Loaded %d per-match break-point stats rows", len(bp_stats))
    bp_stats_with_ids = attach_player_ids_and_chronology(bp_stats, frozen_join)
    bp_rolling = compute_rolling_serve_return_features(
        bp_stats_with_ids, rate_cols=["bp_return_win_pct"],
    )
    bp_feature_cols = [c for c in bp_rolling.columns if c.startswith("bp_return_win_pct")]
    logger.info("Computed %d break-point-specific feature columns", len(bp_feature_cols))

    feature_cols = [c for c in rolling.columns if any(c.startswith(r) for r in RATE_COLS)]
    # NOTE: neither (tourney_id, match_num) alone NOR (tml_tourney_id, tml_match_num) in
    # frozen_join are safe merge keys — Day 5 has 480 rows with NaN match_num (a batch of
    # 2025 matches TML hasn't fully backfilled), and frozen_join itself has 43 duplicate
    # (tml_tourney_id, tml_match_num) pairs. Both were confirmed via direct inspection.
    # `rolling` already carries tml_tourney_id/tml_match_num from attach_player_ids_and_
    # chronology's earlier merge — only bring in tml_winner_id/tml_loser_id here (bringing
    # tourney_id/match_num in again would create ambiguous suffixed duplicate columns).
    # The final merge onto day5 uses the full 4-column (tourney_id, match_num, winner_id,
    # loser_id) key, confirmed unique in day5 by direct inspection.
    rolling_with_ids = rolling.merge(
        frozen_join[["mcp_match_id", "tml_winner_id", "tml_loser_id"]].drop_duplicates(subset="mcp_match_id"),
        left_on="match_id", right_on="mcp_match_id", how="inner",
    )

    winner_mask = rolling_with_ids["player_id"] == rolling_with_ids["tml_winner_id"]
    winner_side = rolling_with_ids.loc[winner_mask, ["tml_tourney_id", "tml_match_num", "tml_winner_id", "tml_loser_id"] + feature_cols]
    winner_side.columns = ["tourney_id", "match_num", "winner_id", "loser_id"] + [f"winner_{c}" for c in feature_cols]

    loser_mask = rolling_with_ids["player_id"] == rolling_with_ids["tml_loser_id"]
    loser_side = rolling_with_ids.loc[loser_mask, ["tml_tourney_id", "tml_match_num", "tml_winner_id", "tml_loser_id"] + feature_cols]
    loser_side.columns = ["tourney_id", "match_num", "winner_id", "loser_id"] + [f"loser_{c}" for c in feature_cols]

    # Same pivot pattern for the surface-conditioned features, using an independent merge
    # against frozen_join (surface_rolling has its own row set from attach_surface, distinct
    # from rolling_with_ids above — kept separate rather than reused, since surface_rolling's
    # sort order and any surface-attachment NaN-drops could differ from rolling's).
    surface_rolling_with_ids = surface_rolling.merge(
        frozen_join[["mcp_match_id", "tml_winner_id", "tml_loser_id"]].drop_duplicates(subset="mcp_match_id"),
        left_on="match_id", right_on="mcp_match_id", how="inner",
    )
    surf_winner_mask = surface_rolling_with_ids["player_id"] == surface_rolling_with_ids["tml_winner_id"]
    surface_winner_side = surface_rolling_with_ids.loc[
        surf_winner_mask, ["tml_tourney_id", "tml_match_num", "tml_winner_id", "tml_loser_id"] + surface_feature_cols
    ]
    surface_winner_side.columns = ["tourney_id", "match_num", "winner_id", "loser_id"] + \
        [f"winner_{c}" for c in surface_feature_cols]

    surf_loser_mask = surface_rolling_with_ids["player_id"] == surface_rolling_with_ids["tml_loser_id"]
    surface_loser_side = surface_rolling_with_ids.loc[
        surf_loser_mask, ["tml_tourney_id", "tml_match_num", "tml_winner_id", "tml_loser_id"] + surface_feature_cols
    ]
    surface_loser_side.columns = ["tourney_id", "match_num", "winner_id", "loser_id"] + \
        [f"loser_{c}" for c in surface_feature_cols]

    # Same pivot pattern for the break-point-specific features, using an independent merge
    # against frozen_join (bp_rolling has its own row set from load_and_prepare_key_points_stats,
    # distinct from rolling_with_ids/surface_rolling_with_ids above — kept separate rather
    # than reused, since bp_rolling's sort order and any BP/BPO-filtering NaN-drops could
    # differ from the other two).
    bp_rolling_with_ids = bp_rolling.merge(
        frozen_join[["mcp_match_id", "tml_winner_id", "tml_loser_id"]].drop_duplicates(subset="mcp_match_id"),
        left_on="match_id", right_on="mcp_match_id", how="inner",
    )
    bp_winner_mask = bp_rolling_with_ids["player_id"] == bp_rolling_with_ids["tml_winner_id"]
    bp_winner_side = bp_rolling_with_ids.loc[
        bp_winner_mask, ["tml_tourney_id", "tml_match_num", "tml_winner_id", "tml_loser_id"] + bp_feature_cols
    ]
    bp_winner_side.columns = ["tourney_id", "match_num", "winner_id", "loser_id"] + \
        [f"winner_{c}" for c in bp_feature_cols]

    bp_loser_mask = bp_rolling_with_ids["player_id"] == bp_rolling_with_ids["tml_loser_id"]
    bp_loser_side = bp_rolling_with_ids.loc[
        bp_loser_mask, ["tml_tourney_id", "tml_match_num", "tml_winner_id", "tml_loser_id"] + bp_feature_cols
    ]
    bp_loser_side.columns = ["tourney_id", "match_num", "winner_id", "loser_id"] + \
        [f"loser_{c}" for c in bp_feature_cols]

    # NEW (Elo-trend features, 2026-07): operates directly on day5's TML-native
    # structure (winner_id/loser_id/tourney_date already present) — no
    # attach_player_ids_and_chronology needed, unlike the MCP-derived features above.
    # See compute_elo_trend_features's own docstring for the full derivation, including
    # a DIFFERENT leakage-safety argument than the rate-based features (elo_pre_match is
    # already a prior-only value by construction, verified directly by perturbation
    # testing that a later match's Elo cannot affect an earlier match's elo_change).
    elo_trend = compute_elo_trend_features(day5)
    elo_feature_cols = [c for c in elo_trend.columns if c.startswith("elo_change_last")]
    logger.info("Computed %d Elo-trend feature columns", len(elo_feature_cols))

    # BUG FIX (caught by the row-count safety net on the very first real run: 198,062 ->
    # 198,894 rows): an EARLIER version merged using only ["tourney_id", "match_num",
    # "winner_id"/"loser_id"], on the unverified assumption that (tourney_id, match_num)
    # alone uniquely identifies a match — it does not, on the real data. Now selects
    # BOTH winner_id and loser_id from elo_trend (available since compute_elo_trend_features
    # carries both through regardless of which side player_id represents) and uses the
    # SAME full, already-established 4-column merge_key as every other merge below.
    elo_winner_side = elo_trend.loc[
        elo_trend["is_winner_row"], ["tourney_id", "match_num", "winner_id", "loser_id"] + elo_feature_cols
    ].rename(columns={c: f"winner_{c}" for c in elo_feature_cols})

    elo_loser_side = elo_trend.loc[
        ~elo_trend["is_winner_row"], ["tourney_id", "match_num", "winner_id", "loser_id"] + elo_feature_cols
    ].rename(columns={c: f"loser_{c}" for c in elo_feature_cols})

    merge_key = ["tourney_id", "match_num", "winner_id", "loser_id"]
    merged = day5.merge(winner_side, on=merge_key, how="left")
    merged = merged.merge(loser_side, on=merge_key, how="left")
    merged = merged.merge(surface_winner_side, on=merge_key, how="left")
    merged = merged.merge(surface_loser_side, on=merge_key, how="left")
    merged = merged.merge(bp_winner_side, on=merge_key, how="left")
    merged = merged.merge(bp_loser_side, on=merge_key, how="left")
    merged = merged.merge(elo_winner_side, on=merge_key, how="left")
    merged = merged.merge(elo_loser_side, on=merge_key, how="left")

    # SAFETY NET: a left-merge must never change row count. Checked once here, after ALL
    # eight merges (surface-blind winner/loser + surface-conditioned winner/loser +
    # break-point-specific winner/loser + Elo-trend winner/loser) — a fan-out in any
    # single one of them would show up in this final count, so one check after all
    # merges is sufficient and equivalent to checking after each individually.
    if len(merged) != len(day5):
        raise AssertionError(
            f"Row count changed during merge: {len(day5):,} -> {len(merged):,}. "
            "This indicates a non-unique merge key (fan-out bug) — do not trust this "
            "output. Investigate before proceeding."
        )

    n_with_features = merged[f"winner_{RATE_COLS[0]}_career"].notna().sum()
    print(f"\nMatches with at least career-to-date serve/return data: {n_with_features:,} "
          f"({n_with_features / len(merged):.1%} of {len(merged):,} total)")

    n_with_surface_features = merged[f"winner_{SURFACE_RATE_COLS[0]}_surface_career"].notna().sum()
    print(f"Matches with at least surface-conditioned career data: {n_with_surface_features:,} "
          f"({n_with_surface_features / len(merged):.1%} of {len(merged):,} total)")

    merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(merged):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()