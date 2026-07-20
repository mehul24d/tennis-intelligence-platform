"""
join_tml_mcp.py — Stages 1, 2, 3, 4, 6 of the TML-Database <-> Match Charting Project join.

Pipeline:
    Stage 1  Load both datasets
    Stage 2  Normalize player names, tournament names, dates, surfaces, rounds
    Stage 3  Deterministic join on (tournament, round, player-pair) with date-band sanity check
    Stage 4  Fallback matching for near-misses, with every fallback decision logged
    Stage 6  Write joined_matches.parquet

Stage 5 (validation report) lives in join_validation.py and consumes this module's output —
keep join logic and validation reporting separate so the report can be re-run without
re-executing the (more expensive) join itself.

Design notes:
    - The join key deliberately does NOT include date as an exact-match field. TML's
      `tourney_date` is the TOURNAMENT START date, not the date of any individual match —
      for a two-week Slam, a final's TML date and its actual MCP `Date` can differ by 10+
      days. Using date as a hard join key would silently fail most non-first-round matches.
      Instead, date is used as a POST-JOIN sanity check: the MCP match date should fall
      within [tourney_date, tourney_date + MAX_TOURNAMENT_SPAN_DAYS].
    - Player pairs are compared as an unordered set, not (winner, loser), because MCP's
      "Player 1" / "Player 2" columns are not guaranteed to be in winner-first order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pandas as pd

from tennis_intel.data.canonical_player_names import (
    apply_alias,
    normalize_player_name,
    normalize_round,
    normalize_tournament_name,
)

logger = logging.getLogger(__name__)

MAX_TOURNAMENT_SPAN_DAYS = 21  # generous upper bound; longest ATP events run ~2 weeks


@dataclass
class JoinLogEntry:
    """One record of a fallback-matching decision, kept for auditability (Stage 4 requirement:
    every fallback must be logged, not silently applied)."""

    mcp_match_id: str
    strategy: str  # e.g. "exact", "swapped_names", "alias_applied", "date_out_of_band_accepted"
    tml_row_index: int | None
    detail: str = ""


@dataclass
class JoinResult:
    joined: pd.DataFrame
    log: list[JoinLogEntry] = field(default_factory=list)
    unmatched_mcp: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Stage 1 — Load
# ---------------------------------------------------------------------------

def load_tml_matches(tml_dir: Path) -> pd.DataFrame:
    """Load TML-Database match-year files only (excludes ATP_Database.csv player master and
    ongoing_tourneys.csv, which have different schemas — see notebooks/01_eda.py Section 2c)."""
    exclude = {"ATP_Database.csv", "ongoing_tourneys.csv"}
    files = sorted(f for f in tml_dir.glob("*.csv") if f.name not in exclude)
    if not files:
        raise FileNotFoundError(f"No TML match-year files found in {tml_dir}")

    def _read(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")

    df = pd.concat([_read(f) for f in files], ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    logger.info("Loaded %d TML matches from %d files", len(df), len(files))
    return df


def load_mcp_matches(mcp_dir: Path, gender: str = "m") -> pd.DataFrame:
    """Load MCP match metadata file for the given gender ('m' or 'w')."""
    path = mcp_dir / f"charting-{gender}-matches.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    logger.info("Loaded %d MCP (%s) matches from %s", len(df), gender, path.name)
    return df


# ---------------------------------------------------------------------------
# Stage 2 — Normalize
# ---------------------------------------------------------------------------

def normalize_tml(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["winner_name_norm"] = df["winner_name"].apply(normalize_player_name).apply(apply_alias)
    df["loser_name_norm"] = df["loser_name"].apply(normalize_player_name).apply(apply_alias)
    df["tourney_name_norm"] = df["tourney_name"].apply(normalize_tournament_name)
    df["round_norm"] = df["round"].apply(normalize_round)
    df["player_pair"] = df.apply(
        lambda r: frozenset({r["winner_name_norm"], r["loser_name_norm"]}), axis=1
    )
    return df


def normalize_mcp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["player1_norm"] = df["Player 1"].apply(normalize_player_name).apply(apply_alias)
    df["player2_norm"] = df["Player 2"].apply(normalize_player_name).apply(apply_alias)
    df["tournament_norm"] = df["Tournament"].apply(normalize_tournament_name)
    df["round_norm"] = df["Round"].apply(normalize_round)
    df["player_pair"] = df.apply(
        lambda r: frozenset({r["player1_norm"], r["player2_norm"]}), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# Stage 3 — Deterministic join
# ---------------------------------------------------------------------------

def _nearest_by_date(candidates: list[int], tml: pd.DataFrame, mcp_date) -> tuple[int | None, str]:
    """
    Given multiple TML candidate row indices sharing a join key, pick the one whose
    tourney_date is closest to the MCP match's actual played date, provided:
      - the closest candidate's date difference is unambiguous (no exact tie with the
        second-closest), and
      - the difference falls within a sane bound (MAX_TOURNAMENT_SPAN_DAYS)

    Returns (chosen_index_or_None, reason_string). This replaces a boolean in-band check,
    which fails whenever a recurring event/rivalry means MULTIPLE years' TML rows are all
    independently "in band" for their own tourney_date — boolean pass/fail can't break that
    tie, but nearest-date can.
    """
    if pd.isna(mcp_date):
        return None, "mcp_date_missing"

    scored = []
    for idx in candidates:
        t_date = tml.loc[idx, "tourney_date"]
        if pd.isna(t_date):
            continue
        diff = abs((mcp_date - t_date).days)
        if diff <= MAX_TOURNAMENT_SPAN_DAYS:
            scored.append((diff, idx))

    if not scored:
        return None, "no_candidate_in_range"

    scored.sort(key=lambda x: x[0])
    if len(scored) == 1:
        return scored[0][1], "single_in_range"

    best_diff, best_idx = scored[0]
    second_diff, _ = scored[1]
    if best_diff == second_diff:
        return None, "tied_distance"  # genuinely ambiguous, do not guess

    return best_idx, "closest_date"


def deterministic_join(tml: pd.DataFrame, mcp: pd.DataFrame) -> JoinResult:
    """
    Exact join on (tournament_norm, round_norm, player_pair). When multiple TML rows share
    a key (common for recurring events/rivalries meeting in the same round across different
    years), disambiguate by nearest match date rather than a boolean date-band check — see
    _nearest_by_date for why a boolean check isn't sufficient here.

    Consumed TML indices are tracked and removed from the candidate pool as they're matched,
    so two different MCP rows (e.g. the same rivalry's meetings in different years) cannot
    both be silently matched to the same single TML row.
    """
    log: list[JoinLogEntry] = []
    matched_rows = []
    unmatched_idx = []
    consumed_tml: set[int] = set()

    tml_index: dict[tuple, list[int]] = {}
    for idx, row in tml.iterrows():
        key = (row["tourney_name_norm"], row["round_norm"], row["player_pair"])
        tml_index.setdefault(key, []).append(idx)

    # Process MCP rows in chronological order so that, when a genuine ambiguity is broken by
    # consumption order rather than date proximity, earlier real matches claim their TML row
    # first — this is a deliberate, documented tie-break, not an arbitrary one.
    mcp_sorted = mcp.sort_values("Date", na_position="last")

    for mcp_idx, mrow in mcp_sorted.iterrows():
        key = (mrow["tournament_norm"], mrow["round_norm"], mrow["player_pair"])
        candidates = [c for c in tml_index.get(key, []) if c not in consumed_tml]

        if len(candidates) == 1:
            tml_idx = candidates[0]
            matched_rows.append((mcp_idx, tml_idx))
            consumed_tml.add(tml_idx)
            log.append(JoinLogEntry(mrow["match_id"], "exact", tml_idx))
        elif len(candidates) > 1:
            chosen, reason = _nearest_by_date(candidates, tml, mrow["Date"])
            if chosen is not None:
                matched_rows.append((mcp_idx, chosen))
                consumed_tml.add(chosen)
                log.append(JoinLogEntry(
                    mrow["match_id"], "exact_disambiguated_by_date", chosen,
                    detail=f"{len(candidates)} candidates, resolved via {reason}",
                ))
            else:
                unmatched_idx.append(mcp_idx)
                log.append(JoinLogEntry(
                    mrow["match_id"], "ambiguous_unresolved", None,
                    detail=f"{len(candidates)} candidates, {reason}",
                ))
        else:
            unmatched_idx.append(mcp_idx)

    joined = _build_joined_frame(tml, mcp, matched_rows)
    unmatched_mcp = mcp.loc[unmatched_idx]
    return JoinResult(joined=joined, log=log, unmatched_mcp=unmatched_mcp)


# ---------------------------------------------------------------------------
# Stage 4 — Fallback matching for the Stage 3 leftovers
# ---------------------------------------------------------------------------

def fallback_join(
    tml: pd.DataFrame,
    unmatched_mcp: pd.DataFrame,
    existing_log: list[JoinLogEntry],
    consumed_tml: set[int] | None = None,
) -> JoinResult:
    """
    Attempts additional matching strategies on rows Stage 3 couldn't resolve:
      1. Round-relaxed match: same tournament + player_pair, ignore round (handles cases where
         round naming conventions genuinely diverge, e.g. round-robin stage labeling)
      2. Nearest-date disambiguation when round-relaxing produces multiple candidates (same
         logic as Stage 3 — see _nearest_by_date)

    `consumed_tml` carries forward every TML index already claimed in Stage 3, so a fallback
    match here cannot double-assign a TML row that a Stage 3 exact match already used.

    Every match made here is tagged with its strategy in the log for later audit. Rows that
    still don't match after all fallbacks are returned as genuinely unmatched — do not force
    a match; an incorrect join silently corrupts every downstream feature.
    """
    log = list(existing_log)
    matched_rows = []
    still_unmatched_idx = []
    consumed = set(consumed_tml) if consumed_tml else set()

    relaxed_index: dict[tuple, list[int]] = {}
    for idx, row in tml.iterrows():
        key = (row["tourney_name_norm"], row["player_pair"])
        relaxed_index.setdefault(key, []).append(idx)

    unmatched_sorted = unmatched_mcp.sort_values("Date", na_position="last")

    for mcp_idx, mrow in unmatched_sorted.iterrows():
        key = (mrow["tournament_norm"], mrow["player_pair"])
        candidates = [c for c in relaxed_index.get(key, []) if c not in consumed]

        if len(candidates) == 1:
            tml_idx = candidates[0]
            matched_rows.append((mcp_idx, tml_idx))
            consumed.add(tml_idx)
            log.append(JoinLogEntry(
                mrow["match_id"], "round_relaxed", tml_idx,
                detail="matched on tournament+players, round label diverged",
            ))
            continue

        if len(candidates) > 1:
            chosen, reason = _nearest_by_date(candidates, tml, mrow["Date"])
            if chosen is not None:
                matched_rows.append((mcp_idx, chosen))
                consumed.add(chosen)
                log.append(JoinLogEntry(
                    mrow["match_id"], "round_relaxed_disambiguated_by_date", chosen,
                    detail=f"{len(candidates)} candidates, resolved via {reason}",
                ))
                continue

        # No safe match found — leave unmatched rather than guessing
        still_unmatched_idx.append(mcp_idx)
        log.append(JoinLogEntry(mrow["match_id"], "unmatched", None))

    joined = _build_joined_frame(tml, unmatched_mcp, matched_rows)
    still_unmatched = unmatched_mcp.loc[still_unmatched_idx]
    return JoinResult(joined=joined, log=log, unmatched_mcp=still_unmatched)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_joined_frame(tml: pd.DataFrame, mcp: pd.DataFrame, matched_rows: list[tuple[int, int]]) -> pd.DataFrame:
    if not matched_rows:
        return pd.DataFrame()
    mcp_idxs, tml_idxs = zip(*matched_rows)
    # Drop 'player_pair' (a frozenset) before prefixing — it's an internal join key only,
    # not serializable to parquet, and not needed downstream since the individual name
    # columns (winner_name_norm, player1_norm, etc.) are already preserved.
    mcp_clean = mcp.drop(columns=["player_pair"], errors="ignore")
    tml_clean = tml.drop(columns=["player_pair"], errors="ignore")
    left = mcp_clean.loc[list(mcp_idxs)].reset_index(drop=True).add_prefix("mcp_")
    right = tml_clean.loc[list(tml_idxs)].reset_index(drop=True).add_prefix("tml_")
    return pd.concat([left, right], axis=1)


def run_full_join(tml_dir: Path, mcp_dir: Path, gender: str = "m") -> JoinResult:
    """Orchestrates Stages 1-4 end to end for one gender. Stage 6 (write) is caller's
    responsibility — see pipelines/build_joined_dataset.py."""
    tml_raw = load_tml_matches(tml_dir)
    mcp_raw = load_mcp_matches(mcp_dir, gender=gender)

    tml_norm = normalize_tml(tml_raw)
    mcp_norm = normalize_mcp(mcp_raw)

    stage3 = deterministic_join(tml_norm, mcp_norm)

    # Recover which TML rows Stage 3 already claimed, so Stage 4 can't double-assign them.
    consumed_from_stage3 = {
        entry.tml_row_index for entry in stage3.log
        if entry.tml_row_index is not None
    }

    stage4 = fallback_join(tml_norm, stage3.unmatched_mcp, stage3.log, consumed_tml=consumed_from_stage3)

    combined_joined = pd.concat([stage3.joined, stage4.joined], ignore_index=True)
    return JoinResult(joined=combined_joined, log=stage4.log, unmatched_mcp=stage4.unmatched_mcp)


def write_joined_dataset(result: JoinResult, output_path: Path) -> None:
    """Stage 6 — persist the joined dataset. Everything downstream reads only this file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.joined.to_parquet(output_path, index=False)
    logger.info("Wrote %d joined matches to %s", len(result.joined), output_path)