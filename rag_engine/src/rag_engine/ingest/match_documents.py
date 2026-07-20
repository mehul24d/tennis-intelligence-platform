"""match_documents.py — turns rows of the v1 platform's full match-level corpus into
retrievable RagDocuments.

SOURCE CHOICE: reads `data/processed/matches_with_elo.parquet` directly (the full
~198k-match day6+Elo corpus) rather than going through
`tennis_intel.serving.match_summary_service`, which is scoped to the ~5,988-match
frozen-join (Match Charting Project-covered) subset and requires loading the full,
heavy ReplayContext (classifier + point-level dataset) just to summarize a match.
`career_stats_service.py` made the same call for player profiles, for the same
reason (see its own module docstring: "day6... instead of the frozen_join-merged
corpus... only those with Match Charting Project point-by-point data too"). A
match-summary RAG document doesn't need point-by-point data — it needs exactly what
matches_with_elo.parquet already has: score, serve stats, ranks, and pre-match Elo.

Synthetic id construction mirrors career_stats_service.py's own
`synthetic_match_id` (tourney_id-match_num-winner_id-loser_id) — the same composite
key already used elsewhere in this project to identify a match without a single
canonical match_id column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from rag_engine.ingest.types import RagDocument

DEFAULT_MATCHES_PATH = (
    Path(__file__).resolve().parents[4]
    / "tennis-intelligence-platform" / "data" / "processed" / "matches_with_elo.parquet"
)


def _fmt(value, digits: int = 1) -> str | None:
    """Formats a numeric field for display, or None if missing (NaN) — callers use
    None to skip a clause entirely rather than print 'nan' into the document text."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and digits is not None:
        return f"{value:.{digits}f}"
    return str(value)


def _match_text(row: pd.Series) -> str:
    winner, loser = row["winner_name"], row["loser_name"]
    tournament, surface, round_ = row["tourney_name"], row["surface"], row["round"]
    date = row["tourney_date"]
    date_str = date.date().isoformat() if pd.notna(date) else "unknown date"
    score = row["score"] if pd.notna(row["score"]) else "unknown score"

    sentence = (
        f"{winner} defeated {loser} {score} in the {round_} of the {tournament} "
        f"({surface}) on {date_str}."
    )

    w_rank, l_rank = _fmt(row.get("winner_rank"), 0), _fmt(row.get("loser_rank"), 0)
    if w_rank and l_rank:
        sentence += f" {winner}, ranked #{w_rank}, beat #{l_rank}-ranked {loser}."

    w_1stin, w_1stwon = _fmt(row.get("w_1stIn"), 0), _fmt(row.get("w_1stWon"), 0)
    w_ace, w_df = _fmt(row.get("w_ace"), 0), _fmt(row.get("w_df"), 0)
    if w_1stin and w_1stwon:
        serve_clause = f" {winner} won {w_1stwon}/{w_1stin} first-serve points"
        if w_ace is not None and w_df is not None:
            serve_clause += f" ({w_ace} aces, {w_df} double faults)"
        sentence += serve_clause + "."

    l_ace, l_df = _fmt(row.get("l_ace"), 0), _fmt(row.get("l_df"), 0)
    if l_ace is not None and l_df is not None:
        sentence += f" {loser} hit {l_ace} aces and {l_df} double faults."

    w_bp_saved, w_bp_faced = _fmt(row.get("w_bpSaved"), 0), _fmt(row.get("w_bpFaced"), 0)
    if w_bp_saved is not None and w_bp_faced is not None:
        sentence += f" {winner} saved {w_bp_saved} of {w_bp_faced} break points faced."

    elo_w, elo_l = _fmt(row.get("elo_pre_match_winner"), 1), _fmt(row.get("elo_pre_match_loser"), 1)
    if elo_w and elo_l:
        sentence += f" Pre-match Elo: {winner} {elo_w} vs {loser} {elo_l}."

    return sentence


def _match_metadata(row: pd.Series, match_id: str) -> dict:
    date = row["tourney_date"]
    return {
        "doc_type": "match_summary",
        "match_id": match_id,
        "winner": str(row["winner_name"]),
        "loser": str(row["loser_name"]),
        "surface": str(row["surface"]) if pd.notna(row["surface"]) else "",
        "tourney_level": str(row["tourney_level"]) if pd.notna(row["tourney_level"]) else "",
        "round": str(row["round"]) if pd.notna(row["round"]) else "",
        "tournament": str(row["tourney_name"]) if pd.notna(row["tourney_name"]) else "",
        "date": date.date().isoformat() if pd.notna(date) else "",
        "year": int(date.year) if pd.notna(date) else -1,
    }


def build_match_documents(
    matches_path: Path = DEFAULT_MATCHES_PATH, limit: int | None = None
) -> Iterator[RagDocument]:
    """Yields one RagDocument per match. `limit` caps the number of rows read (from
    the end of the corpus, i.e. most recent matches first after sorting by date) — for
    fast iteration while developing/verifying the index before running the full corpus."""
    df = pd.read_parquet(matches_path)
    df = df.dropna(subset=["winner_name", "loser_name", "tourney_date"])
    df = df.sort_values("tourney_date", ascending=False)
    if limit is not None:
        df = df.head(limit)

    for _, row in df.iterrows():
        match_id = (
            f"{row['tourney_id']}-{row['match_num']}-{row['winner_id']}-{row['loser_id']}"
        )
        yield RagDocument(
            doc_id=f"match:{match_id}",
            text=_match_text(row),
            metadata=_match_metadata(row, match_id),
        )
