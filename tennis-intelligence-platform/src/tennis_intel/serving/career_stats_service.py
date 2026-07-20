"""
career_stats_service.py — Rankings and Player Profile, rebuilt to use the FULL day6
table (~198,062 matches) instead of the frozen_join-merged corpus (~5,988 matches —
only those with Match Charting Project point-by-point data too).

WHY THIS REBUILD EXISTS: the original player_profile_service.py/rankings_service.py
were built on MatchListContext, which is frozen_join-derived and silently limited to
~3% of TML's real historical match count. Correct for Match Explorer (needs
point-by-point replay to be useful) and anything needing per-point data — but Player
Profile and Rankings need NEITHER; they only need day6's own match-level
Elo/tournament/score columns, present for the FULL TML corpus regardless of MCP
coverage.

COLUMN NAMES DIFFER FROM match_list_service.py's OWN — day6 has NO mcp_/tml_
prefixes (confirmed via pipelines/diagnose_frozen_join_schema.py's real output) and
NO single match_id string, only the composite (tourney_id, match_num, winner_id,
loser_id) key — a synthetic display identifier is constructed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CareerStatsContext:
    """The full, un-joined TML match table (day6) — every match in this project's
    corpus, independent of Match Charting Project coverage."""
    day6: pd.DataFrame


def load_career_stats_context(day6: pd.DataFrame) -> CareerStatsContext:
    """Pass the SAME day6 dataframe already loaded for ReplayContext — no reason to
    reload the same parquet file a second time."""
    df = day6.copy()
    df["synthetic_match_id"] = (
        df["tourney_id"].astype(str) + "-" + df["match_num"].astype(str) + "-"
        + df["winner_id"].astype(str) + "-" + df["loser_id"].astype(str)
    )
    return CareerStatsContext(day6=df)


def search_players_by_name(ctx: CareerStatsContext, name_query: str, limit: int = 20) -> list[dict]:
    """Finds distinct (player_id, player_name) pairs matching a substring, case-
    insensitive, across BOTH winner_name and loser_name."""
    df = ctx.day6
    q = name_query.lower()

    winner_matches = df[df["winner_name"].str.lower().str.contains(q, na=False)][
        ["winner_id", "winner_name"]
    ].rename(columns={"winner_id": "player_id", "winner_name": "player_name"})
    loser_matches = df[df["loser_name"].str.lower().str.contains(q, na=False)][
        ["loser_id", "loser_name"]
    ].rename(columns={"loser_id": "player_id", "loser_name": "player_name"})

    combined = pd.concat([winner_matches, loser_matches], ignore_index=True)
    combined = combined.drop_duplicates(subset="player_id").head(limit)
    return combined.to_dict("records")


def get_player_profile(ctx: CareerStatsContext, player_id: str) -> dict:
    """Full Player Profile payload across the FULL TML corpus. Raises ValueError if
    not found."""
    df = ctx.day6
    as_winner = df[df["winner_id"] == player_id].copy()
    as_loser = df[df["loser_id"] == player_id].copy()

    if len(as_winner) == 0 and len(as_loser) == 0:
        raise ValueError(f"player_id '{player_id}' not found as either winner or loser in day6.")

    player_name = (
        as_winner["winner_name"].iloc[0] if len(as_winner) else as_loser["loser_name"].iloc[0]
    )

    as_winner_view = pd.DataFrame({
        "match_id": as_winner["synthetic_match_id"], "tourney_date": as_winner["tourney_date"],
        "tournament": as_winner["tourney_name"], "surface": as_winner["surface"],
        "round": as_winner["round"], "tourney_level": as_winner["tourney_level"],
        "opponent_name": as_winner["loser_name"], "opponent_id": as_winner["loser_id"],
        "won": True, "score": as_winner["score"],
        "elo_pre": as_winner["elo_pre_match_winner"], "elo_surface_pre": as_winner["elo_surface_pre_match_winner"],
    })
    as_loser_view = pd.DataFrame({
        "match_id": as_loser["synthetic_match_id"], "tourney_date": as_loser["tourney_date"],
        "tournament": as_loser["tourney_name"], "surface": as_loser["surface"],
        "round": as_loser["round"], "tourney_level": as_loser["tourney_level"],
        "opponent_name": as_loser["winner_name"], "opponent_id": as_loser["winner_id"],
        "won": False, "score": as_loser["score"],
        "elo_pre": as_loser["elo_pre_match_loser"], "elo_surface_pre": as_loser["elo_surface_pre_match_loser"],
    })
    career = pd.concat([as_winner_view, as_loser_view], ignore_index=True)
    career = career.sort_values("tourney_date").reset_index(drop=True)

    current_elo = None
    peak_elo = None
    if career["elo_pre"].notna().any():
        current_elo = float(career["elo_pre"].dropna().iloc[-1])
        peak_elo = float(career["elo_pre"].max())

    elo_timeline = [
        {
            "match_id": row["match_id"],
            "date": row["tourney_date"].isoformat() if pd.notna(row["tourney_date"]) else None,
            "elo": round(float(row["elo_pre"]), 1) if pd.notna(row["elo_pre"]) else None,
            "surface_elo": round(float(row["elo_surface_pre"]), 1) if pd.notna(row["elo_surface_pre"]) else None,
        }
        for _, row in career.iterrows()
    ]

    n_matches = len(career)
    n_wins = int(career["won"].sum())
    n_losses = n_matches - n_wins

    surface_stats = {}
    for surface in career["surface"].dropna().unique():
        sub = career[career["surface"] == surface]
        surface_stats[surface] = {
            "matches": len(sub), "wins": int(sub["won"].sum()),
            "win_pct": round(float(sub["won"].mean()), 4) if len(sub) else None,
        }

    recent_form = career.tail(10)
    recent_form_list = [
        {
            "match_id": row["match_id"],
            "date": row["tourney_date"].isoformat() if pd.notna(row["tourney_date"]) else None,
            "opponent": row["opponent_name"], "won": bool(row["won"]),
            "surface": row["surface"], "tournament": row["tournament"],
        }
        for _, row in recent_form.iterrows()
    ]

    grand_slam = career[career["tourney_level"] == "G"]
    grand_slam_stats = {
        "matches": len(grand_slam), "wins": int(grand_slam["won"].sum()),
        "win_pct": round(float(grand_slam["won"].mean()), 4) if len(grand_slam) else None,
    }

    return {
        "player_id": player_id, "player_name": player_name,
        "current_elo": round(current_elo, 1) if current_elo is not None else None,
        "peak_elo": round(peak_elo, 1) if peak_elo is not None else None,
        "career_matches": n_matches, "career_wins": n_wins, "career_losses": n_losses,
        "career_win_pct": round(n_wins / n_matches, 4) if n_matches else None,
        "surface_stats": surface_stats,
        "grand_slam_stats": grand_slam_stats,
        "recent_form": recent_form_list,
        "elo_timeline": elo_timeline,
    }


def get_head_to_head(ctx: CareerStatsContext, player_id_a: str, player_id_b: str) -> dict:
    """Head-to-head record between two players, across the FULL TML corpus."""
    df = ctx.day6
    a_beat_b = df[(df["winner_id"] == player_id_a) & (df["loser_id"] == player_id_b)]
    b_beat_a = df[(df["winner_id"] == player_id_b) & (df["loser_id"] == player_id_a)]

    matches = []
    for _, row in a_beat_b.iterrows():
        matches.append({
            "match_id": row["synthetic_match_id"],
            "date": row["tourney_date"].isoformat() if pd.notna(row["tourney_date"]) else None,
            "tournament": row["tourney_name"], "surface": row["surface"],
            "winner_id": player_id_a, "score": row["score"],
        })
    for _, row in b_beat_a.iterrows():
        matches.append({
            "match_id": row["synthetic_match_id"],
            "date": row["tourney_date"].isoformat() if pd.notna(row["tourney_date"]) else None,
            "tournament": row["tourney_name"], "surface": row["surface"],
            "winner_id": player_id_b, "score": row["score"],
        })
    matches.sort(key=lambda m: m["date"] or "")

    return {
        "player_id_a": player_id_a, "player_id_b": player_id_b,
        "a_wins": len(a_beat_b), "b_wins": len(b_beat_a),
        "matches": matches,
    }


def _build_player_elo_long_form(ctx: CareerStatsContext) -> pd.DataFrame:
    """Long-form (player_id, player_name, date, elo, surface, surface_elo) table
    across the FULL day6 corpus."""
    df = ctx.day6
    as_winner = pd.DataFrame({
        "player_id": df["winner_id"], "player_name": df["winner_name"],
        "date": df["tourney_date"], "elo": df["elo_pre_match_winner"],
        "surface": df["surface"], "surface_elo": df["elo_surface_pre_match_winner"],
    })
    as_loser = pd.DataFrame({
        "player_id": df["loser_id"], "player_name": df["loser_name"],
        "date": df["tourney_date"], "elo": df["elo_pre_match_loser"],
        "surface": df["surface"], "surface_elo": df["elo_surface_pre_match_loser"],
    })
    long_form = pd.concat([as_winner, as_loser], ignore_index=True)
    return long_form.sort_values(["player_id", "date"], kind="mergesort").reset_index(drop=True)


def get_current_elo_rankings(ctx: CareerStatsContext, limit: int = 100) -> list[dict]:
    long_form = _build_player_elo_long_form(ctx)
    long_form = long_form.dropna(subset=["elo"])
    current = long_form.groupby("player_id").last().reset_index()
    current = current.sort_values("elo", ascending=False).head(limit)
    return [
        {"rank": i + 1, "player_id": row["player_id"], "player_name": row["player_name"],
         "elo": round(float(row["elo"]), 1)}
        for i, (_, row) in enumerate(current.iterrows())
    ]


def get_peak_elo_rankings(ctx: CareerStatsContext, limit: int = 100) -> list[dict]:
    long_form = _build_player_elo_long_form(ctx)
    long_form = long_form.dropna(subset=["elo"])
    peak_idx = long_form.groupby("player_id")["elo"].idxmax()
    peak = long_form.loc[peak_idx].sort_values("elo", ascending=False).head(limit)
    return [
        {"rank": i + 1, "player_id": row["player_id"], "player_name": row["player_name"],
         "peak_elo": round(float(row["elo"]), 1),
         "date_achieved": row["date"].isoformat() if pd.notna(row["date"]) else None}
        for i, (_, row) in enumerate(peak.iterrows())
    ]


def get_surface_elo_rankings(ctx: CareerStatsContext, surface: str, limit: int = 100) -> list[dict]:
    long_form = _build_player_elo_long_form(ctx)
    long_form = long_form[long_form["surface"] == surface].dropna(subset=["surface_elo"])
    current = long_form.groupby("player_id").last().reset_index()
    current = current.sort_values("surface_elo", ascending=False).head(limit)
    return [
        {"rank": i + 1, "player_id": row["player_id"], "player_name": row["player_name"],
         "surface_elo": round(float(row["surface_elo"]), 1)}
        for i, (_, row) in enumerate(current.iterrows())
    ]


def get_peak_surface_elo_rankings(ctx: CareerStatsContext, surface: str, limit: int = 100) -> list[dict]:
    """Peak (career-high) Elo rankings for ONE surface — the maximum surface_elo
    ever reached by each player ON THAT SURFACE specifically, at any point in their
    career, not just their most recent surface_elo (that's get_surface_elo_rankings).
    Same idxmax pattern as get_peak_elo_rankings, applied to the surface-filtered
    subset rather than the whole corpus."""
    long_form = _build_player_elo_long_form(ctx)
    long_form = long_form[long_form["surface"] == surface].dropna(subset=["surface_elo"])
    if len(long_form) == 0:
        return []
    peak_idx = long_form.groupby("player_id")["surface_elo"].idxmax()
    peak = long_form.loc[peak_idx].sort_values("surface_elo", ascending=False).head(limit)
    return [
        {"rank": i + 1, "player_id": row["player_id"], "player_name": row["player_name"],
         "peak_surface_elo": round(float(row["surface_elo"]), 1),
         "date_achieved": row["date"].isoformat() if pd.notna(row["date"]) else None}
        for i, (_, row) in enumerate(peak.iterrows())
    ]


def get_biggest_upsets(ctx: CareerStatsContext, limit: int = 100) -> list[dict]:
    df = ctx.day6.copy()
    df = df.dropna(subset=["elo_pre_match_winner", "elo_pre_match_loser"])
    df["elo_gap"] = df["elo_pre_match_loser"] - df["elo_pre_match_winner"]
    upsets = df[df["elo_gap"] > 0].sort_values("elo_gap", ascending=False).head(limit)
    return [
        {
            "rank": i + 1, "match_id": row["synthetic_match_id"],
            "date": row["tourney_date"].isoformat() if pd.notna(row["tourney_date"]) else None,
            "tournament": row["tourney_name"], "round": row["round"],
            "winner_name": row["winner_name"], "winner_elo": round(float(row["elo_pre_match_winner"]), 1),
            "loser_name": row["loser_name"], "loser_elo": round(float(row["elo_pre_match_loser"]), 1),
            "elo_gap": round(float(row["elo_gap"]), 1),
        }
        for i, (_, row) in enumerate(upsets.iterrows())
    ]


def get_full_match_list(
    ctx: CareerStatsContext, frozen_join: pd.DataFrame,
    player: str | None = None, surface: str | None = None,
    year: int | None = None, tourney_level: str | None = None,
    limit: int = 100, offset: int = 0,
) -> dict:
    """
    Match Explorer, searching the FULL TML corpus (~198,062 matches) rather than
    only the ~6,000 matches with Match Charting Project point-by-point coverage —
    the same "use the full corpus" fix already applied to Player Profile and
    Rankings (see this module's own docstring for the full reasoning), now applied
    to Match Explorer per explicit request: every real TML match should be
    browsable, even if most of them can only show a brief score summary rather than
    a full point-by-point replay.

    has_replay_data: True for matches that DO have MCP point-by-point coverage
    (safe to link into GET /api/matches/{id}/replay) — computed by cross-referencing
    against frozen_join's own (tourney_id, match_num, winner_id, loser_id) composite
    key, the SAME key that links a TML match to its MCP replay data everywhere else
    in this project (build_point_dataset.py's own join). match_id in the response is
    the REAL mcp_match_id when has_replay_data is True (so the frontend's existing
    "Open analysis" link keeps working unchanged for those rows), and the synthetic
    TML-only id otherwise (which cannot be replayed, only displayed).
    """
    df = ctx.day6

    charted_keys = frozen_join[
        ["tml_tourney_id", "tml_match_num", "tml_winner_id", "tml_loser_id", "mcp_match_id"]
    ].rename(columns={
        "tml_tourney_id": "tourney_id", "tml_match_num": "match_num",
        "tml_winner_id": "winner_id", "tml_loser_id": "loser_id",
    })
    merged = df.merge(
        charted_keys, on=["tourney_id", "match_num", "winner_id", "loser_id"], how="left",
    )
    if len(merged) != len(df):
        raise AssertionError(
            f"Row count changed during charted-match cross-reference: "
            f"{len(df):,} -> {len(merged):,}. This indicates a non-unique merge key "
            f"(fan-out) — do not trust this output."
        )

    if player:
        p = player.lower()
        mask = (
            merged["winner_name"].str.lower().str.contains(p, na=False)
            | merged["loser_name"].str.lower().str.contains(p, na=False)
        )
        merged = merged[mask]
    if surface:
        merged = merged[merged["surface"].str.lower() == surface.lower()]
    if year is not None:
        merged = merged[merged["tourney_date"].dt.year == year]
    if tourney_level:
        merged = merged[merged["tourney_level"] == tourney_level]

    merged = merged.sort_values("tourney_date", ascending=False)
    total = len(merged)
    page = merged.iloc[offset:offset + limit]

    matches = []
    for _, row in page.iterrows():
        has_replay = pd.notna(row["mcp_match_id"])
        matches.append({
            "match_id": row["mcp_match_id"] if has_replay else row["synthetic_match_id"],
            "has_replay_data": bool(has_replay),
            "tournament": row["tourney_name"],
            "year": int(row["tourney_date"].year) if pd.notna(row["tourney_date"]) else None,
            "surface": row["surface"],
            "round": row["round"],
            "winner": row["winner_name"],
            "loser": row["loser_name"],
            "final_score": row.get("score"),
            "tournament_level": row.get("tourney_level"),
            "best_of": int(row["best_of"]) if pd.notna(row.get("best_of")) else None,
            "winner_elo": round(float(row["elo_pre_match_winner"]), 1) if pd.notna(row["elo_pre_match_winner"]) else None,
            "loser_elo": round(float(row["elo_pre_match_loser"]), 1) if pd.notna(row["elo_pre_match_loser"]) else None,
        })

    return {"total": total, "limit": limit, "offset": offset, "matches": matches}