"""
validate_tiebreak_notation.py — tests whether MCP's `Pts` column in TIEBREAKS is
server-first (server_count, receiver_count) or fixed-player (p1_count, p2_count) notation,
by cross-checking against `PtWinner` (unambiguous regardless of Pts convention) across
EVERY charted tiebreak in the dataset — not just one match — before changing the parser.

Motivation: a single-match replay (2025 Roland Garros final) showed physically impossible
score jumps (6-0 -> 0-7) under the fixed-player assumption used since Day 7. Manual
verification against that one match's PtWinner values showed server-first notation is
perfectly consistent for all 12 checked transitions. This script checks whether that holds
at scale, since one dramatic match is not sufficient evidence to change a parser used
across the entire project (same "verify against real data" discipline as every prior stage).

For each tiebreak point (Gm1==6 and Gm2==6, i.e. after the first point of the tiebreak),
computes whether PtWinner matches the player whose count increased under each candidate
interpretation, and reports the consistency rate for each.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW_MCP = Path(__file__).resolve().parents[1] / "data" / "raw" / "tennis_MatchChartingProject"
POINT_FILES = [
    RAW_MCP / "charting-m-points-to-2009.csv",
    RAW_MCP / "charting-m-points-2010s.csv",
    RAW_MCP / "charting-m-points-2020s.csv",
]


def parse_tb_pair(pts: str) -> tuple[int, int] | None:
    if not isinstance(pts, str):
        return None
    m = re.match(r"^(\d+)-(\d+)$", pts.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    frames = [pd.read_csv(p, low_memory=False) for p in POINT_FILES]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["match_id", "Pt"], kind="mergesort").reset_index(drop=True)

    df["is_tb"] = (df["Gm1"] == 6) & (df["Gm2"] == 6)
    tb = df[df["is_tb"]].copy()
    print(f"Total tiebreak-flagged points (Gm1==6 and Gm2==6): {len(tb):,}")

    tb["pair"] = tb["Pts"].apply(parse_tb_pair)
    tb = tb[tb["pair"].notna()].copy()
    print(f"With cleanly-parseable numeric 'N-M' scores: {len(tb):,}")

    same_row_fixed_match = 0
    same_row_server_match = 0
    shifted_server_match = 0
    total_same_row_checked = 0
    total_shifted_checked = 0

    for match_id, g in tb.groupby("match_id"):
        g = g.sort_values("Pt").reset_index(drop=True)
        prev_fixed = None
        prev_server_abs = None
        prev_pt_winner = None

        for i, row in g.iterrows():
            svr = row["Svr"]
            a, b = row["pair"]
            pt_winner = row["PtWinner"]

            fixed_p1, fixed_p2 = a, b
            if svr == 1:
                server_p1, server_p2 = a, b
            else:
                server_p1, server_p2 = b, a

            if prev_fixed is not None:
                total_same_row_checked += 1
                d1f, d2f = fixed_p1 - prev_fixed[0], fixed_p2 - prev_fixed[1]
                implied_fixed = 1 if d1f > d2f else (2 if d2f > d1f else 0)
                same_row_fixed_match += (implied_fixed == pt_winner)

                d1s, d2s = server_p1 - prev_server_abs[0], server_p2 - prev_server_abs[1]
                implied_server = 1 if d1s > d2s else (2 if d2s > d1s else 0)
                same_row_server_match += (implied_server == pt_winner)

                # Shifted hypothesis: does the PREVIOUS row's PtWinner match this transition?
                if prev_pt_winner is not None:
                    total_shifted_checked += 1
                    shifted_server_match += (implied_server == prev_pt_winner)

            prev_fixed = (fixed_p1, fixed_p2)
            prev_server_abs = (server_p1, server_p2)
            prev_pt_winner = pt_winner

    print(f"\nTotal same-row transitions checked: {total_same_row_checked:,}")
    print(f"Total shifted transitions checked: {total_shifted_checked:,}\n")
    print(f"{'Interpretation':<45} {'Matches':>10} {'Rate':>8}")
    print(f"{'Fixed player1/player2 (same row)':<45} {same_row_fixed_match:>10,} "
          f"{100*same_row_fixed_match/total_same_row_checked:>7.2f}%")
    print(f"{'Server-first (same row)':<45} {same_row_server_match:>10,} "
          f"{100*same_row_server_match/total_same_row_checked:>7.2f}%")
    print(f"{'Server-first (PtWinner shifted back 1 row)':<45} {shifted_server_match:>10,} "
          f"{100*shifted_server_match/total_shifted_checked:>7.2f}%")

    print("\nWhichever interpretation has a consistency rate close to 100% (allowing for a")
    print("small, expected fraction of genuine charting errors, consistent with the 192")
    print("unparseable points already found in regular-game scores during Day 7) is the")
    print("correct global convention for this column.")


if __name__ == "__main__":
    main()