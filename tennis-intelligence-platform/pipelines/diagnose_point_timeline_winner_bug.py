"""
diagnose_point_timeline_winner_bug.py — dumps the RAW Svr/PtWinner values alongside
the computed server/winner names for a specific match and point range, to pin down
exactly why "Server" and "Won by" showed the same player across an entire game where
the score (after the score_before fix) shows that player clearly losing every point.

Usage:
    python pipelines/diagnose_point_timeline_winner_bug.py <match_id> [start_point] [end_point]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Add BOTH the project root (for "from pipelines.replay_match import ...") AND the
# pipelines/ directory itself (for "from replay_match import ..." with no package
# prefix) — the exact convention in replay_service.py's own import statement isn't
# guaranteed to match what's in this sandbox's copy, so support both rather than
# guess wrong and produce a second broken diagnostic.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_intel.serving.replay_service import load_replay_context
from tennis_intel.serving.point_timeline_service import get_point_timeline


def main() -> None:
    args = sys.argv[1:]
    match_id = None
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--match-id" and i + 1 < len(args):
            match_id = args[i + 1]
            i += 2
        else:
            positional.append(args[i])
            i += 1

    if match_id is None and positional:
        match_id = positional.pop(0)
    start = int(positional[0]) if len(positional) > 0 else 1
    end = int(positional[1]) if len(positional) > 1 else 20

    if not match_id:
        print("Usage: python pipelines/diagnose_point_timeline_winner_bug.py <match_id> [start] [end]")
        print("   or: python pipelines/diagnose_point_timeline_winner_bug.py --match-id <match_id> [start] [end]")
        return

    print("Loading replay context (this takes a moment)...")
    ctx = load_replay_context()

    match_df = ctx.points[ctx.points["match_id"] == match_id].sort_values("Pt").reset_index(drop=True)
    if len(match_df) == 0:
        print(f"No match found with id: {match_id}")
        return

    print(f"\nRAW data for points {start}-{end}:")
    print(f"{'Pt':>4} {'Svr':>4} {'PtWinner':>9} {'p1_pts':>7} {'p2_pts':>7} {'is_tb':>6} "
          f"{'Set1':>5} {'Set2':>5} {'Gm1':>4} {'Gm2':>4}")
    subset = match_df[(match_df["Pt"] >= start) & (match_df["Pt"] <= end)]
    for _, row in subset.iterrows():
        p1_pts = row.get("p1_points")
        p2_pts = row.get("p2_points")
        is_tb = row.get("is_tiebreak_game")
        print(f"{row['Pt']:>4} {row['Svr']:>4} {row['PtWinner']:>9} {p1_pts!s:>7} {p2_pts!s:>7} "
              f"{str(is_tb):>6} {row['Set1']:>5} {row['Set2']:>5} {row['Gm1']:>4} {row['Gm2']:>4}")

    print(f"\nComputed timeline (via get_point_timeline) for the same range:")
    result = get_point_timeline(ctx, match_id)
    for entry in result["points"]:
        if start <= entry["point_index"] <= end:
            print(f"Pt {entry['point_index']:>3}: server={entry['server']!r}, "
                  f"receiver={entry['receiver']!r}, winner={entry['winner']!r}, "
                  f"score_before={entry['score_before']!r}, is_break_point={entry['is_break_point']}")


if __name__ == "__main__":
    main()