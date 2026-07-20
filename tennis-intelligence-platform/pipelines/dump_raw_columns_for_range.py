"""
dump_raw_columns_for_range.py — dumps EVERY relevant raw column (Svr, PtWinner,
p1_points, p2_points, Gm1, Gm2, Set1, Set2, Pts) for a specific point range, with
zero derived logic in between.

Usage:
    python pipelines/dump_raw_columns_for_range.py --match-id <match_id> <start> <end>
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_intel.serving.replay_service import load_replay_context


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

    if not match_id:
        print("Usage: python pipelines/dump_raw_columns_for_range.py --match-id <id> <start> <end>")
        return
    start = int(positional[0]) if len(positional) > 0 else 1
    end = int(positional[1]) if len(positional) > 1 else 20

    print("Loading replay context (this takes a moment)...")
    ctx = load_replay_context()

    match_df = ctx.points[ctx.points["match_id"] == match_id].sort_values("Pt").reset_index(drop=True)
    if len(match_df) == 0:
        print(f"No match found with id: {match_id}")
        return

    subset = match_df[(match_df["Pt"] >= start) & (match_df["Pt"] <= end)]
    cols = ["Pt", "Svr", "PtWinner", "p1_points", "p2_points", "Gm1", "Gm2", "Set1", "Set2", "Pts"]
    available_cols = [c for c in cols if c in subset.columns]
    print(f"\n{subset[available_cols].to_string(index=False)}")


if __name__ == "__main__":
    main()