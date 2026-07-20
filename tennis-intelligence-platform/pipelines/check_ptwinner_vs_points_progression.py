"""
check_ptwinner_vs_points_progression.py — for one match, checks how often PtWinner
agrees with the IMPLIED winner from the points progression between consecutive rows
of the SAME game.

============================================================================
PRIOR OUTPUT/CONCLUSIONS FROM THIS SCRIPT SHOULD BE DISREGARDED, NOT ASSUMED STILL
VALID (2026-07) — and for a SUBTLER reason than a simple wrong-answer bug: this
script previously paired LITERAL PtWinner (PtWinner==1 means player 1 won, the now-
confirmed-correct convention) against p1_points/p2_points progression parsed as
FIXED-PLAYER (first printed Pts number = player 1, unconditionally). That specific
pairing is self-INCONSISTENT — literal PtWinner only pairs correctly with SERVER-
FIRST Pts (first printed number = whoever is currently serving, reordered via Svr).
Comparing literal PtWinner against fixed-player Pts (the old code below) would show
~50% "disagreement" even under the confirmed-correct convention, for any match with
mixed serving — not because PtWinner is wrong, but because the ground truth it was
checked against was parsed with the wrong (mismatched) Pts convention. See
docs/ptwinner_convention_correction.md for the full investigation, including the
corpus-scale test that resolved which of the two internally-self-consistent pairings
(server-relative PtWinner + fixed-player Pts, vs. literal PtWinner + server-first
Pts) actually matches the independently-recorded Gm1/Gm2 columns (literal + server-
first won, 99.91% vs. ~51%/chance). This script now parses Pts as server-first
(reordered via Svr) below, so it correctly pairs with literal PtWinner.
============================================================================

Usage:
    python pipelines/check_ptwinner_vs_points_progression.py --match-id <match_id>
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
    i = 0
    while i < len(args):
        if args[i] == "--match-id" and i + 1 < len(args):
            match_id = args[i + 1]
            i += 2
        else:
            i += 1

    if not match_id:
        print("Usage: python pipelines/check_ptwinner_vs_points_progression.py --match-id <match_id>")
        return

    print("Loading replay context (this takes a moment)...")
    ctx = load_replay_context()

    match_df = ctx.points[ctx.points["match_id"] == match_id].sort_values("Pt").reset_index(drop=True)
    if len(match_df) == 0:
        print(f"No match found with id: {match_id}")
        return

    records = match_df.to_dict("records")
    agree, disagree, skipped = 0, 0, 0
    disagreements = []

    for i in range(len(records) - 1):
        row, next_row = records[i], records[i + 1]
        if row.get("is_tiebreak_game") or next_row.get("is_tiebreak_game"):
            skipped += 1
            continue
        if row["Gm1"] != next_row["Gm1"] or row["Gm2"] != next_row["Gm2"]:
            skipped += 1
            continue

        # p1_points/p2_points (from ctx.points) are parsed FIXED-PLAYER (first printed
        # Pts number = p1, unconditionally). To correctly pair with literal PtWinner,
        # re-derive SERVER-FIRST p1/p2 by translating via Svr: when Svr==1, server-first
        # and fixed-player agree (p1 = p1_points); when Svr==2, they're swapped (p1, the
        # RECEIVER when Svr==2, holds the value fixed-player parsing put in p2_points).
        # See module docstring for why this matters.
        raw_p1_before, raw_p2_before = row.get("p1_points"), row.get("p2_points")
        raw_p1_after, raw_p2_after = next_row.get("p1_points"), next_row.get("p2_points")
        if any(v is None for v in [raw_p1_before, raw_p2_before, raw_p1_after, raw_p2_after]):
            skipped += 1
            continue

        server_is_p1 = row["Svr"] == 1
        p1_before = raw_p1_before if server_is_p1 else raw_p2_before
        p2_before = raw_p2_before if server_is_p1 else raw_p1_before
        p1_after = raw_p1_after if server_is_p1 else raw_p2_after
        p2_after = raw_p2_after if server_is_p1 else raw_p1_after

        p1_increased = p1_after > p1_before
        p2_increased = p2_after > p2_before
        if p1_increased == p2_increased:
            skipped += 1
            continue

        implied_winner_is_p1 = p1_increased
        pt_winner_is_p1 = row["PtWinner"] == 1

        if implied_winner_is_p1 == pt_winner_is_p1:
            agree += 1
        else:
            disagree += 1
            disagreements.append(int(row["Pt"]))

    total_checked = agree + disagree
    print(f"\nChecked {total_checked} point transitions ({skipped} skipped: tiebreak or game-boundary or ambiguous)")
    if total_checked:
        print(f"Agree:    {agree} ({100*agree/total_checked:.1f}%)")
        print(f"Disagree: {disagree} ({100*disagree/total_checked:.1f}%)")
    else:
        print("No comparable transitions found.")
    if disagreements:
        print(f"\nDisagreeing point numbers: {disagreements}")

    # NEW: for each disagreeing point, print Svr and Gm1/Gm2 directly -- this
    # localizes whether Svr itself is constant (unchanging) across each disagreeing
    # run (consistent with an entire game being mislabeled with the wrong server
    # identity) versus scattered/inconsistent (which would point elsewhere).
    if disagreements:
        print("\nSvr and game context for each disagreeing point (to check whether entire")
        print("games, not scattered points, are affected):")
        pt_lookup = {int(r["Pt"]): r for r in records}
        for p in disagreements:
            r = pt_lookup[p]
            print(f"  Pt {p:>4}: Svr={r['Svr']}, Gm1={r['Gm1']}, Gm2={r['Gm2']}, "
                  f"PtWinner={r['PtWinner']}, p1_pts={r.get('p1_points')}, p2_pts={r.get('p2_points')}")

    # NEW: check Svr's distribution across the WHOLE match, not just disagreeing
    # points -- if Svr is stuck at a single value the ENTIRE match (never alternates
    # to the other player at all), that's a fundamentally different, simpler bug
    # than "server mislabeled for specific games" -- and would mean Svr can't be
    # trusted as a signal here at all, regardless of parity reasoning.
    svr_values = [r["Svr"] for r in records]
    svr_counts = {v: svr_values.count(v) for v in set(svr_values)}
    print(f"\nSvr value distribution across the ENTIRE match ({len(records)} points): {svr_counts}")
    if len(svr_counts) == 1:
        print("Svr NEVER changes across the whole match -- it is frozen at a single")
        print("value throughout, which cannot possibly be correct (server always")
        print("alternates game-to-game in tennis) -- Svr itself is broken for this")
        print("match, not just mismatched with PtWinner in specific games.")
    else:
        print("Svr DOES take multiple values across the match -- check separately")
        print("whether it correctly alternates roughly every game, or is still")
        print("suspicious in some other way (e.g. stuck for long stretches).")

    print("\nInterpretation:")
    print("- If disagreement is isolated to a small cluster of points, this looks like")
    print("  a genuine charting error for that specific stretch, consistent with this")
    print("  project's own documented ~2.3% baseline charting-error rate (see")
    print("  point_level_features.py's own docstring) -- an upstream MCP data-quality")
    print("  issue, not a bug in this project's code.")
    print("- If disagreement is widespread across the WHOLE match, that would instead")
    print("  suggest a systemic issue worth investigating further.")


if __name__ == "__main__":
    main()