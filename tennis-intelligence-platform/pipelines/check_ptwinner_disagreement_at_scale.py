"""
check_ptwinner_disagreement_at_scale.py — runs the same PtWinner-vs-points-
progression disagreement check across a SAMPLE of matches (not just one), to
determine whether the ~51% disagreement rate found for one specific match is an
isolated anomaly in that one charted file, or a genuine, widespread pattern worth
investigating as a project-wide data-quality issue.

============================================================================
THIS SCRIPT'S "RELATIVE hypothesis... confirmed at 0.00% disagreement" CONCLUSION IS
A DEAD END, NOT A SETTLED FACT (2026-07) — see docs/ptwinner_convention_correction.md
for the full investigation. The blind spot: `compute_disagreement_rate` below
explicitly skips every game-boundary row (`if row["Gm1"] != next_row["Gm1"] or
row["Gm2"] != next_row["Gm2"]: continue`) and checks PtWinner only against
p1_points/p2_points (parsed fixed-player). That can only ever test INTERNAL
self-consistency between two DERIVED quantities on interior points — and there are
TWO internally self-consistent pairings possible here (server-relative PtWinner +
fixed-player Pts, which is what "RELATIVE" tests; and literal PtWinner + server-first
Pts, never tested here), which coincide whenever Svr==1 and are exact mirror
opposites whenever Svr==2. A same-row self-consistency check cannot distinguish them
— it can only rule out MIXED pairings (which is all "FIXED" vs "RELATIVE" here
actually demonstrates). This script never checks against Gm1/Gm2, the one
independently-recorded signal that CAN distinguish the two. Checked there instead:
literal PtWinner matches Gm1/Gm2 at 99.91% corpus-wide (symmetric across Svr==1/2);
server-relative (this script's "RELATIVE", declared the winner below) matches only
~51% (chance) at game boundaries. PtWinner is LITERAL, fixed-player-relative — not
server-relative. Do not re-derive conclusions from this script's own comparison
without also checking against Gm1/Gm2 at boundaries; see
check_game_counter_consistency_at_scale.py for that check.
============================================================================

Usage:
    python pipelines/check_ptwinner_disagreement_at_scale.py [--n-matches 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_intel.serving.replay_service import load_replay_context

RANDOM_STATE = 42


def compute_disagreement_rate(match_df) -> tuple[int, int, int, int]:
    """
    Returns (agree_fixed, disagree_fixed, agree_relative, disagree_relative) --
    TWO parallel comparisons against the SAME p1_points/p2_points progression
    signal, to directly test which interpretation of PtWinner is actually correct:

    FIXED hypothesis (the original, assumed-standard interpretation): PtWinner==1
    means player 1 won, PtWinner==2 means player 2 won, regardless of who served.

    RELATIVE hypothesis (found via manual tracing of a real match that showed ~50%
    disagreement under the FIXED interpretation): PtWinner==1 means the SERVER won,
    PtWinner==2 means the RECEIVER won -- i.e. PtWinner is server/receiver-relative,
    not a fixed player identity, and must be combined with Svr to recover which
    actual player won.
    """
    records = match_df.to_dict("records")
    agree_fixed, disagree_fixed = 0, 0
    agree_relative, disagree_relative = 0, 0

    for i in range(len(records) - 1):
        row, next_row = records[i], records[i + 1]
        if row.get("is_tiebreak_game") or next_row.get("is_tiebreak_game"):
            continue
        if row["Gm1"] != next_row["Gm1"] or row["Gm2"] != next_row["Gm2"]:
            continue

        p1_before, p2_before = row.get("p1_points"), row.get("p2_points")
        p1_after, p2_after = next_row.get("p1_points"), next_row.get("p2_points")
        if any(v is None for v in [p1_before, p2_before, p1_after, p2_after]):
            continue

        p1_increased = p1_after > p1_before
        p2_increased = p2_after > p2_before
        if p1_increased == p2_increased:
            continue

        implied_winner_is_p1 = p1_increased

        # FIXED hypothesis
        pt_winner_is_p1_fixed = row["PtWinner"] == 1
        if implied_winner_is_p1 == pt_winner_is_p1_fixed:
            agree_fixed += 1
        else:
            disagree_fixed += 1

        # RELATIVE hypothesis: PtWinner==1 means SERVER won, PtWinner==2 means
        # RECEIVER won -- combine with Svr to get the actual fixed-player winner.
        server_is_p1 = row["Svr"] == 1
        server_won = row["PtWinner"] == 1
        pt_winner_is_p1_relative = server_is_p1 if server_won else (not server_is_p1)
        if implied_winner_is_p1 == pt_winner_is_p1_relative:
            agree_relative += 1
        else:
            disagree_relative += 1

    return agree_fixed, disagree_fixed, agree_relative, disagree_relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-matches", type=int, default=200)
    args = parser.parse_args()

    print("Loading replay context (this takes a moment)...")
    ctx = load_replay_context()

    all_match_ids = sorted(ctx.match_ids)
    n_use = min(args.n_matches, len(all_match_ids))
    selected = np.random.RandomState(RANDOM_STATE).choice(all_match_ids, size=n_use, replace=False)

    print(f"Checking {n_use} matches...")
    match_rates = []
    for i, match_id in enumerate(selected):
        if i % 50 == 0 and i > 0:
            print(f"  {i} / {n_use} matches checked")
        match_df = ctx.points[ctx.points["match_id"] == match_id].sort_values("Pt").reset_index(drop=True)
        agree_f, disagree_f, agree_r, disagree_r = compute_disagreement_rate(match_df)
        total_f = agree_f + disagree_f
        total_r = agree_r + disagree_r
        if total_f >= 10:
            match_rates.append({
                "match_id": match_id,
                "disagree_pct_fixed": 100 * disagree_f / total_f,
                "disagree_pct_relative": 100 * disagree_r / total_r,
                "n": total_f,
            })

    rates_fixed = np.array([m["disagree_pct_fixed"] for m in match_rates])
    rates_relative = np.array([m["disagree_pct_relative"] for m in match_rates])

    print(f"\n=== FIXED hypothesis (PtWinner==1 means player 1 won) — "
          f"disagreement rate across {len(match_rates)} matches ===")
    print(f"Mean:   {rates_fixed.mean():.2f}%")
    print(f"Median: {np.median(rates_fixed):.2f}%")
    print(f"p90:    {np.percentile(rates_fixed, 90):.2f}%")
    print(f"Max:    {rates_fixed.max():.2f}%")

    print(f"\n=== RELATIVE hypothesis (PtWinner==1 means SERVER won) — "
          f"disagreement rate across {len(match_rates)} matches ===")
    print(f"Mean:   {rates_relative.mean():.2f}%")
    print(f"Median: {np.median(rates_relative):.2f}%")
    print(f"p90:    {np.percentile(rates_relative, 90):.2f}%")
    print(f"Max:    {rates_relative.max():.2f}%")

    print("\nInterpretation:")
    print("- Whichever hypothesis shows disagreement near the documented ~2.3%")
    print("  charting-error baseline (not ~50%) is the CORRECT interpretation of")
    print("  PtWinner in this dataset. If RELATIVE wins decisively, PtWinner is")
    print("  server/receiver-coded, not fixed-player-coded -- a major, project-wide")
    print("  finding requiring is_break_point/is_set_point/is_match_point/")
    print("  server_is_winner and the point-level classifier's own training target")
    print("  to all be re-examined, since they may depend on the FIXED")
    print("  interpretation being correct.")


if __name__ == "__main__":
    main()