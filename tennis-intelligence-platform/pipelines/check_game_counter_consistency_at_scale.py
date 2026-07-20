"""
check_game_counter_consistency_at_scale.py — checks, across a SAMPLE of matches,
whether the raw Gm1/Gm2 game counter agrees with which player's own point count
actually reached the game-winning majority within each game — the exact class of
contradiction found in the Athens match
(20251102-M-Athens-R32-Stefanos_Sakellaridis-Nuno_Borges), where the point score
(0-0 -> 15-0 -> 30-0 -> 40-0, player 1 dominating) directly contradicted Gm2
incrementing (crediting player 2 with winning that same game).

============================================================================
PRIOR OUTPUT/CONCLUSIONS FROM THIS SCRIPT SHOULD BE DISREGARDED, NOT ASSUMED
STILL VALID (2026-07). This script's own ground-truth logic
(check_match_game_consistency, below) used a server-relative interpretation of
PtWinner that was traced and reverted the same day — see
docs/ptwinner_convention_correction.md for the full investigation. Any prior run of
this script (including whatever it originally concluded about the Athens match being
"isolated" or "systemic") was measured against that wrong ground truth and must be
re-derived, not cited. The corrected, literal-PtWinner version of this exact question
was answered directly and definitively during that investigation: literal PtWinner
matches Gm1/Gm2 at 99.91% corpus-wide (167 mismatches / 181,258 game boundaries),
symmetric across Svr==1/2 — consistent with ordinary rare charting error, not a
systemic issue. This script has been updated to use the corrected convention below,
so re-running it now should reproduce that same ~99.9% figure; if it doesn't, that's
itself worth investigating.
============================================================================

This is a DIFFERENT check from check_ptwinner_disagreement_at_scale.py (which tests
internal self-consistency between PtWinner and point-by-point score progression — see
that script's own now-updated docstring for why it cannot, by itself, distinguish
literal from server-relative PtWinner) — this one tests whether the GAME-LEVEL
counter (Gm1/Gm2) is itself consistent with the point-level data, a genuinely
separate, independent data-quality question, and the one that actually settled which
PtWinner convention is correct.

Usage:
    python pipelines/check_game_counter_consistency_at_scale.py [--n-matches 200]
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


def check_match_game_consistency(match_df) -> tuple[int, int]:
    """
    For one match, splits points into games (by Gm1/Gm2 value), then for each game
    checks whether the player who won the MAJORITY of that game's own points
    (using the confirmed-correct LITERAL PtWinner interpretation — PtWinner==1 means
    player 1 won, period; see module docstring) matches who Gm1/Gm2 says actually won
    the game.

    Returns (agree, disagree) game counts. Tiebreak games are excluded (Gm1/Gm2
    freezes during a tiebreak by convention, so this specific check doesn't apply
    there).
    """
    records = match_df.to_dict("records")
    current_game_records = []
    current_gm1, current_gm2 = None, None
    game_boundaries = []

    for row in records:
        if row.get("is_tiebreak_game"):
            continue
        gm1, gm2 = int(row["Gm1"]), int(row["Gm2"])
        if current_gm1 is not None and (gm1 != current_gm1 or gm2 != current_gm2):
            game_boundaries.append((current_game_records, current_gm1, current_gm2))
            current_game_records = []
        current_gm1, current_gm2 = gm1, gm2
        current_game_records.append(row)
    if current_game_records:
        game_boundaries.append((current_game_records, current_gm1, current_gm2))

    agree, disagree = 0, 0
    for i in range(len(game_boundaries) - 1):
        game_records, gm1, gm2 = game_boundaries[i]
        _, next_gm1, next_gm2 = game_boundaries[i + 1]

        if next_gm1 > gm1:
            actual_winner = "P1"
        elif next_gm2 > gm2:
            actual_winner = "P2"
        else:
            continue

        p1_wins_in_game, p2_wins_in_game = 0, 0
        for r in game_records:
            point_winner_is_p1 = r["PtWinner"] == 1
            if point_winner_is_p1:
                p1_wins_in_game += 1
            else:
                p2_wins_in_game += 1

        if p1_wins_in_game == p2_wins_in_game:
            continue
        majority_winner = "P1" if p1_wins_in_game > p2_wins_in_game else "P2"

        if majority_winner == actual_winner:
            agree += 1
        else:
            disagree += 1

    return agree, disagree


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
    matches_with_any_mismatch = []

    for i, match_id in enumerate(selected):
        if i % 50 == 0 and i > 0:
            print(f"  {i} / {n_use} matches checked")
        match_df = ctx.points[ctx.points["match_id"] == match_id].sort_values("Pt").reset_index(drop=True)
        agree, disagree = check_match_game_consistency(match_df)
        total = agree + disagree
        if total >= 5:
            rate = 100 * disagree / total
            match_rates.append({"match_id": match_id, "disagree_pct": rate, "n": total})
            if disagree > 0:
                matches_with_any_mismatch.append({"match_id": match_id, "disagree_pct": rate,
                                                    "n_disagree": disagree, "n_total": total})

    rates = np.array([m["disagree_pct"] for m in match_rates])
    print(f"\n=== Game-counter consistency across {len(match_rates)} matches ===")
    print(f"Mean disagreement rate:   {rates.mean():.2f}%")
    print(f"Median disagreement rate: {np.median(rates):.2f}%")
    print(f"Matches with ZERO mismatched games: {sum(1 for r in rates if r == 0)} / {len(match_rates)} "
          f"({100*sum(1 for r in rates if r == 0)/len(match_rates):.1f}%)")
    print(f"Matches with AT LEAST ONE mismatched game: {len(matches_with_any_mismatch)} / {len(match_rates)} "
          f"({100*len(matches_with_any_mismatch)/len(match_rates):.1f}%)")

    if matches_with_any_mismatch:
        print(f"\nMatches with mismatches (showing up to 15, sorted by disagreement rate):")
        for m in sorted(matches_with_any_mismatch, key=lambda x: -x["disagree_pct"])[:15]:
            print(f"  {m['match_id']}: {m['n_disagree']}/{m['n_total']} games mismatched "
                  f"({m['disagree_pct']:.1f}%)")

    print("\nInterpretation:")
    print("- If the vast majority of matches show ZERO mismatched games, and only a")
    print("  small handful (consistent with the documented ~2.3% baseline charting-")
    print("  error rate) show any mismatch at all, that confirms the Athens match's")
    print("  Gm1/Gm2 inconsistency is an isolated, known-class charting error in one")
    print("  file -- not a systemic issue, and no further code changes are needed.")
    print("- If a large fraction of matches show mismatches, that would indicate a")
    print("  genuine, systemic game-counter issue worth investigating further.")


if __name__ == "__main__":
    main()