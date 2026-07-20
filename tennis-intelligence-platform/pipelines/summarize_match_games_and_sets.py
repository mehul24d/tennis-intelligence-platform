"""
summarize_match_games_and_sets.py — computes the AUTHORITATIVE, error-free game-by-
game score directly from the real Gm1/Gm2/Set1/Set2 columns for one match, rather
than relying on manual counting of a pasted point-by-point table.

Usage:
    python pipelines/summarize_match_games_and_sets.py --match-id <match_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_intel.serving.replay_service import load_replay_context, compute_five_engine_trajectory
from tennis_intel.serving.point_timeline_service import get_point_timeline


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
        print("Usage: python pipelines/summarize_match_games_and_sets.py --match-id <match_id>")
        return

    print("Loading replay context (this takes a moment)...")
    ctx = load_replay_context()

    match_df = ctx.points[ctx.points["match_id"] == match_id].sort_values("Pt").reset_index(drop=True)
    if len(match_df) == 0:
        print(f"No match found with id: {match_id}")
        return

    records = match_df.to_dict("records")
    timeline = get_point_timeline(ctx, match_id)
    timeline_by_pt = {p["point_index"]: p for p in timeline["points"]}

    print(f"\nTotal points in match: {len(records)}\n")
    print(f"{'Pt':>4} {'Set1':>5} {'Set2':>5} {'Gm1':>4} {'Gm2':>4} {'Winner (from API)':<25} {'Event':<20}")

    prev_gm1, prev_gm2, prev_set1, prev_set2 = None, None, None, None
    game_winners = []

    for row in records:
        pt = int(row["Pt"])
        gm1, gm2 = int(row["Gm1"]), int(row["Gm2"])
        set1, set2 = int(row["Set1"]), int(row["Set2"])
        api_winner = timeline_by_pt.get(pt, {}).get("winner", "?")

        event = ""
        if prev_gm1 is not None:
            if set1 != prev_set1 or set2 != prev_set2:
                event = "SET BOUNDARY"
            elif gm1 > prev_gm1:
                event = "P1 WON PRIOR GAME"
                game_winners.append("P1")
            elif gm2 > prev_gm2:
                event = "P2 WON PRIOR GAME"
                game_winners.append("P2")

        print(f"{pt:>4} {set1:>5} {set2:>5} {gm1:>4} {gm2:>4} {api_winner:<25} {event:<20}")
        prev_gm1, prev_gm2, prev_set1, prev_set2 = gm1, gm2, set1, set2

    print(f"\n=== Game winners, in order (derived purely from Gm1/Gm2 incrementing) ===")
    print(game_winners)
    p1_games, p2_games = game_winners.count("P1"), game_winners.count("P2")
    print(f"\nP1 (player1) won {p1_games} games, P2 (player2) won {p2_games} games (by this count, excluding tiebreak games)")

    # NEW: direct, decisive cross-check -- for EACH game, does the majority of the
    # API's own per-point "winner" field for that game's points agree with which
    # player actually won the game (per Gm1/Gm2 incrementing, an independent, raw
    # data source that does NOT depend on the point_timeline_service's own
    # PtWinner/Svr-based "winner" computation at all)? A game where these
    # DISAGREE is a direct, internal contradiction worth flagging specifically.
    print("\n=== Cross-check: does each game's point-level 'winner' majority match")
    print("    who actually won that game (per Gm1/Gm2)? ===")

    computed = compute_five_engine_trajectory(ctx, match_id)
    p1_name, p2_name = computed["p1_name"], computed["p2_name"]
    print(f"(player1 = {p1_name!r}, player2 = {p2_name!r})\n")

    current_game_points = []
    current_gm1, current_gm2 = None, None
    game_idx = 0
    mismatches = []

    def _flush_game(points, idx):
        if not points or idx >= len(game_winners):
            return
        p1_count = sum(1 for p in points if p == "P1")
        p2_count = len(points) - p1_count
        majority_winner = "P1" if p1_count > p2_count else "P2"
        actual_winner = game_winners[idx]
        status = "OK" if majority_winner == actual_winner else "MISMATCH"
        print(f"  Game {idx+1}: actual winner={actual_winner}, point-majority winner="
              f"{majority_winner} ({p1_count}-{p2_count} points) [{status}]")
        if status == "MISMATCH":
            mismatches.append(idx + 1)

    for row in records:
        pt = int(row["Pt"])
        gm1, gm2 = int(row["Gm1"]), int(row["Gm2"])
        api_winner = timeline_by_pt.get(pt, {}).get("winner", "?")

        if current_gm1 is not None and (gm1 != current_gm1 or gm2 != current_gm2):
            _flush_game(current_game_points, game_idx)
            game_idx += 1
            current_game_points = []
        current_gm1, current_gm2 = gm1, gm2
        current_game_points.append("P1" if api_winner == p1_name else "P2")
    _flush_game(current_game_points, game_idx)

    print(f"\nMismatches found: {len(mismatches)} -- game numbers: {mismatches}")

    # DECISIVE CHECK: does the API's own "winner" field at row i match whichever
    # player's OWN p1_points/p2_points count increases between row i and row i+1
    # (the same score-progression method that found the original PtWinner bug,
    # now applied directly to the FIXED "winner" field itself, to check for a
    # possible row-alignment issue between the winner computation and the score
    # progression)?
    print("\n=== Decisive check: does 'winner' at each row match the score-progression")
    print("    implied winner (whichever player's own point count increases next)? ===")
    p1_name, p2_name = computed["p1_name"], computed["p2_name"]
    agree, disagree, disagreeing_pts = 0, 0, []
    for idx in range(len(records) - 1):
        row, next_row = records[idx], records[idx + 1]
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
        implied_winner = p1_name if p1_increased else p2_name
        api_winner = timeline_by_pt.get(int(row["Pt"]), {}).get("winner")
        if implied_winner == api_winner:
            agree += 1
        else:
            disagree += 1
            disagreeing_pts.append(int(row["Pt"]))

    total = agree + disagree
    print(f"Checked {total} transitions: {agree} agree, {disagree} disagree")
    if total:
        print(f"Disagreement rate: {100*disagree/total:.1f}%")
    if disagreeing_pts:
        print(f"Disagreeing points: {disagreeing_pts}")


if __name__ == "__main__":
    main()