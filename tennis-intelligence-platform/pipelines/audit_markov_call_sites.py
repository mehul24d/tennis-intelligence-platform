"""
audit_markov_call_sites.py — static audit: finds EVERY call site of every Markov
probability function across the entire codebase (src/ and pipelines/, including files not
otherwise touched in this session) and checks whether p_return is constructed correctly
(1 - opponent's real serve rate) or incorrectly (the player's own generic return stat).

Built after discovering the same p_return construction bug independently in FOUR separate
files across three different "days" of the project (Day 9's build_day9_point_model.py,
Day 10's evaluate_live_engines.py, Day 11's evaluate_live_engines_v2.py, and
generate_publication_trajectory.py) — a manual grep found them one at a time; this script
exists so the check is repeatable and automatic rather than relying on catching it by hand
again next time a new call site is added.

This is a heuristic static check (regex over source, not an AST-level guarantee), so it
reports SUSPECTS for human review, not a certified proof of correctness — the actual
authority is the runtime cross-check in `validate_markov_inputs.py` (companion script).
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MARKOV_FUNCS = [
    "prob_win_match", "prob_win_set", "prob_win_game", "prob_win_tiebreak",
    "prob_a_wins_match_from_state",
]

# Patterns that indicate the WRONG construction: a bare "return_pts_won_pct" or
# "_return" column being assigned directly to a variable that looks like p_return,
# WITHOUT a "1 -" or "1.0 -" prefix and WITHOUT referencing the opponent's serve column.
SUSPECT_PATTERN = re.compile(
    r"(p_return|pr|p_a_return)\s*=\s*(?!1\.?0?\s*-).*return_pts_won_pct", re.IGNORECASE
)
CORRECT_PATTERN = re.compile(
    r"(p_return|pr|p_a_return|opponent_serve)\s*=.*first_serve_win_pct", re.IGNORECASE
)


def find_python_files() -> list[Path]:
    files = []
    for d in ["src", "pipelines"]:
        files.extend((PROJECT_ROOT / d).rglob("*.py"))
    return [f for f in files if "__pycache__" not in str(f)]


def audit_file(path: Path) -> list[dict]:
    findings = []
    text = path.read_text()
    if not any(fn in text for fn in MARKOV_FUNCS):
        return findings

    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        if SUSPECT_PATTERN.search(line):
            findings.append({
                "file": str(path.relative_to(PROJECT_ROOT)), "line": i,
                "content": line.strip(), "verdict": "SUSPECT — uses own return_pts_won_pct "
                                                     "directly as p_return without deriving "
                                                     "from opponent's serve rate",
            })
        elif CORRECT_PATTERN.search(line) and "1" in line and "-" in line:
            findings.append({
                "file": str(path.relative_to(PROJECT_ROOT)), "line": i,
                "content": line.strip(),
                "verdict": "LOOKS CORRECT — derives from (1 - opponent's serve rate)",
            })
    return findings


def main() -> None:
    all_findings = []
    for f in find_python_files():
        all_findings.extend(audit_file(f))

    if not all_findings:
        print("No Markov-related p_return construction found in any file.")
        return

    suspects = [f for f in all_findings if "SUSPECT" in f["verdict"]]
    correct = [f for f in all_findings if "CORRECT" in f["verdict"]]

    print(f"{'='*70}\nMARKOV p_return CONSTRUCTION AUDIT\n{'='*70}\n")
    print(f"Files scanned: {len(find_python_files())}")
    print(f"Total findings: {len(all_findings)}  (suspect: {len(suspects)}, "
          f"looks-correct: {len(correct)})\n")

    if suspects:
        print("--- SUSPECT LINES (require manual fix) ---")
        for s in suspects:
            print(f"  {s['file']}:{s['line']}")
            print(f"    {s['content']}")
        print()

    if correct:
        print("--- LOOKS-CORRECT LINES (derived from opponent's serve rate) ---")
        for c in correct:
            print(f"  {c['file']}:{c['line']}")
            print(f"    {c['content']}")

    print(f"\n{'='*70}")
    if suspects:
        print(f"RESULT: {len(suspects)} suspect construction(s) found — fix before trusting "
              f"any Markov-derived number from these locations.")
    else:
        print("RESULT: no suspect constructions found by this heuristic check. Still run "
              "validate_markov_inputs.py for a runtime cross-check, since this is a "
              "regex-based heuristic, not a certified guarantee.")


if __name__ == "__main__":
    main()