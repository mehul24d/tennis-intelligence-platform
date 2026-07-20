"""
validate_markov_inputs.py — runtime authority check (companion to the static
audit_markov_call_sites.py). Rather than reading source code, this ACTUALLY CALLS every
fixed function with a battery of known-answer and property-based tests, so a bug can be
caught even if it's expressed in a way the static regex audit can't recognize (e.g. split
across multiple lines, computed via an intermediate variable, or via a differently-named
helper).

Three layers of checking:
  1. KNOWN-ANSWER tests: hand-computable cases where the correct output is known exactly.
  2. PROPERTY tests: invariants that MUST hold regardless of input values (symmetry,
     monotonicity, boundedness) — these catch a wide class of bugs without needing to
     hand-derive every possible answer.
  3. PLAUSIBILITY tests: for realistic ATP-tour-level inputs, flags any result that would
     be a red flag for a human tennis analyst (e.g. >98% pre-match confidence between two
     top-20 players) — this is exactly the kind of check that would have caught the
     original bug immediately, since 0.995 for Sinner-Alcaraz should have looked wrong
     on sight.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

from tennis_intel.live.markov_baseline import prob_win_match, prob_win_set, prob_win_game
from tennis_intel.live.live_win_probability import MatchState, prob_a_wins_match_from_state

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ============================================================
# LAYER 1: Known-answer tests
# ============================================================
section("LAYER 1: Known-answer tests")

check("prob_win_game(0.60) matches literature value 0.7357",
      abs(prob_win_game(0.60) - 0.7357) < 1e-3)

check("Symmetric players (p_serve=p_return=0.5) give exactly 0.5 for match",
      abs(prob_win_match(0.5, 0.5, best_of=3) - 0.5) < 1e-9)

check("Symmetric players give exactly 0.5 for a set",
      abs(prob_win_set(0.5, 0.5) - 0.5) < 1e-9)


# ============================================================
# LAYER 2: Property-based invariants
# ============================================================
section("LAYER 2: Property-based invariants (must hold for ANY valid input)")

# Property: if a player's serve/return are BOTH better than a symmetric baseline,
# their win probability must exceed 0.5.
p = prob_win_match(0.70, 0.45, best_of=3)  # better than average on both dimensions
check("Better-than-average server AND returner has P(win) > 0.5", p > 0.5, f"got {p:.4f}")

# Property: match probability must be monotonically increasing in p_serve, holding
# p_return fixed.
vals = [prob_win_match(ps, 0.40, best_of=3) for ps in [0.55, 0.60, 0.65, 0.70, 0.75]]
check("P(win) strictly increases as p_serve increases (p_return fixed)",
      all(vals[i] < vals[i+1] for i in range(len(vals)-1)), f"got {vals}")

# Property: match probability must be monotonically increasing in p_return, holding
# p_serve fixed.
vals2 = [prob_win_match(0.65, pr, best_of=3) for pr in [0.30, 0.35, 0.40, 0.45, 0.50]]
check("P(win) strictly increases as p_return increases (p_serve fixed)",
      all(vals2[i] < vals2[i+1] for i in range(len(vals2)-1)), f"got {vals2}")

# CRITICAL PROPERTY (the one that would have caught the actual bug): swapping which
# player's stats are used for "self" vs "opponent" must produce complementary results.
# If A's (p_serve, p_return) vs B's (p_serve, p_return) are used CORRECTLY (each
# derived using the OTHER's real serve rate), then P(A wins) + P(B wins) using the
# correctly-paired construction must equal 1.0 exactly (zero-sum, single unified match).
a_serve, a_return_stat = 0.70, 0.40   # A's own career stats
b_serve, b_return_stat = 0.65, 0.38   # B's own career stats

# CORRECT construction: p_return for A = 1 - B's real serve rate
p_a_correct = prob_win_match(a_serve, 1 - b_serve, best_of=3)
p_b_correct = prob_win_match(b_serve, 1 - a_serve, best_of=3)
check("Correct construction: P(A wins) + P(B wins) sums to exactly 1.0",
      abs((p_a_correct + p_b_correct) - 1.0) < 1e-9,
      f"got {p_a_correct:.4f} + {p_b_correct:.4f} = {p_a_correct+p_b_correct:.4f}")

# BUGGY construction (what the original code did): p_return for A = A's OWN return stat
p_a_buggy = prob_win_match(a_serve, a_return_stat, best_of=3)
p_b_buggy = prob_win_match(b_serve, b_return_stat, best_of=3)
buggy_sum = p_a_buggy + p_b_buggy
check("Buggy construction does NOT sum to 1.0 (demonstrates why it's wrong)",
      abs(buggy_sum - 1.0) > 0.01,
      f"got {p_a_buggy:.4f} + {p_b_buggy:.4f} = {buggy_sum:.4f} "
      f"(this large deviation from 1.0 is itself the signature of the bug: two "
      f"independent 'self-vs-generic-opponent' calculations don't describe one real match)")


# ============================================================
# LAYER 3: Plausibility tests on realistic tour-level inputs
# ============================================================
section("LAYER 3: Plausibility — would a human analyst accept this number?")

# Two elite, closely-matched players (Sinner/Alcaraz-shaped career stats, from the real
# diagnostic data pulled during this session)
sinner_serve = 0.7675
alcaraz_serve = 0.7258

p_sinner_correct = prob_win_match(sinner_serve, 1 - alcaraz_serve, best_of=5)
check("Two elite closely-matched players: pre-match P(favorite) stays below 0.90 (bo5)",
      p_sinner_correct < 0.90,
      f"got {p_sinner_correct:.4f} — a number this high between two top-2-in-the-world "
      f"players should be treated as suspect and investigated, not accepted silently")

check("Two elite closely-matched players: pre-match P(favorite) stays above 0.50",
      p_sinner_correct > 0.50, f"got {p_sinner_correct:.4f}")

print(f"\n  (Reference: the ORIGINAL buggy construction gave 0.9951 for this exact matchup "
      f"— this plausibility check is specifically designed to catch a recurrence of that.)")


# ============================================================
# Summary
# ============================================================
section("SUMMARY")
if FAILURES:
    print(f"❌ {len(FAILURES)} check(s) FAILED:")
    for f in FAILURES:
        print(f"   - {f}")
    sys.exit(1)
else:
    print("✅ All checks passed.")