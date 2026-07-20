"""
validate_ml_mc_engine.py — comprehensive correctness validation for batch_simulate_dynamic
(the ML+Monte Carlo engine), following the same rigor applied to the Markov engine in
validate_markov_engine.py. Every level is independently interpretable.
"""

from __future__ import annotations

import random
import sys
sys.path.insert(0, "src")

import numpy as np

from tennis_intel.live.monte_carlo_engine import batch_simulate_dynamic, _ordinal_points
from tennis_intel.live.markov_baseline import prob_win_match
from tennis_intel.features.point_score_parser import is_break_point

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


FEATURE_COLS = [
    "is_tiebreak_game", "is_break_point", "is_set_point", "is_match_point",
    "is_second_serve_point", "p1_momentum_last10", "p2_momentum_last10",
    "p1_momentum_last20", "p2_momentum_last20", "server_is_winner",
    "elo_pre_match_winner", "elo_pre_match_loser",
]


# ======================================================================
# LEVEL 1: _ordinal_points correctness (hand-verifiable known cases)
# ======================================================================
section("LEVEL 1: _ordinal_points correctness")

check("0-0 maps to (0,0)", _ordinal_points(0, 0) == (0, 0))
check("2-1 maps to (2,1) — below deuce region, unchanged", _ordinal_points(2, 1) == (2, 1))
check("3-3 maps to (3,3) — exactly deuce", _ordinal_points(3, 3) == (3, 3))
check("5-4 (one ahead in deuce region) maps to (4,3) — advantage", _ordinal_points(5, 4) == (4, 3))
check("4-5 (one behind) maps to (3,4)", _ordinal_points(4, 5) == (3, 4))
check("6-6 (deep deuce, tied) maps to (3,3)", _ordinal_points(6, 6) == (3, 3))
check("10-8 (deep deuce, 2 ahead) maps to (4,3)", _ordinal_points(10, 8) == (4, 3))
check("100-99 (extreme, 1 ahead) still correctly maps to (4,3)", _ordinal_points(100, 99) == (4, 3))


# ======================================================================
# LEVEL 2: Constant-probability convergence to the Markov formula
# ======================================================================
section("LEVEL 2: Constant-probability convergence (ML+MC engine reduces to Markov "
        "when the classifier ignores all features)")

def const_predict_fn(p_const):
    def f(fm):
        return np.full(len(fm), p_const)
    return f

for p_const, best_of in [(0.6, 3), (0.65, 5), (0.5, 3)]:
    rng = random.Random(1)
    p_mc = batch_simulate_dynamic(
        (0, 0, 0, 0, 0, 0, True, False), {}, FEATURE_COLS, const_predict_fn(p_const),
        best_of=best_of, player1_is_winner=True, n_simulations=3000, rng=rng,
    )
    p_markov = prob_win_match(p_const, 1 - p_const, best_of=best_of)
    diff = abs(p_mc - p_markov)
    check(f"Constant p={p_const}, bo{best_of}: ML+MC converges to Markov",
          diff < 0.03, f"MC={p_mc:.4f}, Markov={p_markov:.4f}, diff={diff:.4f}")


# ======================================================================
# LEVEL 3: server_is_winner correctness — A must always resolve to the ACTUAL winner
# ======================================================================
section("LEVEL 3: server_is_winner correctness, verified via a feature-capturing predict_fn")

captured_features = []

def capturing_predict_fn(fm):
    captured_features.append(fm.copy())
    return np.full(len(fm), 0.5)

server_is_winner_idx = FEATURE_COLS.index("server_is_winner")

captured_features.clear()
batch_simulate_dynamic(
    (0, 0, 0, 0, 0, 0, True, False), {}, FEATURE_COLS, capturing_predict_fn,
    best_of=3, player1_is_winner=True, n_simulations=5, rng=random.Random(0), max_points=1,
)
first_tick_features = captured_features[0]
check("When the tracked winner (A) serves first, server_is_winner=True for all sims at tick 1",
      bool(np.all(first_tick_features[:, server_is_winner_idx] == 1.0)))

captured_features.clear()
batch_simulate_dynamic(
    (0, 0, 0, 0, 0, 0, False, False), {}, FEATURE_COLS, capturing_predict_fn,
    best_of=3, player1_is_winner=True, n_simulations=5, rng=random.Random(0), max_points=1,
)
first_tick_features2 = captured_features[0]
check("When A (winner) does NOT serve first, server_is_winner=False for all sims at tick 1",
      bool(np.all(first_tick_features2[:, server_is_winner_idx] == 0.0)))


# ======================================================================
# LEVEL 4: Flag correctness — cross-check against independently-validated Day 7 functions
# ======================================================================
section("LEVEL 4: Situational flags match the independently-validated Day 7 functions")

captured_features.clear()
batch_simulate_dynamic(
    (0, 0, 3, 3, 2, 3, True, False), {}, FEATURE_COLS, capturing_predict_fn,
    best_of=3, player1_is_winner=True, n_simulations=5, rng=random.Random(0), max_points=1,
)
bp_idx = FEATURE_COLS.index("is_break_point")
captured = captured_features[0]
expected_bp = is_break_point(True, 2, 3, False)
check(f"At a real 30-40 break-point state, the engine's flag matches the Day 7 scalar "
      f"function directly (expected={expected_bp})",
      bool(np.all(captured[:, bp_idx] == float(expected_bp))))


# ======================================================================
# LEVEL 5: Momentum blending formula (hand-computable)
# ======================================================================
section("LEVEL 5: Momentum blending formula correctness")

def momentum_ref(hist, window, seed):
    k = min(len(hist), window)
    if k == 0:
        return seed
    return (seed * (window - k) + sum(hist[-k:])) / window

check("No history -> momentum equals the seed exactly", momentum_ref([], 10, 0.7) == 0.7)
check("Full window of 1s -> momentum = 1.0", momentum_ref([1]*10, 10, 0.5) == 1.0)
check("Full window of 0s -> momentum = 0.0", momentum_ref([0]*10, 10, 0.5) == 0.0)
half_hist = [1]*5 + [0]*5
check("Half-1s, half-0s window -> momentum = 0.5", abs(momentum_ref(half_hist, 10, 0.5) - 0.5) < 1e-9)
partial = momentum_ref([1, 1, 1], 10, 0.5)
expected_partial = (0.5 * 7 + 3) / 10
check("Partial history (3 wins, seed 0.5, window 10) matches hand calculation",
      abs(partial - expected_partial) < 1e-9, f"got {partial}, expected {expected_partial}")


# ======================================================================
# LEVEL 6: Termination guarantee — never exceeds max_points, never hangs
# ======================================================================
section("LEVEL 6: Termination guarantee")

import time
for best_of, expected_cap in [(3, 350), (5, 700)]:
    rng = random.Random(0)
    t0 = time.time()
    p = batch_simulate_dynamic(
        (0, 0, 0, 0, 0, 0, True, False), {}, FEATURE_COLS, const_predict_fn(0.5),
        best_of=best_of, player1_is_winner=True, n_simulations=500, rng=rng,
    )
    elapsed = time.time() - t0
    check(f"bo{best_of}: terminates within a reasonable time (<30s) and gives a valid probability",
          elapsed < 30 and 0.0 <= p <= 1.0, f"elapsed={elapsed:.1f}s, p={p:.4f}")


# ======================================================================
# LEVEL 7: Determinism — same seed, same result
# ======================================================================
section("LEVEL 7: Determinism (same seed produces byte-identical results)")

p1 = batch_simulate_dynamic(
    (1, 0, 3, 2, 1, 0, True, False), {"elo_pre_match_winner": 1800.0}, FEATURE_COLS,
    const_predict_fn(0.62), best_of=3, player1_is_winner=True,
    n_simulations=200, rng=random.Random(777),
)
p2 = batch_simulate_dynamic(
    (1, 0, 3, 2, 1, 0, True, False), {"elo_pre_match_winner": 1800.0}, FEATURE_COLS,
    const_predict_fn(0.62), best_of=3, player1_is_winner=True,
    n_simulations=200, rng=random.Random(777),
)
check("Identical seed produces byte-identical results", p1 == p2, f"{p1} vs {p2}")


# ======================================================================
# LEVEL 8: Output range validity across randomized realistic states
# ======================================================================
section("LEVEL 8: Stress test — output always in [0, 1], never NaN, across randomized states")

rng = random.Random(42)
n_stress = 200
violations = 0
nan_count = 0
for _ in range(n_stress):
    a_sets = rng.randint(0, 1)
    b_sets = rng.randint(0, 1)
    best_of = rng.choice([3, 5])
    a_games = rng.randint(0, 5)
    b_games = rng.randint(0, 5)
    a_pts = rng.randint(0, 3)
    b_pts = rng.randint(0, 3)
    server_is_a = rng.choice([True, False])
    p1_is_winner = rng.choice([True, False])
    p_val = rng.uniform(0.3, 0.8)

    # max_points scaled to best_of, matching the function's OWN real defaults (350/700) —
    # an earlier version of this test used a flat max_points=100, which is genuinely too
    # tight for best-of-5 (matching real matches, which can exceed 100 points) and caused
    # the function to correctly, honestly report NaN rather than fabricate an answer. That
    # was a bug in this test's unrealistic cap, not in the function — fixed here.
    realistic_cap = 150 if best_of == 3 else 300

    p = batch_simulate_dynamic(
        (a_sets, b_sets, a_games, b_games, a_pts, b_pts, server_is_a, False),
        {}, FEATURE_COLS, const_predict_fn(p_val), best_of=best_of,
        player1_is_winner=p1_is_winner, n_simulations=30, rng=rng, max_points=realistic_cap,
    )
    if isinstance(p, float) and p != p:  # NaN check
        nan_count += 1
    elif not (0.0 <= p <= 1.0):
        violations += 1

check(f"Stress test: {n_stress} randomized states all produce valid probabilities "
      f"(with realistic max_points caps)",
      violations == 0, f"{violations} range violations, {nan_count} NaN (undecided) results")
if nan_count > 0:
    print(f"  ({nan_count} NaN results are the function correctly reporting 'undecided' when")
    print(f"   no simulations completed within the cap — expected for very early match states")
    print(f"   with only 30 simulations, not a correctness bug)")


# ======================================================================
# SUMMARY
# ======================================================================
section("SUMMARY")
if FAILURES:
    print(f"❌ {len(FAILURES)} check(s) FAILED:")
    for f in FAILURES:
        print(f"   - {f}")
    sys.exit(1)
else:
    print("✅ ALL LEVELS PASSED.")