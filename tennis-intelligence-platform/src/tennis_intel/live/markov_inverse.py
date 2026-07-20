"""
markov_inverse.py — inverts the pre-match win-probability composite (from the richer,
feature-informed pre-match model — Elo, surface Elo, H2H, tournament form) back into a
point-level serve-probability prior consistent with prob_win_match's actual recursion.

WHY THIS EXISTS (found via direct external critique of ml_informed_markov.py's smoothing
design): the original ServeReturnPosterior seeded its prior DIRECTLY from
winner_first_serve_win_pct_career — a raw career POINT-level average, completely bypassing
every richer pre-match feature (surface Elo, H2H, tournament form) this project spent this
entire session building and validating. That is a genuinely different, poorer-quality
prior than "the point-level rate that would REPRODUCE the actual, best-available pre-match
match-win probability."

PARAMETERIZATION: one target scalar (P0 = P(A wins the match)) cannot uniquely determine
two unknowns (p_a_serve, p_a_return) — the system is underdetermined without an additional
constraint. Rather than inventing an arbitrary point on the resulting indifference curve,
p_a_return is FIXED at its own independently, already-correctly-derived value
(1 - opponent's real surface-conditioned serve rate — the same construction validated and
used everywhere else in this project since the p_return bug fix), and p_a_serve is solved
via bisection to be WHATEVER value, combined with that already-trusted return rate,
reproduces the richer P0 estimate exactly. This anchors the solution to a real, independently
estimated quantity rather than an arbitrary tie-breaking rule.
"""

from __future__ import annotations

from tennis_intel.live.markov_baseline import prob_win_match


def invert_prematch_probability(
    target_p0: float, p_a_return: float, best_of: int,
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    """
    Solves for p_a_serve such that prob_win_match(p_a_serve, p_a_return, best_of) ==
    target_p0, via bisection (prob_win_match is strictly monotonically increasing in
    p_a_serve for fixed p_a_return, so bisection is guaranteed to converge).

    BUG FIX (found on real data, 2026-07): target_p0 can legitimately arrive as EXACTLY
    0.0 or 1.0 — compute_ml_pre_match_probability estimates it via a FINITE (200-trial)
    Monte Carlo rollout, and a large enough skill gap makes "the favorite wins all 200
    simulated trials" a genuine, expected outcome of sampling variance, not a malformed
    input. The original version raised ValueError on exactly this legitimate case,
    crashing evaluate_ml_informed_markov.py partway through a real 150-match evaluation.
    Fixed by clipping defensively to a narrow-but-valid range BEFORE validating, mirroring
    the same defensive-clipping pattern already used in ml_informed_markov_predict for the
    identical underlying reason (a classifier/simulator can legitimately emit an extreme
    0/1 value that would otherwise break degenerate-probability math downstream).

    Returns p_a_serve in (0, 1). If target_p0 is itself at or beyond the achievable range
    for the given p_a_return (e.g. target_p0=0.99 but p_a_return is very low, such that
    even p_a_serve=0.999 can't reach it), returns the boundary value (0.001 or 0.999)
    rather than failing — a defensively bounded answer rather than a crash.
    """
    target_p0 = min(max(target_p0, 1e-4), 1 - 1e-4)
    p_a_return = min(max(p_a_return, 1e-4), 1 - 1e-4)

    lo, hi = 0.001, 0.999
    p_at_lo = prob_win_match(lo, p_a_return, best_of=best_of)
    p_at_hi = prob_win_match(hi, p_a_return, best_of=best_of)

    if target_p0 <= p_at_lo:
        return lo
    if target_p0 >= p_at_hi:
        return hi

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        p_mid = prob_win_match(mid, p_a_return, best_of=best_of)
        if abs(p_mid - target_p0) < tol:
            return mid
        if p_mid < target_p0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0