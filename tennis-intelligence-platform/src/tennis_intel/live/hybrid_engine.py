"""
*** DEPRECATED — DO NOT USE FOR PRODUCTION OR AS A DEFAULT ENGINE ***

Measured and confirmed underperforming BOTH of its own inputs (evaluate_hybrid_engine.py):
this fixed-weight blend was tested against pure Markov and pure ML+MC individually, and it
was WORSE than both on every metric. Handing more weight to Markov whenever ML+MC was
confident systematically punished ML+MC for being correctly informed — see the "WHY THIS
RESOLVES THE HYBRID'S FAILURE" note in ml_informed_markov.py for the full explanation of
why the actual, working replacement architecture (a single point-level probability feeding
one correct recursion, no blending step) succeeds where this design failed.

Kept in the codebase — NOT deleted — because it remains a legitimate, explicitly-requested
diagnostic/comparison line in replay_match.py's multi-engine chart (seeing a known-inferior
baseline plotted alongside working engines has real comparative value). It must never be
wired into any default evaluation path, production recommendation, or "which engine should
I use" decision. Per external audit (Architecture Review, finding A): isolated behind
explicit, by-name-only usage rather than hard-deleted, since deletion would remove a
capability that was deliberately, explicitly asked for.

*** END DEPRECATION NOTICE — original module docstring follows ***

hybrid_engine.py — dynamic weighting between the Markov and ML+MC engines for point-by-
point live win probability, using a FIXED, hand-specified (not fit/tuned) weighting
function to avoid introducing a new leakage risk (a trained meta-model would need its own
held-out split on top of everything else in this project).

WEIGHTING RATIONALE, directly evidence-based (not arbitrary): Day 11's own reliability
tables (docs/day11_head_to_head_v2_freeze.md) found Markov calibrates well and is
appropriately SHARP at extreme predictions (near 0 or 1), while ML+MC calibrates BETTER
specifically in the MODERATE 0.6-0.8 confidence range (see the corrected Day 11 addendum:
"ML+MC has better ECE... better calibrated in the 0.6-0.8 moderate-confidence range").
This motivates a weighting scheme that trusts Markov more as a prediction becomes extreme,
and trusts ML+MC more as a prediction sits closer to a genuine toss-up — the opposite of
"always use whichever engine sounds more sophisticated."

This is a v1, deliberately simple design: a monotonic function of ML+MC's OWN predicted
distance from 0.5, not a learned blend. A genuinely learned meta-model (logistic
regression on [markov_p, ml_mc_p] -> outcome, or similar) is a natural v2 extension, but
requires ITS OWN held-out split distinct from both this evaluation and the tuning
validation split already used elsewhere in this project — scoped as a clear follow-up,
not attempted here to avoid quietly reintroducing the exact class of leakage this project
has repeatedly found and fixed.
"""

from __future__ import annotations


def hybrid_weight_markov(ml_mc_p: float) -> float:
    """
    Weight given to the Markov prediction, as a function of ML+MC's own predicted
    probability. Ranges from 0 (at ml_mc_p=0.5, a genuine toss-up — trust ML+MC entirely)
    to 1 (at ml_mc_p=0 or 1, an extreme prediction — trust Markov entirely), linearly
    in between. This is 2 * |ml_mc_p - 0.5|.
    """
    if not (0.0 <= ml_mc_p <= 1.0):
        raise ValueError(f"ml_mc_p must be in [0, 1], got {ml_mc_p}")
    return 2.0 * abs(ml_mc_p - 0.5)


def hybrid_predict(markov_p: float, ml_mc_p: float) -> float:
    """
    Blends the two engines' predictions (both must already represent P(the SAME tracked
    player wins) — same orientation convention as everywhere else in this project) into a
    single hybrid probability, using hybrid_weight_markov's fixed, evidence-based weighting.
    """
    if not (0.0 <= markov_p <= 1.0):
        raise ValueError(f"markov_p must be in [0, 1], got {markov_p}")
    w_markov = hybrid_weight_markov(ml_mc_p)
    w_ml_mc = 1.0 - w_markov
    return w_markov * markov_p + w_ml_mc * ml_mc_p