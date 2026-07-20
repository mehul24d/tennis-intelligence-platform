"""
ml_informed_markov.py — a third engine, distinct from both pure Markov (constant
career-average serve/return rates) and ML+MC (a full Monte Carlo rollout): feeds the
ALREADY-TRAINED Day 9 point classifier's context-aware P(server wins point) directly into
the ALREADY-VALIDATED Markov recursion (prob_a_wins_match_from_state), recomputed fresh at
every point — exactly the architecture recommended independently by two external design
reviews of this project, and the natural fix for the fixed-weight hybrid's failed premise
(evaluate_hybrid_engine.py): rather than blending two match-level outputs with a
hand-specified weight, there is only one point-level probability, informed by real context,
feeding one correct recursion. No weight to get wrong.

SMOOTHING (added in response to direct external critique, 2026-07): the FIRST version of
this engine fed the classifier's raw single-point prediction straight into the recursion,
with no smoothing over recent actual outcomes and no accounting for how much the recursion
AMPLIFIES a given amount of input noise depending on the current score state. Both gaps
are addressed here:

  1. BETA-BINOMIAL SMOOTHING: rather than trusting one classifier prediction in isolation,
     it is blended with a running Beta posterior over ACTUAL service points observed so far
     in the match (not classifier outputs — real outcomes), so a single noisy point-level
     prediction can't swing the estimate as much as a sustained trend in real results can.

  2. SENSITIVITY-AWARE SMOOTHING: the recursion's local sensitivity
     (d P(win) / d p_serve) at the CURRENT score state is computed via finite differences
     on the existing, unmodified prob_a_wins_match_from_state — no new derivation, reusing
     the already-validated recursion as-is. States where a small change in p_serve produces
     a large change in P(win) (deuce, break point, set point) are smoothed HARDER — the
     stable Beta posterior is trusted more, the noisy instantaneous classifier prediction
     less — exactly the opposite of what a naive constant-weight blend would do.

WHY THIS RESOLVES THE HYBRID'S FAILURE: the hybrid failed because handing more weight to
Markov whenever ML+MC was confident punished ML+MC for being correctly informed. This
architecture doesn't blend two independent match-level answers at all — it uses the ML
model for exactly what it's good at (context-aware POINT probability) and the Markov
recursion for exactly what it's good at (correctly composing point probability into game/
set/match probability), with no arbitrary weighting step in between.

SERVE/RETURN CONSTRUCTION: the recursion needs BOTH p_a_serve (A's win-prob serving) and
p_a_return (A's win-prob returning) at every point, but a classifier prediction only exists
for whoever is ACTUALLY serving the current real point. Rather than falling back to a
static career-average for the unobserved half (inconsistent — one context-aware number,
one static number, fed into the same recursion), TWO synthetic feature rows are built for
the SAME real point-in-time context (identical break-point/tiebreak/momentum state), one
with server_is_a=True and one server_is_a=False, and the classifier is queried for both —
giving two context-aware numbers that are directly comparable, since both reflect the exact
same moment in the match.

APPROXIMATION BEING MADE, stated explicitly: the recursion assumes today's point-in-time
p_serve/p_return hold for the ENTIRE remainder of the match, recomputed fresh at the next
point with a new prediction. This is the same approximation quality as feeding ANY single
number into a Markov recursion at each step (including the existing pure-Markov engine's
constant career average) — not a new weakness introduced here, just made more visible by
using a number that itself changes point to point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tennis_intel.live.live_win_probability import MatchState, prob_a_wins_match_from_state
from tennis_intel.live.markov_inverse import invert_prematch_probability


def recursion_sensitivity(state: MatchState, p_a_serve: float, p_a_return: float,
                          with_respect_to: str = "serve", eps: float = 0.01) -> float:
    """
    |d P(A wins match) / d p| at the current state, via central finite differences on the
    existing, unmodified recursion — where p is EITHER p_a_serve or p_a_return, selected
    by with_respect_to ("serve" or "return"). Reused directly rather than re-derived — this
    is the SAME prob_a_wins_match_from_state validated extensively elsewhere in this
    project, just probed at two nearby points to estimate its local slope.

    High sensitivity (e.g. near 1.0 at deuce, break point, or a close set) means a small
    error in the corresponding input estimate gets AMPLIFIED into a large error in the
    output match probability — exactly where smoothing should be strongest. Low sensitivity
    (e.g. already up 2 sets to love with a big game lead) means even a fairly wrong estimate
    barely changes the answer — smoothing matters less there.
    """
    if with_respect_to == "serve":
        p_hi = min(p_a_serve + eps, 0.999)
        p_lo = max(p_a_serve - eps, 0.001)
        y_hi = prob_a_wins_match_from_state(state, p_hi, p_a_return)
        y_lo = prob_a_wins_match_from_state(state, p_lo, p_a_return)
    elif with_respect_to == "return":
        p_hi = min(p_a_return + eps, 0.999)
        p_lo = max(p_a_return - eps, 0.001)
        y_hi = prob_a_wins_match_from_state(state, p_a_serve, p_hi)
        y_lo = prob_a_wins_match_from_state(state, p_a_serve, p_lo)
    else:
        raise ValueError(f"with_respect_to must be 'serve' or 'return', got {with_respect_to!r}")
    return abs(y_hi - y_lo) / (p_hi - p_lo)


@dataclass
class ServeReturnPosterior:
    """
    Running Beta posterior over a player's ACTUAL service-point win rate observed so far
    in the current match — updated from real outcomes (points actually won/lost on serve),
    NOT from classifier predictions. This is the "smoothed, accumulated evidence" half of
    the blend; the classifier's per-point prediction is the "instantaneous, context-aware"
    half. Separate posteriors are kept for serve and return, since these are genuinely
    different skills with different in-match trajectories.

    Prior: seeded from the SAME career first-serve-win-rate already used by the pure Markov
    engine, with a modest effective sample size (alpha0+beta0) — informative enough to
    resist being swung by 2-3 early points, weak enough to be overtaken by a real, sustained
    in-match trend within the first set or two. This mirrors exactly the "Bayesian
    updating, not a naive rolling window" architecture both external reviews recommended.

    TEMPORAL DISCOUNTING (added per external review, 2026-07 — ranked #1 ROI, "fixes the
    single biggest known failure mode: posterior pinning/inertia later in match"): static
    Beta-Binomial updating assumes the true rate is constant for the entire match. It
    isn't — fatigue, tactical adjustment, and momentum mean the true rate can genuinely
    drift over 3+ hours, and without a way to FORGET old evidence, the posterior's
    effective sample size only ever grows, making it progressively more rigid exactly
    when a real, sustained shift in form should be able to move it. Before every update,
    existing counts are discounted by lambda_decay (in (0, 1], default 1.0 = no
    discounting, exactly reproducing all prior behavior for full backward compatibility):

        alpha_t = lambda_decay * alpha_{t-1} + 1{server won point}
        beta_t  = lambda_decay * beta_{t-1}  + 1{server lost point}

    This decays the PRIOR's own contribution too, not just accumulated match evidence —
    deliberately, per the reviewer's own framing: the prior's influence SHOULD fade as the
    match progresses, same as any other older evidence, not remain permanently anchored.

    IMPORTANT SUBTLETY: decaying alpha+beta on every update breaks the previous
    points_observed_serve/return derivation, which assumed alpha+beta = n0 + points
    observed (true only WITHOUT decay, since decay shrinks alpha+beta independently of
    how many real points have been added). effective_points_serve/return are now tracked
    as EXPLICIT, SEPARATELY-DECAYED fields for exactly this reason — they represent the
    decayed effective count of REAL observations only, cleanly separable from the prior's
    own (also-decaying) contribution, and are what sensitivity_aware_blend's evidence
    floor should actually use.
    """
    alpha_serve: float
    beta_serve: float
    alpha_return: float
    beta_return: float
    n0_serve: float = 20.0
    n0_return: float = 20.0
    # Effective (decayed) count of REAL points observed so far — see the class docstring's
    # "IMPORTANT SUBTLETY" for why this must be tracked explicitly rather than derived by
    # subtracting n0 from alpha+beta once decay is in play.
    effective_points_serve: float = 0.0
    effective_points_return: float = 0.0
    # Decay factor applied to EXISTING counts before each update. 1.0 = no decay, exactly
    # reproducing pre-discounting behavior. Reviewer-suggested range: 0.97-0.995.
    lambda_decay: float = 1.0

    @classmethod
    def from_career_rate(cls, career_serve_rate: float, career_return_rate: float,
                         prior_strength: float = 20.0) -> "ServeReturnPosterior":
        """
        DEPRECATED PATTERN, kept for backward compatibility with existing callers/tests:
        seeds directly from a raw career POINT-level rate with a fixed prior_strength,
        bypassing every richer pre-match feature (surface Elo, H2H, tournament form) this
        project has built. See from_pretrained_prior below for the corrected construction
        (Markov-inverted from the full pre-match win-probability model, with a
        confidence-derived n0) — new code should prefer that, not this.
        """
        career_serve_rate = 0.65 if career_serve_rate is None or np.isnan(career_serve_rate) else career_serve_rate
        career_return_rate = 0.38 if career_return_rate is None or np.isnan(career_return_rate) else career_return_rate
        return cls(
            alpha_serve=career_serve_rate * prior_strength,
            beta_serve=(1 - career_serve_rate) * prior_strength,
            alpha_return=career_return_rate * prior_strength,
            beta_return=(1 - career_return_rate) * prior_strength,
            n0_serve=prior_strength, n0_return=prior_strength,
        )

    @classmethod
    def from_pretrained_prior(
        cls, p_serve0: float, n0_serve: float, p_return0: float, n0_return: float,
        lambda_decay: float = 1.0,
    ) -> "ServeReturnPosterior":
        """
        Seeds the posterior from an ALREADY Markov-inverted, ALREADY confidence-weighted
        prior (see build_pretrained_prior below) — the corrected construction, in place of
        from_career_rate's direct-from-raw-career-average approach. p_serve0/p_return0 are
        point-level probabilities consistent with the richer pre-match win-probability
        model (Elo, surface Elo, H2H, tournament form), NOT a raw career average;
        n0_serve/n0_return are the effective prior sample sizes, driving how quickly the
        posterior departs from this prior as real points are observed. lambda_decay
        defaults to 1.0 (no discounting) for exact backward compatibility — pass a value
        below 1.0 to enable temporal forgetting.
        """
        if not (0.0 < lambda_decay <= 1.0):
            raise ValueError(f"lambda_decay must be in (0, 1], got {lambda_decay}")
        return cls(
            alpha_serve=p_serve0 * n0_serve, beta_serve=(1 - p_serve0) * n0_serve,
            alpha_return=p_return0 * n0_return, beta_return=(1 - p_return0) * n0_return,
            n0_serve=n0_serve, n0_return=n0_return,
            effective_points_serve=0.0, effective_points_return=0.0,
            lambda_decay=lambda_decay,
        )

    def mean_serve(self) -> float:
        return self.alpha_serve / (self.alpha_serve + self.beta_serve)

    def mean_return(self) -> float:
        return self.alpha_return / (self.alpha_return + self.beta_return)

    def points_observed_serve(self) -> float:
        """Effective (decayed) count of REAL service points observed so far. Tracked
        explicitly (see class docstring) rather than derived from alpha+beta, since decay
        breaks that derivation once lambda_decay < 1.0. With lambda_decay=1.0 this is
        identical to the pre-discounting behavior."""
        return self.effective_points_serve

    def points_observed_return(self) -> float:
        return self.effective_points_return

    def update_serve(self, a_won_point: bool) -> "ServeReturnPosterior":
        """Returns a NEW posterior (immutable update) reflecting one more observed
        service point — a real outcome, not a prediction. Existing counts are decayed by
        lambda_decay BEFORE adding this point's evidence (lambda_decay=1.0 reproduces the
        exact pre-discounting update)."""
        return ServeReturnPosterior(
            alpha_serve=self.lambda_decay * self.alpha_serve + (1.0 if a_won_point else 0.0),
            beta_serve=self.lambda_decay * self.beta_serve + (0.0 if a_won_point else 1.0),
            alpha_return=self.alpha_return, beta_return=self.beta_return,
            n0_serve=self.n0_serve, n0_return=self.n0_return,
            effective_points_serve=self.lambda_decay * self.effective_points_serve + 1.0,
            effective_points_return=self.effective_points_return,
            lambda_decay=self.lambda_decay,
        )

    def update_return(self, a_won_point: bool) -> "ServeReturnPosterior":
        return ServeReturnPosterior(
            alpha_serve=self.alpha_serve, beta_serve=self.beta_serve,
            alpha_return=self.lambda_decay * self.alpha_return + (1.0 if a_won_point else 0.0),
            beta_return=self.lambda_decay * self.beta_return + (0.0 if a_won_point else 1.0),
            n0_serve=self.n0_serve, n0_return=self.n0_return,
            effective_points_serve=self.effective_points_serve,
            effective_points_return=self.lambda_decay * self.effective_points_return + 1.0,
            lambda_decay=self.lambda_decay,
        )


def build_pretrained_prior(
    p0_a_wins: float, p_a_return_seed: float, best_of: int,
    elo_matches_played_a: float | None = None, elo_matches_played_b: float | None = None,
    h2h_meetings: float | None = None, tourney_h2h_meetings: float | None = None,
    base_n0: float = 20.0, min_n0: float = 5.0, max_n0: float = 60.0,
    reference_matches: float = 150.0, reference_h2h_meetings: float = 5.0,
    reference_tourney_h2h_meetings: float = 3.0,
) -> tuple[float, float, float, float]:
    """
    Implements the corrected prior construction: inverts the ALREADY-COMPUTED, richer
    pre-match win probability (p0_a_wins — from whatever model the caller uses; typically
    the Elo/surface-Elo/H2H/tournament-form-informed ML pre-match estimate) back into a
    point-level serve-probability prior consistent with the Markov recursion, and sets a
    confidence-derived effective sample size n0 rather than one fixed constant for every
    match regardless of how much real history backs the estimate.

    Returns (p_serve0_a, n0_serve_a, p_return0_a, n0_return_a).

    p_a_return_seed is A's OWN independently-estimated return rate (1 - opponent's real
    surface-conditioned serve rate, per the same p_return construction validated and used
    everywhere else in this project) — held FIXED as the anchor the inversion solves
    around, per the documented parameterization choice above.

    n0 SCALING — UPGRADED TO A COMPOSITE SIGNAL (external audit, 2026-07, Architecture
    Review finding C): the original version scaled n0 PURELY off elo_matches_played_pre_*,
    ignoring every other available signal of "how much should this pre-match estimate be
    trusted." A player with 300 career matches but ZERO prior meetings against THIS
    specific opponent has real, matchup-specific uncertainty that raw career match count
    cannot see — two genuinely different kinds of "experience" were being conflated into
    one proxy. Now a composite of three independent, already-available signals, each
    contributing its own share of the [base_n0, max_n0] range:
      1. Career match count (elo_matches_played_pre_*) — general experience/rating
         reliability, the original signal, unchanged in kind.
      2. Head-to-head meeting count (winner_h2h_wins_pre_match + loser_h2h_wins_pre_match)
         — matchup-specific history; two players who have met 5+ times carry real,
         specific information about how they play EACH OTHER that career stats alone miss.
      3. Tournament-specific H2H meeting count — narrower still (how they've played each
         other AT THIS SPECIFIC EVENT), a real but usually thin signal.
    If h2h_meetings/tourney_h2h_meetings are not supplied (backward compatible with any
    existing caller), those two components contribute zero and n0 reduces to EXACTLY the
    original match-count-only behavior — no change in output for callers that don't pass
    the new arguments.

    BUG FIX (found via code review, 2026-07): n0_return_a was previously computed from
    elo_matches_played_a (A's own match count) — but p_a_return_seed's VALUE is derived
    from the OPPONENT's (B's) serve rate, not A's. The confidence measure and the
    quantity it's supposed to express confidence about were keyed to different players;
    elo_matches_played_b was accepted as a parameter and silently never used, which is
    what let this go unnoticed. Now n0_return_a correctly uses elo_matches_played_b: how
    much we trust B's serve rate (and therefore A's return prior, which is derived from
    it) should depend on how much real history exists for B, not for A. The SAME
    h2h_meetings/tourney_h2h_meetings values are used for both n0_serve_a and n0_return_a
    — matchup history is symmetric (how often A and B have played each other doesn't
    depend on which of them the current sub-posterior is about).
    """
    p_serve0_a = invert_prematch_probability(p0_a_wins, p_a_return_seed, best_of)

    # Composite range is split three ways when both new signals are available; each
    # component's own scale is capped at 1.0, so no single signal can push n0 past max_n0
    # on its own, and a component that's unavailable (None) simply contributes nothing
    # rather than being imputed or guessed.
    n_components = 1 + (h2h_meetings is not None) + (tourney_h2h_meetings is not None)
    share = (max_n0 - base_n0) / n_components

    def _match_count_contribution(matches_played: float | None) -> float:
        if matches_played is None or np.isnan(matches_played):
            return 0.0
        return min(matches_played / reference_matches, 1.0) * share

    def _h2h_contribution(meetings: float | None, reference: float) -> float:
        if meetings is None or np.isnan(meetings):
            return 0.0
        return min(meetings / reference, 1.0) * share

    def _confidence_n0(matches_played: float | None) -> float:
        total = base_n0
        total += _match_count_contribution(matches_played)
        total += _h2h_contribution(h2h_meetings, reference_h2h_meetings)
        total += _h2h_contribution(tourney_h2h_meetings, reference_tourney_h2h_meetings)
        return float(np.clip(total, min_n0, max_n0))

    n0_serve_a = _confidence_n0(elo_matches_played_a)
    n0_return_a = _confidence_n0(elo_matches_played_b)

    return p_serve0_a, n0_serve_a, p_a_return_seed, n0_return_a


def sensitivity_aware_blend(
    classifier_p: float, posterior_mean: float, sensitivity: float,
    points_observed: int | None = None,
    max_sensitivity_for_scaling: float = 3.0,
    evidence_floor_reference_points: float = 200.0,
) -> float:
    """
    Blends the classifier's instantaneous, context-aware prediction with the smoothed
    Beta posterior mean.

    BUG FIX (found via direct external critique + a real, precisely-traced jump on the
    2025 Roland Garros final, 2026-07): the ORIGINAL version weighted PURELY by the
    recursion's local sensitivity at the current score state, capping at a MAXIMUM of 80%
    weight on the posterior — meaning a MINIMUM of 20% weight on a single raw classifier
    reading was ALWAYS present, even at the very first point of the match. Traced this
    exactly: at the true start of a best-of-5 match, recursion sensitivity to p_return is
    genuinely high (~3.55, since the whole match outcome is still undecided) — a real,
    correct property of the tennis scoring recursion, not a bug — but combined with even
    the CAPPED 20% raw-classifier weight, this was enough to produce a ~5-point jump in
    match probability from ONE point's worth of noisy evidence, directly violating the
    project's stated objective ("start at a historically-grounded pre-match baseline...
    updates coherently... rather than ignoring the scoreboard" — a single point is not
    yet meaningful evidence and should not move the estimate this much).

    FIXED (v1) by adding a SECOND, independent floor on the classifier's weight, based on
    how much real evidence has accumulated relative to n0 (the prior's own effective
    sample size).

    BUG FIX (v2, found via a real chart showing the SMOOTHED line becoming MORE volatile
    than the UNSMOOTHED one within the first 30-50 points of a match — backwards from the
    whole point of smoothing): v1's evidence floor used n0 (~20-60, sized for the PRIOR's
    own confidence strength) as the reference scale for how many IN-MATCH points it
    should take before trusting live evidence more. These are two different things that
    should not share one number — verified directly: at points_observed=30 with a
    realistic n0=46.67, weight_from_evidence had already dropped to ~0.61, and by
    points_observed=100 (still well within a single match) to ~0.32 — meaning the floor's
    "protection" was substantially gone by roughly one set in, letting the raw, sharp,
    is_second_serve_point-driven classifier signal dominate long before any real
    match-level trend could have emerged. Fixed by introducing
    evidence_floor_reference_points (default 200, a realistic full-match point count) as
    a SEPARATE scale from n0 — the floor now decays over the timescale of an entire
    match, not over the timescale of the prior's own (much smaller) confidence strength.
    n0 continues to govern ONLY the Beta-Binomial posterior's own mean and variance
    (unchanged, already validated), not how quickly the blend defers to raw per-point
    noise.

    weight_on_posterior ranges from ~0.2 (low sensitivity, trust the live signal) to ~0.8
    (high sensitivity, trust the stable prior) via the sensitivity-based component, and is
    floored by evidence_floor_reference_points/(evidence_floor_reference_points+points_observed)
    via the evidence-based component — never fully replacing either source, since both
    carry genuine information (the posterior can be stale; the classifier can be noisy on
    any single point).

    If points_observed is not provided (backward compatible with any caller that
    doesn't have them), falls back to the original sensitivity-only behavior.
    """
    normalized_sensitivity = min(sensitivity / max_sensitivity_for_scaling, 1.0)
    weight_from_sensitivity = 0.2 + 0.6 * normalized_sensitivity  # ranges [0.2, 0.8]

    if points_observed is not None:
        # Evidence-based floor: when points_observed << evidence_floor_reference_points,
        # weight_from_evidence -> 1.0 (trust the prior almost completely); as
        # points_observed grows relative to a REALISTIC MATCH LENGTH (not n0), the floor
        # relaxes and hands control back to the sensitivity-based logic above, over the
        # timescale of an actual match rather than the prior's own confidence strength.
        weight_from_evidence = evidence_floor_reference_points / (
            evidence_floor_reference_points + points_observed
        )
        weight_on_posterior = max(weight_from_sensitivity, weight_from_evidence)
    else:
        weight_on_posterior = weight_from_sensitivity

    weight_on_posterior = min(weight_on_posterior, 1.0)
    return weight_on_posterior * posterior_mean + (1 - weight_on_posterior) * classifier_p


def ml_informed_point_probabilities(
    row: dict, model, feature_cols: list[str],
) -> tuple[float, float]:
    """
    Returns (p_a_serve, p_a_return): A's context-aware probability of winning a point when
    A serves, and when A returns, at this exact point-in-time context (break point,
    tiebreak, momentum, pre-match features all held at their real current values) — built
    from TWO synthetic feature rows differing only in server_is_a, queried against the
    SAME trained classifier used everywhere else in this project.
    """
    row_a_serves = dict(row)
    row_b_serves = dict(row)

    # LEAKAGE FIX (external audit, 2026-07): this previously set server_is_winner directly
    # (True for the "A serves" hypothetical, False for "B serves") — but server_is_winner
    # was confirmed to be a leaky, outcome-dependent feature (see build_point_dataset.py
    # and build_day9_point_model.py for the full explanation) and has been removed from
    # the classifier's actual feature list. This hypothetical-row construction now uses
    # server_is_player1 instead — a genuinely safe, real-time-observable feature (who is
    # physically serving) — translated correctly for the "A=tracked winner" convention via
    # player1_is_winner: if Player 1 IS the winner (A), then "A serves" means Player 1
    # serves; if Player 1 is the loser, "A serves" means Player 1 does NOT serve.
    p1_is_winner = bool(row.get("player1_is_winner", True))
    row_a_serves["server_is_player1"] = p1_is_winner
    row_b_serves["server_is_player1"] = not p1_is_winner

    X_a = np.array([[row_a_serves.get(c, np.nan) for c in feature_cols]], dtype=float)
    X_b = np.array([[row_b_serves.get(c, np.nan) for c in feature_cols]], dtype=float)

    p_a_serve = float(model.predict_proba(X_a)[0, 1])
    p_b_serve_when_b_serves = float(model.predict_proba(X_b)[0, 1])
    p_a_return = 1.0 - p_b_serve_when_b_serves

    return p_a_serve, p_a_return


def ml_informed_markov_predict(
    state: MatchState, row: dict, model, feature_cols: list[str],
    posterior: "ServeReturnPosterior",
) -> tuple[float, "ServeReturnPosterior"]:
    """
    Returns (P(A wins the match), updated posterior).

    The classifier's raw, instantaneous per-point prediction is blended with the running
    Beta posterior (real observed outcomes, not classifier predictions), with the blend
    weight determined by the recursion's LOCAL SENSITIVITY at the current state — high
    sensitivity favors the stable posterior, low sensitivity favors the live classifier
    signal. The posterior is then updated with the REAL outcome of this specific point
    (from row["PtWinner"]/row["player1_is_winner"]) — a genuine Bayesian update on
    evidence, not a self-referential smoothing of the classifier's own output.
    """
    p_a_serve_raw, p_a_return_raw = ml_informed_point_probabilities(row, model, feature_cols)

    sens_serve = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "serve")
    sens_return = recursion_sensitivity(state, p_a_serve_raw, p_a_return_raw, "return")

    p_a_serve = sensitivity_aware_blend(
        p_a_serve_raw, posterior.mean_serve(), sens_serve,
        points_observed=posterior.points_observed_serve(),
    )
    p_a_return = sensitivity_aware_blend(
        p_a_return_raw, posterior.mean_return(), sens_return,
        points_observed=posterior.points_observed_return(),
    )

    # Clip defensively: the recursion assumes valid probabilities; a classifier could in
    # principle emit exactly 0.0 or 1.0 for an unusual input, which would make some
    # terminal branches of the recursion degenerate (e.g. a player who can literally never
    # win a point). Clipping to a narrow but non-degenerate range avoids that failure mode
    # without meaningfully changing any reasonable prediction.
    p_a_serve = float(np.clip(p_a_serve, 0.01, 0.99))
    p_a_return = float(np.clip(p_a_return, 0.01, 0.99))

    p_match = prob_a_wins_match_from_state(state, p_a_serve, p_a_return)

    # Update the posterior with the REAL outcome of this point — "A" is always the tracked
    # winner (uniform project convention). player1_is_winner tells us whether A is player1
    # or player2; PtWinner tells us which NAMED player (1 or 2) won this specific point.
    #
    # CONVENTION, SETTLED 2026-07 AFTER A FALSE FIX AND A CORRECTION (read this before
    # touching this logic again): PtWinner IS literal, fixed-player-relative —
    # PtWinner==1 means player 1 (the same "Player 1" named in charting-m-matches.csv)
    # won the point, PtWinner==2 means player 2 won it, PERIOD — independent of who was
    # serving. This directly determines a_won_this_point via player1_is_winner alone;
    # Svr/state.server_is_a play NO role in interpreting PtWinner (they still matter
    # for routing the update to update_serve vs update_return below, and for what the
    # classifier features mean, but not for who won the point).
    #
    # A prior version of this comment (and code) claimed the OPPOSITE — that PtWinner is
    # SERVER-relative (PtWinner==1 means "the server won", requiring Svr to recover the
    # named winner) — and "fixed" this exact line to implement that. That fix was WRONG,
    # traced and reverted the same day it was made. What happened, for the record:
    #
    #   1. The server-relative claim originally rested on
    #      check_ptwinner_disagreement_at_scale.py reporting "0.00% disagreement" for a
    #      server-relative interpretation. That script's ground truth is p1_points/
    #      p2_points (parsed as fixed-player Pts), and it explicitly SKIPS every
    #      game-boundary row (`if row["Gm1"] != next_row["Gm1"] or row["Gm2"] !=
    #      next_row["Gm2"]: continue`). It only ever tests INTERIOR, within-game point
    #      transitions — and for those, BOTH "server-relative PtWinner + fixed-player
    #      Pts" AND "literal PtWinner + server-first Pts" are internally self-consistent
    #      (they coincide whenever Svr==1 and are exact mirror opposites whenever
    #      Svr==2, but a same-row self-consistency check can't tell the two apart — it
    #      only rules out MIXED pairings, which is what its "0.00%" figure actually
    #      shows). It never checked against Gm1/Gm2 — the one independently-recorded
    #      signal that can externally distinguish the two candidate conventions.
    #   2. Checking against Gm1/Gm2 at real game boundaries (the transitions that script
    #      skips) resolves it: literal PtWinner (this version) matches which Gm column
    #      increments at 99.91% (167 mismatches / 181,258 boundaries, corpus-wide,
    #      symmetric across Svr==1 and Svr==2 — consistent with ordinary charting
    #      error, not a systematic issue). Server-relative PtWinner matches only ~51%
    #      of boundaries (chance level), invisible to the old script only because it
    #      never looked at boundaries at all.
    #   3. Hand-traced in plain tennis terms on two real matches (Laver/Ashe 1969,
    #      Nadal/Shapovalov 2019): literal PtWinner's account is the ordinary one (the
    #      server grinds out a deuce game and holds); server-relative's account required
    #      the receiver to break serve after multiple deuces in both hand-checked cases
    #      — not impossible, but the less common outcome, and the one NOT matching the
    #      recorded Gm column in either case.
    #
    # See docs/critical_issue_gm_attribution_mismatch.md for the full investigation and
    # docs/ptwinner_convention_correction.md for the final resolution and the list of
    # every other file/finding that depended on the now-reverted assumption.
    pt_winner = row.get("PtWinner")
    if pt_winner is not None and not (isinstance(pt_winner, float) and np.isnan(pt_winner)):
        p1_is_winner = bool(row.get("player1_is_winner", True))
        a_won_this_point = (int(pt_winner) == 1) if p1_is_winner else (int(pt_winner) == 2)
        if state.server_is_a:
            new_posterior = posterior.update_serve(a_won_this_point)
        else:
            new_posterior = posterior.update_return(a_won_this_point)
    else:
        # No real outcome available for this row (e.g. unparseable score) — carry the
        # posterior forward unchanged rather than update on missing information.
        new_posterior = posterior

    return p_match, new_posterior