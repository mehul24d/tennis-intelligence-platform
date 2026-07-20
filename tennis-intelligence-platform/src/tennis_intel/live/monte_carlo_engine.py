"""
monte_carlo_engine.py — Day 9: Monte Carlo simulation of remaining points given a
point-outcome probability, producing live match win probability.

WHY NOT PURE RECURSION: the Day 8 Markov baseline uses exact analytical recursion because
it assumes i.i.d. points (constant p per server). The ML classifier produces a DIFFERENT
probability for each point (conditioning on in-match state + pre-match features), so there
is no closed-form rollup. Instead: simulate many possible continuations from the current
state, using the classifier's per-point probabilities, and average the outcomes.

This is the standard Monte Carlo approach for in-match tennis simulation and is described
in the live_win_probability_extension_analysis.md roadmap (Section 4, Option B).

SIMULATION FIDELITY: the simulation correctly handles serve alternation (within a game and
across games), tiebreak triggering at 6-6, set completion rules, and best-of-N match
completion — the same rules proven correct in Day 7's point_score_parser.py tests.

SPEED: for a typical ~150-point remaining match at 1,000 simulations this completes in
well under a second per match, suitable for live use.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def _advance_point(
    a_sets: int, b_sets: int,
    a_games: int, b_games: int,
    a_points: int, b_points: int,
    server_is_a: bool,
    is_tiebreak: bool,
    best_of: int,
) -> tuple[int, int, int, int, int, int, bool, bool, bool | None]:
    """
    Advances the match state by exactly one point. Returns:
    (a_sets, b_sets, a_games, b_games, a_points, b_points,
     server_is_a, is_tiebreak, match_winner_is_a)

    match_winner_is_a is None if the match is still in progress, True/False if it ended.
    Uses standard ad-scoring and tiebreak rules (same as Day 7's point_score_parser.py).
    """
    sets_needed = (best_of // 2) + 1

    def game_won_by_a() -> bool:
        """Did player A just win the current game (based on current point scores)?"""
        if is_tiebreak:
            return a_points >= 7 and (a_points - b_points) >= 2
        # Regular game: first to 4 with 2-clear
        if a_points >= 4 and (a_points - b_points) >= 2:
            return True
        return False

    def game_won_by_b() -> bool:
        if is_tiebreak:
            return b_points >= 7 and (b_points - a_points) >= 2
        if b_points >= 4 and (b_points - a_points) >= 2:
            return True
        return False

    # BUG FIX (found via a rigorous multi-run averaged symmetric-skill test: 10 independent
    # simulations of a genuinely 50/50 matchup landed at a mean of 0.524-0.528, ~11 standard
    # errors from the true 0.5 — a real, reproducible bias, not noise): this function
    # previously only flipped server_is_a when a GAME concluded, meaning during an ongoing
    # tiebreak (which is one long "game" in the data model but requires serve to alternate
    # every 2 points after the first, per real tennis rules), the SAME player served every
    # single point of the entire tiebreak — a massive, illegitimate advantage for whoever
    # happened to be serving when the tiebreak began. Fixed by alternating serve mid-tiebreak
    # using the same rule already correctly implemented and validated in
    # live_win_probability.py's _prob_a_wins_tiebreak_from (point 1: server A; then B serves
    # 2, A serves 2, B serves 2, ... — i.e. serve switches whenever the total points played
    # so far is odd, matching the standard 1-2-2-2 tiebreak rotation).
    if is_tiebreak and not game_won_by_a() and not game_won_by_b():
        total_tb_points = a_points + b_points
        if total_tb_points % 2 == 1:
            server_is_a = not server_is_a
        return a_sets, b_sets, a_games, b_games, a_points, b_points, server_is_a, is_tiebreak, None

    # Check if a game just ended after the point we ENTERED WITH (pre-advance state)
    # Actually we're called BEFORE the point is played; the caller supplies the state
    # *before* the point — so we need to advance one point then check.
    # (This function is called with the state before a point is played.)

    # Game ends?
    if game_won_by_a():
        a_games += 1
        a_points, b_points = 0, 0
        server_is_a = not server_is_a
        # Set ends?
        is_tiebreak = False
        set_won = False
        if a_games >= 6 and (a_games - b_games) >= 2:
            set_won = True
        elif a_games == 7 and b_games == 6:
            set_won = True
        if set_won:
            a_sets += 1
            a_games, b_games = 0, 0
            if a_sets >= sets_needed:
                return a_sets, b_sets, 0, 0, 0, 0, server_is_a, False, True
        elif a_games == 6 and b_games == 6:
            is_tiebreak = True
    elif game_won_by_b():
        b_games += 1
        a_points, b_points = 0, 0
        server_is_a = not server_is_a
        is_tiebreak = False
        set_won = False
        if b_games >= 6 and (b_games - a_games) >= 2:
            set_won = True
        elif b_games == 7 and a_games == 6:
            set_won = True
        if set_won:
            b_sets += 1
            a_games, b_games = 0, 0
            if b_sets >= sets_needed:
                return a_sets, b_sets, 0, 0, 0, 0, server_is_a, False, False
        elif a_games == 6 and b_games == 6:
            is_tiebreak = True

    return a_sets, b_sets, a_games, b_games, a_points, b_points, server_is_a, is_tiebreak, None


def simulate_match_from_state(
    a_sets: int, b_sets: int,
    a_games: int, b_games: int,
    a_points: int, b_points: int,
    server_is_a: bool,
    is_tiebreak: bool,
    best_of: int,
    p_server_wins_point: float,  # constant serve-win prob (Markov-compatible mode)
    n_simulations: int = 1000,
    rng: random.Random | None = None,
    max_points: int | None = None,
) -> float:
    """
    Estimates P(A wins match from current state) via Monte Carlo simulation, assuming a
    constant p_server_wins_point on each point regardless of who is serving — this is the
    Markov-compatible mode used for validation (should reproduce the Day 8 baseline).

    For the ML mode (variable per-point probability), use simulate_match_with_classifier.

    BUG FIX #1 (external audit, 2026-07, Critical): this function previously entered the
    simulation loop even when the input state was ALREADY terminal (a_sets or b_sets
    already >= sets_needed), returning a non-1.0/0.0 value instead of the correct
    deterministic answer — e.g. simulate_match_from_state(2,0,...,best_of=3) returned 0.8
    instead of 1.0. Fixed with an explicit terminal-state guard before the simulation loop
    even starts, matching the same check batch_simulate_dynamic performs implicitly via
    _advance_point (see that function's docstring), made explicit here since this
    function's own inline loop does not delegate to _advance_point for its FIRST point.

    BUG FIX #2 (external audit, 2026-07, Critical): at p_server_wins_point=1.0 exactly,
    this function hung indefinitely rather than returning a degenerate-but-valid answer.
    This is a GENUINE mathematical infinite loop, not merely a long-running one: if the
    server always wins every point, whoever is serving always holds their own service
    game — and since serve alternates every game, NEITHER player can ever build the
    2-game lead needed to win a set (or, in a tiebreak, the 2-point lead needed to win
    it) — the match provably never terminates under this exact degenerate input. No
    amount of waiting resolves this; only a hard step cap can. Fixed with the same
    max_points pattern already used in batch_simulate_dynamic (defaults to 700 for
    best-of-5, 350 otherwise) — if the cap is hit, the simulation is treated as a coin
    flip for that trial (a defensible, documented fallback for a case that should not
    occur for any real, non-degenerate serve probability).
    """
    if rng is None:
        rng = random.Random(42)
    sets_needed = (best_of // 2) + 1

    if a_sets >= sets_needed:
        return 1.0
    if b_sets >= sets_needed:
        return 0.0

    if max_points is None:
        max_points = 700 if best_of == 5 else 350

    a_wins = 0

    for _ in range(n_simulations):
        # Copy state
        as_, bs_, ag, bg, ap, bp = a_sets, b_sets, a_games, b_games, a_points, b_points
        srv_a, tb = server_is_a, is_tiebreak
        winner = None
        n_points_this_sim = 0

        while winner is None:
            n_points_this_sim += 1
            if n_points_this_sim > max_points:
                logger.warning(
                    "simulate_match_from_state: hit max_points=%d cap (p_server_wins_point=%.4f) "
                    "— treating this trial as a coin flip. This should not occur for any "
                    "realistic, non-degenerate serve probability; if it does, check the input.",
                    max_points, p_server_wins_point,
                )
                winner = rng.random() < 0.5
                break

            # Play one point
            server_wins = rng.random() < p_server_wins_point
            if server_wins:
                if srv_a:
                    ap += 1
                else:
                    bp += 1
            else:
                if srv_a:
                    bp += 1
                else:
                    ap += 1

            # Advance state (checks for game/set/match completion)
            as_, bs_, ag, bg, ap, bp, srv_a, tb, winner = _advance_point(
                as_, bs_, ag, bg, ap, bp, srv_a, tb, best_of
            )

        if winner is True:
            a_wins += 1

    return a_wins / n_simulations


def batch_simulate_with_classifier(
    initial_state: tuple,
    feature_matrix_fn,  # callable(list[state_dict]) -> np.ndarray of shape (n_active, n_features)
    predict_fn,         # callable(feature_matrix) -> np.ndarray of P(server wins), shape (n_active,)
    best_of: int,
    n_simulations: int = 300,
    rng: random.Random | None = None,
    max_points: int = 500,
) -> float:
    """
    Same purpose as simulate_match_with_classifier, but BATCHED: at every simulation tick,
    ALL still-active simulations' current states are collected into one feature matrix and
    scored with ONE call to predict_fn, instead of one call per simulation. This turns
    O(n_simulations) classifier calls per tick into O(1) — the difference between a
    tractable "every point, every match" evaluation and an intractable one.

    max_points: safety cap on total points per simulation, to guarantee termination even in
    a pathological non-terminating edge case (should never be hit in practice, but a live
    engine must never hang).
    """
    if rng is None:
        rng = random.Random(42)

    # Internal state representation: tuples, not dicts. _advance_point already returns a
    # tuple natively, and rebuilding a dict literal on every active simulation on every
    # tick was a real, measurable source of Python-level overhead in the hot loop —
    # tuple assignment/unpacking is substantially cheaper. Dicts are only constructed at
    # the point of calling feature_matrix_fn, since that's an external interface that
    # needs named access to state fields.
    initial = tuple(initial_state)  # (a_sets, b_sets, a_games, b_games, a_pts, b_pts, server_is_a, is_tiebreak)
    states = [initial] * n_simulations
    winners = [None] * n_simulations
    active = list(range(n_simulations))

    def state_to_dict(s: tuple) -> dict:
        return {
            "a_sets": s[0], "b_sets": s[1], "a_games": s[2], "b_games": s[3],
            "a_points": s[4], "b_points": s[5], "server_is_a": s[6],
            "is_tiebreak": s[7], "best_of": best_of,
        }

    tick = 0
    while active and tick < max_points:
        tick += 1
        active_states = [state_to_dict(states[i]) for i in active]
        feature_matrix = feature_matrix_fn(active_states)
        probs = predict_fn(feature_matrix)

        still_active = []
        for idx_in_batch, sim_idx in enumerate(active):
            a_sets, b_sets, a_games, b_games, ap, bp, server_is_a, is_tiebreak = states[sim_idx]
            p_srv = float(probs[idx_in_batch])
            server_wins = rng.random() < p_srv

            if server_wins:
                if server_is_a:
                    ap += 1
                else:
                    bp += 1
            else:
                if server_is_a:
                    bp += 1
                else:
                    ap += 1

            new_state = _advance_point(
                a_sets, b_sets, a_games, b_games, ap, bp, server_is_a, is_tiebreak, best_of,
            )
            match_winner = new_state[8]
            states[sim_idx] = new_state[:8]  # cheap tuple slice, no dict rebuild

            if match_winner is not None:
                winners[sim_idx] = match_winner
            else:
                still_active.append(sim_idx)

        active = still_active

    if active:
        logger.warning("%d simulation(s) hit max_points=%d without terminating — "
                       "treating as undecided (excluded from the estimate).",
                       len(active), max_points)

    decided = [w for w in winners if w is not None]
    if not decided:
        return float("nan")
    return sum(1 for w in decided if w) / len(decided)


import numpy as np


def _ordinal_points(ap: int, bp: int) -> tuple[int, int]:
    """Maps unbounded simulated regular-game point counts to the ordinal 0..4 (4=AD)
    representation the flag functions expect. In the deuce region (both >= 3) only the
    difference matters: tied -> deuce (3,3); one ahead -> that player at advantage."""
    if ap >= 3 and bp >= 3:
        if ap == bp:
            return 3, 3
        return (4, 3) if ap > bp else (3, 4)
    return min(ap, 3), min(bp, 3)


def batch_simulate_dynamic(
    initial_state: tuple,
    static_features: dict,           # feature_name -> value, for pre-match/static features
    feature_cols: list,              # full ordered feature list the classifier expects
    predict_fn,                      # callable(np.ndarray) -> np.ndarray of P(server wins)
    best_of: int,
    player1_is_winner: bool,         # maps sim's A/B (A = tracked winner) to MCP Player1/2
    seed_momentum: dict | None = None,  # starting-point momentum values for seeding
    n_simulations: int = 300,
    rng: random.Random | None = None,
    max_points: int | None = None,
) -> float:
    """
    Task 7 rollout: like batch_simulate_with_classifier, but SITUATIONAL FLAGS
    (break/set/match point, tiebreak) are RE-DERIVED from each simulation's actual current
    state at every tick, and MOMENTUM is updated from each simulation's own simulated point
    outcomes — fixing the stale-context limitation identified in the Day 10 evaluation
    (docs/day10_head_to_head_freeze.md).

    Momentum seeding: for the first k simulated points (k < window N), momentum blends the
    starting value with simulated outcomes as (seed*(N-k) + sum(last k)) / N — i.e. the
    pre-simulation portion of the window is assumed to average the observed starting
    momentum. Exact reconstruction would require the real match's per-point history inside
    the simulator; this blend converges to the fully-simulated value within N points and is
    documented rather than silently approximated.

    is_second_serve_point is held False during simulation (serve outcome detail is not
    simulated) — a documented limitation, identical for every simulated point, so it cannot
    bias one continuation against another.

    max_points defaults to 350 for best-of-3 and 700 for best-of-5 (Task 6: the previous
    flat 400 cap was genuinely reachable by real-length best-of-5 continuations — long
    5-setters exceed 400 points — so those warnings were substantially expected behavior
    on best-of-5 evaluations, not only a bug).
    """
    from tennis_intel.features.point_score_parser import (
        is_break_point, is_set_point, is_match_point,
    )

    if rng is None:
        rng = random.Random(42)
    if max_points is None:
        max_points = 700 if best_of == 5 else 350

    # DEFENSIVE GUARD (external audit, 2026-07): added in parallel with the confirmed fix
    # to simulate_match_from_state, not because this function was shown to produce wrong
    # results in this project's actual calling pattern (it delegates to _advance_point,
    # which correctly detects termination once a point causes it — and every real call
    # site here evaluates a genuinely in-progress point, never an already-decided state)
    # — but because relying on that calling-pattern assumption forever is fragile. Cheap
    # insurance against a future caller passing an already-terminal state.
    sets_needed = (best_of // 2) + 1
    a_sets0 = initial_state[0]
    b_sets0 = initial_state[1]
    if a_sets0 >= sets_needed:
        return 1.0
    if b_sets0 >= sets_needed:
        return 0.0

    n_feat = len(feature_cols)
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    # float32, not float64: halves memory bandwidth for the feature matrix built at every
    # simulation tick. Safe here — sklearn tree splits are simple threshold comparisons,
    # and float32's ~7-digit precision is far finer than the model's own irreducible noise
    # (point-level Brier ~0.2), so this cannot meaningfully change any prediction.
    base_row = np.full(n_feat, np.nan, dtype=np.float32)
    for c, v in static_features.items():
        if c in col_idx:
            base_row[col_idx[c]] = v

    seed_momentum = seed_momentum or {}

    def _clean_seed(v):
        try:
            v = float(v)
            return 0.5 if np.isnan(v) else v
        except (TypeError, ValueError):
            return 0.5

    seed10 = _clean_seed(seed_momentum.get("p1_momentum_last10", 0.5))
    seed20 = _clean_seed(seed_momentum.get("p1_momentum_last20", 0.5))

    initial = tuple(initial_state)
    states = [initial] * n_simulations
    winners = [None] * n_simulations
    # Per-sim rolling history of PLAYER 1 point outcomes (capped at 20, the largest window)
    histories: list[list[int]] = [[] for _ in range(n_simulations)]
    active = list(range(n_simulations))

    def momentum(hist: list, window: int, seed: float) -> float:
        k = min(len(hist), window)
        if k == 0:
            return seed
        return (seed * (window - k) + sum(hist[-k:])) / window

    tick = 0
    while active and tick < max_points:
        tick += 1
        fm = np.empty((len(active), n_feat), dtype=np.float32)
        for j, sim_idx in enumerate(active):
            a_sets, b_sets, a_games, b_games, ap, bp, server_is_a, is_tb = states[sim_idx]
            row = base_row.copy()

            # Map A/B (A = the tracked player) back to MCP Player1/2 for flag conventions
            if player1_is_winner:
                p1s, p2s, p1g, p2g = a_sets, b_sets, a_games, b_games
                p1p_raw, p2p_raw = ap, bp
                svr_is_p1 = server_is_a
            else:
                p1s, p2s, p1g, p2g = b_sets, a_sets, b_games, a_games
                p1p_raw, p2p_raw = bp, ap
                svr_is_p1 = not server_is_a

            if is_tb:
                p1p, p2p, tb1, tb2 = 0, 0, p1p_raw, p2p_raw
            else:
                p1p, p2p = _ordinal_points(p1p_raw, p2p_raw)
                tb1, tb2 = None, None

            bp_flag = is_break_point(svr_is_p1, p1p, p2p, is_tb, tb1, tb2)
            sp_flag = is_set_point(svr_is_p1, p1p, p2p, is_tb, p1g, p2g, tb1, tb2)
            mp_flag = is_match_point(svr_is_p1, p1p, p2p, is_tb, p1g, p2g,
                                     p1s, p2s, best_of, tb1, tb2)

            hist = histories[sim_idx]
            m10 = momentum(hist, 10, seed10)
            m20 = momentum(hist, 20, seed20)

            for name, val in (
                ("is_tiebreak_game", is_tb), ("is_break_point", bp_flag),
                ("is_set_point", sp_flag), ("is_match_point", mp_flag),
                ("is_second_serve_point", False),
                ("p1_momentum_last10", m10), ("p2_momentum_last10", 1 - m10),
                ("p1_momentum_last20", m20), ("p2_momentum_last20", 1 - m20),
                ("server_is_winner", server_is_a),
            ):
                if name in col_idx:
                    row[col_idx[name]] = val
            fm[j] = row

        probs = predict_fn(fm)

        still_active = []
        for j, sim_idx in enumerate(active):
            a_sets, b_sets, a_games, b_games, ap, bp, server_is_a, is_tb = states[sim_idx]
            server_wins = rng.random() < float(probs[j])
            a_won_point = server_wins if server_is_a else (not server_wins)
            if a_won_point:
                ap += 1
            else:
                bp += 1
            p1_won = a_won_point if player1_is_winner else (not a_won_point)
            h = histories[sim_idx]
            h.append(1 if p1_won else 0)
            if len(h) > 20:
                del h[0]

            new_state = _advance_point(a_sets, b_sets, a_games, b_games, ap, bp,
                                       server_is_a, is_tb, best_of)
            match_winner = new_state[8]
            states[sim_idx] = new_state[:8]
            if match_winner is not None:
                winners[sim_idx] = match_winner
            else:
                still_active.append(sim_idx)
        active = still_active

    if active:
        logger.debug("%d simulation(s) hit max_points=%d without terminating.",
                     len(active), max_points)

    decided = [w for w in winners if w is not None]
    if not decided:
        return float("nan")
    return sum(1 for w in decided if w) / len(decided)


def simulate_match_with_classifier(
    initial_state: tuple,
    point_feature_fn,  # callable(state_dict) -> float, returns P(server wins point)
    best_of: int,
    n_simulations: int = 500,
    rng: random.Random | None = None,
) -> float:
    """
    P(A wins) via Monte Carlo with a per-point probability from a trained classifier.
    `point_feature_fn` receives the current state as a dict and returns the probability
    that the server wins the next point.

    For each simulation: at each step, call point_feature_fn to get the current point's
    win probability, sample an outcome, advance the state, repeat until match ends.
    """
    if rng is None:
        rng = random.Random(42)

    a_sets0, b_sets0, a_games0, b_games0, a_points0, b_points0, server_is_a0, is_tiebreak0 = initial_state
    sets_needed = (best_of // 2) + 1
    a_wins = 0

    for _ in range(n_simulations):
        as_, bs_ = a_sets0, b_sets0
        ag, bg = a_games0, b_games0
        ap, bp = a_points0, b_points0
        srv_a, tb = server_is_a0, is_tiebreak0
        winner = None

        while winner is None:
            state_dict = {
                "a_sets": as_, "b_sets": bs_, "a_games": ag, "b_games": bg,
                "a_points": ap, "b_points": bp, "server_is_a": srv_a,
                "is_tiebreak": tb, "best_of": best_of,
            }
            p_srv = point_feature_fn(state_dict)
            server_wins = rng.random() < p_srv

            if server_wins:
                if srv_a:
                    ap += 1
                else:
                    bp += 1
            else:
                if srv_a:
                    bp += 1
                else:
                    ap += 1

            as_, bs_, ag, bg, ap, bp, srv_a, tb, winner = _advance_point(
                as_, bs_, ag, bg, ap, bp, srv_a, tb, best_of
            )

        if winner is True:
            a_wins += 1

    return a_wins / n_simulations