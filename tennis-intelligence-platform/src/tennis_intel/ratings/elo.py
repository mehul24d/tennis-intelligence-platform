"""
elo.py — standard Elo rating system.

Cold-start policy: every player's first-seen rating is `initial_rating` (default 1500.0),
defined once in RatingSystem.__init__ and never silently overridden elsewhere in this file
or in processor.py. If a different cold-start policy is ever needed (e.g. seeding by junior
ranking), it must be an explicit, documented parameter change here — not a hidden default
somewhere downstream.

K-factor: fixed at 32.0 by default (standard chess/Elo convention, and what the project
owner's synthetic test expectations are built against — see tests/unit/test_elo.py Test 1).
A future variant (e.g. K scaled by tournament level or match count) would be a new
RatingSystem subclass or a k_factor_fn parameter — not a change to this class's default.
"""

from __future__ import annotations

from tennis_intel.ratings.base import RatingSystem


class EloRating(RatingSystem):
    default_k: float = 32.0

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, rating_winner: float, rating_loser: float, k: float | None = None) -> tuple[float, float]:
        k = self.default_k if k is None else k
        expected_winner = self.expected_score(rating_winner, rating_loser)
        # Winner's actual score = 1, loser's actual score = 0. Zero-sum: whatever the winner
        # gains, the loser loses exactly the same amount, since expected_winner + expected_loser == 1.
        delta = k * (1.0 - expected_winner)
        return rating_winner + delta, rating_loser - delta