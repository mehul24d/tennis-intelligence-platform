"""
base.py — abstract interface all rating systems implement.

Designed so pipelines/build_elo.py and processor.py never need to know which concrete
rating system they're running: standard Elo (elo.py) ships first, but Glicko-2 (glicko.py),
surface-specific Elo (surface_elo.py), and time-decayed variants (decay.py) can all be added
later as new implementations of this same interface without touching the chronological
processing loop in processor.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RatingSystem(ABC):
    """Every rating system must define: how to compute a win probability from two ratings
    (expected_score), and how a match outcome updates both players' ratings (update_ratings).
    initial_rating is the cold-start value assigned the first time a player is seen — defined
    once here, per player, and never silently changed downstream."""

    def __init__(self, initial_rating: float = 1500.0):
        self.initial_rating = initial_rating

    @abstractmethod
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability that player A beats player B, given both current ratings."""
        raise NotImplementedError

    @abstractmethod
    def update_ratings(self, rating_winner: float, rating_loser: float, **kwargs) -> tuple[float, float]:
        """Returns (new_winner_rating, new_loser_rating) after a single match outcome."""
        raise NotImplementedError