"""live_features.py — the shape live (in-match) data takes on its way into the agent's
prompt. Deliberately a thin, caller-supplied snapshot rather than something that loads
a full ReplayContext itself (that's a 15-20s+ load — see
tennis_intel.serving.replay_service.load_replay_context) — callers (dashboard,
compute_five_engine_trajectory, or a test) build one of these from whatever v1 serving
call they already made, and hand it to the agent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LiveFeature:
    """One fact about the live match state, tagged with which engine/source it came
    from and whether it's an estimate (a model output) vs. an observed fact (e.g. the
    current score, which is just data, not a prediction)."""

    label: str
    value: str
    is_estimate: bool = False


@dataclass(frozen=True)
class LiveFeatureSnapshot:
    """The live-feature context for one question. `features` is ordered — that order
    becomes the [L1], [L2], ... tags in the prompt, so callers should put the most
    relevant facts first."""

    match_id: str
    p1_name: str
    p2_name: str
    features: list[LiveFeature] = field(default_factory=list)

    @classmethod
    def from_trajectory_point(
        cls,
        match_id: str,
        p1_name: str,
        p2_name: str,
        point_index: int,
        markov_p1: float | None = None,
        ml_p1: float | None = None,
        ml_informed_p1: float | None = None,
        hybrid_p1: float | None = None,
        score_summary: str | None = None,
    ) -> "LiveFeatureSnapshot":
        """Builds a snapshot from one point of
        replay_service.compute_five_engine_trajectory's output -- the common case of
        "what does the model think right now, at this point in the match". Omits any
        engine whose value wasn't supplied rather than fabricating a placeholder."""
        features: list[LiveFeature] = []
        if score_summary is not None:
            features.append(LiveFeature(f"Score at point {point_index}", score_summary))
        engine_values = [
            ("Markov engine", markov_p1),
            ("ML + Monte Carlo engine", ml_p1),
            ("ML-informed Markov engine", ml_informed_p1),
            ("Hybrid engine", hybrid_p1),
        ]
        for engine_name, p in engine_values:
            if p is not None:
                features.append(
                    LiveFeature(
                        f"{engine_name} estimate of {p1_name} win probability",
                        f"{p:.1%}",
                        is_estimate=True,
                    )
                )
        return cls(match_id=match_id, p1_name=p1_name, p2_name=p2_name, features=features)

    def to_tagged_lines(self, start_index: int = 1) -> list[str]:
        """Renders features as '[L{n}] label: value' lines, matching the [L#]/[D#]
        citation convention the system prompt instructs the model to use."""
        return [
            f"[L{start_index + i}] {feat.label}: {feat.value}"
            + (" (model estimate)" if feat.is_estimate else "")
            for i, feat in enumerate(self.features)
        ]
