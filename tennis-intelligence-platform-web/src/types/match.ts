// types/match.ts — matches api/schemas/match.py exactly (MatchReplayResponse,
// MatchSearchResponse). Field names and nullability are taken directly from the
// real, confirmed API responses returned by the running FastAPI backend, not
// guessed — see api/schemas/match.py for the Pydantic source of truth.

export interface PlayerRef {
  name: string;
}

export interface PrematchProbabilities {
  markov: number | null;
  ml_mc: number | null;
  ml_informed_unsmoothed: number | null;
  ml_informed_smoothed: number | null;
  hybrid: number | null;
}

export interface PointPrediction {
  point_index: number;
  set1: number;
  set2: number;
  gm1: number;
  gm2: number;
  markov_p1: number;
  ml_mc_p1: number;
  ml_informed_unsmoothed_p1: number;
  ml_informed_smoothed_p1: number;
  hybrid_p1: number;
}

export interface SetBoundary {
  set_number: number;
  point_index: number;
  score: string;
  winner_is_p1: boolean;
}

export interface MatchReplayResponse {
  match_id: string;
  player1: PlayerRef;
  player2: PlayerRef;
  winner: string;
  n_points: number;
  tournament: string | null;
  date: string | null;
  final_score: string | null;
  prematch: PrematchProbabilities;
  points: PointPrediction[];
  set_boundaries: SetBoundary[];
}

export interface MatchSearchResponse {
  match_ids: string[];
}

// The five prediction engines, used consistently across the UI for labels/colors —
// matches this project's own ENGINE_DISPLAY_NAMES convention (replay_service.py,
// export_model_comparison.py) rather than inventing separate frontend naming.
export type EngineKey = "markov" | "ml_mc" | "ml_informed_unsmoothed" | "ml_informed_smoothed" | "hybrid";

export const ENGINE_LABELS: Record<EngineKey, string> = {
  markov: "Analytical Markov",
  ml_mc: "Machine Learning + Monte Carlo",
  ml_informed_unsmoothed: "ML-Informed Markov (Unsmoothed)",
  ml_informed_smoothed: "ML-Informed Markov (Smoothed)",
  hybrid: "Hybrid Engine",
};

// Maps each engine to its point-level field name in PointPrediction — since the
// real API uses a per-engine suffix convention (markov_p1, ml_mc_p1, etc.) rather
// than a nested object, this lookup avoids repeating that mapping in every
// component that needs to pull a specific engine's trajectory.
export const ENGINE_POINT_FIELD: Record<EngineKey, keyof PointPrediction> = {
  markov: "markov_p1",
  ml_mc: "ml_mc_p1",
  ml_informed_unsmoothed: "ml_informed_unsmoothed_p1",
  ml_informed_smoothed: "ml_informed_smoothed_p1",
  hybrid: "hybrid_p1",
};