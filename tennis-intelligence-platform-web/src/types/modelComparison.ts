// types/modelComparison.ts — matches api/schemas/model_comparison.py exactly.

export interface EngineStats {
  display_name: string;
  n_points: number;
  log_loss: number;
  brier: number;
  ece: number;
}

export interface ModelComparisonResponse {
  n_matches: number;
  n_points: number;
  holdout_year: number;
  is_full_holdout: boolean;
  engines: Record<string, EngineStats>;
}
