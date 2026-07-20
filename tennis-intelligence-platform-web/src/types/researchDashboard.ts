// types/researchDashboard.ts — matches api/schemas/research_dashboard.py exactly.

export interface MetricWithCI {
  point_estimate: number;
  ci_lower: number;
  ci_upper: number;
}

export interface ReliabilityPoint {
  bin_index: number;
  n: number;
  mean_predicted: number;
  observed_win_rate: number;
  calibration_gap: number;
}

export interface PredictionHistogram {
  bin_edges: number[];
  counts: number[];
}

export interface EngineResearchStats {
  display_name: string;
  n_points: number;
  log_loss: MetricWithCI;
  brier: MetricWithCI;
  ece: number;
  sharpness: number;
  reliability_diagram: ReliabilityPoint[];
  prediction_histogram: PredictionHistogram;
}

export interface ResearchDashboardResponse {
  n_matches: number;
  n_points: number;
  holdout_year: number;
  is_full_holdout: boolean;
  n_calibration_bins: number;
  n_bootstrap: number;
  engines: Record<string, EngineResearchStats>;
}
