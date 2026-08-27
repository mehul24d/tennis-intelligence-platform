// types/matchAnalysis.ts — matches api/schemas/match_summary.py and
// api/schemas/model_agreement.py exactly.

export interface ProbabilitySwing {
  point_index: number;
  probability_before: number;
  probability_after: number;
  swing: number;
}

export interface LargestComeback {
  lowest_win_probability: number;
  comeback_margin: number;
  point_index_of_low: number;
}

export interface StreakByPlayer {
  player1: number;
  player2: number;
}

export interface BreakPoints {
  player1_created: number;
  player1_converted: number;
  player2_created: number;
  player2_converted: number;
}

export interface MatchSummaryResponse {
  match_id: string;
  largest_probability_swing: ProbabilitySwing;
  largest_comeback: LargestComeback;
  longest_winning_streak_points: StreakByPlayer;
  longest_service_hold_points: number;
  break_points: BreakPoints;
  total_winners: number | null;
  total_unforced_errors: number | null;
  serve_percentage: number | null;
}

export interface ModelAgreementPoint {
  point_index: number;
  highest_probability: number;
  highest_probability_engine: string;
  lowest_probability: number;
  lowest_probability_engine: string;
  average_probability: number;
  std_dev: number;
  max_disagreement: number;
  most_confident_engine: string;
  least_confident_engine: string;
  changing_fastest_engine: string | null;
}

export interface DisagreementSummary {
  points_disagreeing_over_5pct: number;
  points_disagreeing_over_10pct: number;
  points_disagreeing_over_20pct: number;
}

export interface ModelAgreementResponse {
  match_id: string;
  n_points: number;
  points: ModelAgreementPoint[];
  disagreement_summary: DisagreementSummary;
}

export interface PointTimelineEntry {
  point_index: number;
  server: string;
  receiver: string;
  winner: string;
  score_before: string | null;
  set1: number;
  set2: number;
  gm1: number;
  gm2: number;
  probability_before_p1: number;
  probability_after_p1: number;
  probability_swing: number;
  is_break_point: boolean;
  is_set_point: boolean;
  is_match_point: boolean;
  is_tiebreak_point: boolean;
  is_largest_swing: boolean;
}

export interface PointTimelineFilters {
  break_points_only?: boolean;
  set_points_only?: boolean;
  match_points_only?: boolean;
  tiebreak_only?: boolean;
  min_swing?: number;
}

export interface PointTimelineResponse {
  match_id: string;
  n_points_total: number;
  n_points_returned: number;
  points: PointTimelineEntry[];
}