// types/fullMatchList.ts — matches api/schemas/full_match_list.py exactly.

export interface FullMatchSummaryRow {
  match_id: string;
  has_replay_data: boolean;
  tournament: string;
  year: number | null;
  surface: string;
  round: string;
  winner: string;
  loser: string;
  final_score: string | null;
  tournament_level: string | null;
  best_of: number | null;
  winner_elo: number | null;
  loser_elo: number | null;
}

export interface FullMatchListResponse {
  total: number;
  limit: number;
  offset: number;
  matches: FullMatchSummaryRow[];
}

export interface FullMatchListFilters {
  player?: string;
  surface?: string;
  year?: number;
  tourney_level?: string;
  limit?: number;
  offset?: number;
}