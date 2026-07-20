// types/matchList.ts — matches api/schemas/match_list.py exactly.

export interface MatchSummaryRow {
  match_id: string;
  tournament: string;
  year: number | null;
  surface: string;
  round: string;
  winner: string;
  loser: string;
  final_score: string | null;
  duration_minutes: number | null;
  tournament_level: string | null;
  best_of: number | null;
  winner_elo: number | null;
  loser_elo: number | null;
  prematch_favourite: string | null;
}

export interface MatchListResponse {
  total: number;
  limit: number;
  offset: number;
  matches: MatchSummaryRow[];
}

export interface MatchListFilters {
  player?: string;
  tournament?: string;
  year?: number;
  surface?: string;
  round?: string;
  tourney_level?: string;
  best_of?: number;
  limit?: number;
  offset?: number;
}
