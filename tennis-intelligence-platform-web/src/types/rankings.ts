// types/rankings.ts — matches api/schemas/rankings.py exactly.

export interface CurrentEloEntry {
  rank: number;
  player_id: string;
  player_name: string;
  elo: number;
}

export interface PeakEloEntry {
  rank: number;
  player_id: string;
  player_name: string;
  peak_elo: number;
  date_achieved: string | null;
}

export interface SurfaceEloEntry {
  rank: number;
  player_id: string;
  player_name: string;
  surface_elo: number;
}

export interface PeakSurfaceEloEntry {
  rank: number;
  player_id: string;
  player_name: string;
  peak_surface_elo: number;
  date_achieved: string | null;
}

export interface UpsetEntry {
  rank: number;
  match_id: string;
  date: string | null;
  tournament: string;
  round: string;
  winner_name: string;
  winner_elo: number;
  loser_name: string;
  loser_elo: number;
  elo_gap: number;
}
