// types/playerProfile.ts — matches api/schemas/player_profile.py exactly.

export interface PlayerSearchResult {
  player_id: string;
  player_name: string;
}

export interface SurfaceStat {
  matches: number;
  wins: number;
  win_pct: number | null;
}

export interface GrandSlamStats {
  matches: number;
  wins: number;
  win_pct: number | null;
}

export interface RecentFormEntry {
  match_id: string;
  date: string | null;
  opponent: string;
  won: boolean;
  surface: string;
  tournament: string;
}

export interface EloTimelinePoint {
  match_id: string;
  date: string | null;
  elo: number | null;
  surface_elo: number | null;
}

export interface PlayerProfileResponse {
  player_id: string;
  player_name: string;
  current_elo: number | null;
  peak_elo: number | null;
  career_matches: number;
  career_wins: number;
  career_losses: number;
  career_win_pct: number | null;
  surface_stats: Record<string, SurfaceStat>;
  grand_slam_stats: GrandSlamStats;
  recent_form: RecentFormEntry[];
  elo_timeline: EloTimelinePoint[];
}

export interface HeadToHeadMatch {
  match_id: string;
  date: string | null;
  tournament: string;
  surface: string;
  winner_id: string;
  score: string | null;
}

export interface HeadToHeadResponse {
  player_id_a: string;
  player_id_b: string;
  a_wins: number;
  b_wins: number;
  matches: HeadToHeadMatch[];
}
