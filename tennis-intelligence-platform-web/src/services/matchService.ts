// services/matchService.ts — calls the real, confirmed endpoints in
// api/routers/matches.py.

import { apiClient } from "@/api/client";
import { MatchReplayResponse, MatchSearchResponse } from "@/types/match";
import {
  MatchSummaryResponse,
  ModelAgreementResponse,
  PointTimelineFilters,
  PointTimelineResponse,
} from "@/types/matchAnalysis";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export const matchService = {
  search(searchTerms: string[]): Promise<MatchSearchResponse> {
    // The real endpoint takes repeated ?search= query params, not a single value,
    // so this bypasses apiClient's single-value param map and builds the URL
    // manually.
    const url = new URL("/api/matches", BASE_URL);
    searchTerms.forEach((t) => url.searchParams.append("search", t));
    return fetch(url.toString()).then((r) => r.json());
  },

  getReplay(matchId: string): Promise<MatchReplayResponse> {
    return apiClient.get<MatchReplayResponse>(`/api/matches/${encodeURIComponent(matchId)}/replay`);
  },

  getSummary(matchId: string): Promise<MatchSummaryResponse> {
    return apiClient.get<MatchSummaryResponse>(`/api/matches/${encodeURIComponent(matchId)}/summary`);
  },

  getModelAgreement(matchId: string): Promise<ModelAgreementResponse> {
    return apiClient.get<ModelAgreementResponse>(`/api/matches/${encodeURIComponent(matchId)}/model-agreement`);
  },

  getPointTimeline(matchId: string, filters: PointTimelineFilters): Promise<PointTimelineResponse> {
    return apiClient.get<PointTimelineResponse>(`/api/matches/${encodeURIComponent(matchId)}/timeline`, {
      break_points_only: filters.break_points_only,
      set_points_only: filters.set_points_only,
      match_points_only: filters.match_points_only,
      tiebreak_only: filters.tiebreak_only,
      min_swing: filters.min_swing,
    });
  },
};