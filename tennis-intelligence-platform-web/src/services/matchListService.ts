// services/matchListService.ts — GET /api/matches/search (Match Explorer table,
// MCP-limited to ~6,000 charted matches) and GET /api/matches/search-full (the
// FULL TML corpus, ~198,062 matches, per explicit request to browse every real
// match even when only a brief score/tournament summary is available).

import { apiClient } from "@/api/client";
import { MatchListResponse, MatchListFilters } from "@/types/matchList";
import { FullMatchListResponse, FullMatchListFilters } from "@/types/fullMatchList";

export const matchListService = {
  search(filters: MatchListFilters): Promise<MatchListResponse> {
    return apiClient.get<MatchListResponse>(
      "/api/matches/search",
      filters as Record<string, string | number | boolean | undefined>
    );
  },

  searchFull(filters: FullMatchListFilters): Promise<FullMatchListResponse> {
    return apiClient.get<FullMatchListResponse>(
      "/api/matches/search-full",
      filters as Record<string, string | number | boolean | undefined>
    );
  },
};