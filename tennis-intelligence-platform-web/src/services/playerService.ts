// services/playerService.ts — GET /api/players/search, GET /api/players/{id},
// GET /api/players/{a}/head-to-head/{b}.

import { apiClient } from "@/api/client";
import { PlayerSearchResult, PlayerProfileResponse, HeadToHeadResponse } from "@/types/playerProfile";

export const playerService = {
  search(query: string, limit = 20): Promise<PlayerSearchResult[]> {
    return apiClient.get<PlayerSearchResult[]>("/api/players/search", { q: query, limit });
  },

  getProfile(playerId: string): Promise<PlayerProfileResponse> {
    return apiClient.get<PlayerProfileResponse>(`/api/players/${encodeURIComponent(playerId)}`);
  },

  getHeadToHead(playerIdA: string, playerIdB: string): Promise<HeadToHeadResponse> {
    return apiClient.get<HeadToHeadResponse>(
      `/api/players/${encodeURIComponent(playerIdA)}/head-to-head/${encodeURIComponent(playerIdB)}`
    );
  },
};
