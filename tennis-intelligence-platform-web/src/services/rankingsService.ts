// services/rankingsService.ts — GET /api/rankings/current-elo, /peak-elo,
// /surface-elo, /peak-surface-elo, /biggest-upsets.

import { apiClient } from "@/api/client";
import {
  CurrentEloEntry,
  PeakEloEntry,
  SurfaceEloEntry,
  PeakSurfaceEloEntry,
  UpsetEntry,
} from "@/types/rankings";

export const rankingsService = {
  getCurrentElo(limit = 100): Promise<CurrentEloEntry[]> {
    return apiClient.get<CurrentEloEntry[]>("/api/rankings/current-elo", { limit });
  },

  getPeakElo(limit = 100): Promise<PeakEloEntry[]> {
    return apiClient.get<PeakEloEntry[]>("/api/rankings/peak-elo", { limit });
  },

  getSurfaceElo(surface: string, limit = 100): Promise<SurfaceEloEntry[]> {
    return apiClient.get<SurfaceEloEntry[]>("/api/rankings/surface-elo", { surface, limit });
  },

  getPeakSurfaceElo(surface: string, limit = 100): Promise<PeakSurfaceEloEntry[]> {
    return apiClient.get<PeakSurfaceEloEntry[]>("/api/rankings/peak-surface-elo", { surface, limit });
  },

  getBiggestUpsets(limit = 100): Promise<UpsetEntry[]> {
    return apiClient.get<UpsetEntry[]>("/api/rankings/biggest-upsets", { limit });
  },
};
