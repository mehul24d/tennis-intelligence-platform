import { useQuery } from "@tanstack/react-query";
import { rankingsService } from "@/services/rankingsService";

export function useCurrentEloRankings(limit = 100) {
  return useQuery({
    queryKey: ["rankings", "current-elo", limit],
    queryFn: () => rankingsService.getCurrentElo(limit),
  });
}

export function usePeakEloRankings(limit = 100) {
  return useQuery({
    queryKey: ["rankings", "peak-elo", limit],
    queryFn: () => rankingsService.getPeakElo(limit),
  });
}

export function useSurfaceEloRankings(surface: string, limit = 100) {
  return useQuery({
    queryKey: ["rankings", "surface-elo", surface, limit],
    queryFn: () => rankingsService.getSurfaceElo(surface, limit),
  });
}

export function usePeakSurfaceEloRankings(surface: string, limit = 100) {
  return useQuery({
    queryKey: ["rankings", "peak-surface-elo", surface, limit],
    queryFn: () => rankingsService.getPeakSurfaceElo(surface, limit),
  });
}

export function useBiggestUpsets(limit = 100) {
  return useQuery({
    queryKey: ["rankings", "biggest-upsets", limit],
    queryFn: () => rankingsService.getBiggestUpsets(limit),
  });
}