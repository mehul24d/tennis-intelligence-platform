import { useQuery } from "@tanstack/react-query";
import { playerService } from "@/services/playerService";

export function usePlayerSearch(query: string) {
  return useQuery({
    queryKey: ["player-search", query],
    queryFn: () => playerService.search(query),
    enabled: query.length > 1,
  });
}

export function usePlayerProfile(playerId: string | undefined) {
  return useQuery({
    queryKey: ["player-profile", playerId],
    queryFn: () => playerService.getProfile(playerId as string),
    enabled: Boolean(playerId),
  });
}

export function useHeadToHead(playerIdA: string | undefined, playerIdB: string | undefined) {
  return useQuery({
    queryKey: ["head-to-head", playerIdA, playerIdB],
    queryFn: () => playerService.getHeadToHead(playerIdA as string, playerIdB as string),
    enabled: Boolean(playerIdA) && Boolean(playerIdB),
  });
}