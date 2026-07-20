import { useQuery } from "@tanstack/react-query";
import { matchService } from "@/services/matchService";

export function useMatchReplay(matchId: string | undefined) {
  return useQuery({
    queryKey: ["match", matchId, "replay"],
    queryFn: () => matchService.getReplay(matchId as string),
    enabled: Boolean(matchId),
  });
}

export function useMatchSummary(matchId: string | undefined) {
  return useQuery({
    queryKey: ["match", matchId, "summary"],
    queryFn: () => matchService.getSummary(matchId as string),
    enabled: Boolean(matchId),
  });
}

export function useModelAgreement(matchId: string | undefined) {
  return useQuery({
    queryKey: ["match", matchId, "model-agreement"],
    queryFn: () => matchService.getModelAgreement(matchId as string),
    enabled: Boolean(matchId),
  });
}

export function useMatchSearch(searchTerms: string[]) {
  return useQuery({
    queryKey: ["match-search", searchTerms],
    queryFn: () => matchService.search(searchTerms),
    enabled: searchTerms.length > 0,
  });
}