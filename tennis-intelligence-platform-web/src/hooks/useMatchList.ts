import { useQuery } from "@tanstack/react-query";
import { matchListService } from "@/services/matchListService";
import { MatchListFilters } from "@/types/matchList";
import { FullMatchListFilters } from "@/types/fullMatchList";

export function useMatchList(filters: MatchListFilters) {
  return useQuery({
    queryKey: ["match-list", filters],
    queryFn: () => matchListService.search(filters),
    placeholderData: (prev) => prev,
  });
}

export function useFullMatchList(filters: FullMatchListFilters) {
  return useQuery({
    queryKey: ["full-match-list", filters],
    queryFn: () => matchListService.searchFull(filters),
    placeholderData: (prev) => prev,
  });
}