import { useQuery } from "@tanstack/react-query";
import { modelComparisonService } from "@/services/modelComparisonService";

export function useModelComparison() {
  return useQuery({
    queryKey: ["model-comparison"],
    queryFn: () => modelComparisonService.get(),
  });
}