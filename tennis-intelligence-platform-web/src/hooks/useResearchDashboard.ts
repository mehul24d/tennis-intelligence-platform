import { useQuery } from "@tanstack/react-query";
import { researchDashboardService } from "@/services/modelComparisonService";

export function useResearchDashboard() {
  return useQuery({
    queryKey: ["research-dashboard"],
    queryFn: () => researchDashboardService.get(),
  });
}