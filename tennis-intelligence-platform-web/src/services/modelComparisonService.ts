// services/modelComparisonService.ts — GET /api/model-comparison,
// GET /api/research-dashboard.

import { apiClient } from "@/api/client";
import { ModelComparisonResponse } from "@/types/modelComparison";
import { ResearchDashboardResponse } from "@/types/researchDashboard";

export const modelComparisonService = {
  get(): Promise<ModelComparisonResponse> {
    return apiClient.get<ModelComparisonResponse>("/api/model-comparison");
  },
};

export const researchDashboardService = {
  get(): Promise<ResearchDashboardResponse> {
    return apiClient.get<ResearchDashboardResponse>("/api/research-dashboard");
  },
};
