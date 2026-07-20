"use client";

import { useResearchDashboard } from "@/hooks/useResearchDashboard";
import { ReliabilityDiagram } from "@/components/charts/ReliabilityDiagram";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ENGINE_COLORS } from "@/components/charts/ProbabilityChart";
import { EngineKey } from "@/types/match";

const FALLBACK_COLOR = "#8888aa";

function engineColor(key: string): string {
  return ENGINE_COLORS[key as EngineKey] ?? FALLBACK_COLOR;
}

export default function ResearchDashboardPage() {
  const { data, isLoading, isError } = useResearchDashboard();

  return (
    <div className="mx-auto max-w-6xl p-8 space-y-4">
      <div>
        <h1 className="text-page-title">Research Dashboard</h1>
        <p className="text-caption mt-1">
          Calibration, reliability, and bootstrap confidence intervals across every
          prediction engine.
        </p>
      </div>

      {isLoading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-80" />
          ))}
        </div>
      )}

      {isError && (
        <p className="text-sm text-accent-red">
          No research data yet. Run{" "}
          <code className="font-stat text-xs bg-muted px-1.5 py-0.5 rounded">
            pipelines/export_research_dashboard.py
          </code>{" "}
          to generate it.
        </p>
      )}

      {data && (
        <>
          <p className="text-caption">
            <span className="font-stat">{data.n_matches}</span> matches,{" "}
            <span className="font-stat">{data.n_points.toLocaleString()}</span> points
            {data.is_full_holdout ? " (full holdout)" : " (sample)"} &middot;{" "}
            <span className="font-stat">{data.n_bootstrap.toLocaleString()}</span> bootstrap resamples
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {Object.entries(data.engines).map(([key, engine]) => {
              const color = engineColor(key);
              return (
                <Card key={key}>
                  <CardHeader>
                    <CardTitle className="text-foreground normal-case tracking-normal text-section-title flex items-center gap-2">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                      {engine.display_name}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-3 gap-3">
                      <div>
                        <p className="text-label">Log Loss</p>
                        <p className="text-stat-lg mt-0.5">{engine.log_loss.point_estimate.toFixed(4)}</p>
                        <p className="text-caption">
                          [{engine.log_loss.ci_lower.toFixed(4)}, {engine.log_loss.ci_upper.toFixed(4)}]
                        </p>
                      </div>
                      <div>
                        <p className="text-label">Brier</p>
                        <p className="text-stat-lg mt-0.5">{engine.brier.point_estimate.toFixed(4)}</p>
                        <p className="text-caption">
                          [{engine.brier.ci_lower.toFixed(4)}, {engine.brier.ci_upper.toFixed(4)}]
                        </p>
                      </div>
                      <div>
                        <p className="text-label">ECE</p>
                        <p className="text-stat-lg mt-0.5">{engine.ece.toFixed(4)}</p>
                      </div>
                    </div>

                    <div>
                      <p className="text-label mb-1">Reliability diagram</p>
                      <ReliabilityDiagram data={engine.reliability_diagram} color={color} />
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}