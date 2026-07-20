"use client";

import { useModelComparison } from "@/hooks/useModelComparison";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export default function ModelComparisonPage() {
  const { data, isLoading, isError } = useModelComparison();

  return (
    <div className="mx-auto max-w-6xl p-8 space-y-4">
      <h1 className="text-page-title">Model Comparison</h1>
      <p className="text-caption -mt-2">Log loss, Brier score, and calibration error on held-out matches.</p>

      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      )}
      {isError && (
        <p className="text-sm text-accent-red">
          No comparison data yet. Run{" "}
          <code className="font-stat text-xs bg-muted px-1.5 py-0.5 rounded">
            pipelines/export_model_comparison.py
          </code>{" "}
          to generate it.
        </p>
      )}

      {data && (
        <>
          <p className="text-caption">
            <span className="font-stat">{data.n_matches}</span> matches,{" "}
            <span className="font-stat">{data.n_points.toLocaleString()}</span> points
            {data.is_full_holdout ? " (full holdout)" : " (sample)"}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(data.engines).map(([key, engine]) => (
              <Card key={key}>
                <CardHeader>
                  <CardTitle className="text-foreground normal-case tracking-normal text-section-title">
                    {engine.display_name}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-1.5">
                  <div className="flex justify-between items-baseline text-sm">
                    <span className="text-caption">Log loss</span>
                    <span className="font-stat font-semibold">{engine.log_loss.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between items-baseline text-sm">
                    <span className="text-caption">Brier</span>
                    <span className="font-stat font-semibold">{engine.brier.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between items-baseline text-sm">
                    <span className="text-caption">ECE</span>
                    <span className="font-stat font-semibold">{engine.ece.toFixed(4)}</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      )}
    </div>
  );
}