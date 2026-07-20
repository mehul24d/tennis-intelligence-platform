"use client";

import { useModelAgreement } from "@/hooks/useMatch";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AnimatedNumber } from "@/components/ui/AnimatedNumber";

export function ModelAgreementPanel({ matchId }: { matchId: string }) {
  const { data, isLoading, isError } = useModelAgreement(matchId);

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
    );
  }
  if (isError || !data) {
    return <p className="text-sm text-accent-red">Couldn&apos;t load model agreement.</p>;
  }

  const { disagreement_summary } = data;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      <Card>
        <CardHeader>
          <CardTitle>Points disagreeing &gt; 5%</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-stat-lg">
            <AnimatedNumber value={disagreement_summary.points_disagreeing_over_5pct} />
          </p>
          <p className="text-caption mt-1">of {data.n_points} points</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Points disagreeing &gt; 10%</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-stat-lg text-accent-gold">
            <AnimatedNumber value={disagreement_summary.points_disagreeing_over_10pct} />
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Points disagreeing &gt; 20%</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-stat-lg text-accent-red">
            <AnimatedNumber value={disagreement_summary.points_disagreeing_over_20pct} />
          </p>
        </CardContent>
      </Card>
    </div>
  );
}