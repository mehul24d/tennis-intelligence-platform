"use client";

import Link from "next/link";
import { CirclePlay } from "lucide-react";
import { useMatchReplay, useMatchSummary } from "@/hooks/useMatch";
import { ProbabilityChart } from "@/components/charts/ProbabilityChart";
import { ModelAgreementPanel } from "@/components/charts/ModelAgreementPanel";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AnimatedNumber } from "@/components/ui/AnimatedNumber";
import { Badge } from "@/components/ui/badge";
import { ENGINE_LABELS, EngineKey } from "@/types/match";

const ENGINE_ORDER: EngineKey[] = [
  "markov",
  "ml_mc",
  "ml_informed_unsmoothed",
  "ml_informed_smoothed",
  "hybrid",
];

export default function MatchAnalysisPage({ params }: { params: { matchId: string } }) {
  const matchId = decodeURIComponent(params.matchId);
  const { data: match, isLoading, isError, error } = useMatchReplay(matchId);
  const { data: summary } = useMatchSummary(matchId);

  if (isLoading) {
    return (
      <div className="mx-auto max-w-6xl p-8 space-y-6">
        <div className="space-y-2">
          <Skeleton className="h-9 w-96" />
          <Skeleton className="h-4 w-56" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-96 w-full" />
        <p className="text-caption text-center">
          Running all five engines point by point. Long matches take a few minutes on
          first load.
        </p>
      </div>
    );
  }

  if (isError || !match) {
    return (
      <div className="mx-auto max-w-6xl p-8">
        <Card>
          <CardContent className="pt-5">
            <p className="text-sm text-accent-red">
              Couldn&apos;t load this match: {error instanceof Error ? error.message : "unknown error"}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl p-8 space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-page-title">
            {match.player1.name} <span className="text-muted-foreground font-normal">vs</span>{" "}
            {match.player2.name}
          </h1>
          {(match.tournament || match.date) && (
            <p className="text-caption mt-1.5">
              {match.tournament && <span>{match.tournament}</span>}
              {match.tournament && match.date && <span> &middot; </span>}
              {match.date && (
                <span>
                  {new Date(match.date).toLocaleDateString(undefined, {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </span>
              )}
            </p>
          )}
          <div className="flex items-center gap-2 mt-2">
            <Badge variant="gold">Winner: {match.winner}</Badge>
            {match.final_score && <span className="text-caption font-stat">{match.final_score}</span>}
            <span className="text-caption">&middot; {match.n_points} points</span>
          </div>
        </div>
        <Link
          href={`/replay/${encodeURIComponent(match.match_id)}`}
          className="shrink-0 inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-sm font-medium hover:border-accent-blue/40 hover:text-accent-blue transition-colors"
        >
          <CirclePlay className="h-4 w-4" />
          Live Replay
        </Link>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {ENGINE_ORDER.map((engine) => {
          const prob = match.prematch[engine];
          return (
            <Card key={engine}>
              <CardHeader>
                <CardTitle>{ENGINE_LABELS[engine]}</CardTitle>
              </CardHeader>
              <CardContent>
                {prob !== null ? (
                  <p className="text-stat-lg text-accent-blue">
                    <AnimatedNumber value={prob * 100} decimals={1} suffix="%" />
                  </p>
                ) : (
                  <p className="text-stat-lg text-muted-foreground">—</p>
                )}
                <p className="text-caption mt-1">
                  {prob !== null ? `${match.player1.name} to win, pre-match` : "No pre-match read"}
                </p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Live Trajectory</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-caption -mt-2 mb-3">
            {match.player1.name} win probability, point by point, across all five engines.
          </p>
          <ProbabilityChart match={match} />
        </CardContent>
      </Card>

      {summary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <Card>
            <CardHeader>
              <CardTitle>Largest comeback</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-stat-lg">
                <AnimatedNumber value={summary.largest_comeback.comeback_margin * 100} decimals={1} suffix="%" />
              </p>
              <p className="text-caption mt-1">
                Down to {(summary.largest_comeback.lowest_win_probability * 100).toFixed(1)}% before turning it around
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Largest probability swing</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-stat-lg">
                <AnimatedNumber value={summary.largest_probability_swing.swing * 100} decimals={1} suffix="%" />
              </p>
              <p className="text-caption mt-1">
                At point {summary.largest_probability_swing.point_index}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Longest winning streak</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-stat-lg">
                <AnimatedNumber
                  value={Math.max(
                    summary.longest_winning_streak_points.player1,
                    summary.longest_winning_streak_points.player2
                  )}
                  suffix=" pts"
                />
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Longest service hold</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-stat-lg">
                <AnimatedNumber value={summary.longest_service_hold_points} suffix=" pts" />
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Model agreement</CardTitle>
        </CardHeader>
        <CardContent>
          <ModelAgreementPanel matchId={matchId} />
        </CardContent>
      </Card>
    </div>
  );
}