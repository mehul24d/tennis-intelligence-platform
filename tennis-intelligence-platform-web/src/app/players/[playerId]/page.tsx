"use client";

import { usePlayerProfile } from "@/hooks/usePlayer";
import { EloTimelineChart } from "@/components/charts/EloTimelineChart";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SurfaceBadge } from "@/components/ui/SurfaceBadge";
import { Skeleton } from "@/components/ui/skeleton";
import { AnimatedNumber } from "@/components/ui/AnimatedNumber";

export default function PlayerProfilePage({ params }: { params: { playerId: string } }) {
  const playerId = decodeURIComponent(params.playerId);
  const { data: profile, isLoading, isError } = usePlayerProfile(playerId);

  if (isLoading) {
    return (
      <div className="mx-auto max-w-5xl p-8 space-y-6">
        <div className="space-y-2">
          <Skeleton className="h-9 w-72" />
          <Skeleton className="h-4 w-48" />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (isError || !profile) {
    return (
      <div className="mx-auto max-w-5xl p-8">
        <p className="text-sm text-accent-red">Couldn&apos;t load this player.</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl p-8 space-y-6">
      <div>
        <h1 className="text-page-title">{profile.player_name}</h1>
        <p className="text-caption mt-1">
          <span className="font-stat">{profile.career_matches}</span> career matches &middot;{" "}
          <span className="font-stat">
            {profile.career_wins}–{profile.career_losses}
          </span>
          {profile.career_win_pct !== null && ` (${(profile.career_win_pct * 100).toFixed(1)}%)`}
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Card>
          <CardHeader>
            <CardTitle>Current Elo</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-stat-xl text-accent-blue">
              {profile.current_elo !== null ? (
                <AnimatedNumber value={profile.current_elo} />
              ) : (
                "—"
              )}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Peak Elo</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-stat-xl text-accent-gold">
              {profile.peak_elo !== null ? <AnimatedNumber value={profile.peak_elo} /> : "—"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Grand Slam record</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-stat-lg">
              {profile.grand_slam_stats.wins}–{profile.grand_slam_stats.matches - profile.grand_slam_stats.wins}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Grand Slam win %</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-stat-lg">
              {profile.grand_slam_stats.win_pct !== null
                ? `${(profile.grand_slam_stats.win_pct * 100).toFixed(1)}%`
                : "—"}
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Elo over career</CardTitle>
        </CardHeader>
        <CardContent>
          <EloTimelineChart timeline={profile.elo_timeline} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Surface record</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-3 gap-4">
          {Object.entries(profile.surface_stats).map(([surface, stat]) => (
            <div key={surface}>
              <SurfaceBadge surface={surface} />
              <p className="text-stat-lg mt-2">
                {stat.wins}–{stat.matches - stat.wins}
                {stat.win_pct !== null && (
                  <span className="text-caption font-sans"> ({(stat.win_pct * 100).toFixed(1)}%)</span>
                )}
              </p>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent form</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {profile.recent_form
            .slice()
            .reverse()
            .map((entry) => (
              <div
                key={entry.match_id}
                className="flex items-center justify-between text-sm py-1.5 border-b border-border/40 last:border-0"
              >
                <span className="flex items-center gap-2">
                  <Badge variant={entry.won ? "success" : "danger"} className="w-6 justify-center">
                    {entry.won ? "W" : "L"}
                  </Badge>
                  vs {entry.opponent}
                </span>
                <span className="text-caption">
                  {entry.tournament} &middot; {entry.surface}
                </span>
              </div>
            ))}
        </CardContent>
      </Card>
    </div>
  );
}