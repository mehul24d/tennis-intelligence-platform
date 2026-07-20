"use client";

import { useState } from "react";
import Link from "next/link";
import {
  useCurrentEloRankings,
  usePeakEloRankings,
  useSurfaceEloRankings,
  usePeakSurfaceEloRankings,
  useBiggestUpsets,
} from "@/hooks/useRankings";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/utils/cn";
import {
  CurrentEloEntry,
  PeakEloEntry,
  SurfaceEloEntry,
  PeakSurfaceEloEntry,
  UpsetEntry,
} from "@/types/rankings";

type PlayerRankRow = CurrentEloEntry | PeakEloEntry | SurfaceEloEntry | PeakSurfaceEloEntry;
type AnyRankRow = PlayerRankRow | UpsetEntry;

function isUpsetRow(row: AnyRankRow): row is UpsetEntry {
  return "match_id" in row;
}

// Top-3 ranks get a gold/silver/bronze rank-number treatment — a common
// leaderboard convention (per the brief's "beautiful leaderboard" requirement)
// that instantly signals "this row is special" without needing a legend.
const RANK_COLORS: Record<number, string> = {
  1: "text-accent-gold",
  2: "text-slate-300",
  3: "text-amber-700",
};

function RankTable<T extends AnyRankRow>({
  rows,
  getValue,
  valueLabel,
  extraLabel,
}: {
  rows: T[];
  getValue: (row: T) => number;
  valueLabel: string;
  extraLabel?: string;
}) {
  return (
    <Card className="overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left">
            <th className="px-4 py-3 w-12 text-label">#</th>
            <th className="px-4 py-3 text-label">Player</th>
            <th className="px-4 py-3 text-right text-label">{valueLabel}</th>
            {extraLabel && <th className="px-4 py-3 text-right text-label">{extraLabel}</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={isUpsetRow(row) ? row.match_id : row.player_id}
              className="border-b border-border/50 hover:bg-muted/40 transition-colors"
            >
              <td className={cn("px-4 py-3 font-stat font-semibold", RANK_COLORS[row.rank] ?? "text-muted-foreground")}>
                {row.rank}
              </td>
              <td className="px-4 py-3">
                {isUpsetRow(row) ? (
                  <span>
                    <span className="font-medium">{row.winner_name}</span>
                    <span className="text-muted-foreground"> d. {row.loser_name}</span>
                  </span>
                ) : (
                  <Link
                    href={`/players/${encodeURIComponent(row.player_id)}`}
                    className="hover:text-accent-blue font-medium transition-colors"
                  >
                    {row.player_name}
                  </Link>
                )}
              </td>
              <td className="px-4 py-3 text-right font-stat font-semibold">{getValue(row).toFixed(1)}</td>
              {extraLabel && (
                <td className="px-4 py-3 text-right text-caption font-stat">
                  {"date_achieved" in row && row.date_achieved
                    ? new Date(row.date_achieved).getFullYear()
                    : ""}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function TableSkeleton() {
  return (
    <Card className="p-4 space-y-2">
      {Array.from({ length: 8 }).map((_, i) => (
        <Skeleton key={i} className="h-9 w-full" />
      ))}
    </Card>
  );
}

export default function RankingsPage() {
  const [surface, setSurface] = useState("Hard");

  const current = useCurrentEloRankings(50);
  const peak = usePeakEloRankings(50);
  const surfaceCurrent = useSurfaceEloRankings(surface, 50);
  const surfacePeak = usePeakSurfaceEloRankings(surface, 50);
  const upsets = useBiggestUpsets(50);

  return (
    <div className="mx-auto max-w-4xl p-8 space-y-4">
      <h1 className="text-page-title">Rankings</h1>
      <p className="text-caption -mt-2">Elo by player, surface, and career peak.</p>

      <Tabs defaultValue="current">
        <TabsList>
          <TabsTrigger value="current">Current Elo</TabsTrigger>
          <TabsTrigger value="peak">Peak Elo</TabsTrigger>
          <TabsTrigger value="surface">Surface Elo</TabsTrigger>
          <TabsTrigger value="upsets">Biggest Upsets</TabsTrigger>
        </TabsList>

        <TabsContent value="current">
          {current.data ? (
            <RankTable rows={current.data} getValue={(r) => r.elo} valueLabel="Elo" />
          ) : (
            <TableSkeleton />
          )}
        </TabsContent>

        <TabsContent value="peak">
          {peak.data ? (
            <RankTable rows={peak.data} getValue={(r) => r.peak_elo} valueLabel="Peak Elo" extraLabel="Year" />
          ) : (
            <TableSkeleton />
          )}
        </TabsContent>

        <TabsContent value="surface">
          <div className="flex gap-2 mb-3">
            {["Hard", "Clay", "Grass"].map((s) => (
              <button
                key={s}
                onClick={() => setSurface(s)}
                className={cn(
                  "rounded-md border px-3 py-1 text-sm font-medium transition-colors",
                  surface === s
                    ? "border-accent-blue/40 bg-accent-blue/10 text-accent-blue"
                    : "border-border text-muted-foreground hover:border-accent-blue/20"
                )}
              >
                {s}
              </button>
            ))}
          </div>
          {surfaceCurrent.data ? (
            <div className="space-y-6">
              <div>
                <p className="text-label mb-2">Current {surface} Elo</p>
                <RankTable rows={surfaceCurrent.data} getValue={(r) => r.surface_elo} valueLabel="Elo" />
              </div>
              {surfacePeak.data && (
                <div>
                  <p className="text-label mb-2">Peak {surface} Elo (all-time)</p>
                  <RankTable
                    rows={surfacePeak.data}
                    getValue={(r) => r.peak_surface_elo}
                    valueLabel="Peak Elo"
                    extraLabel="Year"
                  />
                </div>
              )}
            </div>
          ) : (
            <TableSkeleton />
          )}
        </TabsContent>

        <TabsContent value="upsets">
          {upsets.data ? (
            <RankTable rows={upsets.data} getValue={(r) => r.elo_gap} valueLabel="Elo gap" />
          ) : (
            <TableSkeleton />
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}