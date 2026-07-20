"use client";

import { useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  createColumnHelper,
} from "@tanstack/react-table";
import { usePointTimeline } from "@/hooks/useMatch";
import { PointTimelineEntry, PointTimelineFilters } from "@/types/matchAnalysis";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const columnHelper = createColumnHelper<PointTimelineEntry>();

const columns = [
  columnHelper.accessor("point_index", { header: "#" }),
  columnHelper.accessor("server", { header: "Server" }),
  columnHelper.accessor("score_before", { header: "Score", cell: (info) => info.getValue() ?? "—" }),
  columnHelper.accessor("winner", { header: "Won by" }),
  columnHelper.accessor("probability_swing", {
    header: "Swing",
    cell: (info) => {
      const swing = info.getValue();
      const isLargest = info.row.original.is_largest_swing;
      return (
        <span className={isLargest ? "font-medium text-accent" : ""}>
          {(swing * 100).toFixed(1)}%{isLargest && " ★"}
        </span>
      );
    },
  }),
  columnHelper.display({
    id: "flags",
    header: "Situation",
    cell: (info) => {
      const row = info.row.original;
      return (
        <div className="flex gap-1">
          {row.is_break_point && <Badge variant="danger">BP</Badge>}
          {row.is_set_point && <Badge variant="accent">SP</Badge>}
          {row.is_match_point && <Badge variant="success">MP</Badge>}
          {row.is_tiebreak_point && <Badge>TB</Badge>}
        </div>
      );
    },
  }),
];

type FilterKey = "all" | "break_points_only" | "set_points_only" | "match_points_only" | "tiebreak_only";

const FILTER_OPTIONS: { key: FilterKey; label: string }[] = [
  { key: "all", label: "All points" },
  { key: "break_points_only", label: "Break points" },
  { key: "set_points_only", label: "Set points" },
  { key: "match_points_only", label: "Match points" },
  { key: "tiebreak_only", label: "Tiebreak" },
];

function buildFilters(filter: FilterKey, minSwing: number | undefined): PointTimelineFilters {
  // Built explicitly, field by field, rather than a dynamic computed key
  // ({ [filter]: true }) — that pattern doesn't type-check against
  // PointTimelineFilters' concrete named fields, the same class of error just
  // fixed in rankings/page.tsx.
  return {
    break_points_only: filter === "break_points_only" || undefined,
    set_points_only: filter === "set_points_only" || undefined,
    match_points_only: filter === "match_points_only" || undefined,
    tiebreak_only: filter === "tiebreak_only" || undefined,
    min_swing: minSwing,
  };
}

export function PointTimeline({ matchId }: { matchId: string }) {
  const [filter, setFilter] = useState<FilterKey>("all");
  const [minSwing, setMinSwing] = useState<number | undefined>(undefined);

  const { data, isLoading } = usePointTimeline(matchId, buildFilters(filter, minSwing));

  const table = useReactTable({
    data: data?.points ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 mb-3">
        {FILTER_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            onClick={() => setFilter(opt.key)}
            className={`rounded-md border px-3 py-1 text-xs ${
              filter === opt.key ? "border-accent text-accent" : "border-border text-muted-foreground"
            }`}
          >
            {opt.label}
          </button>
        ))}
        <button
          onClick={() => setMinSwing(minSwing === 0.05 ? undefined : 0.05)}
          className={`rounded-md border px-3 py-1 text-xs ${
            minSwing === 0.05 ? "border-accent text-accent" : "border-border text-muted-foreground"
          }`}
        >
          Swing &gt; 5%
        </button>
        <button
          onClick={() => setMinSwing(minSwing === 0.1 ? undefined : 0.1)}
          className={`rounded-md border px-3 py-1 text-xs ${
            minSwing === 0.1 ? "border-accent text-accent" : "border-border text-muted-foreground"
          }`}
        >
          Swing &gt; 10%
        </button>
      </div>

      <Card className="overflow-x-auto max-h-[480px] overflow-y-auto">
        {isLoading && <div className="p-6 text-sm text-muted-foreground">Loading…</div>}
        {data && (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card">
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id} className="border-b border-border">
                  {hg.headers.map((header) => (
                    <th key={header.id} className="px-3 py-2 text-left font-medium text-muted-foreground">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr key={row.id} className="border-b border-border/50">
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-3 py-2">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
      {data && (
        <p className="text-xs text-muted-foreground mt-2">
          Showing {data.n_points_returned} of {data.n_points_total} points
        </p>
      )}
    </div>
  );
}