"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  SortingState,
} from "@tanstack/react-table";
import { useFullMatchList } from "@/hooks/useMatchList";
import { FullMatchSummaryRow } from "@/types/fullMatchList";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SurfaceBadge } from "@/components/ui/SurfaceBadge";
import { Skeleton } from "@/components/ui/skeleton";

const columnHelper = createColumnHelper<FullMatchSummaryRow>();

const columns = [
  columnHelper.accessor("tournament", { header: "Tournament" }),
  columnHelper.accessor("year", { header: "Year" }),
  columnHelper.accessor("surface", {
    header: "Surface",
    cell: (info) => <SurfaceBadge surface={info.getValue()} />,
  }),
  columnHelper.accessor("round", { header: "Round" }),
  columnHelper.accessor("winner", {
    header: "Winner",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  columnHelper.accessor("loser", { header: "Loser" }),
  columnHelper.accessor("final_score", {
    header: "Score",
    cell: (info) => <span className="font-stat text-caption">{info.getValue() ?? "—"}</span>,
  }),
  columnHelper.accessor("has_replay_data", {
    header: "Data",
    cell: (info) =>
      info.getValue() ? (
        <Badge variant="success">Full replay</Badge>
      ) : (
        <Badge>Brief score only</Badge>
      ),
  }),
];

export default function MatchExplorerPage() {
  const [player, setPlayer] = useState("");
  const [surface, setSurface] = useState("");
  const [year, setYear] = useState("");
  const [sorting, setSorting] = useState<SortingState>([]);
  const [offset, setOffset] = useState(0);
  const limit = 25;

  const filters = useMemo(
    () => ({
      player: player || undefined,
      surface: surface || undefined,
      year: year ? Number(year) : undefined,
      limit,
      offset,
    }),
    [player, surface, year, offset]
  );

  const { data, isLoading, isError } = useFullMatchList(filters);

  const table = useReactTable({
    data: data?.matches ?? [],
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="mx-auto max-w-6xl p-8 space-y-4">
      <div>
        <h1 className="text-page-title">Match Explorer</h1>
        <p className="text-caption mt-1">
          Every match in the corpus. Full replay where point-by-point data exists —
          score and Elo only where it doesn&apos;t.
        </p>
      </div>

      <div className="flex flex-wrap gap-3">
        <Input
          placeholder="Player name"
          value={player}
          onChange={(e) => {
            setPlayer(e.target.value);
            setOffset(0);
          }}
          className="max-w-xs"
        />
        <Input
          placeholder="Surface (Hard, Clay, Grass)"
          value={surface}
          onChange={(e) => {
            setSurface(e.target.value);
            setOffset(0);
          }}
          className="max-w-xs"
        />
        <Input
          placeholder="Year"
          value={year}
          onChange={(e) => {
            setYear(e.target.value);
            setOffset(0);
          }}
          className="max-w-[120px]"
        />
      </div>

      <Card className="overflow-x-auto">
        {isLoading && (
          <div className="p-4 space-y-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        )}
        {isError && <div className="p-8 text-sm text-accent-red">Couldn&apos;t load matches.</div>}
        {data && (
          <table className="w-full text-sm">
            <thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id} className="border-b border-border">
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      className="cursor-pointer select-none px-4 py-3 text-left text-label"
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() === "asc" && " ↑"}
                      {header.column.getIsSorted() === "desc" && " ↓"}
                    </th>
                  ))}
                  <th className="px-4 py-3" />
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr key={row.id} className="border-b border-border/50 hover:bg-muted/40 transition-colors">
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-4 py-3">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                  <td className="px-4 py-3">
                    {row.original.has_replay_data ? (
                      <Link
                        href={`/match/${encodeURIComponent(row.original.match_id)}`}
                        className="text-accent-blue hover:underline font-medium"
                      >
                        Open analysis
                      </Link>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      {data && (
        <div className="flex items-center justify-between text-caption">
          <span>
            Showing {offset + 1}–{Math.min(offset + limit, data.total)} of{" "}
            <span className="font-stat">{data.total.toLocaleString()}</span>
          </span>
          <div className="flex gap-2">
            <button
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - limit))}
              className="rounded-md border border-border px-3 py-1 hover:border-accent-blue/40 transition-colors disabled:opacity-40 disabled:hover:border-border"
            >
              Previous
            </button>
            <button
              disabled={offset + limit >= data.total}
              onClick={() => setOffset(offset + limit)}
              className="rounded-md border border-border px-3 py-1 hover:border-accent-blue/40 transition-colors disabled:opacity-40 disabled:hover:border-border"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}