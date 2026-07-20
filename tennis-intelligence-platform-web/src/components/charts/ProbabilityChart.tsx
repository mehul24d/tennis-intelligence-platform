"use client";

import { useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ReferenceLine,
  Brush,
  TooltipProps,
} from "recharts";
import { Download } from "lucide-react";
import { MatchReplayResponse, EngineKey, ENGINE_LABELS, ENGINE_POINT_FIELD } from "@/types/match";
import { Button } from "@/components/ui/button";

export const ENGINE_COLORS: Record<EngineKey, string> = {
  markov: "#7F77DD",
  ml_mc: "#1D9E75",
  ml_informed_unsmoothed: "#EF9F27",
  ml_informed_smoothed: "#378ADD",
  hybrid: "#D4537E",
};

const ALL_ENGINES: EngineKey[] = [
  "markov",
  "ml_mc",
  "ml_informed_unsmoothed",
  "ml_informed_smoothed",
  "hybrid",
];

const SET_SHADE_COLORS = ["rgba(255,255,255,0.015)", "rgba(255,255,255,0.04)"];

interface ChartRow {
  point_index: number;
  set1: number;
  set2: number;
  gm1: number;
  gm2: number;
  [key: string]: number;
}

/**
 * Custom tooltip surfacing everything the design brief calls for that this
 * response actually contains: point number, games, sets, and every visible
 * engine's win probability at that point. Server identity and the literal
 * score string (e.g. "40-15") lived in the point-timeline endpoint, which was
 * removed from this app due to unresolved data-quality issues — so this
 * tooltip deliberately does NOT fabricate those fields; it shows only what's
 * genuinely backed by the replay response.
 */
function ChartTooltip({
  active,
  payload,
  label,
  player1Name,
}: TooltipProps<number, string> & { player1Name: string }) {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0].payload as ChartRow;

  return (
    <div className="glass rounded-lg px-4 py-3 text-xs shadow-xl min-w-[200px]">
      <p className="text-label mb-1">Point {label}</p>
      <p className="text-caption mb-2">
        Sets {row.set1}–{row.set2} &middot; Games {row.gm1}–{row.gm2}
      </p>
      <div className="space-y-1">
        {payload.map((entry) => (
          <div key={entry.dataKey} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-muted-foreground">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: entry.color }} />
              {ENGINE_LABELS[entry.dataKey as EngineKey]}
            </span>
            <span className="font-stat font-semibold">{(entry.value as number).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <p className="text-caption mt-2 pt-2 border-t border-border/50">{player1Name} win probability</p>
    </div>
  );
}

export function ProbabilityChart({ match }: { match: MatchReplayResponse }) {
  const [visibleEngines, setVisibleEngines] = useState<Set<EngineKey>>(new Set(ALL_ENGINES));
  const chartWrapperRef = useRef<HTMLDivElement>(null);

  const data: ChartRow[] = useMemo(() => {
    const point0: ChartRow = { point_index: 0, set1: 0, set2: 0, gm1: 0, gm2: 0 };
    for (const engine of ALL_ENGINES) {
      const val = match.prematch[engine];
      if (val !== null) point0[engine] = val * 100;
    }

    const rows: ChartRow[] = match.points.map((p) => {
      const row = {
        point_index: p.point_index,
        set1: p.set1, set2: p.set2, gm1: p.gm1, gm2: p.gm2,
      } as ChartRow;
      for (const engine of ALL_ENGINES) {
        row[engine] = p[ENGINE_POINT_FIELD[engine]] * 100;
      }
      return row;
    });

    if (rows.length > 0) {
      const last = rows[rows.length - 1];
      const p1Won = match.winner === match.player1.name;
      for (const engine of ALL_ENGINES) {
        last[engine] = p1Won ? 100 : 0;
      }
    }

    return [point0, ...rows];
  }, [match]);

  const toggleEngine = (engine: EngineKey) => {
    setVisibleEngines((prev) => {
      const next = new Set(prev);
      if (next.has(engine)) next.delete(engine);
      else next.add(engine);
      return next;
    });
  };

  const handleExportPng = () => {
    const svg = chartWrapperRef.current?.querySelector("svg");
    if (!svg) return;
    const svgData = new XMLSerializer().serializeToString(svg);
    const canvas = document.createElement("canvas");
    const bbox = svg.getBoundingClientRect();
    canvas.width = bbox.width * 2;
    canvas.height = bbox.height * 2;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = new Image();
    const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(svgBlob);
    img.onload = () => {
      ctx.fillStyle = "#0a0e17";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      const pngUrl = canvas.toDataURL("image/png");
      const link = document.createElement("a");
      link.href = pngUrl;
      link.download = `${match.match_id}-win-probability.png`;
      link.click();
    };
    img.src = url;
  };

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex flex-wrap gap-2">
          {ALL_ENGINES.map((engine) => {
            const active = visibleEngines.has(engine);
            return (
              <button
                key={engine}
                onClick={() => toggleEngine(engine)}
                className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-all"
                style={{
                  borderColor: active ? ENGINE_COLORS[engine] : "hsl(var(--border))",
                  opacity: active ? 1 : 0.4,
                  backgroundColor: active ? `${ENGINE_COLORS[engine]}14` : "transparent",
                }}
              >
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: ENGINE_COLORS[engine] }} />
                {ENGINE_LABELS[engine]}
              </button>
            );
          })}
        </div>
        <Button variant="outline" size="sm" onClick={handleExportPng}>
          <Download className="h-3.5 w-3.5" />
          Export PNG
        </Button>
      </div>

      <div ref={chartWrapperRef}>
        <ResponsiveContainer width="100%" height={440}>
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />

            {match.set_boundaries.map((boundary, i) => {
              const prevIdx = i === 0 ? 0 : match.set_boundaries[i - 1].point_index;
              return (
                <ReferenceArea
                  key={`set-${boundary.set_number}`}
                  x1={prevIdx}
                  x2={boundary.point_index}
                  fill={SET_SHADE_COLORS[i % 2]}
                  ifOverflow="extendDomain"
                />
              );
            })}

            {match.set_boundaries.map((boundary) => (
              <ReferenceLine
                key={`boundary-${boundary.set_number}`}
                x={boundary.point_index}
                stroke="hsl(var(--border))"
                strokeDasharray="4 4"
                label={{
                  value: `Set ${boundary.set_number}`,
                  position: "top",
                  fill: "hsl(var(--muted-foreground))",
                  fontSize: 11,
                }}
              />
            ))}

            <ReferenceLine y={50} stroke="hsl(var(--muted-foreground))" strokeDasharray="2 2" opacity={0.4} />

            <XAxis
              dataKey="point_index"
              stroke="hsl(var(--muted-foreground))"
              tick={{ fontSize: 11 }}
              label={{
                value: "Point",
                position: "insideBottom",
                offset: -5,
                fontSize: 12,
                fill: "hsl(var(--muted-foreground))",
              }}
            />
            <YAxis
              domain={[0, 100]}
              stroke="hsl(var(--muted-foreground))"
              tick={{ fontSize: 11 }}
              label={{
                value: `${match.player1.name} win probability (%)`,
                angle: -90,
                position: "insideLeft",
                fontSize: 12,
                fill: "hsl(var(--muted-foreground))",
              }}
            />

            <Tooltip content={<ChartTooltip player1Name={match.player1.name} />} />

            {ALL_ENGINES.filter((e) => visibleEngines.has(e)).map((engine, i) => (
              <Line
                key={engine}
                type="monotone"
                dataKey={engine}
                name={ENGINE_LABELS[engine]}
                stroke={ENGINE_COLORS[engine]}
                strokeWidth={2}
                dot={false}
                connectNulls={false}
                isAnimationActive
                animationDuration={900}
                animationEasing="ease-out"
                animationBegin={i * 80}
              />
            ))}

            <Brush
              dataKey="point_index"
              height={28}
              stroke="hsl(var(--accent-blue))"
              fill="hsl(var(--background-elevated))"
              travellerWidth={8}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-caption text-center mt-1">
        Drag the handles below to zoom into any stretch of the match.
      </p>
    </div>
  );
}