"use client";

import { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  ReferenceDot,
} from "recharts";
import { MatchReplayResponse, EngineKey, ENGINE_POINT_FIELD } from "@/types/match";
import { ENGINE_COLORS } from "@/components/charts/ProbabilityChart";

interface ChartRow {
  point_index: number;
  [key: string]: number;
}

/**
 * A progressive-reveal variant of ProbabilityChart, purpose-built for live
 * replay: only shows points up to `currentIndex`, with a highlighted dot
 * marking the current position — deliberately kept separate from the
 * publication-quality ProbabilityChart (which always renders the full match
 * and has its own zoom/export/tooltip machinery not needed here) rather than
 * overloading one component with two very different responsibilities.
 */
export function LiveReplayChart({
  match,
  currentIndex,
  engine = "ml_informed_smoothed",
}: {
  match: MatchReplayResponse;
  currentIndex: number;
  engine?: EngineKey;
}) {
  const fullData: ChartRow[] = useMemo(() => {
    const point0: ChartRow = { point_index: 0 };
    const val = match.prematch[engine];
    if (val !== null) point0[engine] = val * 100;

    const rows: ChartRow[] = match.points.map((p) => ({
      point_index: p.point_index,
      [engine]: p[ENGINE_POINT_FIELD[engine]] * 100,
    }));

    if (rows.length > 0) {
      const last = rows[rows.length - 1];
      const p1Won = match.winner === match.player1.name;
      last[engine] = p1Won ? 100 : 0;
    }

    return [point0, ...rows];
  }, [match, engine]);

  const visibleData = fullData.filter((r) => r.point_index <= currentIndex);
  const currentRow = visibleData[visibleData.length - 1];
  const color = ENGINE_COLORS[engine];

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={visibleData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
        {match.set_boundaries
          .filter((b) => b.point_index <= currentIndex)
          .map((boundary) => (
            <ReferenceLine
              key={boundary.set_number}
              x={boundary.point_index}
              stroke="hsl(var(--border))"
              strokeDasharray="4 4"
            />
          ))}
        <ReferenceLine y={50} stroke="hsl(var(--muted-foreground))" strokeDasharray="2 2" opacity={0.4} />
        <XAxis
          type="number"
          dataKey="point_index"
          domain={[0, match.n_points]}
          stroke="hsl(var(--muted-foreground))"
          tick={{ fontSize: 11 }}
        />
        <YAxis
          domain={[0, 100]}
          stroke="hsl(var(--muted-foreground))"
          tick={{ fontSize: 11 }}
          label={{
            value: `${match.player1.name} win %`,
            angle: -90,
            position: "insideLeft",
            fontSize: 11,
            fill: "hsl(var(--muted-foreground))",
          }}
        />
        <Line
          type="monotone"
          dataKey={engine}
          stroke={color}
          strokeWidth={2.5}
          dot={false}
          isAnimationActive={false}
        />
        {currentRow && (
          <ReferenceDot
            x={currentRow.point_index}
            y={currentRow[engine]}
            r={5}
            fill={color}
            stroke="hsl(var(--background))"
            strokeWidth={2}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}