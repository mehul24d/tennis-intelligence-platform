"use client";

import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { ReliabilityPoint } from "@/types/researchDashboard";

/**
 * A calibration/reliability diagram: for each probability bin, plots the
 * model's own mean predicted probability against the ACTUAL observed win
 * rate in that bin, connected in bin order via Scatter's own documented
 * `line` prop (the officially supported way to get a connected-scatter plot
 * in Recharts — mixing separate <Line>/<Scatter> elements inside a
 * ComposedChart is a known problem area for this library, confirmed via
 * Recharts' own GitHub issue tracker, so this deliberately avoids that).
 *
 * A perfectly calibrated model falls exactly on the dashed diagonal
 * (predicted == observed); points below the diagonal mean the model is
 * overconfident in that range, points above mean underconfident.
 */
export function ReliabilityDiagram({
  data,
  color,
}: {
  data: ReliabilityPoint[];
  color: string;
}) {
  const chartData = data
    .slice()
    .sort((a, b) => a.mean_predicted - b.mean_predicted)
    .map((d) => ({
      predicted: d.mean_predicted * 100,
      observed: d.observed_win_rate * 100,
      n: d.n,
    }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
        <XAxis
          type="number"
          dataKey="predicted"
          domain={[0, 100]}
          stroke="hsl(var(--muted-foreground))"
          tick={{ fontSize: 10 }}
          label={{
            value: "Predicted win probability (%)",
            position: "insideBottom",
            offset: -5,
            fontSize: 11,
            fill: "hsl(var(--muted-foreground))",
          }}
        />
        <YAxis
          type="number"
          dataKey="observed"
          domain={[0, 100]}
          stroke="hsl(var(--muted-foreground))"
          tick={{ fontSize: 10 }}
          label={{
            value: "Observed win rate (%)",
            angle: -90,
            position: "insideLeft",
            fontSize: 11,
            fill: "hsl(var(--muted-foreground))",
          }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: 8,
            fontSize: 11,
          }}
          formatter={(value: number, name: string) => [
            `${value.toFixed(1)}%`,
            name === "observed" ? "Observed" : "Predicted",
          ]}
        />
        <ReferenceLine
          segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
          stroke="hsl(var(--muted-foreground))"
          strokeDasharray="4 4"
          ifOverflow="extendDomain"
        />
        <Scatter data={chartData} fill={color} line={{ stroke: color, strokeWidth: 2 }} lineType="joint" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}