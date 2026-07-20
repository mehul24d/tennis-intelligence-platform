"use client";

import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { EloTimelinePoint } from "@/types/playerProfile";

export function EloTimelineChart({ timeline }: { timeline: EloTimelinePoint[] }) {
  const data = timeline.map((point, i) => ({
    index: i,
    elo: point.elo,
    surface_elo: point.surface_elo,
    date: point.date,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
        <XAxis dataKey="index" stroke="hsl(var(--muted-foreground))" tick={{ fontSize: 11 }} hide />
        <YAxis
          domain={["dataMin - 50", "dataMax + 50"]}
          stroke="hsl(var(--muted-foreground))"
          tick={{ fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelFormatter={(_, payload) => payload?.[0]?.payload?.date ?? ""}
        />
        <Line type="monotone" dataKey="elo" stroke="#378ADD" strokeWidth={2} dot={false} name="Elo" />
      </LineChart>
    </ResponsiveContainer>
  );
}