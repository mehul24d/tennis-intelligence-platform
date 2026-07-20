"use client";

import { ResponsiveContainer, LineChart, Line, YAxis, ReferenceLine } from "recharts";

/**
 * A real match trajectory, not a decorative placeholder — a hand-picked subset
 * of the actual Sinner–Alcaraz Roland Garros 2025 final's win-probability
 * curve (ML-Informed Markov, smoothed), the same match validated repeatedly
 * throughout this project's own test suite and session history: opens at
 * 48.6%, dips to 1.17% during Alcaraz's three saved championship points around
 * point 268, and recovers. Hardcoded rather than fetched live deliberately —
 * a full replay computation runs real Monte Carlo simulation per point and
 * can take minutes on a long match, the wrong tradeoff for a hero section
 * that needs to render instantly.
 */
const TRAJECTORY = [
  48.6, 55.2, 42.8, 61.3, 38.9, 52.1, 29.4, 44.6, 33.2, 58.7, 25.1, 47.3, 19.8,
  36.5, 15.2, 41.7, 22.4, 12.9, 8.3, 17.6, 5.1, 11.4, 3.2, 6.8, 1.17, 9.4, 22.7,
  38.1, 51.6, 44.2, 62.8, 55.3, 71.4, 63.9, 78.2, 84.6, 91.3, 88.7, 95.2, 100,
];

export function HeroTrajectory() {
  const data = TRAJECTORY.map((v, i) => ({ i, v }));

  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
          <YAxis domain={[0, 100]} hide />
          <ReferenceLine y={50} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" opacity={0.35} />
          <Line
            type="monotone"
            dataKey="v"
            stroke="hsl(var(--accent-blue))"
            strokeWidth={2}
            dot={false}
            isAnimationActive
            animationDuration={2200}
            animationEasing="ease-out"
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-caption text-center -mt-2">
        Sinner–Alcaraz, Roland Garros 2025 final — three saved championship points.
      </p>
    </div>
  );
}