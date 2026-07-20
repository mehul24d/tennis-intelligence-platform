"use client";

import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip,
  Legend,
} from "recharts";
import { PlayerProfileResponse } from "@/types/playerProfile";

/**
 * Compares two players across the dimensions this project's API genuinely
 * exposes for both of them: current Elo, peak Elo, career win %, Grand Slam
 * win %, and per-surface win % (Hard/Clay/Grass, when both players have
 * played that surface). Elo (roughly 1200-2400) and percentages (0-100) are
 * on completely different scales, so every axis is independently normalized
 * to 0-100 RELATIVE TO THE BETTER OF THE TWO PLAYERS on that axis — the
 * standard way radar charts handle mixed units, avoiding an arbitrary
 * absolute Elo scale that would carry no real meaning.
 */
export function ComparisonRadar({
  playerA,
  playerB,
}: {
  playerA: PlayerProfileResponse;
  playerB: PlayerProfileResponse;
}) {
  function normalizedPair(a: number | null, b: number | null): [number, number] {
    if (a === null || b === null) return [0, 0];
    const max = Math.max(a, b, 1e-9);
    return [(a / max) * 100, (b / max) * 100];
  }

  const surfaces = ["Hard", "Clay", "Grass"];
  const axes: { subject: string; a: number; b: number }[] = [];

  const [eloA, eloB] = normalizedPair(playerA.current_elo, playerB.current_elo);
  axes.push({ subject: "Current Elo", a: eloA, b: eloB });

  const [peakA, peakB] = normalizedPair(playerA.peak_elo, playerB.peak_elo);
  axes.push({ subject: "Peak Elo", a: peakA, b: peakB });

  const [winA, winB] = normalizedPair(
    playerA.career_win_pct !== null ? playerA.career_win_pct * 100 : null,
    playerB.career_win_pct !== null ? playerB.career_win_pct * 100 : null
  );
  axes.push({ subject: "Career Win %", a: winA, b: winB });

  const [gsA, gsB] = normalizedPair(
    playerA.grand_slam_stats.win_pct !== null ? playerA.grand_slam_stats.win_pct * 100 : null,
    playerB.grand_slam_stats.win_pct !== null ? playerB.grand_slam_stats.win_pct * 100 : null
  );
  axes.push({ subject: "Grand Slam Win %", a: gsA, b: gsB });

  for (const surface of surfaces) {
    const statA = playerA.surface_stats[surface];
    const statB = playerB.surface_stats[surface];
    if (!statA || !statB) continue;
    const [sA, sB] = normalizedPair(
      statA.win_pct !== null ? statA.win_pct * 100 : null,
      statB.win_pct !== null ? statB.win_pct * 100 : null
    );
    axes.push({ subject: `${surface} Win %`, a: sA, b: sB });
  }

  return (
    <ResponsiveContainer width="100%" height={360}>
      <RadarChart data={axes} outerRadius="75%">
        <PolarGrid stroke="hsl(var(--border))" />
        <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} />
        <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9 }} axisLine={false} />
        <Radar
          name={playerA.player_name}
          dataKey="a"
          stroke="#378ADD"
          fill="#378ADD"
          fillOpacity={0.35}
          isAnimationActive
          animationDuration={800}
        />
        <Radar
          name={playerB.player_name}
          dataKey="b"
          stroke="#D4537E"
          fill="#D4537E"
          fillOpacity={0.35}
          isAnimationActive
          animationDuration={800}
          animationBegin={150}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: 8,
            fontSize: 11,
          }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
      </RadarChart>
    </ResponsiveContainer>
  );
}