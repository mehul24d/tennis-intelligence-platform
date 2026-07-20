"use client";

import { useState } from "react";
import { PlayerCombobox } from "@/components/players/PlayerCombobox";
import { ComparisonRadar } from "@/components/charts/ComparisonRadar";
import { usePlayerProfile, useHeadToHead } from "@/hooks/usePlayer";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { PlayerSearchResult } from "@/types/playerProfile";

function StatRow({ label, a, b }: { label: string; a: string; b: string }) {
  return (
    <div className="grid grid-cols-3 items-center py-2 border-b border-border/40 last:border-0">
      <span className="text-stat-lg text-right pr-4">{a}</span>
      <span className="text-label text-center">{label}</span>
      <span className="text-stat-lg pl-4">{b}</span>
    </div>
  );
}

export default function ComparePlayersPage() {
  const [playerA, setPlayerA] = useState<PlayerSearchResult | null>(null);
  const [playerB, setPlayerB] = useState<PlayerSearchResult | null>(null);

  const profileA = usePlayerProfile(playerA?.player_id);
  const profileB = usePlayerProfile(playerB?.player_id);
  const h2h = useHeadToHead(playerA?.player_id, playerB?.player_id);

  const bothSelected = Boolean(playerA && playerB);
  const bothLoaded = Boolean(profileA.data && profileB.data);

  return (
    <div className="mx-auto max-w-5xl p-8 space-y-6">
      <div>
        <h1 className="text-page-title">Compare Players</h1>
        <p className="text-caption mt-1">
          Side-by-side career stats, surface performance, and head-to-head record.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <PlayerCombobox label="Player A" selected={playerA} onSelect={setPlayerA} />
        <PlayerCombobox label="Player B" selected={playerB} onSelect={setPlayerB} />
      </div>

      {!bothSelected && (
        <Card>
          <CardContent className="pt-5 text-center text-caption">
            Select two players above to compare them.
          </CardContent>
        </Card>
      )}

      {bothSelected && !bothLoaded && (
        <div className="space-y-4">
          <Skeleton className="h-96 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      )}

      {bothSelected && bothLoaded && profileA.data && profileB.data && (
        (() => {
          const dataA = profileA.data;
          const dataB = profileB.data;
          return (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Profile comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <ComparisonRadar playerA={dataA} playerB={dataB} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Career stats</CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <div className="grid grid-cols-3 pb-2 mb-1 border-b border-border">
                <span className="text-section-title text-right pr-4">{dataA.player_name}</span>
                <span />
                <span className="text-section-title pl-4">{dataB.player_name}</span>
              </div>
              <StatRow
                label="Current Elo"
                a={dataA.current_elo?.toFixed(0) ?? "—"}
                b={dataB.current_elo?.toFixed(0) ?? "—"}
              />
              <StatRow
                label="Peak Elo"
                a={dataA.peak_elo?.toFixed(0) ?? "—"}
                b={dataB.peak_elo?.toFixed(0) ?? "—"}
              />
              <StatRow
                label="Career Record"
                a={`${dataA.career_wins}–${dataA.career_losses}`}
                b={`${dataB.career_wins}–${dataB.career_losses}`}
              />
              <StatRow
                label="Career Win %"
                a={dataA.career_win_pct !== null ? `${(dataA.career_win_pct * 100).toFixed(1)}%` : "—"}
                b={dataB.career_win_pct !== null ? `${(dataB.career_win_pct * 100).toFixed(1)}%` : "—"}
              />
              <StatRow
                label="Grand Slam Record"
                a={`${dataA.grand_slam_stats.wins}–${
                  dataA.grand_slam_stats.matches - dataA.grand_slam_stats.wins
                }`}
                b={`${dataB.grand_slam_stats.wins}–${
                  dataB.grand_slam_stats.matches - dataB.grand_slam_stats.wins
                }`}
              />
            </CardContent>
          </Card>

          {h2h.data && (
            <Card>
              <CardHeader>
                <CardTitle>Head-to-head</CardTitle>
              </CardHeader>
              <CardContent>
                {h2h.data.matches.length === 0 ? (
                  <p className="text-caption">These two players have never met.</p>
                ) : (
                  <>
                    <div className="flex items-center justify-center gap-6 mb-4">
                      <span className="text-stat-xl text-accent-blue">{h2h.data.a_wins}</span>
                      <span className="text-caption">head-to-head</span>
                      <span className="text-stat-xl" style={{ color: "#D4537E" }}>
                        {h2h.data.b_wins}
                      </span>
                    </div>
                    <div className="space-y-1">
                      {h2h.data.matches.map((m) => (
                        <div
                          key={m.match_id}
                          className="flex items-center justify-between text-sm py-1.5 border-b border-border/40 last:border-0"
                        >
                          <span>
                            {m.winner_id === playerA?.player_id ? dataA.player_name : dataB.player_name}{" "}
                            <span className="text-muted-foreground">won</span>
                          </span>
                          <span className="text-caption">
                            {m.tournament} &middot; {m.surface}
                            {m.date && ` · ${new Date(m.date).getFullYear()}`}
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </>
          );
        })()
      )}
    </div>
  );
}