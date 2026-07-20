"use client";

import { useEffect, useRef, useState } from "react";
import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight } from "lucide-react";
import { useMatchReplay } from "@/hooks/useMatch";
import { LiveReplayChart } from "@/components/charts/LiveReplayChart";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/utils/cn";

const SPEEDS = [0.5, 1, 2, 4];

export default function LiveReplayPage({ params }: { params: { matchId: string } }) {
  const matchId = decodeURIComponent(params.matchId);
  const { data: match, isLoading, isError } = useMatchReplay(matchId);

  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  useEffect(() => {
    if (!isPlaying || !match) return;
    intervalRef.current = setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev >= match.n_points) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 350 / speed);
    return () => clearInterval(intervalRef.current);
  }, [isPlaying, speed, match]);

  if (isLoading) {
    return (
      <div className="mx-auto max-w-5xl p-8 space-y-4">
        <Skeleton className="h-8 w-80" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  if (isError || !match) {
    return (
      <div className="mx-auto max-w-5xl p-8">
        <p className="text-sm text-accent-red">Couldn&apos;t load this match.</p>
      </div>
    );
  }

  const currentPoint = match.points[Math.min(currentIndex, match.points.length) - 1];
  const isFinished = currentIndex >= match.n_points;

  const jumpToSet = (setNumber: number) => {
    const boundary = match.set_boundaries.find((b) => b.set_number === setNumber);
    if (boundary) {
      setCurrentIndex(boundary.point_index);
      setIsPlaying(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl p-8 space-y-4">
      <div>
        <h1 className="text-page-title">Live Replay</h1>
        <p className="text-caption mt-1">
          {match.player1.name} vs {match.player2.name}
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>
            {isFinished ? "Final" : `Point ${currentIndex} of ${match.n_points}`}
            {isFinished && (
              <span className="ml-2">
                <Badge variant="gold">Winner: {match.winner}</Badge>
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <LiveReplayChart match={match} currentIndex={currentIndex} />

          {currentPoint && !isFinished && (
            <div className="flex items-center justify-center gap-6 mt-2 text-sm">
              <span className="font-stat">
                Sets {currentPoint.set1}–{currentPoint.set2}
              </span>
              <span className="font-stat">
                Games {currentPoint.gm1}–{currentPoint.gm2}
              </span>
            </div>
          )}

          <input
            type="range"
            min={0}
            max={match.n_points}
            value={currentIndex}
            onChange={(e) => {
              setCurrentIndex(Number(e.target.value));
              setIsPlaying(false);
            }}
            className="w-full mt-4 accent-accent-blue"
          />

          <div className="flex items-center justify-center gap-2 mt-4">
            <button
              onClick={() => setCurrentIndex(0)}
              className="p-2 rounded-md border border-border hover:border-accent-blue/40 transition-colors"
              aria-label="Jump to start"
            >
              <SkipBack className="h-4 w-4" />
            </button>
            <button
              onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))}
              className="p-2 rounded-md border border-border hover:border-accent-blue/40 transition-colors"
              aria-label="Previous point"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              onClick={() => setIsPlaying((p) => !p)}
              className="p-3 rounded-full bg-accent-blue text-white hover:bg-accent-blue/90 transition-colors"
              aria-label={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
            </button>
            <button
              onClick={() => setCurrentIndex((i) => Math.min(match.n_points, i + 1))}
              className="p-2 rounded-md border border-border hover:border-accent-blue/40 transition-colors"
              aria-label="Next point"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
            <button
              onClick={() => setCurrentIndex(match.n_points)}
              className="p-2 rounded-md border border-border hover:border-accent-blue/40 transition-colors"
              aria-label="Jump to end"
            >
              <SkipForward className="h-4 w-4" />
            </button>
          </div>

          <div className="flex items-center justify-center gap-1.5 mt-3">
            <span className="text-label mr-1">Speed</span>
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={cn(
                  "px-2 py-0.5 rounded text-xs font-medium transition-colors",
                  speed === s
                    ? "bg-accent-blue/15 text-accent-blue"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {s}x
              </button>
            ))}
          </div>

          {match.set_boundaries.length > 0 && (
            <div className="flex items-center justify-center gap-1.5 mt-3">
              <span className="text-label mr-1">Jump to set</span>
              {match.set_boundaries.map((b) => (
                <button
                  key={b.set_number}
                  onClick={() => jumpToSet(b.set_number)}
                  className="px-2 py-0.5 rounded text-xs font-medium text-muted-foreground hover:text-accent-blue transition-colors border border-border"
                >
                  Set {b.set_number}
                </button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}