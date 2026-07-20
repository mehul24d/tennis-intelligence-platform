"use client";

import Link from "next/link";
import { TrendingUp, Trophy, GitCompare, FlaskConical } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AnimatedNumber } from "@/components/ui/AnimatedNumber";
import { HeroTrajectory } from "@/components/landing/HeroTrajectory";

const NAV_CARDS = [
  {
    href: "/explorer",
    title: "Match Explorer",
    desc: "Every charted match. Filter by surface, round, and year.",
    icon: TrendingUp,
  },
  {
    href: "/rankings",
    title: "Rankings",
    desc: "Current, peak, and surface-specific Elo.",
    icon: Trophy,
  },
  {
    href: "/model-comparison",
    title: "Engine Analysis",
    desc: "Log loss, Brier score, and calibration error, side by side.",
    icon: GitCompare,
  },
  {
    href: "/research",
    title: "Research",
    desc: "Reliability curves and bootstrap confidence intervals.",
    icon: FlaskConical,
  },
];

const STATS = [
  { label: "Prediction engines", value: 5, suffix: "", useGrouping: false },
  { label: "Charted matches", value: 6000, suffix: "+", useGrouping: true },
  { label: "Matches in corpus", value: 198000, suffix: "+", useGrouping: true },
  { label: "Holdout points", value: 356000, suffix: "+", useGrouping: true },
];

export default function HomePage() {
  return (
    <div>
      <div className="court-texture">
        <div className="mx-auto max-w-6xl px-8 py-16 grid grid-cols-1 lg:grid-cols-5 gap-10 items-center">
          <div className="lg:col-span-3 text-center lg:text-left">
            <h1 className="text-display">Tennis Intelligence Platform</h1>
            <p className="mt-4 text-muted-foreground max-w-xl mx-auto lg:mx-0 text-base">
              Five win-probability engines, tested point by point against{" "}
              <span className="font-stat">356,000+</span> holdout points. Analytical
              Markov chains against gradient-boosted classifiers — same matches, same
              metrics, no thumb on the scale.
            </p>
            <div className="flex flex-wrap items-center justify-center lg:justify-start gap-4 mt-8">
              <Button asChild size="default">
                <Link href="/explorer">Explore Matches</Link>
              </Button>
              <Button asChild variant="outline">
                <Link href="/players">Player Analytics</Link>
              </Button>
              <Link
                href="/research"
                className="text-sm font-medium text-muted-foreground hover:text-accent-blue transition-colors underline underline-offset-4 decoration-border"
              >
                Research →
              </Link>
            </div>
          </div>

          <div className="lg:col-span-2">
            <Card className="p-4">
              <HeroTrajectory />
            </Card>
          </div>
        </div>
      </div>
      <div className="surface-accent-line" />

      <div className="mx-auto max-w-6xl p-8 space-y-10">
        <div>
          <p className="text-label mb-3">By the numbers</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {STATS.map((stat) => (
              <Card key={stat.label} className="bg-background-elevated">
                <CardContent className="pt-5 text-center">
                  <p className="text-stat-xl text-accent-blue">
                    <AnimatedNumber value={stat.value} suffix={stat.suffix} useGrouping={stat.useGrouping} />
                  </p>
                  <p className="text-label mt-2">{stat.label}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        <div>
          <p className="text-label mb-3">Explore the platform</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {NAV_CARDS.map((card) => (
              <Link key={card.href} href={card.href}>
                <Card className="h-full">
                  <CardHeader className="pb-2">
                    <card.icon className="h-5 w-5 text-accent-blue mb-2" />
                    <CardTitle className="text-section-title text-foreground normal-case tracking-normal">
                      {card.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-caption">{card.desc}</p>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}