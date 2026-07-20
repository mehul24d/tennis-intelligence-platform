"use client";

import { useModelComparison } from "@/hooks/useModelComparison";
import { EnginePipelineDiagram } from "@/components/models/EnginePipelineDiagram";
import { MonteCarloDiagram } from "@/components/models/MonteCarloDiagram";
import { MarkovRecursionDiagram } from "@/components/models/MarkovRecursionDiagram";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { ENGINE_COLORS } from "@/components/charts/ProbabilityChart";
import { EngineKey } from "@/types/match";

// Maps each engine to its OWN dedicated diagram, since the three families of
// engines work fundamentally differently (closed-form recursion vs. repeated
// random simulation vs. Bayesian-updated recursion) and deserve their own
// visual rather than one generic "AI pipeline" graphic reused everywhere.
const ENGINE_DIAGRAMS: Partial<Record<EngineKey, () => React.ReactElement>> = {
  markov: MarkovRecursionDiagram,
  ml_mc: MonteCarloDiagram,
  ml_informed_smoothed: EnginePipelineDiagram,
};

interface EngineExplainer {
  key: EngineKey;
  name: string;
  summary: string;
  howItWorks: string[];
}

const ENGINES: EngineExplainer[] = [
  {
    key: "markov",
    name: "Analytical Markov",
    summary: "A closed-form recursion over the standard tennis scoring hierarchy.",
    howItWorks: [
      "Takes a single fixed serve-win probability per player and recurses it exactly through game → set → match, with no simulation involved.",
      "Fast and exact given its inputs, but cannot react to anything that happens mid-match — the serve probability never updates as the match unfolds.",
    ],
  },
  {
    key: "ml_mc",
    name: "Machine Learning + Monte Carlo",
    summary: "A trained classifier predicts each remaining point; Monte Carlo simulates the rest of the match many times.",
    howItWorks: [
      "At any given match state, a gradient-boosted classifier estimates the probability the server wins the CURRENT point, using momentum, pressure, and streak features.",
      "The match is then simulated forward from that state hundreds of times (this project runs 300 simulations per point by default), and the win probability is simply the fraction of simulated matches player 1 wins.",
    ],
  },
  {
    key: "ml_informed_unsmoothed",
    name: "ML-Informed Markov (Unsmoothed)",
    summary: "Bayesian-updates a pre-match prior toward the live classifier's read on serve strength, point by point.",
    howItWorks: [
      "Starts from a Beta-Binomial prior built by inverting the pre-match ML win probability into a serve-probability distribution, with confidence (effective sample size) scaled by how much real history — career matches, head-to-head meetings, tournament-specific H2H — actually backs that estimate.",
      "Each point, the per-point classifier's read is blended into this posterior via a sensitivity-aware update, then fed through the same exact Markov recursion as the analytical engine — this is the 'unsmoothed' raw version, before the exponential-smoothing pass.",
    ],
  },
  {
    key: "ml_informed_smoothed",
    name: "ML-Informed Markov (Smoothed)",
    summary: "The same engine as above, with exponential smoothing to suppress point-to-point noise.",
    howItWorks: [
      "Identical underlying mechanics to the unsmoothed version, but the resulting probability trajectory is smoothed to avoid visually jarring swings driven by classifier noise on any single point.",
      "This is the best-performing engine on this project's own calibration metrics (lowest LogLoss and Brier of all five, see the live comparison below) and is what the Match Analysis page's pre-match card shows by default.",
    ],
  },
  {
    key: "hybrid",
    name: "Hybrid Engine",
    summary: "A direct blend of the Analytical Markov and ML+Monte Carlo outputs.",
    howItWorks: [
      "Averages the Analytical Markov and ML+Monte Carlo predictions at each point — a simple ensemble intended to hedge between a fast, exact-but-static method and a slower, adaptive-but-noisier one.",
    ],
  },
];

export default function ModelsPage() {
  const { data, isLoading } = useModelComparison();

  return (
    <div className="mx-auto max-w-5xl p-8 space-y-6">
      <div>
        <h1 className="text-page-title">Models</h1>
        <p className="text-caption mt-1">
          Five engines, one scoring hierarchy. How each computes a win probability, and
          how they rank on held-out matches.
        </p>
      </div>

      <div className="space-y-4">
        {ENGINES.map((engine) => {
          const stats = data?.engines[engine.key];
          const color = ENGINE_COLORS[engine.key];
          return (
            <Card key={engine.key}>
              <CardHeader>
                <CardTitle className="text-foreground normal-case tracking-normal text-section-title flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                  {engine.name}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm">{engine.summary}</p>
                <ul className="space-y-1.5">
                  {engine.howItWorks.map((point, i) => (
                    <li key={i} className="text-caption pl-4 relative before:content-['—'] before:absolute before:left-0">
                      {point}
                    </li>
                  ))}
                </ul>

                {ENGINE_DIAGRAMS[engine.key] && (
                  <div className="pt-2">
                    {(() => {
                      const Diagram = ENGINE_DIAGRAMS[engine.key]!;
                      return <Diagram />;
                    })()}
                  </div>
                )}

                {isLoading && <Skeleton className="h-16 w-full mt-2" />}
                {stats && (
                  <div className="flex flex-wrap gap-2 pt-2">
                    <Badge variant="accent">Log Loss {stats.log_loss.toFixed(4)}</Badge>
                    <Badge variant="accent">Brier {stats.brier.toFixed(4)}</Badge>
                    <Badge variant="accent">ECE {stats.ece.toFixed(4)}</Badge>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}