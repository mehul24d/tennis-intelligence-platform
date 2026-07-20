"use client";

/**
 * The real hierarchical recursion markov_baseline.py implements: point-win
 * probability feeds a closed-form game-win formula (including the actual
 * deuce-subgame math), which feeds set, which feeds match. No simulation
 * anywhere in this engine — everything here is an exact closed-form
 * computation.
 *
 * NOTE: colors are written as fully literal className strings per level,
 * NOT constructed via template-literal interpolation (e.g. `fill-${color}`)
 * — Tailwind's build-time scanner only detects statically-analyzable class
 * names in source text, so a dynamically-built class name would silently
 * never get generated in the actual CSS output, leaving every box unstyled.
 */
const LEVEL_STYLES = [
  { box: "fill-accent-blue/10 stroke-accent-blue/40", text: "fill-accent-blue" },
  { box: "fill-accent-green/10 stroke-accent-green/40", text: "fill-accent-green" },
  { box: "fill-accent-gold/10 stroke-accent-gold/40", text: "fill-accent-gold" },
  { box: "fill-accent-blue/10 stroke-accent-blue/40", text: "fill-accent-blue" },
];

const LEVELS = [
  { label: "Point", detail: "p = serve-win probability (input)" },
  { label: "Game", detail: "closed-form binomial + deuce subgame" },
  { label: "Set", detail: "games won-by-2, or tiebreak at 6-6" },
  { label: "Match", detail: "best-of-3 or best-of-5 sets" },
];

export function MarkovRecursionDiagram() {
  return (
    <svg viewBox="0 0 900 160" className="w-full h-auto" role="img" aria-label="Markov recursion hierarchy">
      <defs>
        <marker id="mk-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="hsl(var(--muted-foreground))" />
        </marker>
      </defs>

      {LEVELS.map((level, i) => {
        const x = 10 + i * 225;
        const style = LEVEL_STYLES[i];
        return (
          <g key={level.label}>
            <rect x={x} y="20" width="195" height="60" rx="8" className={style.box} strokeWidth="1" />
            <text x={x + 97} y="44" textAnchor="middle" className={`${style.text} text-[13px] font-semibold`}>
              {level.label}
            </text>
            <text x={x + 97} y="62" textAnchor="middle" className="fill-muted-foreground text-[9px]">
              {level.detail}
            </text>
            {i < LEVELS.length - 1 && (
              <line
                x1={x + 200}
                y1="50"
                x2={x + 220}
                y2="50"
                stroke="hsl(var(--muted-foreground))"
                strokeWidth="1.5"
                markerEnd="url(#mk-arrow)"
              />
            )}
          </g>
        );
      })}

      <text x="10" y="115" className="fill-foreground text-[10px] font-mono">
        Deuce subgame win probability = p² / (p² + q²), where q = 1 − p
      </text>
      <text x="10" y="135" className="fill-muted-foreground text-[9px]">
        Every level above is an exact formula — no randomness, no simulation. This is what
      </text>
      <text x="10" y="150" className="fill-muted-foreground text-[9px]">
        makes the Analytical Markov engine fast, but also why it can&apos;t react mid-match.
      </text>
    </svg>
  );
}