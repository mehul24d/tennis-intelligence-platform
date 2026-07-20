"use client";

/**
 * A genuine diagram of monte_carlo_engine.py's own simulate_match_from_state
 * loop — including its two real, documented safeguards (a terminal-state
 * guard before the loop starts, and a max_points cap to prevent the
 * mathematically-real infinite loop at p=1.0), not just the textbook "run N
 * trials and average" description.
 */
export function MonteCarloDiagram() {
  return (
    <svg viewBox="0 0 900 220" className="w-full h-auto" role="img" aria-label="Monte Carlo simulation loop">
      <defs>
        <marker id="mc-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="hsl(var(--muted-foreground))" />
        </marker>
      </defs>

      <g>
        <rect x="10" y="20" width="170" height="50" rx="8" className="fill-background-elevated stroke-border" strokeWidth="1" />
        <text x="95" y="40" textAnchor="middle" className="fill-foreground text-[11px] font-medium">Current match state</text>
        <text x="95" y="56" textAnchor="middle" className="fill-muted-foreground text-[9px]">sets, games, points, server</text>
      </g>

      <line x1="180" y1="45" x2="230" y2="45" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#mc-arrow)" />

      <g>
        <rect x="235" y="20" width="160" height="50" rx="8" className="fill-accent-gold/10 stroke-accent-gold/40" strokeWidth="1" />
        <text x="315" y="40" textAnchor="middle" className="fill-accent-gold text-[11px] font-semibold">Already decided?</text>
        <text x="315" y="56" textAnchor="middle" className="fill-accent-gold text-[10px]">return 1.0 / 0.0 directly</text>
      </g>

      <line x1="315" y1="70" x2="315" y2="105" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#mc-arrow)" />
      <text x="330" y="90" className="fill-muted-foreground text-[9px]">no</text>

      <g>
        <rect x="235" y="110" width="160" height="50" rx="8" className="fill-accent-blue/10 stroke-accent-blue/40" strokeWidth="1" />
        <text x="315" y="130" textAnchor="middle" className="fill-accent-blue text-[11px] font-semibold">Run 300 trials</text>
        <text x="315" y="146" textAnchor="middle" className="fill-accent-blue text-[10px]">each simulated independently</text>
      </g>

      <line x1="395" y1="135" x2="440" y2="135" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#mc-arrow)" />

      <g>
        <rect x="445" y="90" width="200" height="100" rx="8" className="fill-background-elevated stroke-border" strokeWidth="1" />
        <text x="545" y="112" textAnchor="middle" className="fill-foreground text-[11px] font-semibold">Per trial, per point:</text>
        <text x="545" y="130" textAnchor="middle" className="fill-muted-foreground text-[10px]">draw a random outcome vs.</text>
        <text x="545" y="145" textAnchor="middle" className="fill-muted-foreground text-[10px]">server-win probability,</text>
        <text x="545" y="160" textAnchor="middle" className="fill-muted-foreground text-[10px]">advance the state, repeat</text>
        <text x="545" y="178" textAnchor="middle" className="fill-accent-red text-[9px]">capped at 350/700 points —</text>
      </g>

      <line x1="645" y1="140" x2="690" y2="140" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#mc-arrow)" />

      <g>
        <rect x="695" y="115" width="190" height="50" rx="8" className="fill-accent-green/10 stroke-accent-green/40" strokeWidth="1" />
        <text x="790" y="135" textAnchor="middle" className="fill-accent-green text-[11px] font-semibold">wins / 300 trials</text>
        <text x="790" y="151" textAnchor="middle" className="fill-accent-green text-[10px]">= P(player 1 wins)</text>
      </g>

      <text x="10" y="205" className="fill-muted-foreground text-[9px]">
        At p=1.0 exactly, neither player can ever break serve — a real, provable infinite loop this cap exists specifically to stop.
      </text>
    </svg>
  );
}