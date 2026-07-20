"use client";

/**
 * A genuine architecture diagram of this project's own ML-Informed Markov
 * pipeline — not a generic "AI pipeline" stock illustration. Every box/arrow
 * here corresponds to a real, named step in ml_informed_markov.py's own
 * build_pretrained_prior/sensitivity_aware_blend functions.
 */
export function EnginePipelineDiagram() {
  return (
    <svg viewBox="0 0 900 260" className="w-full h-auto" role="img" aria-label="ML-Informed Markov pipeline">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="hsl(var(--muted-foreground))" />
        </marker>
      </defs>

      <g>
        <rect x="10" y="20" width="150" height="56" rx="8" className="fill-background-elevated stroke-border" strokeWidth="1" />
        <text x="85" y="44" textAnchor="middle" className="fill-foreground text-[11px] font-medium">Elo, surface Elo,</text>
        <text x="85" y="60" textAnchor="middle" className="fill-foreground text-[11px] font-medium">H2H, form</text>
      </g>

      <line x1="160" y1="48" x2="210" y2="48" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#arrow)" />

      <g>
        <rect x="215" y="20" width="150" height="56" rx="8" className="fill-accent-blue/10 stroke-accent-blue/40" strokeWidth="1" />
        <text x="290" y="44" textAnchor="middle" className="fill-accent-blue text-[11px] font-semibold">XGBoost</text>
        <text x="290" y="60" textAnchor="middle" className="fill-accent-blue text-[11px] font-semibold">pre-match model</text>
      </g>

      <line x1="365" y1="48" x2="415" y2="48" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#arrow)" />

      <g>
        <rect x="420" y="20" width="170" height="56" rx="8" className="fill-background-elevated stroke-border" strokeWidth="1" />
        <text x="505" y="38" textAnchor="middle" className="fill-foreground text-[11px] font-medium">Invert to serve prior</text>
        <text x="505" y="54" textAnchor="middle" className="fill-foreground text-[11px] font-medium">+ composite n₀</text>
        <text x="505" y="68" textAnchor="middle" className="fill-muted-foreground text-[9px]">(career + H2H + tourney H2H)</text>
      </g>

      <line x1="590" y1="48" x2="640" y2="48" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#arrow)" />

      <g>
        <rect x="645" y="20" width="150" height="56" rx="8" className="fill-accent-gold/10 stroke-accent-gold/40" strokeWidth="1" />
        <text x="720" y="44" textAnchor="middle" className="fill-accent-gold text-[11px] font-semibold">Beta-Binomial</text>
        <text x="720" y="60" textAnchor="middle" className="fill-accent-gold text-[11px] font-semibold">prior</text>
      </g>

      <line x1="720" y1="76" x2="720" y2="120" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#arrow)" />

      <g>
        <rect x="620" y="125" width="200" height="56" rx="8" className="fill-background-elevated stroke-border" strokeWidth="1" />
        <text x="720" y="149" textAnchor="middle" className="fill-foreground text-[11px] font-medium">Per-point classifier</text>
        <text x="720" y="165" textAnchor="middle" className="fill-foreground text-[11px] font-medium">(momentum, pressure, streaks)</text>
      </g>

      <line x1="620" y1="153" x2="450" y2="153" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#arrow)" />

      <g>
        <rect x="250" y="125" width="200" height="56" rx="8" className="fill-accent-green/10 stroke-accent-green/40" strokeWidth="1" />
        <text x="350" y="149" textAnchor="middle" className="fill-accent-green text-[11px] font-semibold">Sensitivity-aware blend</text>
        <text x="350" y="165" textAnchor="middle" className="fill-accent-green text-[11px] font-semibold">(posterior mean + classifier)</text>
      </g>

      <line x1="250" y1="153" x2="200" y2="153" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#arrow)" />

      <g>
        <rect x="10" y="125" width="180" height="56" rx="8" className="fill-background-elevated stroke-border" strokeWidth="1" />
        <text x="100" y="149" textAnchor="middle" className="fill-foreground text-[11px] font-medium">Markov recursion</text>
        <text x="100" y="165" textAnchor="middle" className="fill-foreground text-[11px] font-medium">(point → game → set → match)</text>
      </g>

      <line x1="100" y1="181" x2="100" y2="220" stroke="hsl(var(--muted-foreground))" strokeWidth="1.5" markerEnd="url(#arrow)" />
      <g>
        <rect x="10" y="222" width="180" height="34" rx="8" className="fill-accent-blue/15 stroke-accent-blue/50" strokeWidth="1" />
        <text x="100" y="243" textAnchor="middle" className="fill-accent-blue text-[11px] font-semibold">P(win | point i)</text>
      </g>
    </svg>
  );
}