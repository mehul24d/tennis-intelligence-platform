import { cn } from "@/utils/cn";

const SURFACE_STYLES: Record<string, string> = {
  Hard: "border-surface-hard/40 bg-surface-hard/10 text-surface-hard",
  Clay: "border-surface-clay/40 bg-surface-clay/10 text-surface-clay",
  Grass: "border-surface-grass/40 bg-surface-grass/10 text-surface-grass",
};

const FALLBACK_STYLE = "border-border bg-muted text-muted-foreground";

export function SurfaceBadge({ surface, className }: { surface: string; className?: string }) {
  const style = SURFACE_STYLES[surface] ?? FALLBACK_STYLE;
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium",
        style,
        className
      )}
    >
      {surface}
    </span>
  );
}