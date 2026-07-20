// CvStatusValue.jsx — renders one cv_pipeline Status-enum-tagged field
// (homography, detection rates, pose results). This is the component this whole
// phase is actually about: `measured`, `not_detected`, `not_attempted`,
// `excluded_known_issue`, `insufficient_sample`, `unvalidated`, and
// `not_applicable` must never look the same on screen. Each gets its own color,
// icon, and label — and critically, the accompanying `note`/`reason` text is
// ALWAYS shown for anything non-measured, not tucked behind a click, because a
// viewer shouldn't have to go hunting for why a number isn't there.
const STATUS_STYLE = {
  measured: { color: "text-emerald-400", border: "border-emerald-500/30", bg: "bg-emerald-500/10", icon: "●", label: "measured" },
  not_detected: { color: "text-orange-400", border: "border-orange-500/30", bg: "bg-orange-500/10", icon: "✕", label: "attempted — not detected" },
  not_attempted: { color: "text-slate-400", border: "border-slate-600/40", bg: "bg-slate-700/20", icon: "–", label: "not attempted" },
  excluded_known_issue: { color: "text-purple-400", border: "border-purple-500/30", bg: "bg-purple-500/10", icon: "⚠", label: "excluded (known issue)" },
  insufficient_sample: { color: "text-yellow-400", border: "border-yellow-500/30", bg: "bg-yellow-500/10", icon: "△", label: "insufficient sample" },
  unvalidated: { color: "text-sky-400", border: "border-sky-500/30", bg: "bg-sky-500/10", icon: "?", label: "unvalidated" },
  not_applicable: { color: "text-slate-500", border: "border-slate-700/40", bg: "bg-slate-800/30", icon: "—", label: "not applicable" },
  sentinel_excluded: { color: "text-pink-400", border: "border-pink-500/30", bg: "bg-pink-500/10", icon: "⊘", label: "sentinel excluded" },
};

const FALLBACK_STYLE = { color: "text-slate-300", border: "border-slate-600/40", bg: "bg-slate-700/20", icon: "?", label: "unknown status" };

// Method badge: distinct from the Status badge above -- Status says whether
// THIS number is trustworthy right now (measured/unvalidated/etc.), method
// says WHICH detection approach produced it. Currently only
// ball_detection_live_estimate carries a `method` field (added when the
// combined ball-detection method, ball_detection_combined.py, was wired in as
// the regime-gated default alongside the original stock-YOLO path) -- renders
// nothing for fields without one, so this is a no-op everywhere else.
const METHOD_STYLE = {
  combined_v2: { color: "text-emerald-300", bg: "bg-emerald-900/40", label: "improved method (validated)" },
  stock_yolo: { color: "text-amber-300", bg: "bg-amber-900/40", label: "best-effort, known limitations" },
};

export default function CvStatusValue({ label, entry }) {
  if (!entry) return null;
  const style = STATUS_STYLE[entry.status] || FALLBACK_STYLE;
  const noteText = entry.note || entry.reason;
  const methodStyle = entry.method ? METHOD_STYLE[entry.method] : null;

  // The one real value worth headlining -- a rate/value field, when the status
  // is genuinely 'measured'. Every other status shows the note instead of a number.
  const headline =
    entry.status === "measured"
      ? entry.rate !== undefined
        ? `${(entry.rate * 100).toFixed(1)}%${entry.n !== undefined ? ` (n=${entry.n})` : ""}`
        : entry.success_rate !== undefined
          ? `${(entry.success_rate * 100).toFixed(1)}% success (n=${entry.n_attempted})`
          : entry.value ?? null
      : null;

  return (
    <div className={`rounded-md border ${style.border} ${style.bg} px-3 py-2`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-slate-300">{label}</span>
        <span className={`inline-flex items-center gap-1 text-xs font-semibold ${style.color}`}>
          <span aria-hidden="true">{style.icon}</span>
          {style.label}
        </span>
      </div>
      {methodStyle && (
        <span className={`mt-1 inline-block rounded px-1.5 py-0.5 text-[10px] font-medium ${methodStyle.color} ${methodStyle.bg}`}>
          {methodStyle.label}
        </span>
      )}
      {headline !== null && <p className={`mt-1 text-lg font-semibold ${style.color}`}>{headline}</p>}
      {noteText && (
        <p className="mt-1 text-xs leading-snug text-slate-400">{noteText}</p>
      )}
    </div>
  );
}
