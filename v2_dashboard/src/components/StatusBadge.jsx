// StatusBadge.jsx — one place that maps a job status string to a color/label, so
// "pending" vs "processing" vs "complete" vs "failed" are always visually distinct,
// never just plain text a viewer has to read carefully to distinguish.
const STYLES = {
  pending: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  processing: "bg-sky-500/15 text-sky-400 border-sky-500/30 animate-pulse",
  complete: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  failed: "bg-red-500/15 text-red-400 border-red-500/30",
};

export default function StatusBadge({ status }) {
  const style = STYLES[status] || "bg-slate-500/15 text-slate-400 border-slate-500/30";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${style}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {status}
    </span>
  );
}
