import { useState } from "react";
import { api } from "../api";

// WinProbabilityPanel.jsx — GET /win-probability/{job_id}. The panel is ALWAYS
// shown once a job exists, and ALWAYS renders both prematch_baseline and
// live_adjustment, whatever their status -- 'not_available' with its real
// reason text is a real, visible state here, never an omitted section or a
// blank space a viewer has to interpret.
export default function WinProbabilityPanel({ jobId }) {
  const [matchId, setMatchId] = useState("");
  const [result, setResult] = useState(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);

  async function handleCheck() {
    setPending(true);
    setError(null);
    try {
      const data = await api.winProbability(jobId, matchId || undefined);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  }

  if (!jobId) {
    return (
      <div className="max-w-2xl">
        <h2 className="text-sm font-medium text-slate-300">Win probability</h2>
        <p className="mt-2 text-xs text-slate-500">Submit and complete a clip above first — this panel is scoped to a job.</p>
      </div>
    );
  }

  return (
    <div className="max-w-2xl">
      <h2 className="text-sm font-medium text-slate-300">Win probability</h2>
      <div className="mt-3 flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          match_id (optional — a real v1 historical match)
          <input
            className="w-96 rounded border border-slate-700 bg-slate-900 px-2 py-1.5 font-mono text-xs text-slate-200"
            placeholder="e.g. 20190710-M-Wimbledon-QF-Novak_Djokovic-David_Goffin"
            value={matchId}
            onChange={(e) => setMatchId(e.target.value)}
          />
        </label>
        <button
          type="button"
          onClick={handleCheck}
          disabled={pending}
          className="rounded bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-700"
        >
          {pending ? "Checking…" : "Check"}
        </button>
      </div>

      {error && (
        <p className="mt-3 rounded border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400">{error}</p>
      )}

      {result && (
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <ProbabilityCard title="Pre-match baseline (v1 engine)" entry={result.prematch_baseline} />
          <ProbabilityCard title="Live adjustment (from this clip's CV features)" entry={result.live_adjustment} />
        </div>
      )}
    </div>
  );
}

function ProbabilityCard({ title, entry }) {
  const available = entry.status === "available";
  return (
    <div
      className={`rounded-lg border px-4 py-3 ${
        available ? "border-emerald-500/30 bg-emerald-500/10" : "border-slate-700/50 bg-slate-800/30"
      }`}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h3>
        <span className={`text-xs font-semibold ${available ? "text-emerald-400" : "text-slate-500"}`}>
          {available ? "● available" : "– not available"}
        </span>
      </div>

      {available ? (
        <div className="mt-2">
          <p className="text-3xl font-bold text-emerald-300">{(entry.p1_win_probability_prematch * 100).toFixed(2)}%</p>
          <p className="mt-1 text-xs text-slate-400">
            {entry.p1_name} to beat {entry.p2_name} — <span className="italic">pre-match only, no live point data</span>
          </p>
          <p className="mt-1 font-mono text-[11px] text-slate-600">{entry.match_id}</p>
          <p className="mt-2 text-xs text-slate-500">{entry.source}</p>
        </div>
      ) : (
        <p className="mt-2 text-xs leading-relaxed text-slate-400">{entry.reason}</p>
      )}
    </div>
  );
}
