import { useState } from "react";
import { api } from "../api";

// ChatView.jsx — POST /query. The citation-audit distinction is the entire
// point of this view: sources_offered vs. sources_used must be visibly
// different states, not one flattened list. A tag counts as "cited" only if
// its descriptor text appears in sources_used; everything else in
// sources_offered is rendered as "offered but not cited" -- dimmed, not hidden.
export default function ChatView({ jobId, jobLabel }) {
  const [question, setQuestion] = useState("");
  const [player, setPlayer] = useState("");
  const [opponent, setOpponent] = useState("");
  const [useLiveJob, setUseLiveJob] = useState(true);
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setPending(true);
    setError(null);
    try {
      const payload = {
        question,
        job_id: useLiveJob && jobId ? jobId : null,
        player: player || null,
        opponent: opponent || null,
      };
      const response = await api.query(payload);
      setMessages((prev) => [...prev, { question, ...response }]);
      setQuestion("");
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="max-w-3xl">
      <h2 className="text-sm font-medium text-slate-300">Ask the tactical agent</h2>

      <form onSubmit={handleSubmit} className="mt-3 space-y-2">
        <div className="flex gap-2">
          <input
            className="flex-1 rounded border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-200"
            placeholder="Ask a question…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button
            type="submit"
            disabled={pending}
            className="rounded bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-700"
          >
            {pending ? "Asking…" : "Ask"}
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
          <input
            className="w-32 rounded border border-slate-700 bg-slate-900 px-2 py-1"
            placeholder="player (optional)"
            value={player}
            onChange={(e) => setPlayer(e.target.value)}
          />
          <input
            className="w-32 rounded border border-slate-700 bg-slate-900 px-2 py-1"
            placeholder="opponent (optional)"
            value={opponent}
            onChange={(e) => setOpponent(e.target.value)}
          />
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={useLiveJob}
              onChange={(e) => setUseLiveJob(e.target.checked)}
              disabled={!jobId}
            />
            Fuse live CV features from {jobId ? jobLabel : "(no completed job yet)"}
          </label>
        </div>
      </form>

      {error && (
        <p className="mt-3 rounded border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400">{error}</p>
      )}

      <div className="mt-4 space-y-4">
        {messages.map((msg, i) => (
          <ChatMessage key={i} msg={msg} />
        ))}
      </div>
    </div>
  );
}

function ChatMessage({ msg }) {
  const citedSet = new Set([...(msg.sources_used?.live_features || []), ...(msg.sources_used?.retrieved_docs || [])]);
  const offeredEntries = Object.entries(msg.sources_offered || {});
  const cited = offeredEntries.filter(([, text]) => citedSet.has(text));
  const offeredOnly = offeredEntries.filter(([, text]) => !citedSet.has(text));

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
      <p className="text-sm font-medium text-slate-300">Q: {msg.question}</p>
      <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-slate-200">{msg.answer}</p>

      {msg.live_features_used === false && msg.live_features_note && (
        <p className="mt-2 text-xs italic text-slate-500">{msg.live_features_note}</p>
      )}

      <div className="mt-3 border-t border-slate-800 pt-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Sources ({cited.length} cited / {offeredEntries.length} offered)
        </p>

        {cited.length > 0 && (
          <div className="mt-2 space-y-1">
            {cited.map(([tag, text]) => (
              <div key={tag} className="flex items-start gap-2 rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-xs">
                <span className="font-mono font-semibold text-emerald-400">[{tag}]</span>
                <span className="text-emerald-200">✓ cited —</span>
                <span className="text-slate-300">{text}</span>
              </div>
            ))}
          </div>
        )}

        {offeredOnly.length > 0 && (
          <div className="mt-2 space-y-1">
            {offeredOnly.map(([tag, text]) => (
              <div key={tag} className="flex items-start gap-2 rounded border border-slate-700/50 bg-slate-800/30 px-2 py-1 text-xs opacity-60">
                <span className="font-mono font-semibold text-slate-500">[{tag}]</span>
                <span className="text-slate-500">offered, not cited —</span>
                <span className="text-slate-500 line-through decoration-slate-600">{text}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
