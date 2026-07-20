import { useEffect, useState } from "react";
import { api, API_BASE_URL } from "./api";
import AnalyzeView from "./components/AnalyzeView";
import ChatView from "./components/ChatView";
import WinProbabilityPanel from "./components/WinProbabilityPanel";

// Step 1 shell: confirm the app can reach v2_serving before building any real
// feature. Later steps replace this body with the actual views (upload/poll,
// results, player+overlay, chat, win-probability) behind simple tab state.
export default function App() {
  const [health, setHealth] = useState({ status: "checking" });
  const [completedJob, setCompletedJob] = useState(null);

  useEffect(() => {
    api
      .health()
      .then((data) => setHealth({ status: "ok", data }))
      .catch((err) => setHealth({ status: "error", error: err.message }));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-xl font-semibold">Tennis Intelligence Platform — v2 Dashboard</h1>
        <p className="mt-1 text-sm text-slate-400">
          Connected to <code className="text-slate-300">{API_BASE_URL}</code>
        </p>
      </header>

      <main className="p-6">
        <div className="max-w-md rounded-lg border border-slate-800 bg-slate-900 p-4">
          <h2 className="text-sm font-medium text-slate-300">v2_serving connection</h2>
          {health.status === "checking" && (
            <p className="mt-2 text-sm text-slate-400">Checking /health…</p>
          )}
          {health.status === "ok" && (
            <p className="mt-2 flex items-center gap-2 text-sm text-emerald-400">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              Reachable — {JSON.stringify(health.data)}
            </p>
          )}
          {health.status === "error" && (
            <p className="mt-2 flex items-center gap-2 text-sm text-red-400">
              <span className="h-2 w-2 rounded-full bg-red-400" />
              Unreachable — {health.error}
            </p>
          )}
        </div>

        <div className="mt-6">
          <AnalyzeView onJobComplete={setCompletedJob} />
        </div>

        <div className="mt-8 border-t border-slate-800 pt-6">
          <ChatView jobId={completedJob?.job_id} jobLabel={completedJob?.video_path?.split("/").pop()} />
        </div>

        <div className="mt-8 border-t border-slate-800 pt-6">
          <WinProbabilityPanel jobId={completedJob?.job_id} />
        </div>
      </main>
    </div>
  );
}
