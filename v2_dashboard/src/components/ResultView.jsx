import CvStatusValue from "./CvStatusValue";

// ResultView.jsx — the structured cv_pipeline result for a completed job.
// Every Status-tagged field goes through CvStatusValue so measured/not_detected/
// excluded_known_issue/etc. are never visually interchangeable. Nothing here
// collapses a non-measured field to a blank cell.
export default function ResultView({ result }) {
  if (!result) return null;

  return (
    <div className="mt-4 space-y-4">
      <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
        <strong>{result.ground_truth?.split(" -- ")[0] || "Live inference"}</strong> — {result.ground_truth?.split(" -- ")[1]}
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400 sm:grid-cols-4">
        <dt className="text-slate-500">clip</dt>
        <dd className="font-mono text-slate-300">{result.clip}</dd>
        <dt className="text-slate-500">frames processed</dt>
        <dd className="font-mono text-slate-300">{result.n_frames_processed}</dd>
        <dt className="text-slate-500">source fps</dt>
        <dd className="font-mono text-slate-300">{result.source_fps?.toFixed(2)}</dd>
        <dt className="text-slate-500">processing time</dt>
        <dd className="font-mono text-slate-300">{result.processing_time_s}s</dd>
        <dt className="text-slate-500">player selection method</dt>
        <dd className="font-mono text-slate-300">{(result.player_selection_method || []).join(", ")}</dd>
      </dl>

      <div>
        <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Homography</h3>
        <CvStatusValue label="Real-world-distance metrics" entry={result.homography} />
      </div>

      <div>
        <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Detection (live estimate)</h3>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
          <CvStatusValue label="Near player" entry={result.near_player_detection_live_estimate} />
          <CvStatusValue label="Far player" entry={result.far_player_detection_live_estimate} />
          <CvStatusValue label="Ball" entry={result.ball_detection_live_estimate} />
        </div>
      </div>

      <div>
        <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Pose (live estimate)</h3>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <CvStatusValue label="Near player pose" entry={result.near_player_pose_live_estimate} />
          <CvStatusValue label="Far player pose" entry={result.far_player_pose_live_estimate} />
        </div>
      </div>

      {result.tracking && (
        <div>
          <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Tracking</h3>
          <div className="rounded-md border border-slate-700/40 bg-slate-800/30 px-3 py-2 text-xs text-slate-300">
            <p>
              near player: {result.tracking.near_player_distinct_track_ids?.length ?? 0} distinct track ID(s){" "}
              <span className="font-mono text-slate-500">
                [{(result.tracking.near_player_distinct_track_ids || []).join(", ")}]
              </span>
              {" · "}
              far player: {result.tracking.far_player_distinct_track_ids?.length ?? 0} distinct track ID(s){" "}
              <span className="font-mono text-slate-500">
                [{(result.tracking.far_player_distinct_track_ids || []).join(", ")}]
              </span>
            </p>
            <p className="mt-1 text-slate-500">{result.tracking.note}</p>
          </div>
        </div>
      )}
    </div>
  );
}
