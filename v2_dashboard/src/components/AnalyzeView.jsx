import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import StatusBadge from "./StatusBadge";
import ResultView from "./ResultView";
import VideoOverlay from "./VideoOverlay";

// v2_serving's /analyze-video is path-based only (local/dev mode) -- see
// v2_serving/src/v2_serving/models.py's AnalyzeVideoRequest docstring, no file
// upload exists on the backend. This dropdown offers the known demo clips rather
// than pretending a file-upload control would do anything real.
const REPO_ROOT = "/Users/mehuldahiya/Desktop/tennis-intelligence";
const KNOWN_CLIPS = [
  ...Array.from({ length: 10 }, (_, i) => `${REPO_ROOT}/data/cv_annotated/videos/video${i + 1}.mp4`),
  `${REPO_ROOT}/data/tennis_clip.mp4`,
];

const POLL_INTERVAL_MS = 3000;

export default function AnalyzeView({ onJobComplete }) {
  const [videoPath, setVideoPath] = useState(KNOWN_CLIPS[0]);
  const [frameLimit, setFrameLimit] = useState(120);
  const [job, setJob] = useState(null); // { job_id, status, video_path, frame_limit, result, error }
  const [submitError, setSubmitError] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => () => clearInterval(pollRef.current), []);

  async function pollJob(jobId) {
    try {
      const data = await api.getJob(jobId);
      setJob(data);
      if (data.status === "complete" || data.status === "failed") {
        clearInterval(pollRef.current);
        if (data.status === "complete") onJobComplete?.(data);
      }
    } catch (err) {
      setSubmitError(err.message);
      clearInterval(pollRef.current);
    }
  }

  async function handleSubmit() {
    setSubmitError(null);
    setJob(null);
    try {
      const submitted = await api.analyzeVideo(videoPath, Number(frameLimit));
      setJob({ job_id: submitted.job_id, status: submitted.status, video_path: videoPath, frame_limit: Number(frameLimit) });
      pollRef.current = setInterval(() => pollJob(submitted.job_id), POLL_INTERVAL_MS);
    } catch (err) {
      setSubmitError(err.message);
    }
  }

  return (
    <div className="max-w-2xl">
      <h2 className="text-sm font-medium text-slate-300">Analyze a clip</h2>

      <div className="mt-3 flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Clip
          <select
            className="rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-200"
            value={videoPath}
            onChange={(e) => setVideoPath(e.target.value)}
          >
            {KNOWN_CLIPS.map((path) => (
              <option key={path} value={path}>
                {path.split("/").pop()}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs text-slate-400">
          frame_limit
          <input
            type="number"
            min={1}
            max={5000}
            className="w-24 rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-200"
            value={frameLimit}
            onChange={(e) => setFrameLimit(e.target.value)}
          />
        </label>

        <button
          type="button"
          onClick={handleSubmit}
          disabled={job && (job.status === "pending" || job.status === "processing")}
          className="rounded bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        >
          Submit
        </button>
      </div>

      {submitError && (
        <p className="mt-3 rounded border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400">
          {submitError}
        </p>
      )}

      {job && (
        <div className="mt-4 rounded-lg border border-slate-800 bg-slate-900 p-4">
          <div className="flex items-center gap-3">
            <StatusBadge status={job.status} />
            <span className="font-mono text-xs text-slate-500">{job.job_id}</span>
          </div>
          <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs text-slate-400">
            <dt className="text-slate-500">video_path</dt>
            <dd className="truncate font-mono">{job.video_path}</dd>
            <dt className="text-slate-500">frame_limit</dt>
            <dd className="font-mono">{job.frame_limit}</dd>
          </dl>
          {job.status === "processing" && (
            <p className="mt-3 text-xs text-sky-400">
              Polling every {POLL_INTERVAL_MS / 1000}s — cv_pipeline is running YOLO
              detection + ByteTrack + pose estimation across {job.frame_limit} frames…
            </p>
          )}
          {job.status === "failed" && (
            <p className="mt-3 rounded border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-400">
              {job.error}
            </p>
          )}
          {job.status === "complete" && (
            <>
              <p className="mt-3 text-xs text-emerald-400">
                Complete — {job.result?.n_frames_processed} frames processed in {job.result?.processing_time_s}s.
              </p>
              {job.result?.frames && (
                <div className="mt-4">
                  <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Video + overlay
                  </h3>
                  <VideoOverlay result={job.result} />
                </div>
              )}
              <ResultView result={job.result} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
