import { useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "../api";

const COLORS = { near: "#34d399", far: "#38bdf8", ball: "#facc15", court: "#a78bfa", singles: "#22d3ee", shot: "#fb7185" };

// homography_applicable defaults to true when absent (older result shapes
// predating this field) -- only an explicit `false` suppresses the court
// overlay for that frame. See ball_detection_combined.py's
// frame_matches_reference_framing for why this exists at the per-frame level,
// not just the per-clip regime level.
const isHomographyApplicable = (frame) => frame?.homography_applicable !== false;

// VideoOverlay.jsx — plays the source clip and draws real per-frame detection
// data on a canvas synced to video playback time. The honesty requirement this
// component exists to satisfy: when a role (near/far player, ball) has NO box
// for the current frame, this draws NOTHING for it on the canvas (never a
// fabricated/stale box) AND says so explicitly in the legend below the video --
// "not detected this frame" is a distinct, visible state, not silence a viewer
// has to interpret themselves.
export default function VideoOverlay({ result }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [frameIdx, setFrameIdx] = useState(0);

  const frames = result.frames || [];
  const fps = result.source_fps || 30;
  const videoUrl = `${API_BASE_URL}/video-file/${result.clip}`;
  const currentFrame = frames[frameIdx] || null;
  const courtCorners = result.homography?.court_corners || null;
  // Singles lines are computed from the same doubles calibration (standard
  // fixed inset, both centered on the same center line) via
  // CourtHomography.singles_corners_pixels() -- not independently re-clicked.
  // Since this is a singles match, these -- not the doubles lines -- are the
  // ones relevant to in/out-of-bounds context; doubles lines stay as the
  // outer reference frame. Both were visually confirmed against real frames
  // (see PROGRESS.md) before being wired in here.
  const singlesCorners = result.homography?.singles_corners || null;

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    // NOT `timeupdate` -- that event is browser-throttled to ~4/sec (not once
    // per rendered video frame), which at this clip's 59.94fps left ~15 real
    // frames elapsing between overlay updates (only ~6-7 at the ~25-30fps
    // clips this was originally built against, which is why this was never
    // visibly wrong before). The ball's own position data was checked and
    // confirmed accurate frame-by-frame -- this was a display-sync-rate bug,
    // not a detection/interpolation lag. A rAF loop samples video.currentTime
    // every rendered browser frame instead, decoupled from timeupdate's cadence.
    let rafId;
    const syncFrame = () => {
      const idx = Math.min(Math.round(video.currentTime * fps), frames.length - 1);
      setFrameIdx((prev) => (prev === Math.max(0, idx) ? prev : Math.max(0, idx)));
      rafId = requestAnimationFrame(syncFrame);
    };
    rafId = requestAnimationFrame(syncFrame);
    return () => cancelAnimationFrame(rafId);
  }, [fps, frames.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !video.videoWidth) return;
    const ctx = canvas.getContext("2d");
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!showOverlay) return;

    const scaleX = canvas.width / (result.video_width || video.videoWidth);
    const scaleY = canvas.height / (result.video_height || video.videoHeight);

    // data/tennis/2.mp4 has two homographies (a confirmed mid-clip camera pan
    // -- see PROGRESS.md) -- frames after the pan carry their own
    // court_corners/singles_corners, which take priority over the clip-level
    // ones. Frames without an override (pre-pan, or any other clip) fall back
    // to result.homography as before.
    const frameCourtCorners = currentFrame?.court_corners || courtCorners;
    const frameSinglesCorners = currentFrame?.singles_corners || singlesCorners;

    const drawBox = (box, color, label) => {
      if (!box) return;
      const [x1, y1, x2, y2] = box;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
      ctx.fillStyle = color;
      ctx.font = "12px monospace";
      ctx.fillText(label, x1 * scaleX + 2, y1 * scaleY - 4);
    };

    const drawPoint = (box, color, label) => {
      if (!box) return;
      const [x1, y1, x2, y2] = box;
      const cx = ((x1 + x2) / 2) * scaleX;
      const cy = ((y1 + y2) / 2) * scaleY;
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = "12px monospace";
      ctx.fillText(label, cx + 8, cy);
    };

    const drawQuad = (corners, color, dashed) => {
      if (!corners) return;
      const order = ["BL", "BR", "TR", "TL"];
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(dashed ? [6, 4] : []);
      ctx.beginPath();
      order.forEach((key, i) => {
        const [x, y] = corners[key];
        const px = x * scaleX, py = y * scaleY;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    };

    if (isHomographyApplicable(currentFrame)) {
      drawQuad(frameCourtCorners, COLORS.court, false);
      drawQuad(frameSinglesCorners, COLORS.singles, true);
    }

    if (currentFrame) {
      drawBox(currentFrame.near_box, COLORS.near, `near${currentFrame.near_track_id !== null ? ` (id ${currentFrame.near_track_id})` : ""}`);
      drawBox(currentFrame.far_box, COLORS.far, `far${currentFrame.far_track_id !== null ? ` (id ${currentFrame.far_track_id})` : ""}`);
      drawPoint(currentFrame.ball_box, COLORS.ball, "ball");

      // shot_events is only ever populated on the exact ball-anchored contact
      // frame(s) (see shot_classification.py) -- drawn only on that single
      // frame, same "never show something not true for the current frame"
      // rule as every other overlay element here. A LIST, not a single
      // object: two distinct candidate swings can legitimately anchor to the
      // same frame (confirmed on real data), and stacking their labels here
      // keeps both visible instead of one silently overwriting the other. At
      // ~60fps this is genuinely a ~16ms flash during normal playback; the
      // clickable list below the video exists specifically so a viewer can
      // find and inspect these frames without needing to catch them live.
      (currentFrame.shot_events || []).forEach((shotEvent, i) => {
        const box = shotEvent.role === "near" ? currentFrame.near_box : currentFrame.far_box;
        if (!box) return;
        const [x1, y1, x2] = box;
        const label = `${shotEvent.classification.toUpperCase()}${shotEvent.probable_serve ? " (probable serve)" : ""}`;
        ctx.fillStyle = COLORS.shot;
        ctx.font = "bold 13px monospace";
        const cx = ((x1 + x2) / 2) * scaleX;
        ctx.textAlign = "center";
        ctx.fillText(label, cx, y1 * scaleY - 18 - i * 14);
        ctx.textAlign = "left";
      });
    }
  }, [frameIdx, showOverlay, currentFrame, courtCorners, singlesCorners, result.video_width, result.video_height]);

  // Flat, chronological list of every shot event in this clip -- exists so a
  // viewer can find/inspect an event without needing to catch its ~16ms
  // on-canvas flash during normal playback (see the shot_events draw block
  // above). Click seeks the video to that exact frame. Two events can share
  // a frame (see above), so this flattens shot_events per frame, not one
  // entry per frame.
  const shotEvents = frames.flatMap((f, idx) =>
    (f.shot_events || []).map((ev, i) => ({ idx, key: `${idx}-${i}`, ...ev }))
  );

  return (
    <div>
      <div className="mb-2 flex items-center gap-3">
        <label className="flex items-center gap-1.5 text-xs text-slate-300">
          <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
          Show overlay
        </label>
        <span className="text-xs text-slate-500">
          frame {frameIdx} / {frames.length - 1}
        </span>
      </div>

      <div className="relative inline-block">
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          className="max-w-full rounded border border-slate-800"
          style={{ maxHeight: 480 }}
          onLoadedMetadata={() => {
            const canvas = canvasRef.current;
            const video = videoRef.current;
            if (canvas && video) {
              canvas.width = video.clientWidth;
              canvas.height = video.clientHeight;
            }
          }}
        />
        <canvas ref={canvasRef} className="pointer-events-none absolute left-0 top-0" />
      </div>

      {/* Legend: explicit per-role state, so "no box drawn" is never ambiguous
          with "this app is broken" -- distinguishes a real detected box from a
          real absence, for every role, every frame. */}
      <div className="mt-2 flex flex-wrap gap-3 text-xs">
        <LegendEntry color={COLORS.near} label="Near player" present={!!currentFrame?.near_box} />
        <LegendEntry color={COLORS.far} label="Far player" present={!!currentFrame?.far_box} />
        <LegendEntry color={COLORS.ball} label="Ball" present={!!currentFrame?.ball_box} />
        {courtCorners && (
          <span className="flex items-center gap-1.5 text-slate-400">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ background: isHomographyApplicable(currentFrame) ? COLORS.court : "transparent", border: `1.5px solid ${COLORS.court}` }}
            />
            Court lines ({result.homography.status})
            {!isHomographyApplicable(currentFrame) && (
              <span className="text-amber-400"> — suppressed this frame (camera angle doesn't match calibration)</span>
            )}
          </span>
        )}
        {result.shot_classification_live_estimate?.status === "measured" && (
          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ border: `1.5px solid ${COLORS.shot}` }} />
            Shot classification: {shotEvents.length} event{shotEvents.length === 1 ? "" : "s"} this segment
          </span>
        )}
      </div>

      {/* Every shot event is a single ball-anchored frame (see shot_event draw
          block above) -- listed here, clickable, since the on-canvas label
          only appears for the one frame it's true of. Not present at all when
          shot classification didn't run this segment (see
          shot_classification_live_estimate.status in the result JSON, e.g.
          "not_attempted" when this clip's ball detection wasn't combined_v2). */}
      {shotEvents.length > 0 && (
        <div className="mt-2 text-xs text-slate-400">
          <div className="mb-1 text-slate-500">Shot events (click to seek):</div>
          <div className="flex flex-wrap gap-1.5">
            {shotEvents.map((ev) => (
              <button
                key={ev.key}
                type="button"
                onClick={() => {
                  const video = videoRef.current;
                  if (video) video.currentTime = ev.idx / fps;
                }}
                className="rounded border px-1.5 py-0.5 font-mono hover:opacity-80"
                style={{ borderColor: COLORS.shot, color: COLORS.shot }}
              >
                f{ev.idx} {ev.role} {ev.classification}
                {ev.probable_serve ? " (serve?)" : ""}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function LegendEntry({ color, label, present }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: present ? color : "transparent", border: `1.5px solid ${color}` }} />
      <span className={present ? "text-slate-300" : "text-slate-500"}>
        {label}: {present ? "detected this frame" : "not detected this frame"}
      </span>
    </span>
  );
}
