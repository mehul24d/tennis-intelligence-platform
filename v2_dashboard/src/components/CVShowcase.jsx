// CVShowcase.jsx — a static gallery of real cv_pipeline output frames, with an
// honest explanation of what each shows and why the live video-upload feature
// (AnalyzeView, local-dev only) isn't part of this deployment. Added 2026-08
// after this dashboard shipped without ANY visible trace of the CV work once
// AnalyzeView/WinProbabilityPanel were hidden behind VITE_PUBLIC_BUILD -- a real
// gap (a visitor to the public site would have no way to know cv_pipeline exists
// at all), not a deliberate omission. Every image here is a REAL frame from
// cv_pipeline's own dev/stress-test output (scratch_output/), not a mockup or a
// staged screenshot -- same "no fabricated proof" discipline as the rest of this
// project's reports (EVALUATION_REPORT.md, STRESS_TEST_REPORT.md, etc.).
const ITEMS = [
  {
    src: "/cv-showcase/court-homography.jpg",
    title: "Court homography + player position",
    body: "The cyan quadrilateral is a calibrated homography — real pixel " +
      "coordinates mapped to real court geometry, built from measured corner " +
      "points (see cv_pipeline/reference_video1_calibration.py). It's what lets " +
      "player-selection tell an actual player apart from a courtside bystander " +
      "by real-world court position, not just box size.",
  },
  {
    src: "/cv-showcase/pose-estimation.jpg",
    title: "Pose estimation (MediaPipe)",
    body: "Per-player body landmarks on a real ATP broadcast frame, feeding shot " +
      "classification (forehand/backhand/serve). Measured at 87.5% confident " +
      "accuracy on this project's manually-audited reference clips — see " +
      "PROGRESS.md's \"Shot-Type Detection\" entry for the full, un-rounded numbers.",
  },
  {
    src: "/cv-showcase/player-tracking.jpg",
    title: "Multi-object tracking (ByteTrack)",
    body: "Every person YOLO detects gets a persistent ID — including ball kids, " +
      "umpires, and bystanders, not just the two players. This is exactly the " +
      "raw signal that makes court-position-based player selection necessary in " +
      "the first place: a size-based heuristic alone picks the wrong box often " +
      "enough that it's flagged directly in this project's own evaluation report.",
  },
  {
    src: "/cv-showcase/ball-detection.jpg",
    title: "Ball detection",
    body: "The hardest of the four — a small, fast-moving, motion-blurred object. " +
      "A fine-tuned YOLOv8n plus a frequency-based artifact filter reaches 53.91% " +
      "pooled recall on this project's amateur ground-truth clips, up from 7.8% " +
      "for stock YOLO — still openly reported as a live estimate, not a solved " +
      "problem (see ball_detection_combined.py's own accuracy history, including " +
      "a corrected initial overestimate).",
  },
];

export default function CVShowcase() {
  return (
    <div className="max-w-3xl">
      <h2 className="text-sm font-medium text-slate-300">Computer-vision pipeline</h2>
      <p className="mt-2 text-sm text-slate-400">
        This deployment's live API only serves the RAG+LLM tactical Q&amp;A below —
        frame-by-frame YOLO/pose/tracking inference on uploaded video is real,
        working code (see <code className="text-slate-300">cv_pipeline/</code>),
        but too heavy to run on a free-tier host without OOMing or timing out on
        every request. The frames below are real output from that pipeline,
        captured during its own development/stress-testing, not mockups.
      </p>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        {ITEMS.map((item) => (
          <figure key={item.src} className="rounded-lg border border-slate-800 bg-slate-900 p-3">
            <img
              src={item.src}
              alt={item.title}
              className="w-full rounded border border-slate-800"
              loading="lazy"
            />
            <figcaption className="mt-2">
              <p className="text-sm font-medium text-slate-200">{item.title}</p>
              <p className="mt-1 text-xs leading-relaxed text-slate-400">{item.body}</p>
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
