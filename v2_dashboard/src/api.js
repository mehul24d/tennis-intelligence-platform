// api.js — the one place the dashboard knows how to reach v2_serving. Confirmed
// running locally on 8734 during Phase 4 (see PROGRESS.md's Phase 4 entries) --
// no mocked/fake data anywhere in this app, per the phase's own constraint.
export const API_BASE_URL = "http://127.0.0.1:8734";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = response.statusText;
    }
    throw new Error(`${response.status} ${path}: ${detail}`);
  }
  return response.json();
}

export const api = {
  health: () => request("/health"),
  analyzeVideo: (videoPath, frameLimit) =>
    request("/analyze-video", {
      method: "POST",
      body: JSON.stringify({ video_path: videoPath, frame_limit: frameLimit }),
    }),
  getJob: (jobId) => request(`/jobs/${jobId}`),
  query: (payload) => request("/query", { method: "POST", body: JSON.stringify(payload) }),
  winProbability: (jobId, matchId) =>
    request(`/win-probability/${jobId}${matchId ? `?match_id=${encodeURIComponent(matchId)}` : ""}`),
};
