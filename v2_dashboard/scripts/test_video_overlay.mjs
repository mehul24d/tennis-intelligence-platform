import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });

await page.getByLabel("Clip").selectOption({ label: "video1.mp4" });
await page.getByLabel("frame_limit").fill("90");
await page.getByRole("button", { name: "Submit" }).click();

await page.waitForSelector("text=/Complete —/", { timeout: 180000 });
await page.waitForSelector("video");

// Wait for the video element to actually load metadata + first frame, then seek
// to a specific, known frame (frame 10) so we can cross-check the overlay
// against the exact box coordinates the API returned for that frame.
await page.waitForFunction(() => {
  const v = document.querySelector("video");
  return v && v.readyState >= 2;
});

const targetFrame = 10;
await page.evaluate((frame) => {
  const v = document.querySelector("video");
  // fps is baked into the component's internal calc; 60fps source -> frame/60s
  v.currentTime = frame / 60.0;
}, targetFrame);

// give the timeupdate handler + redraw a moment
await page.waitForTimeout(400);
await page.screenshot({ path: "/tmp/step4_overlay_frame10.png", fullPage: true });
console.log("captured overlay at frame 10");

// Toggle overlay off, confirm boxes disappear (screenshot for comparison).
await page.getByLabel("Show overlay").uncheck();
await page.waitForTimeout(200);
await page.screenshot({ path: "/tmp/step4_overlay_off.png", fullPage: true });
console.log("captured overlay-off state");

await browser.close();
