import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });

// Use a smaller frame_limit so the whole flow (pending -> processing -> complete)
// finishes in a reasonable time for this verification run, while still being a
// real cv_pipeline inference run, not a fixture.
await page.getByLabel("frame_limit").fill("60");

await page.getByRole("button", { name: "Submit" }).click();

// Grab whatever status is visible immediately after submit -- pending or
// processing, whichever the real timing lands on.
await page.waitForTimeout(300);
await page.screenshot({ path: "/tmp/step2_1_immediately_after_submit.png", fullPage: true });
console.log("captured immediately-after-submit state");

// The poll interval is 3s, so the first poll only fires ~3s after submit --
// wait past that specifically (not a guess) before expecting "processing".
await page.waitForSelector("text=processing", { timeout: 15000 });
await page.screenshot({ path: "/tmp/step2_2_processing.png", fullPage: true });
console.log("captured processing state");

// Wait for completion (real inference on 60 frames) -- poll for the "Complete —" text.
await page.waitForSelector("text=/Complete —/", { timeout: 120000 });
await page.screenshot({ path: "/tmp/step2_3_complete.png", fullPage: true });
console.log("captured complete state");

await browser.close();
