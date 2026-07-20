import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 1400 } });
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });

// video7 is confirmed (Phase 3) to have a homography scale issue --
// excluded_known_issue -- a real case, not a happy-path-only clip.
await page.getByLabel("Clip").selectOption({ label: "video7.mp4" });
await page.getByLabel("frame_limit").fill("90");
await page.getByRole("button", { name: "Submit" }).click();

await page.waitForSelector("text=/Complete —/", { timeout: 180000 });
await page.waitForTimeout(300);
await page.screenshot({ path: "/tmp/step3_result_view.png", fullPage: true });
console.log("captured result view");

// Also dump the raw result JSON to stdout so we can cross-check the screenshot
// against the actual API data, not just eyeball it.
const jobIdText = await page.locator("span.font-mono.text-xs.text-slate-500").first().textContent();
console.log("job_id:", jobIdText);

await browser.close();
