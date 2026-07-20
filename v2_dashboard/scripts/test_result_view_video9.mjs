import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 1400 } });
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });

// video9's far-player pose is documented (Phase 3) to fail entirely in spot
// checks -- testing whether that reproduces here as a real not_detected case.
await page.getByLabel("Clip").selectOption({ label: "video9.mp4" });
await page.getByLabel("frame_limit").fill("90");
await page.getByRole("button", { name: "Submit" }).click();

await page.waitForSelector("text=/Complete —/", { timeout: 180000 });
await page.waitForTimeout(300);
await page.screenshot({ path: "/tmp/step3_video9_result.png", fullPage: true });

const jobIdText = await page.locator("span.font-mono.text-xs.text-slate-500").first().textContent();
console.log("job_id:", jobIdText);

await browser.close();
