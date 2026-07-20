import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 1500 } });
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });

await page.getByLabel("Clip").selectOption({ label: "video1.mp4" });
await page.getByLabel("frame_limit").fill("60");
await page.getByRole("button", { name: "Submit" }).click();
await page.waitForSelector("text=/Complete —/", { timeout: 180000 });
console.log("job complete");

// Case 1: no match_id -- both fields should honestly report not_available.
await page.getByRole("button", { name: "Check" }).click();
await page.waitForSelector("text=/not available/", { timeout: 15000 });
await page.waitForTimeout(200);
await page.screenshot({ path: "/tmp/step6_no_match_id.png", fullPage: true });
console.log("captured no-match_id case");

// Case 2: real regression match_id (Djokovic/Goffin, 0.7818) -- first call pays
// v1's one-time load_replay_context() cost (~15-20s).
await page.getByPlaceholder(/e.g. 20190710/).fill("20190710-M-Wimbledon-QF-Novak_Djokovic-David_Goffin");
await page.getByRole("button", { name: "Check" }).click();
await page.waitForSelector("text=/78\\.18%/", { timeout: 60000 });
await page.waitForTimeout(200);
await page.screenshot({ path: "/tmp/step6_real_match_id.png", fullPage: true });
console.log("captured real match_id case");

await browser.close();
