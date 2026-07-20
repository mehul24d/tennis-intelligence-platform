import { chromium } from "playwright";

const clip = process.argv[2] || "video2.mp4";
const frameLimit = process.argv[3] || "20";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 1400 } });
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });

await page.getByLabel("Clip").selectOption({ label: clip });
await page.getByLabel("frame_limit").fill(frameLimit);
await page.getByRole("button", { name: "Submit" }).click();

await page.waitForSelector("text=/Complete —/", { timeout: 180000 });
await page.waitForTimeout(300);
await page.screenshot({ path: `/tmp/step3_short_${clip}_${frameLimit}.png`, fullPage: true });

const jobIdText = await page.locator("span.font-mono.text-xs.text-slate-500").first().textContent();
console.log("job_id:", jobIdText);

await browser.close();
