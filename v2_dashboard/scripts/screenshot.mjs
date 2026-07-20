import { chromium } from "playwright";

const url = process.argv[2] || "http://localhost:5173/";
const outPath = process.argv[3] || "/tmp/dashboard_screenshot.png";
const waitMs = Number(process.argv[4] || 800);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto(url, { waitUntil: "networkidle" });
await page.waitForTimeout(waitMs);
await page.screenshot({ path: outPath, fullPage: true });
await browser.close();
console.log("saved", outPath);
