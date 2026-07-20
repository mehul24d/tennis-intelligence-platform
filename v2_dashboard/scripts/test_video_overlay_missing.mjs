import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });

await page.getByLabel("Clip").selectOption({ label: "video2.mp4" });
await page.getByLabel("frame_limit").fill("20");
await page.getByRole("button", { name: "Submit" }).click();

await page.waitForSelector("text=/Complete —/", { timeout: 180000 });
await page.waitForFunction(() => {
  const v = document.querySelector("video");
  return v && v.readyState >= 2;
});

await page.evaluate(() => {
  document.querySelector("video").currentTime = 5 / 60.0;
});
await page.waitForTimeout(400);
await page.screenshot({ path: "/tmp/step4_missing_far_box.png", fullPage: true });
console.log("captured missing-far-box frame");

await browser.close();
