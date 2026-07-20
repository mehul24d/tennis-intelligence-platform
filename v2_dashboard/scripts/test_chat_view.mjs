import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 1400 } });
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });

// Submit a real job first so the chat has live CV features to fuse.
await page.getByLabel("Clip").selectOption({ label: "video1.mp4" });
await page.getByLabel("frame_limit").fill("120");
await page.getByRole("button", { name: "Submit" }).click();
await page.waitForSelector("text=/Complete —/", { timeout: 180000 });
console.log("job complete, asking question now");

// Same question verified in Phase 4 to produce an offered-but-uncited source (L2).
await page.getByPlaceholder("Ask a question…").fill(
  "How is the near player performing so far in this clip, and how does that compare to Cameron Norrie's historical Wimbledon results?"
);
await page.getByPlaceholder("player (optional)").fill("Cameron Norrie");
await page.getByRole("button", { name: "Ask" }).click();

// Real Gemini call -- give it real time.
await page.waitForSelector("text=/cited \\/ /", { timeout: 90000 });
await page.waitForTimeout(300);
await page.screenshot({ path: "/tmp/step5_chat_result.png", fullPage: true });
console.log("captured chat result");

await browser.close();
