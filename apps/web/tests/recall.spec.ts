import { expect, test } from "@playwright/test";

test("recalls an event with bounded evidence", async ({ page }) => {
  await page.goto("/");
  const sessionPicker = page.locator(".session-switcher select");
  if (await sessionPicker.count()) await sessionPicker.selectOption("ses_kitchen_demo");
  await expect(page.getByText("What the agent remembers")).toBeVisible();
  await page.getByRole("button", { name: "Play memory replay" }).click();
  await expect(page.getByRole("button", { name: "Pause memory replay" })).toBeVisible();
  await expect(page.locator(".video-time strong")).not.toHaveText("04:44");
  await page.getByRole("button", { name: "Pause memory replay" }).click();
  await expect(page.getByRole("button", { name: "Play memory replay" })).toBeVisible();
  await page.getByRole("button", { name: /什么时候有人把杯子放到桌上/ }).click();
  await expect(page.getByText(/02:08–02:17/)).toBeVisible();
  await expect(page.getByText("82% confidence")).toBeVisible();
  await expect(page.getByText("Snapshot only")).toBeVisible();
  await expect(page.locator(".evidence-card img")).toHaveCount(3);
});

test("prepares a strict MP4 ingest session", async ({ page }) => {
  await page.route("**/api/v1/sessions/upload", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 400));
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({
        session: { id: "ses_browser_test", status: "queued" },
        job: { id: "job_browser_test", status: "queued", progress: 0, stage: "queued" },
      }),
    });
  });
  await page.route("**/api/v1/jobs/job_browser_test", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "job_browser_test", status: "failed", progress: 0.15, stage: "failed",
        error: { detail: "Snapshot ingest failed. Confirm the file is a decodable MP4 and retry." },
      }),
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "New stream" }).click();
  await expect(page.getByRole("heading", { name: "Ingest an MP4 stream" })).toBeVisible();
  await page.getByPlaceholder("Kitchen · evening stream").fill("Workshop camera");
  await page.locator('input[type="file"]').setInputFiles({
    name: "camera.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("00000018ftypmp42", "utf8"),
  });
  await expect(page.getByText("camera.mp4")).toBeVisible();
  await expect(page.getByRole("button", { name: /Start single-pass ingest/ })).toBeEnabled();
  await expect(page.getByText(/Restart with Hybrid V3|Query workers receive the snapshot/)).toBeVisible();
  await page.getByRole("button", { name: /Start single-pass ingest/ }).click();
  await expect(page.getByRole("button", { name: "Uploading source…" })).toBeDisabled();
  await expect(page.getByText(/Snapshot ingest failed/)).toBeVisible({ timeout: 4000 });
  await expect(page.getByRole("button", { name: /Retry ingest/ })).toBeEnabled();
});

test("attaches a smooth local source preview without uploading it again", async ({ page }) => {
  await page.route(/\/api\/v1\/sessions$/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([{
        id: "ses_snapshot_preview", name: "Uploaded memory", mode: "strict",
        source_type: "upload", status: "ready", duration_s: 30, observed_until_s: 30,
        memory_budget_bytes: 1048576, state_bytes: 280000, retained_frame_count: 1,
        snapshot_id: "rho_1.00", snapshot_sha256: "abc123", suggested_queries: [],
        timeline: [{ frame_ref: "frame.jpg", timestamp_s: 0, importance: 0.65 }],
      }]),
    });
  });
  await page.goto("/");
  const picker = page.locator(".local-source-picker input");
  await expect(page.getByText("Attach local source")).toBeVisible();
  await picker.setInputFiles({
    name: "original.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("00000018ftypmp42", "utf8"),
  });
  const video = page.getByLabel("Local source preview: original.mp4");
  await expect(video).toBeVisible();
  await expect(video).toHaveAttribute("src", /^blob:/);
  await expect(page.getByText("Local source preview")).toBeVisible();
});
