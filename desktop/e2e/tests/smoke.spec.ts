import { test, expect } from '../fixtures/wailsApp';

/**
 * Smoke test: confirm the Wails binary launches, exposes CDP, and the
 * frontend shell mounts its root element.
 *
 * Selector contract (update if HTML changes):
 *   #app          — outermost container div in desktop/frontend/dist/index.html
 *   #chat-messages — primary content area; proves meaningful UI rendered
 *   #message-input — input widget; proves interactive shell is present
 *
 * This is the minimal contract validating ADR-046's harness end-to-end.
 * Coverage specs (agent-invoke, memory-persist, streaming) build on this.
 */

test("wails window opens and frontend shell mounts", async ({ wailsApp }) => {
  const { page } = wailsApp;

  await page.waitForLoadState("domcontentloaded");

  // Root container — defined in desktop/frontend/dist/index.html as
  // <div id="app" class="container">. If this drifts, update that file.
  await expect(page.locator("#app")).toBeVisible({ timeout: 15_000 });

  // Primary content area and input widget confirm meaningful UI rendered.
  await expect(page.locator("#chat-messages")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("#message-input")).toBeVisible({ timeout: 15_000 });
});
