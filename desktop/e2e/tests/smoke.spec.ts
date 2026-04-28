import { test, expect } from '../fixtures/wailsApp';

/**
 * Smoke test: confirm the Wails binary launches, exposes CDP, and the
 * Svelte frontend mounts its root element.
 *
 * This is the minimal contract validating ADR-046's harness end-to-end.
 * Coverage specs (agent-invoke, memory-persist, streaming) build on this.
 */

test('wails window opens and svelte root mounts', async ({ wailsApp }) => {
  const { page } = wailsApp;

  // Wait for the Svelte app shell. The frontend mounts into <div id="app">.
  // If selector drifts, check desktop/frontend/src/main.ts and update accordingly.
  await page.waitForLoadState('domcontentloaded');
  const root = page.locator('#app');
  await expect(root).toBeVisible({ timeout: 15_000 });
});
