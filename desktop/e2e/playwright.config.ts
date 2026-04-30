import { defineConfig } from '@playwright/test';

/**
 * Playwright config for Ryzanstein Wails desktop E2E.
 *
 * Per ADR-046:
 *  - Wails apps are singleton processes -> workers MUST be 1.
 *  - The wailsApp fixture spawns desktop.exe with WEBVIEW2 CDP port and
 *    connects via chromium.connectOverCDP, so we do NOT use Playwright's
 *    built-in webServer.
 */
export default defineConfig({
  testDir: './tests',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
});
