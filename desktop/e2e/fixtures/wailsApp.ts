import { test as base, chromium, type Browser, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { createServer } from 'node:net';
import { resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

/**
 * Wails E2E fixture per ADR-046.
 *
 * Strategy:
 *   1. Pick a free TCP port via net.createServer({ port: 0 }).
 *   2. Spawn ../desktop.exe with WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS
 *      passing --remote-debugging-port=<port> to the embedded WebView2.
 *   3. Poll until chromium.connectOverCDP succeeds.
 *   4. Return the first context's first page (the Wails main window).
 *
 * Wails apps are singleton: workers=1 in playwright.config.ts.
 */

const BINARY_PATH = resolve(__dirname, '..', '..', 'desktop.exe');
const CDP_CONNECT_TIMEOUT_MS = 30_000;
const CDP_POLL_INTERVAL_MS = 250;

async function getFreePort(): Promise<number> {
  return new Promise((resolveFn, rejectFn) => {
    const srv = createServer();
    srv.unref();
    srv.on('error', rejectFn);
    srv.listen(0, () => {
      const addr = srv.address();
      if (addr && typeof addr === 'object') {
        const port = addr.port;
        srv.close(() => resolveFn(port));
      } else {
        srv.close();
        rejectFn(new Error('failed to acquire free port'));
      }
    });
  });
}

async function connectWithRetry(port: number): Promise<Browser> {
  const deadline = Date.now() + CDP_CONNECT_TIMEOUT_MS;
  let lastErr: unknown;
  while (Date.now() < deadline) {
    try {
      return await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
    } catch (err) {
      lastErr = err;
      await sleep(CDP_POLL_INTERVAL_MS);
    }
  }
  throw new Error(
    `CDP connect timeout on port ${port} after ${CDP_CONNECT_TIMEOUT_MS}ms: ${String(lastErr)}`,
  );
}

type WailsFixtures = {
  wailsApp: { page: Page; browser: Browser; proc: ChildProcess; port: number };
};

export const test = base.extend<WailsFixtures>({
  // eslint-disable-next-line no-empty-pattern
  wailsApp: async ({}, use) => {
    const port = await getFreePort();
    const proc = spawn(BINARY_PATH, [], {
      env: {
        ...process.env,
        WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${port}`,
      },
      stdio: 'pipe',
      windowsHide: false,
    });

    proc.on('error', (err) => {
      // Surface spawn failures (missing binary, permissions, etc.)
      console.error(`[wailsApp] spawn error: ${err.message}`);
    });

    const browser = await connectWithRetry(port);

    // Wails opens exactly one BrowserContext with one Page (the main window).
    const contexts = browser.contexts();
    if (contexts.length === 0) {
      throw new Error('no BrowserContext exposed by Wails CDP');
    }
    const ctx = contexts[0];
    const pages = ctx.pages();
    const page = pages.length > 0 ? pages[0] : await ctx.waitForEvent('page');

    await use({ page, browser, proc, port });

    // Teardown: close CDP, then terminate the Wails process.
    try {
      await browser.close();
    } catch {
      /* ignore */
    }
    if (proc.exitCode === null && !proc.killed) {
      proc.kill();
    }
  },
});

export const expect = test.expect;
