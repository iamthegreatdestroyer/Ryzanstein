import { test, expect } from "../fixtures/wailsApp";

/**
 * Sprint 8 — agent invoke coverage.
 *
 * Verifies that the Wails-bound `App.SendMessage` is reachable from the
 * renderer over the Playwright/CDP bridge and returns a non-empty string.
 *
 * The desktop binary ships with an offline fallback path (see
 * `desktop/main.go::SendMessage`): when the inference API at
 * RYZANSTEIN_API_URL is unreachable, the bound method still resolves with
 * a string of the form `"[Offline Mode] {agent} received your message..."`.
 * That guarantees this spec is deterministic in CI even without a live
 * backend — we accept either a real model reply OR the offline fallback.
 */

const TEST_PROMPT = "hello from e2e";
const TEST_MODEL = "qwen2.5:0.5b";
const TEST_AGENT = "atlas";

test("App.SendMessage returns a string (real response or offline fallback)", async ({
  wailsApp,
}) => {
  const { page } = wailsApp;

  // Wait for the Wails runtime to expose the bound App methods.
  await page.waitForFunction(
    () => {
      const w = window as unknown as {
        go?: { main?: { App?: { SendMessage?: unknown } } };
      };
      return typeof w.go?.main?.App?.SendMessage === "function";
    },
    null,
    { timeout: 15_000 },
  );

  const response = await page.evaluate(
    async ({ prompt, model, agent }) => {
      const w = window as unknown as {
        go: {
          main: {
            App: {
              SendMessage: (
                userMessage: string,
                modelID: string,
                agentCodename: string,
              ) => Promise<string>;
            };
          };
        };
      };
      return w.go.main.App.SendMessage(prompt, model, agent);
    },
    { prompt: TEST_PROMPT, model: TEST_MODEL, agent: TEST_AGENT },
  );

  expect(typeof response).toBe("string");
  expect(response.length).toBeGreaterThan(0);

  // Accept either a real model reply or the deterministic offline fallback.
  const isOfflineFallback = /\[Offline Mode\]/i.test(response);
  const isRealReply = !isOfflineFallback && response.trim().length > 0;
  expect(isOfflineFallback || isRealReply).toBe(true);
});

test("App.SendMessage emits chat:response event with assistant message", async ({
  wailsApp,
}) => {
  const { page } = wailsApp;

  await page.waitForFunction(
    () => {
      const w = window as unknown as {
        go?: { main?: { App?: { SendMessage?: unknown } } };
        runtime?: { EventsOn?: unknown };
      };
      return (
        typeof w.go?.main?.App?.SendMessage === "function" &&
        typeof w.runtime?.EventsOn === "function"
      );
    },
    null,
    { timeout: 15_000 },
  );

  // Subscribe to chat:response BEFORE invoking SendMessage so we don't race.
  const eventPromise = page.evaluate(
    () =>
      new Promise<unknown>((resolve, reject) => {
        const w = window as unknown as {
          runtime: {
            EventsOn: (
              event: string,
              cb: (payload: unknown) => void,
            ) => void;
          };
        };
        const timer = setTimeout(
          () => reject(new Error("chat:response not received within 30s")),
          30_000,
        );
        w.runtime.EventsOn("chat:response", (payload: unknown) => {
          clearTimeout(timer);
          resolve(payload);
        });
      }),
  );

  await page.evaluate(
    async ({ prompt, model, agent }) => {
      const w = window as unknown as {
        go: {
          main: {
            App: {
              SendMessage: (
                userMessage: string,
                modelID: string,
                agentCodename: string,
              ) => Promise<string>;
            };
          };
        };
      };
      await w.go.main.App.SendMessage(prompt, model, agent);
    },
    { prompt: TEST_PROMPT, model: TEST_MODEL, agent: TEST_AGENT },
  );

  const payload = (await eventPromise) as {
    role?: string;
    content?: string;
    id?: string;
  };

  expect(payload).toBeTruthy();
  expect(payload.role).toBe("assistant");
  expect(typeof payload.content).toBe("string");
  expect((payload.content ?? "").length).toBeGreaterThan(0);
});
