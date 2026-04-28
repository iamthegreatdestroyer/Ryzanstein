import { test, expect } from "../fixtures/wailsApp";

/**
 * Sprint 8 — memory persist coverage.
 *
 * Verifies that messages exchanged through `App.SendMessage` are persisted
 * in the chat service history and can be retrieved via `App.GetHistory`.
 * Also exercises `App.ClearHistory`, which mutates state synchronously and
 * emits a `chat:cleared` event over the Wails runtime bus.
 *
 * The offline fallback path in `desktop/main.go::SendMessage` still appends
 * full Message records (user + assistant) to the in-memory history, so the
 * persistence guarantees hold without a live backend.
 *
 * `Service.GetHistory(limit)` returns the LAST `limit` entries when
 * `limit > 0 && limit <= len(history)`; otherwise the full slice. Each
 * Message carries a non-empty `id` and a positive `timestamp`.
 */

const TEST_MODEL = "qwen2.5:0.5b";
const TEST_AGENT = "atlas";

interface Message {
  id: string;
  role: string;
  content: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

interface AppBindings {
  SendMessage: (
    userMessage: string,
    modelID: string,
    agentCodename: string,
  ) => Promise<string>;
  GetHistory: (limit: number) => Promise<Message[]>;
  ClearHistory: () => Promise<void>;
}

interface RuntimeBindings {
  EventsOn: (event: string, cb: (data: unknown) => void) => () => void;
}

declare global {
  interface Window {
    go: { main: { App: AppBindings } };
    runtime: RuntimeBindings;
  }
}

async function waitForBindings(page: import("@playwright/test").Page) {
  await page.waitForFunction(
    () => {
      const w = window as unknown as {
        go?: { main?: { App?: Partial<AppBindings> } };
        runtime?: Partial<RuntimeBindings>;
      };
      const app = w.go?.main?.App;
      return (
        !!app &&
        typeof app.SendMessage === "function" &&
        typeof app.GetHistory === "function" &&
        typeof app.ClearHistory === "function" &&
        typeof w.runtime?.EventsOn === "function"
      );
    },
    null,
    { timeout: 15_000 },
  );
}

test.describe("memory persistence", () => {
  test("SendMessage persists user + assistant Message records", async ({
    wailsApp,
  }) => {
    const { page } = wailsApp;
    await waitForBindings(page);

    // Reset history to a known state.
    await page.evaluate(() => window.go.main.App.ClearHistory());

    const prompts = ["msg1 from e2e", "msg2 from e2e"];
    for (const prompt of prompts) {
      const reply = await page.evaluate(
        async ({ prompt, model, agent }) =>
          window.go.main.App.SendMessage(prompt, model, agent),
        { prompt, model: TEST_MODEL, agent: TEST_AGENT },
      );
      expect(typeof reply).toBe("string");
      expect(reply.length).toBeGreaterThan(0);
    }

    const history = await page.evaluate(() =>
      window.go.main.App.GetHistory(10),
    );

    // Expect at least 2 user + 2 assistant entries.
    expect(Array.isArray(history)).toBe(true);
    expect(history.length).toBeGreaterThanOrEqual(4);

    const userMessages = history.filter((m) => m.role === "user");
    const userContents = userMessages.map((m) => m.content);
    for (const prompt of prompts) {
      expect(userContents).toContain(prompt);
    }

    // Every Message carries an id and positive timestamp.
    for (const msg of history) {
      expect(typeof msg.id).toBe("string");
      expect(msg.id.length).toBeGreaterThan(0);
      expect(typeof msg.timestamp).toBe("number");
      expect(msg.timestamp).toBeGreaterThan(0);
    }
  });

  test("ClearHistory empties the slice and emits chat:cleared", async ({
    wailsApp,
  }) => {
    const { page } = wailsApp;
    await waitForBindings(page);

    // Reset and seed one exchange.
    await page.evaluate(() => window.go.main.App.ClearHistory());
    await page.evaluate(
      async ({ model, agent }) =>
        window.go.main.App.SendMessage("seed message", model, agent),
      { model: TEST_MODEL, agent: TEST_AGENT },
    );

    const seeded = await page.evaluate(() => window.go.main.App.GetHistory(10));
    expect(seeded.length).toBeGreaterThanOrEqual(2);

    // Subscribe to chat:cleared BEFORE invoking the clear, then await it.
    const cleared = await page.evaluate(
      () =>
        new Promise<boolean>((resolve, reject) => {
          const timer = setTimeout(
            () => reject(new Error("chat:cleared event not received")),
            30_000,
          );
          const off = window.runtime.EventsOn("chat:cleared", () => {
            clearTimeout(timer);
            try {
              off?.();
            } catch {
              /* tolerate runtimes that don't return an unsubscribe */
            }
            resolve(true);
          });
          // Fire the clear after the listener is registered.
          window.go.main.App.ClearHistory().catch((err) => {
            clearTimeout(timer);
            reject(err);
          });
        }),
    );
    expect(cleared).toBe(true);

    const after = await page.evaluate(() => window.go.main.App.GetHistory(10));
    expect(Array.isArray(after)).toBe(true);
    expect(after.length).toBe(0);
  });

  test("GetHistory(limit) returns the last N entries", async ({ wailsApp }) => {
    const { page } = wailsApp;
    await waitForBindings(page);

    await page.evaluate(() => window.go.main.App.ClearHistory());

    const prompts = ["limit-1", "limit-2", "limit-3"];
    for (const prompt of prompts) {
      await page.evaluate(
        async ({ prompt, model, agent }) =>
          window.go.main.App.SendMessage(prompt, model, agent),
        { prompt, model: TEST_MODEL, agent: TEST_AGENT },
      );
    }

    const full = await page.evaluate(() => window.go.main.App.GetHistory(0));
    expect(full.length).toBeGreaterThanOrEqual(6); // 3 user + 3 assistant

    const tail = await page.evaluate(() => window.go.main.App.GetHistory(2));
    expect(tail.length).toBe(2);

    // The truncated slice is the suffix of the full slice.
    const fullTail = full.slice(full.length - 2);
    expect(tail.map((m) => m.id)).toEqual(fullTail.map((m) => m.id));
  });
});
