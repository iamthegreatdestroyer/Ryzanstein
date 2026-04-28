import { test, expect } from "../fixtures/wailsApp";

const TEST_PROMPT = "Stream a single short sentence.";
const TEST_MODEL = "qwen2.5:0.5b";
const TEST_AGENT = "atlas";

test("App.SendMessageStream emits streamStart, streamToken*, streamEnd", async ({
  wailsApp,
}) => {
  const { page } = wailsApp;

  await page.waitForFunction(
    () => {
      const w = window as unknown as {
        go?: { main?: { App?: { SendMessageStream?: unknown } } };
        runtime?: { EventsOn?: unknown };
      };
      return (
        typeof w.go?.main?.App?.SendMessageStream === "function" &&
        typeof w.runtime?.EventsOn === "function"
      );
    },
    null,
    { timeout: 15_000 },
  );

  // Subscribe to all stream events BEFORE invoking SendMessageStream.
  const streamPromise = page.evaluate(
    () =>
      new Promise<{
        startReceived: boolean;
        tokens: string[];
        end: { role?: string; content?: string } | null;
        errorMessage: string | null;
      }>((resolve) => {
        const w = window as unknown as {
          runtime: {
            EventsOn: (
              event: string,
              cb: (payload: unknown) => void,
            ) => void;
          };
        };

        const tokens: string[] = [];
        let startReceived = false;
        let errorMessage: string | null = null;

        const finalize = (end: unknown) => {
          resolve({
            startReceived,
            tokens,
            end: end as { role?: string; content?: string } | null,
            errorMessage,
          });
        };

        const timer = setTimeout(() => {
          // Resolve with whatever we collected; assertions below tolerate
          // offline / no-backend conditions.
          finalize(null);
        }, 60_000);

        w.runtime.EventsOn("chat:streamStart", () => {
          startReceived = true;
        });
        w.runtime.EventsOn("chat:streamToken", (payload: unknown) => {
          if (typeof payload === "string") {
            tokens.push(payload);
          }
        });
        w.runtime.EventsOn("chat:streamError", (payload: unknown) => {
          if (typeof payload === "string") {
            errorMessage = payload;
          } else {
            errorMessage = JSON.stringify(payload);
          }
        });
        w.runtime.EventsOn("chat:streamEnd", (payload: unknown) => {
          clearTimeout(timer);
          finalize(payload);
        });
      }),
  );

  await page.evaluate(
    async ({ prompt, model, agent }) => {
      const w = window as unknown as {
        go: {
          main: {
            App: {
              SendMessageStream: (
                userMessage: string,
                modelID: string,
                agentCodename: string,
              ) => Promise<void>;
            };
          };
        };
      };
      await w.go.main.App.SendMessageStream(prompt, model, agent);
    },
    { prompt: TEST_PROMPT, model: TEST_MODEL, agent: TEST_AGENT },
  );

  const result = await streamPromise;

  // streamStart should always fire — it is emitted synchronously before the
  // goroutine kicks off, regardless of backend availability.
  expect(result.startReceived).toBe(true);

  // streamEnd must arrive with an assistant Message payload. In offline
  // mode the backend produces a deterministic "[Offline Mode]" string,
  // and tokens may be empty (no streaming when API is unreachable).
  expect(result.end).toBeTruthy();
  const end = result.end!;
  expect(end.role).toBe("assistant");
  expect(typeof end.content).toBe("string");
  expect((end.content ?? "").length).toBeGreaterThan(0);

  const isOfflineFallback = /\[Offline Mode\]/i.test(end.content ?? "");

  if (!isOfflineFallback) {
    // Real backend: tokens should have streamed and concatenate to content.
    expect(result.tokens.length).toBeGreaterThan(0);
    expect(result.tokens.join("")).toBe(end.content);
  } else {
    // Offline fallback path: tokens may be empty, error may be set. Accept.
    expect(Array.isArray(result.tokens)).toBe(true);
  }
});
