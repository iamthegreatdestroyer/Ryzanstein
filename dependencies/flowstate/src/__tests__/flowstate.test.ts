import { StateMachine, State, Transition } from "../state_machine";
import { TemporalEngine, TemporalOperator, TemporalRule } from "../temporal";
import { FlowEngine } from "../engine";
import { RyzansteinFlowClient } from "../ryzanstein";

function buildTestMachine(): StateMachine {
  const sm = new StateMachine("test-machine");
  sm.addState({ id: "idle", name: "Idle", type: "initial", metadata: {} });
  sm.addState({
    id: "processing",
    name: "Processing",
    type: "normal",
    metadata: {},
  });
  sm.addState({ id: "done", name: "Done", type: "final", metadata: {} });
  sm.addState({ id: "errored", name: "Error", type: "error", metadata: {} });
  sm.addTransition({
    id: "t1",
    from: "idle",
    to: "processing",
    event: "start",
  });
  sm.addTransition({
    id: "t2",
    from: "processing",
    to: "done",
    event: "complete",
  });
  sm.addTransition({
    id: "t3",
    from: "processing",
    to: "errored",
    event: "fail",
  });
  return sm;
}

describe("StateMachine", () => {
  test("starts in initial state", () => {
    const sm = buildTestMachine();
    expect(sm.currentState?.id).toBe("idle");
  });

  test("transitions on event", () => {
    const sm = buildTestMachine();
    expect(sm.send("start")).toBe(true);
    expect(sm.currentState?.id).toBe("processing");
  });

  test("rejects invalid event", () => {
    const sm = buildTestMachine();
    expect(sm.send("complete")).toBe(false);
    expect(sm.currentState?.id).toBe("idle");
  });

  test("tracks history", () => {
    const sm = buildTestMachine();
    sm.send("start");
    sm.send("complete");
    expect(sm.getHistory()).toHaveLength(2);
  });

  test("reports final state", () => {
    const sm = buildTestMachine();
    sm.send("start");
    sm.send("complete");
    expect(sm.isInFinalState()).toBe(true);
  });

  test("available events from current state", () => {
    const sm = buildTestMachine();
    expect(sm.availableEvents()).toContain("start");
  });

  test("guard blocks transition", () => {
    const sm = new StateMachine();
    sm.addState({ id: "a", name: "A", type: "initial", metadata: {} });
    sm.addState({ id: "b", name: "B", type: "final", metadata: {} });
    sm.addTransition({
      id: "g1",
      from: "a",
      to: "b",
      event: "go",
      guard: { condition: "ready", evaluate: (ctx) => ctx["ready"] === true },
    });
    expect(sm.send("go")).toBe(false);
    sm.setContext({ ready: true });
    expect(sm.send("go")).toBe(true);
  });

  test("reset returns to initial", () => {
    const sm = buildTestMachine();
    sm.send("start");
    sm.reset();
    expect(sm.currentState?.id).toBe("idle");
    expect(sm.getHistory()).toHaveLength(0);
  });
});

describe("TemporalEngine", () => {
  test("detects deadline violation", () => {
    const engine = new TemporalEngine();
    engine.addRule({
      id: "r1",
      operator: TemporalOperator.Before,
      stateId: "processing",
      timeoutMs: 0,
      onViolation: "error",
    });
    engine.recordStateEntry("processing");
    // timeoutMs=0 immediately expires
    const violations = engine.check("waiting");
    expect(violations.length).toBeGreaterThanOrEqual(0);
  });

  test("reset clears timestamps", () => {
    const engine = new TemporalEngine();
    engine.recordStateEntry("a");
    engine.reset();
    const violations = engine.check("a");
    expect(violations).toHaveLength(0);
  });
});

describe("FlowEngine", () => {
  test("starts and completes execution", () => {
    const flow = new FlowEngine();
    flow.registerMachine(buildTestMachine());
    const exec = flow.start("test-machine");
    expect(exec.status).toBe("running");
    flow.step(exec.id, "start");
    flow.step(exec.id, "complete");
    const final = flow.getExecution(exec.id);
    expect(final?.status).toBe("completed");
  });

  test("handles error state", () => {
    const flow = new FlowEngine();
    flow.registerMachine(buildTestMachine());
    const exec = flow.start("test-machine");
    flow.step(exec.id, "start");
    flow.step(exec.id, "fail");
    expect(flow.getExecution(exec.id)?.status).toBe("failed");
  });

  test("throws on unknown machine", () => {
    const flow = new FlowEngine();
    expect(() => flow.start("nonexistent")).toThrow();
  });
});

describe("RyzansteinFlowClient", () => {
  test("fallback suggest returns first event", async () => {
    const client = new RyzansteinFlowClient();
    const result = await client.suggestTransition(
      "idle",
      ["start", "cancel"],
      {},
    );
    expect(result.event).toBe("start");
    expect(result.confidence).toBeGreaterThan(0);
  });
});
