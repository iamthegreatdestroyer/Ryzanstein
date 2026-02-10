import { IntentParser } from "../parser";
import { IntentValidator } from "../validator";
import { IntentEngine } from "../engine";
import {
  createIntentSpec,
  createGoal,
  IntentPriority,
  IntentStatus,
} from "../types";
import { RyzansteinIntentClient } from "../ryzanstein";

describe("IntentParser", () => {
  const parser = new IntentParser();

  test("parses valid JSON spec", () => {
    const spec = parser.parseJson({
      name: "test-intent",
      description: "A test",
      goals: [{ action: "inference", description: "run model" }],
    });
    expect(spec.name).toBe("test-intent");
    expect(spec.goals).toHaveLength(1);
  });

  test("throws on null input", () => {
    expect(() => parser.parseJson(null)).toThrow();
  });

  test("parses natural language", () => {
    const spec = parser.parseNaturalLanguage("generate a summary of this code");
    expect(spec.goals).toHaveLength(1);
    expect(spec.goals[0].action).toBe("infer");
  });

  test("handles missing fields gracefully", () => {
    const spec = parser.parseJson({});
    expect(spec.name).toBe("unnamed");
    expect(spec.goals).toHaveLength(0);
  });
});

describe("IntentValidator", () => {
  const validator = new IntentValidator();

  test("valid spec passes", () => {
    const spec = createIntentSpec("test", "desc");
    spec.goals.push(createGoal("inference", "run model"));
    expect(validator.isValid(spec)).toBe(true);
  });

  test("empty name fails", () => {
    const spec = createIntentSpec("", "desc");
    spec.goals.push(createGoal("inference", "run model"));
    const errors = validator.validate(spec);
    expect(errors.some((e) => e.path === "name")).toBe(true);
  });

  test("no goals fails", () => {
    const spec = createIntentSpec("test", "desc");
    const errors = validator.validate(spec);
    expect(errors.some((e) => e.path === "goals")).toBe(true);
  });

  test("circular dependency detected", () => {
    const spec = createIntentSpec("test", "desc");
    const g1 = createGoal("a", "step a");
    const g2 = createGoal("b", "step b");
    g1.dependsOn = [g2.id];
    g2.dependsOn = [g1.id];
    spec.goals = [g1, g2];
    const errors = validator.validate(spec);
    expect(errors.some((e) => e.message.includes("Circular"))).toBe(true);
  });

  test("invalid constraint type", () => {
    const spec = createIntentSpec("test", "desc");
    const goal = createGoal("a", "step a");
    goal.constraints = [
      { type: "invalid" as any, key: "x", operator: "eq", value: 1 },
    ];
    spec.goals = [goal];
    const errors = validator.validate(spec);
    expect(
      errors.some((e) => e.message.includes("Invalid constraint type")),
    ).toBe(true);
  });
});

describe("IntentEngine", () => {
  const engine = new IntentEngine();

  test("executes simple spec", async () => {
    const spec = createIntentSpec("test", "simple test");
    spec.goals.push(createGoal("inference", "run model"));
    const result = await engine.execute(spec);
    expect(result.status).toBe(IntentStatus.Completed);
    expect(result.goalResults.size).toBe(1);
  });

  test("fails on invalid spec", async () => {
    const spec = createIntentSpec("", "no name");
    const result = await engine.execute(spec);
    expect(result.status).toBe(IntentStatus.Failed);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  test("handles dependency chain", async () => {
    const spec = createIntentSpec("chain", "dependency chain");
    const g1 = createGoal("step1", "first step");
    const g2 = createGoal("step2", "second step");
    g2.dependsOn = [g1.id];
    spec.goals = [g1, g2];
    const result = await engine.execute(spec);
    expect(result.status).toBe(IntentStatus.Completed);
    expect(result.goalResults.size).toBe(2);
  });

  test("stores results for retrieval", async () => {
    const spec = createIntentSpec("stored", "check storage");
    spec.goals.push(createGoal("a", "goal a"));
    await engine.execute(spec);
    const retrieved = engine.getResult(spec.id);
    expect(retrieved).toBeDefined();
    expect(retrieved?.specId).toBe(spec.id);
  });
});

describe("RyzansteinIntentClient", () => {
  const client = new RyzansteinIntentClient();

  test("fallback resolve identifies actions", async () => {
    const result = await client.resolveIntent("generate a summary");
    expect(result.action).toBe("inference");
  });

  test("fallback resolve handles unknown", async () => {
    const result = await client.resolveIntent("something random");
    expect(result.action).toBe("generic");
  });

  test("extract parameters returns word count", async () => {
    const params = await client.extractParameters("hello world foo");
    expect(params["wordCount"]).toBe(3);
  });
});
