import {
  IntentSpec,
  IntentGoal,
  IntentConstraint,
  IntentPriority,
  createIntentSpec,
  createGoal,
} from "./types";

/**
 * Parses a declarative intent specification from JSON or structured text.
 */
export class IntentParser {
  /**
   * Parse an intent specification from a JSON object.
   */
  parseJson(input: unknown): IntentSpec {
    if (typeof input !== "object" || input === null) {
      throw new Error("IntentParser: input must be a non-null object");
    }

    const obj = input as Record<string, unknown>;
    const spec = createIntentSpec(
      String(obj["name"] ?? "unnamed"),
      String(obj["description"] ?? ""),
    );

    if (obj["id"] && typeof obj["id"] === "string") {
      spec.id = obj["id"];
    }

    if (Array.isArray(obj["goals"])) {
      spec.goals = (obj["goals"] as unknown[]).map((g) => this.parseGoal(g));
    }

    if (Array.isArray(obj["globalConstraints"])) {
      spec.globalConstraints = (obj["globalConstraints"] as unknown[]).map(
        (c) => this.parseConstraint(c),
      );
    }

    if (obj["metadata"] && typeof obj["metadata"] === "object") {
      spec.metadata = obj["metadata"] as Record<string, unknown>;
    }

    return spec;
  }

  /**
   * Parse a natural language intent description into structured spec.
   * In production, delegates to Ryzanstein LLM for structure extraction.
   */
  parseNaturalLanguage(text: string): IntentSpec {
    const spec = createIntentSpec("nl-intent", text);
    const goal = createGoal(
      "infer",
      text,
      { rawText: text },
      IntentPriority.Normal,
    );
    spec.goals.push(goal);
    return spec;
  }

  private parseGoal(input: unknown): IntentGoal {
    if (typeof input !== "object" || input === null) {
      throw new Error("IntentParser: goal must be a non-null object");
    }
    const obj = input as Record<string, unknown>;
    const goal = createGoal(
      String(obj["action"] ?? "unknown"),
      String(obj["description"] ?? ""),
      (obj["parameters"] as Record<string, unknown>) ?? {},
      (obj["priority"] as IntentPriority) ?? IntentPriority.Normal,
    );

    if (obj["id"] && typeof obj["id"] === "string") {
      goal.id = obj["id"];
    }
    if (Array.isArray(obj["dependsOn"])) {
      goal.dependsOn = obj["dependsOn"] as string[];
    }
    if (Array.isArray(obj["constraints"])) {
      goal.constraints = (obj["constraints"] as unknown[]).map((c) =>
        this.parseConstraint(c),
      );
    }
    return goal;
  }

  private parseConstraint(input: unknown): IntentConstraint {
    if (typeof input !== "object" || input === null) {
      throw new Error("IntentParser: constraint must be a non-null object");
    }
    const obj = input as Record<string, unknown>;
    return {
      type: (obj["type"] as IntentConstraint["type"]) ?? "custom",
      key: String(obj["key"] ?? ""),
      operator: (obj["operator"] as IntentConstraint["operator"]) ?? "eq",
      value: obj["value"],
    };
  }
}
