import { IntentSpec, IntentResult, IntentStatus, GoalResult } from "./types";
import { IntentValidator } from "./validator";

/**
 * Core intent execution engine. Resolves intent specs
 * into executable plans and coordinates goal completion.
 */
export class IntentEngine {
  private validator = new IntentValidator();
  private results = new Map<string, IntentResult>();

  /**
   * Execute an intent specification.
   */
  async execute(spec: IntentSpec): Promise<IntentResult> {
    const validationErrors = this.validator.validate(spec);
    const errors = validationErrors.filter((e) => e.severity === "error");

    if (errors.length > 0) {
      return {
        specId: spec.id,
        status: IntentStatus.Failed,
        goalResults: new Map(),
        startedAt: new Date(),
        completedAt: new Date(),
        errors: errors.map((e) => `${e.path}: ${e.message}`),
      };
    }

    const result: IntentResult = {
      specId: spec.id,
      status: IntentStatus.Executing,
      goalResults: new Map(),
      startedAt: new Date(),
      errors: [],
    };

    // Topological sort for dependency resolution
    const order = this.topologicalSort(spec);

    for (const goal of order) {
      // Check dependencies completed
      const depsFailed = goal.dependsOn.some((depId) => {
        const depResult = result.goalResults.get(depId);
        return !depResult || depResult.status === IntentStatus.Failed;
      });

      if (depsFailed) {
        result.goalResults.set(goal.id, {
          goalId: goal.id,
          status: IntentStatus.Failed,
          error: "Dependency failed",
          durationMs: 0,
        });
        continue;
      }

      const start = Date.now();
      try {
        // In production, dispatches to Ryzanstein agent mesh
        const output = await this.executeGoal(goal.action, goal.parameters);
        result.goalResults.set(goal.id, {
          goalId: goal.id,
          status: IntentStatus.Completed,
          output,
          durationMs: Date.now() - start,
        });
      } catch (err) {
        result.goalResults.set(goal.id, {
          goalId: goal.id,
          status: IntentStatus.Failed,
          error: String(err),
          durationMs: Date.now() - start,
        });
      }
    }

    const allCompleted = [...result.goalResults.values()].every(
      (r) => r.status === IntentStatus.Completed,
    );
    result.status = allCompleted ? IntentStatus.Completed : IntentStatus.Failed;
    result.completedAt = new Date();

    this.results.set(spec.id, result);
    return result;
  }

  /**
   * Get a previous execution result.
   */
  getResult(specId: string): IntentResult | undefined {
    return this.results.get(specId);
  }

  private topologicalSort(spec: IntentSpec) {
    const goalMap = new Map(spec.goals.map((g) => [g.id, g]));
    const visited = new Set<string>();
    const order: typeof spec.goals = [];

    const visit = (id: string) => {
      if (visited.has(id)) return;
      visited.add(id);
      const goal = goalMap.get(id);
      if (!goal) return;
      for (const dep of goal.dependsOn) {
        visit(dep);
      }
      order.push(goal);
    };

    for (const goal of spec.goals) {
      visit(goal.id);
    }
    return order;
  }

  private async executeGoal(
    action: string,
    params: Record<string, unknown>,
  ): Promise<unknown> {
    // Stub: in production routes to Ryzanstein agent
    return { action, params, result: "simulated_success" };
  }
}
