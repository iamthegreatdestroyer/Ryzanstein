import { IntentSpec, IntentGoal, IntentConstraint } from "./types";

export interface ValidationError {
  path: string;
  message: string;
  severity: "error" | "warning";
}

/**
 * Validates intent specifications for correctness and completeness.
 */
export class IntentValidator {
  /**
   * Validate a full intent specification.
   */
  validate(spec: IntentSpec): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!spec.name || spec.name.trim().length === 0) {
      errors.push({
        path: "name",
        message: "Intent name is required",
        severity: "error",
      });
    }

    if (spec.goals.length === 0) {
      errors.push({
        path: "goals",
        message: "At least one goal is required",
        severity: "error",
      });
    }

    const goalIds = new Set<string>();
    for (let i = 0; i < spec.goals.length; i++) {
      const goal = spec.goals[i];
      if (goalIds.has(goal.id)) {
        errors.push({
          path: `goals[${i}].id`,
          message: `Duplicate goal ID: ${goal.id}`,
          severity: "error",
        });
      }
      goalIds.add(goal.id);
      errors.push(...this.validateGoal(goal, i, goalIds));
    }

    // Check for circular dependencies
    const cycleErrors = this.checkCycles(spec.goals);
    errors.push(...cycleErrors);

    for (let i = 0; i < spec.globalConstraints.length; i++) {
      errors.push(
        ...this.validateConstraint(
          spec.globalConstraints[i],
          `globalConstraints[${i}]`,
        ),
      );
    }

    return errors;
  }

  /**
   * Check if spec is valid (no errors).
   */
  isValid(spec: IntentSpec): boolean {
    return (
      this.validate(spec).filter((e) => e.severity === "error").length === 0
    );
  }

  private validateGoal(
    goal: IntentGoal,
    index: number,
    validIds: Set<string>,
  ): ValidationError[] {
    const errors: ValidationError[] = [];
    const prefix = `goals[${index}]`;

    if (!goal.action || goal.action.trim().length === 0) {
      errors.push({
        path: `${prefix}.action`,
        message: "Goal action is required",
        severity: "error",
      });
    }

    for (const dep of goal.dependsOn) {
      if (!validIds.has(dep)) {
        errors.push({
          path: `${prefix}.dependsOn`,
          message: `Dependency '${dep}' references unknown goal`,
          severity: "warning",
        });
      }
    }

    for (let i = 0; i < goal.constraints.length; i++) {
      errors.push(
        ...this.validateConstraint(
          goal.constraints[i],
          `${prefix}.constraints[${i}]`,
        ),
      );
    }

    return errors;
  }

  private validateConstraint(
    constraint: IntentConstraint,
    path: string,
  ): ValidationError[] {
    const errors: ValidationError[] = [];
    const validTypes = ["time", "resource", "quality", "cost", "custom"];
    if (!validTypes.includes(constraint.type)) {
      errors.push({
        path: `${path}.type`,
        message: `Invalid constraint type: ${constraint.type}`,
        severity: "error",
      });
    }
    const validOps = ["lt", "gt", "eq", "lte", "gte", "in", "not_in"];
    if (!validOps.includes(constraint.operator)) {
      errors.push({
        path: `${path}.operator`,
        message: `Invalid operator: ${constraint.operator}`,
        severity: "error",
      });
    }
    return errors;
  }

  private checkCycles(goals: IntentGoal[]): ValidationError[] {
    const errors: ValidationError[] = [];
    const adj = new Map<string, string[]>();
    for (const g of goals) {
      adj.set(g.id, g.dependsOn);
    }

    const visited = new Set<string>();
    const inStack = new Set<string>();

    const dfs = (id: string): boolean => {
      if (inStack.has(id)) return true;
      if (visited.has(id)) return false;
      visited.add(id);
      inStack.add(id);
      for (const dep of adj.get(id) ?? []) {
        if (dfs(dep)) return true;
      }
      inStack.delete(id);
      return false;
    };

    for (const g of goals) {
      if (dfs(g.id)) {
        errors.push({
          path: `goals`,
          message: `Circular dependency detected involving goal: ${g.id}`,
          severity: "error",
        });
        break;
      }
    }

    return errors;
  }
}
