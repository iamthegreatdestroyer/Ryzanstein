/**
 * Temporal logic operators for time-constrained workflow transitions.
 */

export enum TemporalOperator {
  /** State must eventually hold */
  Eventually = "eventually",
  /** State must always hold */
  Always = "always",
  /** State holds until another condition */
  Until = "until",
  /** Deadline: condition must hold before time T */
  Before = "before",
  /** Delay: condition holds after time T */
  After = "after",
}

export interface TemporalRule {
  id: string;
  operator: TemporalOperator;
  stateId: string;
  condition?: string;
  timeoutMs?: number;
  targetStateId?: string;
  onViolation?: "error" | "warn" | "retry";
}

/**
 * Evaluates temporal constraints on workflow execution.
 */
export class TemporalEngine {
  private rules: TemporalRule[] = [];
  private stateTimestamps = new Map<string, number>();

  addRule(rule: TemporalRule): void {
    this.rules.push(rule);
  }

  recordStateEntry(stateId: string): void {
    this.stateTimestamps.set(stateId, Date.now());
  }

  /**
   * Check all temporal rules against current state.
   */
  check(currentStateId: string): TemporalViolation[] {
    const violations: TemporalViolation[] = [];
    const now = Date.now();

    for (const rule of this.rules) {
      switch (rule.operator) {
        case TemporalOperator.Before: {
          if (rule.timeoutMs && rule.stateId !== currentStateId) {
            const entryTime = this.stateTimestamps.get(rule.stateId);
            if (entryTime && now - entryTime > rule.timeoutMs) {
              violations.push({
                ruleId: rule.id,
                message: `State '${rule.stateId}' exceeded deadline of ${rule.timeoutMs}ms`,
                severity: rule.onViolation ?? "error",
              });
            }
          }
          break;
        }
        case TemporalOperator.Eventually: {
          if (rule.timeoutMs) {
            const firstEntry = [...this.stateTimestamps.entries()][0];
            if (firstEntry && now - firstEntry[1] > rule.timeoutMs) {
              if (!this.stateTimestamps.has(rule.stateId)) {
                violations.push({
                  ruleId: rule.id,
                  message: `State '${rule.stateId}' was never reached within ${rule.timeoutMs}ms`,
                  severity: rule.onViolation ?? "warn",
                });
              }
            }
          }
          break;
        }
        default:
          break;
      }
    }

    return violations;
  }

  reset(): void {
    this.stateTimestamps.clear();
  }
}

export interface TemporalViolation {
  ruleId: string;
  message: string;
  severity: "error" | "warn" | "retry";
}
