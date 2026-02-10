import { StateMachine } from './state_machine';
import { TemporalEngine, TemporalRule } from './temporal';
import { v4 as uuidv4 } from 'uuid';

export interface FlowExecution {
  id: string;
  machineId: string;
  status: 'running' | 'completed' | 'failed' | 'paused';
  startedAt: Date;
  completedAt?: Date;
  errors: string[];
}

/**
 * Orchestrates state machine execution with temporal constraints.
 */
export class FlowEngine {
  private machines = new Map<string, StateMachine>();
  private temporal = new TemporalEngine();
  private executions = new Map<string, FlowExecution>();

  registerMachine(machine: StateMachine): void {
    this.machines.set(machine.id, machine);
  }

  addTemporalRule(rule: TemporalRule): void {
    this.temporal.addRule(rule);
  }

  /**
   * Start a flow execution.
   */
  start(machineId: string): FlowExecution {
    const machine = this.machines.get(machineId);
    if (!machine) throw new Error(`Machine '${machineId}' not found`);

    machine.reset();
    const exec: FlowExecution = {
      id: uuidv4(),
      machineId,
      status: 'running',
      startedAt: new Date(),
      errors: [],
    };

    if (machine.currentState) {
      this.temporal.recordStateEntry(machine.currentState.id);
    }

    this.executions.set(exec.id, exec);
    return exec;
  }

  /**
   * Send an event to a running execution.
   */
  step(executionId: string, event: string): boolean {
    const exec = this.executions.get(executionId);
    if (!exec || exec.status !== 'running') return false;

    const machine = this.machines.get(exec.machineId);
    if (!machine) return false;

    const transitioned = machine.send(event);

    if (transitioned && machine.currentState) {
      this.temporal.recordStateEntry(machine.currentState.id);

      // Check temporal violations
      const violations = this.temporal.check(machine.currentState.id);
      for (const v of violations) {
        exec.errors.push(v.message);
        if (v.severity === 'error') {
          exec.status = 'failed';
          exec.completedAt = new Date();
          return true;
        }
      }

      if (machine.isInFinalState()) {
        exec.status = machine.currentState.type === 'error' ? 'failed' : 'completed';
        exec.completedAt = new Date();
      }
    }

    return transitioned;
  }

  getExecution(id: string): FlowExecution | undefined {
    return this.executions.get(id);
  }

  getMachine(id: string): StateMachine | undefined {
    return this.machines.get(id);
  }
}
