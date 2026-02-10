import { v4 as uuidv4 } from 'uuid';

/**
 * Priority levels for intent execution.
 */
export enum IntentPriority {
  Critical = 'critical',
  High = 'high',
  Normal = 'normal',
  Low = 'low',
  Background = 'background',
}

/**
 * Execution status of an intent.
 */
export enum IntentStatus {
  Pending = 'pending',
  Parsing = 'parsing',
  Validated = 'validated',
  Executing = 'executing',
  Completed = 'completed',
  Failed = 'failed',
  Cancelled = 'cancelled',
}

/**
 * A constraint on intent execution.
 */
export interface IntentConstraint {
  type: 'time' | 'resource' | 'quality' | 'cost' | 'custom';
  key: string;
  operator: 'lt' | 'gt' | 'eq' | 'lte' | 'gte' | 'in' | 'not_in';
  value: unknown;
}

/**
 * A single goal within an intent specification.
 */
export interface IntentGoal {
  id: string;
  description: string;
  action: string;
  parameters: Record<string, unknown>;
  constraints: IntentConstraint[];
  dependsOn: string[];
  priority: IntentPriority;
}

/**
 * Full intent specification document.
 */
export interface IntentSpec {
  id: string;
  version: string;
  name: string;
  description: string;
  goals: IntentGoal[];
  globalConstraints: IntentConstraint[];
  metadata: Record<string, unknown>;
  createdAt: Date;
}

/**
 * Result of intent execution.
 */
export interface IntentResult {
  specId: string;
  status: IntentStatus;
  goalResults: Map<string, GoalResult>;
  startedAt: Date;
  completedAt?: Date;
  errors: string[];
}

export interface GoalResult {
  goalId: string;
  status: IntentStatus;
  output?: unknown;
  error?: string;
  durationMs: number;
}

/**
 * Create a new empty IntentSpec.
 */
export function createIntentSpec(name: string, description: string): IntentSpec {
  return {
    id: uuidv4(),
    version: '1.0.0',
    name,
    description,
    goals: [],
    globalConstraints: [],
    metadata: {},
    createdAt: new Date(),
  };
}

/**
 * Create a new IntentGoal.
 */
export function createGoal(
  action: string,
  description: string,
  params: Record<string, unknown> = {},
  priority: IntentPriority = IntentPriority.Normal
): IntentGoal {
  return {
    id: uuidv4(),
    description,
    action,
    parameters: params,
    constraints: [],
    dependsOn: [],
    priority,
  };
}
