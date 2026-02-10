import { v4 as uuidv4 } from "uuid";

export type StateType = "initial" | "normal" | "final" | "error";

export interface TransitionGuard {
  condition: string;
  evaluate: (context: Record<string, unknown>) => boolean;
}

export interface Transition {
  id: string;
  from: string;
  to: string;
  event: string;
  guard?: TransitionGuard;
  action?: (context: Record<string, unknown>) => void;
}

export interface State {
  id: string;
  name: string;
  type: StateType;
  onEnter?: (context: Record<string, unknown>) => void;
  onExit?: (context: Record<string, unknown>) => void;
  metadata: Record<string, unknown>;
}

/**
 * State machine definition with states, transitions, and guards.
 */
export class StateMachine {
  readonly id: string;
  private states = new Map<string, State>();
  private transitions: Transition[] = [];
  private currentStateId: string | null = null;
  private context: Record<string, unknown> = {};
  private history: Array<{
    from: string;
    to: string;
    event: string;
    timestamp: Date;
  }> = [];

  constructor(id?: string) {
    this.id = id ?? uuidv4();
  }

  addState(state: State): this {
    this.states.set(state.id, state);
    if (state.type === "initial" && !this.currentStateId) {
      this.currentStateId = state.id;
    }
    return this;
  }

  addTransition(transition: Transition): this {
    this.transitions.push(transition);
    return this;
  }

  get currentState(): State | undefined {
    return this.currentStateId
      ? this.states.get(this.currentStateId)
      : undefined;
  }

  getContext(): Record<string, unknown> {
    return { ...this.context };
  }

  setContext(ctx: Record<string, unknown>): void {
    this.context = { ...this.context, ...ctx };
  }

  /**
   * Send an event to trigger a transition.
   */
  send(event: string): boolean {
    if (!this.currentStateId) return false;

    const matching = this.transitions.filter(
      (t) => t.from === this.currentStateId && t.event === event,
    );

    for (const transition of matching) {
      if (transition.guard && !transition.guard.evaluate(this.context)) {
        continue;
      }

      const fromState = this.states.get(this.currentStateId!);
      const toState = this.states.get(transition.to);
      if (!toState) continue;

      fromState?.onExit?.(this.context);
      transition.action?.(this.context);

      this.history.push({
        from: this.currentStateId!,
        to: transition.to,
        event,
        timestamp: new Date(),
      });

      this.currentStateId = transition.to;
      toState.onEnter?.(this.context);
      return true;
    }

    return false;
  }

  /**
   * Get available events from current state.
   */
  availableEvents(): string[] {
    if (!this.currentStateId) return [];
    return this.transitions
      .filter((t) => t.from === this.currentStateId)
      .map((t) => t.event);
  }

  isInFinalState(): boolean {
    return (
      this.currentState?.type === "final" || this.currentState?.type === "error"
    );
  }

  getHistory() {
    return [...this.history];
  }

  /**
   * Reset to initial state.
   */
  reset(): void {
    const initial = [...this.states.values()].find((s) => s.type === "initial");
    this.currentStateId = initial?.id ?? null;
    this.context = {};
    this.history = [];
  }
}
