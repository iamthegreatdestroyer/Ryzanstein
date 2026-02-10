/**
 * Ryzanstein integration for flowstate workflows.
 */

export interface FlowStateRyzansteinConfig {
  baseUrl: string;
  timeout: number;
}

const DEFAULT_CONFIG: FlowStateRyzansteinConfig = {
  baseUrl: 'http://localhost:8000',
  timeout: 30000,
};

export class RyzansteinFlowClient {
  private config: FlowStateRyzansteinConfig;

  constructor(config: Partial<FlowStateRyzansteinConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async healthCheck(): Promise<boolean> {
    try {
      const r = await fetch(`${this.config.baseUrl}/health`, {
        signal: AbortSignal.timeout(this.config.timeout),
      });
      return r.ok;
    } catch {
      return false;
    }
  }

  /**
   * Suggest next workflow transition based on model inference.
   */
  async suggestTransition(
    currentState: string,
    availableEvents: string[],
    context: Record<string, unknown>
  ): Promise<SuggestedTransition> {
    // In production: calls Ryzanstein inference API
    return this.fallbackSuggest(currentState, availableEvents);
  }

  private fallbackSuggest(currentState: string, events: string[]): SuggestedTransition {
    return {
      event: events[0] ?? 'next',
      confidence: 0.5,
      reasoning: `Fallback: suggesting first available event from state '${currentState}'`,
    };
  }
}

export interface SuggestedTransition {
  event: string;
  confidence: number;
  reasoning: string;
}
