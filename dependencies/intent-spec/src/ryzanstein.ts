/**
 * Ryzanstein integration for intent-spec.
 * Delegates intent resolution to Ryzanstein's inference API.
 */

export interface RyzansteinConfig {
  baseUrl: string;
  apiKey?: string;
  timeout: number;
}

const DEFAULT_CONFIG: RyzansteinConfig = {
  baseUrl: 'http://localhost:8000',
  timeout: 30000,
};

export class RyzansteinIntentClient {
  private config: RyzansteinConfig;

  constructor(config: Partial<RyzansteinConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Check Ryzanstein health.
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.baseUrl}/health`, {
        signal: AbortSignal.timeout(this.config.timeout),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Resolve a natural language intent via Ryzanstein inference.
   */
  async resolveIntent(text: string): Promise<ResolvedIntent> {
    // In production: POST /v1/chat/completions with structured output
    return this.fallbackResolve(text);
  }

  /**
   * Extract structured parameters from natural language.
   */
  async extractParameters(text: string): Promise<Record<string, unknown>> {
    // In production: POST /v1/chat/completions with function calling
    return this.fallbackExtract(text);
  }

  private fallbackResolve(text: string): ResolvedIntent {
    const words = text.toLowerCase().split(/\s+/);
    const actions: Record<string, string> = {
      'generate': 'inference',
      'analyze': 'analysis',
      'search': 'search',
      'compress': 'compression',
      'encrypt': 'encryption',
      'test': 'testing',
      'deploy': 'deployment',
      'review': 'code_review',
    };

    let action = 'generic';
    for (const [keyword, act] of Object.entries(actions)) {
      if (words.includes(keyword)) {
        action = act;
        break;
      }
    }

    return {
      action,
      confidence: 0.5,
      parameters: { rawText: text },
    };
  }

  private fallbackExtract(text: string): Record<string, unknown> {
    return { rawText: text, wordCount: text.split(/\s+/).length };
  }
}

export interface ResolvedIntent {
  action: string;
  confidence: number;
  parameters: Record<string, unknown>;
}
