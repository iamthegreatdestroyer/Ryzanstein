# intent-spec

Intent specification language and runtime for declarative goal expression in the Ryzanstein LLM ecosystem.

## Overview

intent-spec translates high-level user intents (natural language or structured JSON) into executable plans. Goals are resolved with dependency ordering, constraint validation, and dispatched to Ryzanstein agents.

## Quick Start

```typescript
import { IntentEngine, IntentParser, createIntentSpec, createGoal, IntentPriority } from 'intent-spec';

// Structured intent
const spec = createIntentSpec('code-review', 'Review and improve code');
spec.goals.push(createGoal('analyze', 'Static analysis', { path: './src' }, IntentPriority.High));
spec.goals.push(createGoal('suggest', 'Generate suggestions', {}, IntentPriority.Normal));

const engine = new IntentEngine();
const result = await engine.execute(spec);

// Natural language intent
const parser = new IntentParser();
const nlSpec = parser.parseNaturalLanguage('generate a summary of this codebase');
const nlResult = await engine.execute(nlSpec);
```

## Architecture

```
User Intent (NL/JSON) → IntentParser → IntentSpec
    ↓
IntentValidator (constraints, cycles, completeness)
    ↓
IntentEngine (topological sort, dependency resolution)
    ↓
Ryzanstein Agent Mesh (execution)
    ↓
IntentResult (status, outputs, errors)
```

## License

AGPL-3.0
