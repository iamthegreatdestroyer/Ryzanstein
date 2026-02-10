# flowstate

Workflow state machine engine with temporal logic for the Ryzanstein LLM ecosystem.

## Overview

flowstate provides a declarative state machine engine with temporal constraints for orchestrating multi-step LLM workflows. Features guard conditions, temporal deadlines (Eventually, Always, Before, After, Until operators), and Ryzanstein integration for AI-guided transitions.

## Quick Start

```typescript
import { FlowEngine, StateMachine } from "flowstate";

const sm = new StateMachine("inference-pipeline");
sm.addState({ id: "init", name: "Init", type: "initial", metadata: {} });
sm.addState({
  id: "inference",
  name: "Inference",
  type: "normal",
  metadata: {},
});
sm.addState({ id: "done", name: "Done", type: "final", metadata: {} });
sm.addTransition({ id: "t1", from: "init", to: "inference", event: "start" });
sm.addTransition({
  id: "t2",
  from: "inference",
  to: "done",
  event: "complete",
});

const engine = new FlowEngine();
engine.registerMachine(sm);
const exec = engine.start("inference-pipeline");
engine.step(exec.id, "start");
engine.step(exec.id, "complete");
```

## License

AGPL-3.0
