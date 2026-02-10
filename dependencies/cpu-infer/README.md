# cpu-infer

CPU-optimized inference engine with SIMD-accelerated operations for the Ryzanstein LLM ecosystem.

## Overview

cpu-infer provides high-throughput LLM inference on commodity CPU hardware. Features auto-vectorized kernels (dot product, softmax, GELU), INT8/ternary quantization, Rayon thread-pool parallelism, and batch inference support.

## Features

- **SIMD-Accelerated Kernels** — dot product, softmax, ReLU, GELU auto-vectorized by LLVM
- **Quantization** — INT8 (min-max scaling) and ternary (BitNet-style -1/0/+1)
- **Batch Inference** — parallel request processing via Rayon
- **Model Types** — BitNet, Mamba, RWKV, Draft model support

## Quick Start

```rust
use cpu_infer::{CpuInferEngine, CpuInferConfig, InferenceRequest, ModelType};

let engine = CpuInferEngine::new(CpuInferConfig::default());
let result = engine.infer(&InferenceRequest {
    prompt: "Hello world".into(),
    max_tokens: 128,
    temperature: 0.7,
    model_type: ModelType::BitNet,
}).unwrap();
println!("{} tok/s", result.tokens_per_second);
```

## License

AGPL-3.0
