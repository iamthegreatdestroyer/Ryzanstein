//! # cpu-infer
//!
//! CPU-optimized inference engine with SIMD-accelerated operations for
//! the Ryzanstein LLM ecosystem. Provides high-throughput inference on
//! commodity hardware without GPU requirements.

pub mod config;
pub mod engine;
pub mod error;
pub mod kernels;
pub mod quantize;
pub mod ryzanstein_integration;

pub use config::CpuInferConfig;
pub use engine::{CpuInferEngine, InferenceRequest, InferenceResult, ModelType};
pub use error::CpuInferError;
pub use quantize::{QuantFormat, QuantizedTensor};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = CpuInferEngine::new(CpuInferConfig::default());
        assert_eq!(engine.thread_count(), num_cpus());
    }

    #[test]
    fn test_inference_request() {
        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: 128,
            temperature: 0.7,
            model_type: ModelType::BitNet,
        };
        assert_eq!(req.max_tokens, 128);
    }

    fn num_cpus() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}
