//! Core inference engine with thread-pool execution.

use crate::config::CpuInferConfig;
use crate::error::CpuInferError;
use crate::kernels;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    BitNet,
    Mamba,
    Rwkv,
    Draft,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub model_type: ModelType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub text: String,
    pub tokens_generated: usize,
    pub tokens_per_second: f64,
    pub total_time_ms: f64,
}

pub struct CpuInferEngine {
    config: CpuInferConfig,
    pool: rayon::ThreadPool,
}

impl CpuInferEngine {
    pub fn new(config: CpuInferConfig) -> Self {
        let threads = config.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("Failed to build thread pool");

        CpuInferEngine { config, pool }
    }

    /// Run inference on the CPU using the configured thread pool.
    pub fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, CpuInferError> {
        let start = std::time::Instant::now();

        // Tokenize (stub: character-level)
        let input_tokens: Vec<f32> = request.prompt.chars().map(|c| c as u32 as f32).collect();

        // Run matmul-style forward pass (stub)
        let output = self
            .pool
            .install(|| kernels::batch_matmul(&input_tokens, request.max_tokens));

        let elapsed = start.elapsed();
        let tokens_generated = output.len().min(request.max_tokens);

        Ok(InferenceResult {
            text: format!("[cpu-infer: {} tokens generated]", tokens_generated),
            tokens_generated,
            tokens_per_second: tokens_generated as f64 / elapsed.as_secs_f64().max(0.001),
            total_time_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    pub fn thread_count(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Batch inference across multiple requests in parallel.
    pub fn infer_batch(
        &self,
        requests: &[InferenceRequest],
    ) -> Vec<Result<InferenceResult, CpuInferError>> {
        self.pool.install(|| {
            use rayon::prelude::*;
            requests.par_iter().map(|req| self.infer(req)).collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CpuInferConfig;

    #[test]
    fn test_infer_basic() {
        let engine = CpuInferEngine::new(CpuInferConfig::default());
        let req = InferenceRequest {
            prompt: "Hello world".into(),
            max_tokens: 32,
            temperature: 0.7,
            model_type: ModelType::BitNet,
        };
        let result = engine.infer(&req).unwrap();
        assert!(result.tokens_generated > 0);
        assert!(result.tokens_per_second > 0.0);
    }

    #[test]
    fn test_infer_batch() {
        let engine = CpuInferEngine::new(CpuInferConfig::default());
        let requests: Vec<InferenceRequest> = (0..4)
            .map(|i| InferenceRequest {
                prompt: format!("Prompt {}", i),
                max_tokens: 16,
                temperature: 0.5,
                model_type: ModelType::Mamba,
            })
            .collect();
        let results = engine.infer_batch(&requests);
        assert_eq!(results.len(), 4);
        for r in results {
            assert!(r.is_ok());
        }
    }

    #[test]
    fn test_thread_count() {
        let config = CpuInferConfig {
            num_threads: Some(2),
            ..Default::default()
        };
        let engine = CpuInferEngine::new(config);
        assert_eq!(engine.thread_count(), 2);
    }
}
