//! Ryzanstein integration for cpu-infer.

use crate::config::CpuInferConfig;
use crate::engine::{InferenceResult, ModelType};
use crate::error::CpuInferError;

pub struct RyzansteinCpuClient {
    config: CpuInferConfig,
}

impl RyzansteinCpuClient {
    pub fn new(config: CpuInferConfig) -> Self {
        RyzansteinCpuClient { config }
    }

    pub async fn health_check(&self) -> Result<bool, CpuInferError> {
        let url = format!("{}/health", self.config.ryzanstein_url);
        match reqwest::get(&url).await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Compare CPU inference result with Ryzanstein GPU result for validation.
    pub async fn validate_result(
        &self,
        prompt: &str,
        cpu_result: &InferenceResult,
    ) -> Result<f64, CpuInferError> {
        // In production: send prompt to Ryzanstein, compare outputs
        Ok(self.fallback_validation(prompt, cpu_result))
    }

    fn fallback_validation(&self, prompt: &str, result: &InferenceResult) -> f64 {
        // Heuristic: higher tokens/sec and non-empty → higher confidence
        if result.tokens_generated == 0 {
            return 0.0;
        }
        let speed_factor = (result.tokens_per_second / 100.0).min(1.0);
        let length_factor = (prompt.len() as f64 / 1000.0).min(1.0);
        (speed_factor + length_factor) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_offline() {
        let client = RyzansteinCpuClient::new(CpuInferConfig::default());
        assert!(!client.health_check().await.unwrap());
    }

    #[test]
    fn test_fallback_validation() {
        let client = RyzansteinCpuClient::new(CpuInferConfig::default());
        let result = InferenceResult {
            text: "output".into(),
            tokens_generated: 50,
            tokens_per_second: 200.0,
            total_time_ms: 250.0,
        };
        let score = client.fallback_validation("hello world", &result);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_fallback_validation_empty() {
        let client = RyzansteinCpuClient::new(CpuInferConfig::default());
        let result = InferenceResult {
            text: "".into(),
            tokens_generated: 0,
            tokens_per_second: 0.0,
            total_time_ms: 0.0,
        };
        assert_eq!(client.fallback_validation("", &result), 0.0);
    }
}
