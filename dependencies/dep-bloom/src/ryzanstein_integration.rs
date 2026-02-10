//! Ryzanstein integration for dep-bloom.

use crate::config::DepBloomConfig;
use crate::error::DepBloomError;

pub struct RyzansteinDepClient {
    config: DepBloomConfig,
}

impl RyzansteinDepClient {
    pub fn new(config: DepBloomConfig) -> Self {
        RyzansteinDepClient { config }
    }

    /// Check Ryzanstein health.
    pub async fn health_check(&self) -> Result<bool, DepBloomError> {
        let url = format!("{}/health", self.config.ryzanstein_url);
        match reqwest::get(&url).await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Resolve dependency compatibility via Ryzanstein model inference.
    pub async fn check_compatibility(
        &self,
        dep_a: &str,
        dep_b: &str,
    ) -> Result<f64, DepBloomError> {
        // In production: POST /v1/chat/completions for compatibility analysis
        Ok(self.fallback_compatibility(dep_a, dep_b))
    }

    fn fallback_compatibility(&self, a: &str, b: &str) -> f64 {
        // Simple heuristic: same prefix → higher compatibility
        let common = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
        let max_len = a.len().max(b.len()).max(1);
        common as f64 / max_len as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_compatibility_same() {
        let client = RyzansteinDepClient::new(DepBloomConfig::default());
        let score = client.fallback_compatibility("tokio", "tokio-stream");
        assert!(score > 0.3);
    }

    #[test]
    fn test_fallback_compatibility_different() {
        let client = RyzansteinDepClient::new(DepBloomConfig::default());
        let score = client.fallback_compatibility("serde", "rand");
        assert!(score < 0.5);
    }

    #[tokio::test]
    async fn test_health_check_offline() {
        let client = RyzansteinDepClient::new(DepBloomConfig::default());
        let healthy = client.health_check().await.unwrap();
        assert!(!healthy);
    }
}
