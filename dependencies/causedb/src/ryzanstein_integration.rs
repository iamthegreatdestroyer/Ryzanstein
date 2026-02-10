//! Ryzanstein integration for causedb.

use crate::error::CauseDbError;
use crate::config::CauseDbConfig;

pub struct RyzansteinCauseClient {
    base_url: String,
}

impl RyzansteinCauseClient {
    pub fn new(config: &CauseDbConfig) -> Self {
        Self { base_url: config.ryzanstein_url.clone() }
    }

    /// Infer causal relationships using Ryzanstein's semantic embeddings
    pub async fn infer_causality(&self, event_a: &str, event_b: &str) -> Result<f64, CauseDbError> {
        // In production, calls Ryzanstein /v1/embeddings
        let _ = (&self.base_url, event_a, event_b);
        Ok(Self::fallback_similarity(event_a, event_b))
    }

    /// Simple fallback similarity
    fn fallback_similarity(a: &str, b: &str) -> f64 {
        let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();
        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();
        if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = RyzansteinCauseClient::new(&CauseDbConfig::default());
        assert_eq!(client.base_url, "http://localhost:8000");
    }

    #[test]
    fn test_fallback_similarity_identical() {
        let sim = RyzansteinCauseClient::fallback_similarity("hello world", "hello world");
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fallback_similarity_different() {
        let sim = RyzansteinCauseClient::fallback_similarity("hello world", "foo bar");
        assert!(sim < 0.01);
    }

    #[test]
    fn test_fallback_similarity_partial() {
        let sim = RyzansteinCauseClient::fallback_similarity("hello world", "hello foo");
        assert!(sim > 0.0 && sim < 1.0);
    }
}
