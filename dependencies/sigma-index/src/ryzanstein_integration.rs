//! Ryzanstein integration hooks for σ-index
//!
//! Provides ΣLANG semantic encoding via Ryzanstein's CPU inference engine.
//! In standalone mode, falls back to local embedding generation.

use crate::error::{IndexError, IndexResult};
use crate::incremental::Token;

/// Ryzanstein API client for semantic encoding
pub struct RyzansteinClient {
    api_url: String,
    model_id: String,
    embedding_dim: usize,
}

impl RyzansteinClient {
    pub fn new(api_url: &str, model_id: &str, embedding_dim: usize) -> Self {
        Self {
            api_url: api_url.to_string(),
            model_id: model_id.to_string(),
            embedding_dim,
        }
    }

    /// Encode text into semantic vector via Ryzanstein ΣLANG
    pub async fn encode(&self, text: &str) -> IndexResult<Vec<f32>> {
        // In production: HTTP POST to Ryzanstein /v1/embeddings
        // For now: return deterministic placeholder
        Ok(self.fallback_encode(text))
    }

    /// Encode batch of texts
    pub async fn encode_batch(&self, texts: &[&str]) -> IndexResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.encode(text).await?);
        }
        Ok(results)
    }

    /// Check if Ryzanstein is available
    pub async fn health_check(&self) -> bool {
        // In production: GET /health
        false // Default: assume not available, use fallback
    }

    /// Fallback: simple hash-based embedding when Ryzanstein unavailable
    fn fallback_encode(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0f32; self.embedding_dim];
        for (i, word) in text.split_whitespace().enumerate() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();
            let idx = (hash as usize) % self.embedding_dim;
            embedding[idx] += 1.0 / (i as f32 + 1.0);
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        embedding
    }
}

/// Top-level encoding function used by SigmaIndex
pub async fn encode_semantic(
    content: &str,
    _tokens: &[Token],
) -> IndexResult<Vec<f32>> {
    let client = RyzansteinClient::new("http://localhost:8000", "sigma-lang-v1", 1024);
    client.encode(content).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_encode_deterministic() {
        let client = RyzansteinClient::new("http://localhost:8000", "test", 128);
        let e1 = client.fallback_encode("hello world");
        let e2 = client.fallback_encode("hello world");
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_fallback_encode_different_inputs() {
        let client = RyzansteinClient::new("http://localhost:8000", "test", 128);
        let e1 = client.fallback_encode("hello world");
        let e2 = client.fallback_encode("goodbye moon");
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_fallback_encode_normalized() {
        let client = RyzansteinClient::new("http://localhost:8000", "test", 128);
        let e = client.fallback_encode("some code here");
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_encode_semantic() {
        let tokens = vec![];
        let result = encode_semantic("fn hello() {}", &tokens).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1024);
    }

    #[test]
    fn test_client_new() {
        let client = RyzansteinClient::new("http://test:8000", "model-v1", 512);
        assert_eq!(client.embedding_dim, 512);
    }
}
