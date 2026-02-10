//! Ryzanstein integration for ann-hybrid.
//!
//! Provides embedding generation hooks and index management that connect
//! back to the Ryzanstein LLM server for semantic vector operations.

use crate::hnsw::HnswConfig;
use crate::index::{HybridIndex, IndexConfig, SearchQuery, SearchResult};
use serde::{Deserialize, Serialize};

/// Configuration for Ryzanstein upstream connectivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RyzansteinConfig {
    /// Base URL of the Ryzanstein API server.
    pub url: String,
    /// Embedding model identifier.
    pub embedding_model: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl Default for RyzansteinConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8000".into(),
            embedding_model: "ryzanstein-embed-v1".into(),
            timeout_secs: 30,
        }
    }
}

/// Client bridging ann-hybrid indices with Ryzanstein embeddings.
pub struct RyzansteinAnnClient {
    config: RyzansteinConfig,
}

impl RyzansteinAnnClient {
    /// Create a new integration client.
    pub fn new(config: RyzansteinConfig) -> Self {
        Self { config }
    }

    /// Health check URL for readiness probes.
    pub fn health_url(&self) -> String {
        format!("{}/health", self.config.url)
    }

    /// Build an embedding request body for Ryzanstein.
    pub fn embedding_request(&self, texts: &[String]) -> serde_json::Value {
        serde_json::json!({
            "model": self.config.embedding_model,
            "input": texts,
        })
    }

    /// Create a pre-configured `HybridIndex` tuned for Ryzanstein code search.
    pub fn create_code_search_index() -> HybridIndex {
        let config = IndexConfig {
            hnsw: HnswConfig {
                m: 16,
                ef_construction: 200,
                ef_search: 100,
                ..HnswConfig::default()
            },
            cuckoo_capacity: 500_000,
            cms_width: 65_536,
            cms_depth: 7,
            frequency_boost: 0.1,
        };
        HybridIndex::new(config)
    }

    /// Convenience: perform a semantic search against an index.
    pub fn semantic_search<'a>(
        index: &mut HybridIndex,
        query_vector: &'a [f32],
        keyword: Option<&'a str>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let query = SearchQuery {
            vector: Some(query_vector),
            keyword,
            top_k,
        };
        index.search(&query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RyzansteinConfig::default();
        assert_eq!(config.url, "http://localhost:8000");
        assert_eq!(config.embedding_model, "ryzanstein-embed-v1");
    }

    #[test]
    fn test_health_url() {
        let client = RyzansteinAnnClient::new(RyzansteinConfig::default());
        assert_eq!(client.health_url(), "http://localhost:8000/health");
    }

    #[test]
    fn test_embedding_request() {
        let client = RyzansteinAnnClient::new(RyzansteinConfig::default());
        let req = client.embedding_request(&["hello world".to_string()]);
        assert_eq!(req["model"], "ryzanstein-embed-v1");
        assert_eq!(req["input"][0], "hello world");
    }
}
