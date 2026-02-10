//! Configuration for σ-index

use serde::{Deserialize, Serialize};

/// Top-level index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub fm_index_config: FMIndexConfig,
    pub hnsw_config: HNSWConfig,
    pub ryzanstein_url: String,
    pub watch_enabled: bool,
    pub max_file_size_bytes: u64,
    pub excluded_patterns: Vec<String>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            fm_index_config: FMIndexConfig::default(),
            hnsw_config: HNSWConfig::default(),
            ryzanstein_url: "http://localhost:8000".to_string(),
            watch_enabled: true,
            max_file_size_bytes: 10 * 1024 * 1024, // 10 MB
            excluded_patterns: vec![
                "node_modules".into(),
                ".git".into(),
                "target".into(),
                "__pycache__".into(),
                ".venv".into(),
            ],
        }
    }
}

/// FM-Index specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FMIndexConfig {
    pub sample_rate: usize,
    pub alphabet_size: usize,
}

impl Default for FMIndexConfig {
    fn default() -> Self {
        Self {
            sample_rate: 32,
            alphabet_size: 256,
        }
    }
}

/// HNSW layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    pub embedding_dim: usize,
    pub max_elements: usize,
    pub ef_construction: usize,
    pub m: usize,
    pub ef_search: usize,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 1024,
            max_elements: 1_000_000,
            ef_construction: 200,
            m: 16,
            ef_search: 50,
        }
    }
}
