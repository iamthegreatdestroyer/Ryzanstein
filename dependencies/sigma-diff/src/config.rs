//! Configuration for σ-diff

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffConfig {
    /// Ryzanstein endpoint for semantic diffing
    pub ryzanstein_url: String,
    /// Minimum confidence threshold for structural changes
    pub confidence_threshold: f64,
    /// Whether to include cosmetic changes
    pub include_cosmetic: bool,
    /// Maximum file size to process (bytes)
    pub max_file_size: usize,
    /// Enable semantic diff via Ryzanstein
    pub semantic_enabled: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            ryzanstein_url: "http://localhost:8000".to_string(),
            confidence_threshold: 0.5,
            include_cosmetic: true,
            max_file_size: 10 * 1024 * 1024,
            semantic_enabled: false,
        }
    }
}
