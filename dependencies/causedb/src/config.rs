use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CauseDbConfig {
    pub ryzanstein_url: String,
    pub max_chain_depth: usize,
    pub min_confidence_threshold: f64,
    pub enable_auto_inference: bool,
    pub storage_path: Option<String>,
}

impl Default for CauseDbConfig {
    fn default() -> Self {
        Self {
            ryzanstein_url: "http://localhost:8000".to_string(),
            max_chain_depth: 100,
            min_confidence_threshold: 0.1,
            enable_auto_inference: false,
            storage_path: None,
        }
    }
}
