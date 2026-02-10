use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepBloomConfig {
    pub ryzanstein_url: String,
    pub expected_dependencies: usize,
    pub false_positive_rate: f64,
    pub max_depth: usize,
}

impl Default for DepBloomConfig {
    fn default() -> Self {
        DepBloomConfig {
            ryzanstein_url: "http://localhost:8000".into(),
            expected_dependencies: 10000,
            false_positive_rate: 0.01,
            max_depth: 50,
        }
    }
}
