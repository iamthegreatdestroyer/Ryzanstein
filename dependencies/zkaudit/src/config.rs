use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkAuditConfig {
    pub ryzanstein_url: String,
    pub max_chain_length: usize,
    pub auto_merkle_threshold: usize,
}

impl Default for ZkAuditConfig {
    fn default() -> Self {
        ZkAuditConfig {
            ryzanstein_url: "http://localhost:8000".into(),
            max_chain_length: 1_000_000,
            auto_merkle_threshold: 1000,
        }
    }
}
