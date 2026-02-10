//! Ryzanstein integration for zkaudit.

use crate::config::ZkAuditConfig;
use crate::error::ZkAuditError;

pub struct RyzansteinAuditClient {
    config: ZkAuditConfig,
}

impl RyzansteinAuditClient {
    pub fn new(config: ZkAuditConfig) -> Self {
        RyzansteinAuditClient { config }
    }

    pub async fn health_check(&self) -> Result<bool, ZkAuditError> {
        let url = format!("{}/health", self.config.ryzanstein_url);
        match reqwest::get(&url).await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    pub async fn submit_audit_proof(
        &self,
        merkle_root: &str,
        entry_count: usize,
    ) -> Result<String, ZkAuditError> {
        // In production: submit to Ryzanstein for on-chain anchoring
        Ok(self.fallback_receipt(merkle_root, entry_count))
    }

    fn fallback_receipt(&self, merkle_root: &str, entry_count: usize) -> String {
        format!(
            "receipt:{}:entries={}:ts={}",
            &merkle_root[..16],
            entry_count,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_offline() {
        let client = RyzansteinAuditClient::new(ZkAuditConfig::default());
        assert!(!client.health_check().await.unwrap());
    }

    #[test]
    fn test_fallback_receipt() {
        let client = RyzansteinAuditClient::new(ZkAuditConfig::default());
        let receipt = client.fallback_receipt("a".repeat(64).as_str(), 100);
        assert!(receipt.starts_with("receipt:"));
        assert!(receipt.contains("entries=100"));
    }
}
