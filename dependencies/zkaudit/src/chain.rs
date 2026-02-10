//! Hash-chain audit log with tamper detection.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditAction {
    ModelLoad,
    Inference,
    ConfigChange,
    AccessGrant,
    AccessRevoke,
    DataExport,
    SystemAlert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: u64,
    pub action: AuditAction,
    pub description: String,
    pub actor: String,
    pub hash: String,
    pub prev_hash: String,
}

pub struct AuditChain {
    entries: Vec<AuditEntry>,
}

impl AuditChain {
    pub fn new() -> Self {
        AuditChain {
            entries: Vec::new(),
        }
    }

    pub fn append(&mut self, action: AuditAction, description: &str, actor: &str) {
        let prev_hash = self
            .entries
            .last()
            .map(|e| e.hash.clone())
            .unwrap_or_else(|| "0".repeat(64));

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let id = uuid::Uuid::new_v4().to_string();

        let hash_input = format!(
            "{}|{}|{:?}|{}|{}|{}",
            id, timestamp, action, description, actor, prev_hash
        );
        let hash = hex::encode(Sha256::digest(hash_input.as_bytes()));

        self.entries.push(AuditEntry {
            id,
            timestamp,
            action,
            description: description.to_string(),
            actor: actor.to_string(),
            hash,
            prev_hash,
        });
    }

    /// Verify the entire chain's hash integrity.
    pub fn verify_integrity(&self) -> bool {
        for i in 1..self.entries.len() {
            if self.entries[i].prev_hash != self.entries[i - 1].hash {
                return false;
            }
        }
        if let Some(first) = self.entries.first() {
            if first.prev_hash != "0".repeat(64) {
                return false;
            }
        }
        true
    }

    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for AuditChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_integrity() {
        let mut chain = AuditChain::new();
        chain.append(AuditAction::ModelLoad, "load", "admin");
        chain.append(AuditAction::Inference, "run", "user");
        chain.append(AuditAction::ConfigChange, "update", "admin");
        assert!(chain.verify_integrity());
    }

    #[test]
    fn test_empty_chain() {
        let chain = AuditChain::new();
        assert!(chain.verify_integrity());
        assert!(chain.is_empty());
    }

    #[test]
    fn test_tamper_detection() {
        let mut chain = AuditChain::new();
        chain.append(AuditAction::Inference, "query", "user");
        chain.append(AuditAction::Inference, "query2", "user");
        // Tamper with first entry
        chain.entries.first_mut().unwrap().hash = "tampered".into();
        assert!(!chain.verify_integrity());
    }

    #[test]
    fn test_action_variants() {
        let actions = vec![
            AuditAction::ModelLoad,
            AuditAction::Inference,
            AuditAction::ConfigChange,
            AuditAction::AccessGrant,
            AuditAction::AccessRevoke,
            AuditAction::DataExport,
            AuditAction::SystemAlert,
        ];
        assert_eq!(actions.len(), 7);
    }
}
