//! # zkaudit
//!
//! Zero-knowledge audit trail system for Ryzanstein. Provides tamper-evident
//! logging with Merkle-tree verification and hash-chain integrity, allowing
//! proof of audit completeness without revealing sensitive content.

pub mod chain;
pub mod config;
pub mod error;
pub mod merkle;
pub mod proof;
pub mod ryzanstein_integration;

pub use chain::{AuditChain, AuditEntry, AuditAction};
pub use merkle::MerkleTree;
pub use proof::{ZkProof, ProofVerifier};
pub use config::ZkAuditConfig;
pub use error::ZkAuditError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_audit_flow() {
        let mut chain = AuditChain::new();
        chain.append(AuditAction::ModelLoad, "loaded ryzanstein-v2", "system");
        chain.append(AuditAction::Inference, "prompt: hello", "user-1");
        assert_eq!(chain.len(), 2);
        assert!(chain.verify_integrity());

        let tree = MerkleTree::from_entries(chain.entries());
        assert!(tree.root().is_some());
    }
}
