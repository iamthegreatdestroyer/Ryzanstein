//! Merkle tree for efficient audit verification.

use crate::chain::AuditEntry;
use sha2::{Digest, Sha256};

pub struct MerkleTree {
    layers: Vec<Vec<String>>,
}

impl MerkleTree {
    pub fn from_entries(entries: &[AuditEntry]) -> Self {
        if entries.is_empty() {
            return MerkleTree { layers: vec![] };
        }

        let leaf_hashes: Vec<String> = entries.iter().map(|e| e.hash.clone()).collect();
        let mut layers = vec![leaf_hashes.clone()];
        let mut current = leaf_hashes;

        while current.len() > 1 {
            let mut next = Vec::new();
            for chunk in current.chunks(2) {
                let combined = if chunk.len() == 2 {
                    format!("{}{}", chunk[0], chunk[1])
                } else {
                    format!("{}{}", chunk[0], chunk[0])
                };
                next.push(hex::encode(Sha256::digest(combined.as_bytes())));
            }
            layers.push(next.clone());
            current = next;
        }

        MerkleTree { layers }
    }

    /// Get the Merkle root hash.
    pub fn root(&self) -> Option<&str> {
        self.layers.last()?.first().map(|s| s.as_str())
    }

    /// Get a proof path for a leaf index.
    pub fn proof(&self, index: usize) -> Vec<(String, bool)> {
        let mut path = Vec::new();
        let mut idx = index;

        for layer in &self.layers[..self.layers.len().saturating_sub(1)] {
            let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
            let sibling = layer.get(sibling_idx).cloned().unwrap_or_default();
            path.push((sibling, idx % 2 == 0));
            idx /= 2;
        }

        path
    }

    pub fn depth(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{AuditAction, AuditChain};

    #[test]
    fn test_merkle_root() {
        let mut chain = AuditChain::new();
        chain.append(AuditAction::Inference, "a", "u");
        chain.append(AuditAction::Inference, "b", "u");
        let tree = MerkleTree::from_entries(chain.entries());
        assert!(tree.root().is_some());
        assert_eq!(tree.root().unwrap().len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_merkle_proof() {
        let mut chain = AuditChain::new();
        for i in 0..4 {
            chain.append(AuditAction::Inference, &format!("item-{}", i), "user");
        }
        let tree = MerkleTree::from_entries(chain.entries());
        let proof = tree.proof(0);
        assert!(!proof.is_empty());
    }

    #[test]
    fn test_empty_tree() {
        let tree = MerkleTree::from_entries(&[]);
        assert!(tree.root().is_none());
        assert_eq!(tree.depth(), 0);
    }

    #[test]
    fn test_single_entry() {
        let mut chain = AuditChain::new();
        chain.append(AuditAction::ModelLoad, "load", "sys");
        let tree = MerkleTree::from_entries(chain.entries());
        assert!(tree.root().is_some());
    }
}
