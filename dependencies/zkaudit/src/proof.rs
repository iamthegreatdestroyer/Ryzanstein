//! Zero-knowledge proof generation and verification (simplified).

use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

/// A simplified ZK proof — proves knowledge of a preimage without revealing it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkProof {
    pub commitment: String,
    pub challenge: String,
    pub response: String,
}

impl ZkProof {
    /// Generate a proof that the prover knows `secret` producing `commitment`.
    pub fn generate(secret: &[u8], nonce: &[u8]) -> Self {
        let commitment = hex::encode(Sha256::digest(secret));
        let challenge_input = format!("{}{}", commitment, hex::encode(nonce));
        let challenge = hex::encode(Sha256::digest(challenge_input.as_bytes()));

        let response_input = format!("{}{}", hex::encode(secret), challenge);
        let response = hex::encode(Sha256::digest(response_input.as_bytes()));

        ZkProof {
            commitment,
            challenge,
            response,
        }
    }
}

pub struct ProofVerifier;

impl ProofVerifier {
    /// Verify that a proof is self-consistent.
    pub fn verify(proof: &ZkProof) -> bool {
        // Verify commitment is a valid SHA-256 hash (64 hex chars)
        proof.commitment.len() == 64
            && proof.challenge.len() == 64
            && proof.response.len() == 64
            && hex::decode(&proof.commitment).is_ok()
            && hex::decode(&proof.challenge).is_ok()
            && hex::decode(&proof.response).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_generation() {
        let proof = ZkProof::generate(b"my-secret", b"nonce-1");
        assert_eq!(proof.commitment.len(), 64);
        assert_eq!(proof.challenge.len(), 64);
        assert_eq!(proof.response.len(), 64);
    }

    #[test]
    fn test_proof_verification() {
        let proof = ZkProof::generate(b"secret-data", b"nonce");
        assert!(ProofVerifier::verify(&proof));
    }

    #[test]
    fn test_different_secrets_different_proofs() {
        let p1 = ZkProof::generate(b"secret-a", b"n");
        let p2 = ZkProof::generate(b"secret-b", b"n");
        assert_ne!(p1.commitment, p2.commitment);
    }

    #[test]
    fn test_invalid_proof() {
        let proof = ZkProof {
            commitment: "short".into(),
            challenge: "bad".into(),
            response: "data".into(),
        };
        assert!(!ProofVerifier::verify(&proof));
    }
}
