use thiserror::Error;

#[derive(Error, Debug)]
pub enum ZkAuditError {
    #[error("Chain integrity violation at entry {0}")]
    IntegrityViolation(usize),

    #[error("Proof verification failed: {0}")]
    ProofFailed(String),

    #[error("Chain capacity exceeded: {0}")]
    CapacityExceeded(usize),

    #[error("Ryzanstein error: {0}")]
    RyzansteinError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}
