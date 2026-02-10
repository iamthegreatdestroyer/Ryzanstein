use thiserror::Error;

#[derive(Error, Debug)]
pub enum DepBloomError {
    #[error("Bloom filter size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("Cyclic dependency detected in dependency graph")]
    CyclicDependency,

    #[error("Dependency not found: {0}")]
    NotFound(String),

    #[error("Version conflict: {0}")]
    VersionConflict(String),

    #[error("Ryzanstein communication error: {0}")]
    RyzansteinError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}
