//! Error types for σ-index

use thiserror::Error;

/// Result type alias for σ-index operations
pub type IndexResult<T> = Result<T, IndexError>;

/// All possible errors in σ-index
#[derive(Error, Debug)]
pub enum IndexError {
    #[error("IO error: {0}")]
    IoError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("FM-Index error: {0}")]
    FMIndexError(String),

    #[error("HNSW error: {0}")]
    HNSWError(String),

    #[error("Ryzanstein integration error: {0}")]
    RyzansteinError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Index not found: {0}")]
    NotFound(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}
