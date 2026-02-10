use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum CauseDbError {
    #[error("Event not found: {0}")]
    EventNotFound(Uuid),

    #[error("Invalid confidence score: {0} (must be 0.0-1.0)")]
    InvalidConfidence(f64),

    #[error("Cycle detected in causal chain")]
    CycleDetected,

    #[error("Query error: {0}")]
    QueryError(String),

    #[error("Ryzanstein connection error: {0}")]
    RyzansteinError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
