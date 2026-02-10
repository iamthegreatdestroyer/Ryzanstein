use thiserror::Error;

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Authentication failed: {0}")]
    AuthFailed(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Upstream error: {0}")]
    UpstreamError(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}
