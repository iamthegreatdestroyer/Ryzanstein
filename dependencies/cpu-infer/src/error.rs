use thiserror::Error;

#[derive(Error, Debug)]
pub enum CpuInferError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Quantization error: {0}")]
    QuantizationError(String),

    #[error("Thread pool error: {0}")]
    ThreadPoolError(String),

    #[error("Ryzanstein communication error: {0}")]
    RyzansteinError(String),
}
