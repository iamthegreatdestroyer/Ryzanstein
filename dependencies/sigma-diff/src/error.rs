//! Error types for σ-diff

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DiffError {
    #[error("I/O error: {0}")]
    IoError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Ryzanstein error: {0}")]
    RyzansteinError(String),
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),
    #[error("File too large: {size} > {max}")]
    FileTooLarge { size: usize, max: usize },
}

pub type DiffResult<T> = std::result::Result<T, DiffError>;
