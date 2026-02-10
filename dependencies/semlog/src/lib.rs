//! semlog — Semantic Log Compression Library
//!
//! A Drain-based log parsing and compression engine that discovers templates,
//! detects recurring patterns, and produces a queryable compressed format.
//!
//! # Architecture
//!
//! ```text
//! Raw Logs → Drain Parser → Semantic Classifier → Pattern Detector → Compressed Storage
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use semlog::{DrainParser, DrainConfig, CompressStats};
//!
//! let config = DrainConfig::default();
//! let mut parser = DrainParser::new(config);
//! parser.parse_line("2026-02-10 INFO  Starting service on port 8080");
//! parser.parse_line("2026-02-10 INFO  Starting service on port 9090");
//! let templates = parser.templates();
//! assert!(templates.len() >= 1);
//! ```

pub mod compress;
pub mod drain;
pub mod pattern;
pub mod query;
pub mod semantic;
pub mod ryzanstein_integration;

// Re-export primary types for ergonomic access.
pub use compress::{CompressError, CompressStats};
pub use drain::{DrainConfig, DrainParser, LogTemplate};
pub use pattern::{PatternConfig, PatternDetector};
pub use ryzanstein_integration::RyzansteinSemlogClient;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drain_roundtrip() {
        let config = DrainConfig::default();
        let mut parser = DrainParser::new(config);
        parser.parse_line("Connection from 192.168.1.1 accepted");
        parser.parse_line("Connection from 10.0.0.5 accepted");
        let templates = parser.templates();
        assert!(!templates.is_empty(), "Drain should extract at least one template");
    }

    #[test]
    fn test_lib_exports_present() {
        // Verify the library re-exports resolve.
        let _ = DrainConfig::default();
        let _ = PatternConfig::default();
    }
}
