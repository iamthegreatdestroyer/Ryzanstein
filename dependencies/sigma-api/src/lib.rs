//! # sigma-api
//!
//! Unified API gateway for the Ryzanstein LLM ecosystem. Provides
//! authentication, rate limiting, request routing, and OpenAI-compatible
//! inference endpoints.

pub mod auth;
pub mod config;
pub mod error;
pub mod rate_limit;
pub mod router;
pub mod ryzanstein_integration;

pub use config::SigmaApiConfig;
pub use error::ApiError;

use axum::Router;

/// Build the full API router with all middleware and routes.
pub fn build_router(config: SigmaApiConfig) -> Router {
    router::create_router(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_router() {
        let config = SigmaApiConfig::default();
        let _router = build_router(config);
    }
}
