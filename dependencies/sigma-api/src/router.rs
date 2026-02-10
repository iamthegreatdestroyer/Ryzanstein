//! Request routing and API endpoint handlers.

use crate::auth::AuthLayer;
use crate::config::SigmaApiConfig;
use crate::rate_limit::RateLimiter;
use axum::{extract::State, http::StatusCode, response::Json, routing::{get, post}, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub config: SigmaApiConfig,
    pub rate_limiter: Arc<RateLimiter>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f32 { 0.7 }

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

pub fn create_router(config: SigmaApiConfig) -> Router {
    let rate_limiter = Arc::new(RateLimiter::new(
        config.rate_limit_rpm,
        config.rate_limit_window_secs,
    ));
    let state = AppState {
        config,
        rate_limiter,
    };

    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/chat/completions", post(chat_completion_handler))
        .route("/v1/models", get(models_handler))
        .with_state(state)
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

async fn chat_completion_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    // Rate limiting check
    if !state.rate_limiter.allow("default") {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    let response_text = format!(
        "[sigma-api proxy: model={}, {} messages]",
        req.model,
        req.messages.len()
    );

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: response_text,
            },
            finish_reason: "stop".into(),
        }],
    }))
}

async fn models_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            { "id": "ryzanstein-v2", "object": "model", "owned_by": "ryzanstein" },
            { "id": "ryzanstein-draft", "object": "model", "owned_by": "ryzanstein" },
        ]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response() {
        let resp = HealthResponse {
            status: "ok".into(),
            version: "0.1.0".into(),
        };
        assert_eq!(resp.status, "ok");
    }

    #[test]
    fn test_chat_request_defaults() {
        let json = r#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 256);
        assert!((req.temperature - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_create_router() {
        let _ = create_router(SigmaApiConfig::default());
    }
}
