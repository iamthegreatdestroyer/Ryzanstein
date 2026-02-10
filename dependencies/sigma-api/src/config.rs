use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigmaApiConfig {
    pub listen_addr: String,
    pub ryzanstein_url: String,
    pub jwt_secret: String,
    pub rate_limit_rpm: usize,
    pub rate_limit_window_secs: u64,
    pub cors_origins: Vec<String>,
}

impl Default for SigmaApiConfig {
    fn default() -> Self {
        SigmaApiConfig {
            listen_addr: "0.0.0.0:8080".into(),
            ryzanstein_url: "http://localhost:8000".into(),
            jwt_secret: "change-me-in-production".into(),
            rate_limit_rpm: 60,
            rate_limit_window_secs: 60,
            cors_origins: vec!["*".into()],
        }
    }
}
