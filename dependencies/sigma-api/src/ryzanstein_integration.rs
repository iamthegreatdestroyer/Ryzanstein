//! Ryzanstein upstream proxy integration.

use crate::config::SigmaApiConfig;
use crate::error::ApiError;

pub struct RyzansteinProxy {
    config: SigmaApiConfig,
    client: reqwest::Client,
}

impl RyzansteinProxy {
    pub fn new(config: SigmaApiConfig) -> Self {
        RyzansteinProxy {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub async fn health_check(&self) -> Result<bool, ApiError> {
        let url = format!("{}/health", self.config.ryzanstein_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    pub async fn proxy_chat_completion(
        &self,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ApiError> {
        let url = format!("{}/v1/chat/completions", self.config.ryzanstein_url);
        match self.client.post(&url).json(body).send().await {
            Ok(resp) => resp
                .json()
                .await
                .map_err(|e| ApiError::UpstreamError(e.to_string())),
            Err(e) => Err(ApiError::UpstreamError(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_offline() {
        let proxy = RyzansteinProxy::new(SigmaApiConfig::default());
        assert!(!proxy.health_check().await.unwrap());
    }
}
