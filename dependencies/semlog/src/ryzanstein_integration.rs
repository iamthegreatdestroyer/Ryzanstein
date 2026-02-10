//! Ryzanstein integration for semlog.
//!
//! Provides a client that connects back to the Ryzanstein LLM server
//! for AI-enhanced log analysis: semantic enrichment of templates,
//! anomaly narrative generation, and natural-language log querying.

use crate::drain::LogTemplate;
use serde::{Deserialize, Serialize};

/// Configuration for Ryzanstein upstream connectivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RyzansteinConfig {
    /// Base URL of the Ryzanstein API server.
    pub url: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl Default for RyzansteinConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8000".into(),
            timeout_secs: 30,
        }
    }
}

/// Client for Ryzanstein-enhanced semantic log operations.
pub struct RyzansteinSemlogClient {
    config: RyzansteinConfig,
}

impl RyzansteinSemlogClient {
    /// Create a new client from configuration.
    pub fn new(config: RyzansteinConfig) -> Self {
        Self { config }
    }

    /// Check Ryzanstein health endpoint.
    pub fn health_url(&self) -> String {
        format!("{}/health", self.config.url)
    }

    /// Build a prompt asking Ryzanstein to explain a set of Drain templates.
    pub fn template_analysis_prompt(&self, templates: &[LogTemplate]) -> serde_json::Value {
        let template_text: Vec<String> = templates
            .iter()
            .map(|t| format!("ID={}: {} (count={})", t.id, t.to_string_repr(), t.count))
            .collect();

        serde_json::json!({
            "model": "ryzanstein",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a log analysis expert. Summarize the recurring patterns and anomalies."
                },
                {
                    "role": "user",
                    "content": format!("Analyze these log templates:\n{}", template_text.join("\n"))
                }
            ]
        })
    }

    /// Build a prompt for natural-language log querying.
    pub fn natural_query_prompt(&self, query: &str, templates: &[LogTemplate]) -> serde_json::Value {
        let context: Vec<String> = templates
            .iter()
            .take(20)
            .map(|t| t.to_string_repr())
            .collect();

        serde_json::json!({
            "model": "ryzanstein",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a log query assistant. Given log templates, answer questions about system behavior."
                },
                {
                    "role": "user",
                    "content": format!("Templates:\n{}\n\nQuestion: {}", context.join("\n"), query)
                }
            ]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RyzansteinConfig::default();
        assert_eq!(config.url, "http://localhost:8000");
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_health_url() {
        let client = RyzansteinSemlogClient::new(RyzansteinConfig::default());
        assert_eq!(client.health_url(), "http://localhost:8000/health");
    }

    #[test]
    fn test_template_analysis_prompt() {
        let client = RyzansteinSemlogClient::new(RyzansteinConfig::default());
        let templates = vec![LogTemplate {
            id: 1,
            tokens: vec!["Connection".into(), "from".into(), "<*>".into(), "accepted".into()],
            count: 42,
        }];
        let prompt = client.template_analysis_prompt(&templates);
        assert!(prompt["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("Connection"));
    }

    #[test]
    fn test_natural_query_prompt() {
        let client = RyzansteinSemlogClient::new(RyzansteinConfig::default());
        let templates = vec![LogTemplate {
            id: 1,
            tokens: vec!["Error".into(), "<*>".into()],
            count: 5,
        }];
        let prompt = client.natural_query_prompt("What errors occurred?", &templates);
        assert!(prompt["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("What errors occurred?"));
    }
}
