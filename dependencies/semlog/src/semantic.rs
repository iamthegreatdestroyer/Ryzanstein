/// Semantic token classification for log variable encoding.
///
/// Classifies extracted variable tokens into types for optimal encoding:
/// - Timestamps → delta encoding
/// - IP addresses → dictionary encoding
/// - HTTP status codes → enum encoding
/// - Numeric values → variable-length integer encoding
/// - Paths/URLs → prefix tree encoding
/// - Arbitrary strings → LZ4 + dictionary

use regex::Regex;
use std::sync::LazyLock;

/// Semantic type of a log token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    Timestamp,
    IpAddress,
    HttpStatus,
    Integer,
    Float,
    Duration,
    Path,
    Uuid,
    Hex,
    String,
}

/// Result of classifying a token.
#[derive(Debug, Clone)]
pub struct ClassifiedToken {
    pub value: String,
    pub token_type: TokenType,
}

// Pre-compiled regexes for token classification
static RE_TIMESTAMP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}").unwrap()
});
static RE_IP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$").unwrap()
});
static RE_UUID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$").unwrap()
});
static RE_HEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^0x[0-9a-fA-F]+$").unwrap()
});
static RE_DURATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\d+(\.\d+)?(ms|s|us|ns|µs)$").unwrap()
});

/// Classify a single token by its semantic type.
pub fn classify_token(token: &str) -> ClassifiedToken {
    let token_type = if RE_TIMESTAMP.is_match(token) {
        TokenType::Timestamp
    } else if RE_IP.is_match(token) {
        TokenType::IpAddress
    } else if RE_UUID.is_match(token) {
        TokenType::Uuid
    } else if RE_HEX.is_match(token) {
        TokenType::Hex
    } else if RE_DURATION.is_match(token) {
        TokenType::Duration
    } else if is_http_status(token) {
        TokenType::HttpStatus
    } else if token.starts_with('/') || token.starts_with("http") {
        TokenType::Path
    } else if token.parse::<i64>().is_ok() {
        TokenType::Integer
    } else if token.parse::<f64>().is_ok() {
        TokenType::Float
    } else {
        TokenType::String
    };

    ClassifiedToken {
        value: token.to_string(),
        token_type,
    }
}

fn is_http_status(token: &str) -> bool {
    matches!(
        token.parse::<u16>().ok(),
        Some(100..=599)
    ) && token.len() == 3
}

/// Classify all wildcard values extracted from a Drain template match.
pub fn classify_variables(variables: &[String]) -> Vec<ClassifiedToken> {
    variables.iter().map(|v| classify_token(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_timestamp() {
        let r = classify_token("2026-01-15T14:30:00.000Z");
        assert_eq!(r.token_type, TokenType::Timestamp);
    }

    #[test]
    fn test_classify_ip() {
        let r = classify_token("192.168.1.100");
        assert_eq!(r.token_type, TokenType::IpAddress);
    }

    #[test]
    fn test_classify_http_status() {
        assert_eq!(classify_token("200").token_type, TokenType::HttpStatus);
        assert_eq!(classify_token("404").token_type, TokenType::HttpStatus);
        assert_eq!(classify_token("503").token_type, TokenType::HttpStatus);
    }

    #[test]
    fn test_classify_integer() {
        assert_eq!(classify_token("12345").token_type, TokenType::Integer);
    }

    #[test]
    fn test_classify_float() {
        assert_eq!(classify_token("3.14159").token_type, TokenType::Float);
    }

    #[test]
    fn test_classify_duration() {
        assert_eq!(classify_token("42ms").token_type, TokenType::Duration);
        assert_eq!(classify_token("1.5s").token_type, TokenType::Duration);
    }

    #[test]
    fn test_classify_path() {
        assert_eq!(classify_token("/api/v1/users").token_type, TokenType::Path);
        assert_eq!(classify_token("https://example.com").token_type, TokenType::Path);
    }

    #[test]
    fn test_classify_uuid() {
        let r = classify_token("550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(r.token_type, TokenType::Uuid);
    }

    #[test]
    fn test_classify_hex() {
        assert_eq!(classify_token("0xDEADBEEF").token_type, TokenType::Hex);
    }

    #[test]
    fn test_classify_string() {
        assert_eq!(classify_token("hello_world").token_type, TokenType::String);
    }

    #[test]
    fn test_classify_variables_batch() {
        let vars = vec![
            "200".to_string(),
            "42ms".to_string(),
            "192.168.1.1".to_string(),
        ];
        let classified = classify_variables(&vars);
        assert_eq!(classified[0].token_type, TokenType::HttpStatus);
        assert_eq!(classified[1].token_type, TokenType::Duration);
        assert_eq!(classified[2].token_type, TokenType::IpAddress);
    }
}
