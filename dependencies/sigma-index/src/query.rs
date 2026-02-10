//! Query types and result structures for σ-index

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Search mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchMode {
    /// Exact text pattern matching via FM-index (O(m))
    Exact,
    /// Semantic similarity search via HNSW (O(log n))
    Semantic,
    /// Combined exact + semantic with result fusion
    Hybrid,
}

/// Match type in results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchType {
    Exact,
    Semantic,
    Hybrid,
}

/// Search query specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub pattern: String,
    pub mode: SearchMode,
    pub top_k: usize,
    pub file_filter: Option<Vec<String>>,
    pub language_filter: Option<Vec<crate::Language>>,
}

impl SearchQuery {
    pub fn exact(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            mode: SearchMode::Exact,
            top_k: 20,
            file_filter: None,
            language_filter: None,
        }
    }

    pub fn semantic(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            mode: SearchMode::Semantic,
            top_k: 10,
            file_filter: None,
            language_filter: None,
        }
    }

    pub fn hybrid(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            mode: SearchMode::Hybrid,
            top_k: 10,
            file_filter: None,
            language_filter: None,
        }
    }
}

/// Individual search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: PathBuf,
    pub line: usize,
    pub column: usize,
    pub score: f32,
    pub snippet: String,
    pub match_type: MatchType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_query() {
        let q = SearchQuery::exact("fn main");
        assert_eq!(q.mode, SearchMode::Exact);
        assert_eq!(q.top_k, 20);
    }

    #[test]
    fn test_semantic_query() {
        let q = SearchQuery::semantic("retry with backoff");
        assert_eq!(q.mode, SearchMode::Semantic);
        assert_eq!(q.top_k, 10);
    }

    #[test]
    fn test_hybrid_query() {
        let q = SearchQuery::hybrid("error handling");
        assert_eq!(q.mode, SearchMode::Hybrid);
    }
}
