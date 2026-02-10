/// Query engine for .semlog compressed log files.
///
/// Searches compressed logs without full decompression.

use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum QueryError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
}

/// Query a .semlog file and return matching lines/patterns as strings.
pub fn query_file(file: &Path, query: &str) -> Result<Vec<String>, QueryError> {
    let data = fs::read_to_string(file)?;
    let doc: serde_json::Value = serde_json::from_str(&data)?;

    let mut results = Vec::new();
    let query_lower = query.to_lowercase();

    // Search templates
    if let Some(templates) = doc.get("templates").and_then(|t| t.as_array()) {
        for t in templates {
            let pattern = t
                .get("pattern")
                .and_then(|p| p.as_str())
                .unwrap_or("");
            if pattern.to_lowercase().contains(&query_lower) {
                let count = t.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
                results.push(format!(
                    "[template] {} (count: {})",
                    pattern, count,
                ));
            }
        }
    }

    // Search patterns
    if let Some(patterns) = doc.get("patterns").and_then(|p| p.as_array()) {
        for p in patterns {
            let kind = p.get("kind").and_then(|k| k.as_str()).unwrap_or("");
            if kind.to_lowercase().contains(&query_lower) {
                let start = p.get("start_line").and_then(|s| s.as_u64()).unwrap_or(0);
                let end = p.get("end_line").and_then(|e| e.as_u64()).unwrap_or(0);
                results.push(format!(
                    "[pattern] {} (lines {}-{})",
                    kind, start, end,
                ));
            }
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_sample_semlog() -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        let data = serde_json::json!({
            "version": "0.1.0",
            "templates": [
                {"id": 0, "pattern": "GET <*> 200 <*>", "count": 100},
                {"id": 1, "pattern": "ERROR connection refused", "count": 5},
            ],
            "patterns": [
                {"kind": "RetrySequence { count: 5, template_id: 1 }", "start_line": 50, "end_line": 54, "line_count": 5},
            ],
            "stats": {"lines": 105, "templates": 2, "patterns": 1},
        });
        write!(f, "{}", serde_json::to_string(&data).unwrap()).unwrap();
        f
    }

    #[test]
    fn test_query_template() {
        let f = make_sample_semlog();
        let results = query_file(f.path(), "ERROR").unwrap();
        assert!(!results.is_empty());
        assert!(results[0].contains("ERROR"));
    }

    #[test]
    fn test_query_pattern() {
        let f = make_sample_semlog();
        let results = query_file(f.path(), "retry").unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_query_no_match() {
        let f = make_sample_semlog();
        let results = query_file(f.path(), "nonexistent_xyz").unwrap();
        assert!(results.is_empty());
    }
}
