//! σ-diff: Structural AST-aware diff engine
//!
//! Computes semantically-meaningful diffs between code files using
//! tree-sitter AST parsing. Instead of line-level diffs, σ-diff produces
//! structural changes (function added, parameter renamed, logic restructured).
//!
//! # Architecture
//! - Parse both versions with tree-sitter
//! - Compute AST edit script using Zhang-Shasha tree edit distance (simplified)
//! - Classify changes by structural category
//! - Generate Ryzanstein-compatible semantic diff output
//!
//! # Example
//! ```rust,no_run
//! use sigma_diff::{DiffEngine, DiffConfig};
//! let engine = DiffEngine::new(DiffConfig::default());
//! let result = engine.diff_files("old.rs", "new.rs").unwrap();
//! println!("{} structural changes", result.changes.len());
//! ```

pub mod ast;
pub mod config;
pub mod edit_script;
pub mod error;
pub mod ryzanstein_integration;

pub use config::DiffConfig;
pub use error::{DiffError, DiffResult};

use std::path::Path;

/// A structural change between two versions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StructuralChange {
    pub kind: ChangeKind,
    pub category: ChangeCategory,
    pub old_span: Option<Span>,
    pub new_span: Option<Span>,
    pub description: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Span {
    pub start_line: usize,
    pub end_line: usize,
    pub start_col: usize,
    pub end_col: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ChangeKind {
    Added,
    Removed,
    Modified,
    Moved,
    Renamed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ChangeCategory {
    Function,
    Struct,
    Enum,
    Import,
    Variable,
    Parameter,
    TypeSignature,
    Logic,
    Comment,
    Formatting,
}

/// Result of diffing two files
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffResult {
    pub old_path: String,
    pub new_path: String,
    pub changes: Vec<StructuralChange>,
    pub summary: DiffSummary,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffSummary {
    pub total_changes: usize,
    pub structural_changes: usize,
    pub cosmetic_changes: usize,
    pub risk_score: f64,
}

/// Main diff engine
pub struct DiffEngine {
    config: DiffConfig,
}

impl DiffEngine {
    pub fn new(config: DiffConfig) -> Self {
        Self { config }
    }

    /// Diff two files by path
    pub fn diff_files(&self, old_path: &str, new_path: &str) -> DiffResult<crate::DiffResult> {
        let old_content = std::fs::read_to_string(old_path)
            .map_err(|e| DiffError::IoError(e.to_string()))?;
        let new_content = std::fs::read_to_string(new_path)
            .map_err(|e| DiffError::IoError(e.to_string()))?;
        self.diff_content(&old_content, &new_content, old_path, new_path)
    }

    /// Diff two strings of code
    pub fn diff_content(
        &self,
        old_content: &str,
        new_content: &str,
        old_path: &str,
        new_path: &str,
    ) -> DiffResult<crate::DiffResult> {
        let lang = detect_language(old_path);

        // Parse ASTs
        let old_ast = ast::parse_code(old_content, lang)?;
        let new_ast = ast::parse_code(new_content, lang)?;

        // Compute edit script
        let changes = edit_script::compute_structural_diff(&old_ast, &new_ast, &self.config)?;

        let structural = changes.iter()
            .filter(|c| !matches!(c.category, ChangeCategory::Comment | ChangeCategory::Formatting))
            .count();
        let cosmetic = changes.len() - structural;

        let risk_score = compute_risk_score(&changes);

        Ok(crate::DiffResult {
            old_path: old_path.to_string(),
            new_path: new_path.to_string(),
            summary: DiffSummary {
                total_changes: changes.len(),
                structural_changes: structural,
                cosmetic_changes: cosmetic,
                risk_score,
            },
            changes,
        })
    }

    /// Diff with semantic understanding via Ryzanstein
    pub async fn diff_semantic(
        &self,
        old_content: &str,
        new_content: &str,
    ) -> DiffResult<Vec<StructuralChange>> {
        let client = ryzanstein_integration::RyzansteinDiffClient::new(
            &self.config.ryzanstein_url,
        );
        client.semantic_diff(old_content, new_content).await
    }
}

fn detect_language(path: &str) -> ast::Language {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext {
        "rs" => ast::Language::Rust,
        "py" => ast::Language::Python,
        "js" | "jsx" => ast::Language::JavaScript,
        "go" => ast::Language::Go,
        _ => ast::Language::Unknown,
    }
}

fn compute_risk_score(changes: &[StructuralChange]) -> f64 {
    let weights: std::collections::HashMap<ChangeCategory, f64> = [
        (ChangeCategory::Logic, 0.9),
        (ChangeCategory::Function, 0.7),
        (ChangeCategory::TypeSignature, 0.8),
        (ChangeCategory::Parameter, 0.6),
        (ChangeCategory::Struct, 0.5),
        (ChangeCategory::Import, 0.3),
        (ChangeCategory::Variable, 0.4),
        (ChangeCategory::Comment, 0.05),
        (ChangeCategory::Formatting, 0.01),
        (ChangeCategory::Enum, 0.5),
    ].into_iter().collect();

    if changes.is_empty() { return 0.0; }
    let total: f64 = changes.iter()
        .map(|c| weights.get(&c.category).copied().unwrap_or(0.5))
        .sum();
    (total / changes.len() as f64).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("foo.rs") as u8, ast::Language::Rust as u8);
        assert_eq!(detect_language("bar.py") as u8, ast::Language::Python as u8);
    }

    #[test]
    fn test_diff_identical_content() {
        let engine = DiffEngine::new(DiffConfig::default());
        let result = engine.diff_content("fn foo() {}", "fn foo() {}", "a.rs", "b.rs").unwrap();
        assert_eq!(result.summary.total_changes, 0);
        assert_eq!(result.summary.risk_score, 0.0);
    }

    #[test]
    fn test_diff_added_function() {
        let engine = DiffEngine::new(DiffConfig::default());
        let old = "fn foo() {}";
        let new = "fn foo() {}\nfn bar() {}";
        let result = engine.diff_content(old, new, "a.rs", "b.rs").unwrap();
        assert!(result.summary.total_changes > 0);
    }

    #[test]
    fn test_risk_score_empty() {
        assert_eq!(compute_risk_score(&[]), 0.0);
    }

    #[test]
    fn test_risk_score_logic_change() {
        let changes = vec![StructuralChange {
            kind: ChangeKind::Modified,
            category: ChangeCategory::Logic,
            old_span: None, new_span: None,
            description: "logic changed".into(),
            confidence: 0.9,
        }];
        assert!(compute_risk_score(&changes) > 0.8);
    }
}
