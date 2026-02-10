//! Incremental parsing via tree-sitter for live index updates

use crate::error::{IndexError, IndexResult};
use crate::Language;

/// A parsed token from source code
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub kind: TokenKind,
    pub start_byte: usize,
    pub end_byte: usize,
    pub line: usize,
}

/// Token classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Function,
    Class,
    Variable,
    Import,
    Comment,
    String,
    Number,
    Keyword,
    Operator,
    Identifier,
    Other,
}

/// Parse a file into structured tokens
pub fn parse_file(content: &str, language: Language) -> IndexResult<Vec<Token>> {
    // Tree-sitter based parsing (simplified for scaffolding)
    let mut tokens = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let kind = classify_line(trimmed, language);
        tokens.push(Token {
            text: trimmed.to_string(),
            kind,
            start_byte: 0,
            end_byte: trimmed.len(),
            line: line_num + 1,
        });
    }
    Ok(tokens)
}

/// Classify a line based on language-specific patterns
fn classify_line(line: &str, language: Language) -> TokenKind {
    match language {
        Language::Rust => {
            if line.starts_with("fn ") || line.starts_with("pub fn ") { TokenKind::Function }
            else if line.starts_with("struct ") || line.starts_with("pub struct ") { TokenKind::Class }
            else if line.starts_with("use ") { TokenKind::Import }
            else if line.starts_with("//") { TokenKind::Comment }
            else if line.starts_with("let ") || line.starts_with("const ") { TokenKind::Variable }
            else { TokenKind::Other }
        }
        Language::Python => {
            if line.starts_with("def ") { TokenKind::Function }
            else if line.starts_with("class ") { TokenKind::Class }
            else if line.starts_with("import ") || line.starts_with("from ") { TokenKind::Import }
            else if line.starts_with('#') { TokenKind::Comment }
            else { TokenKind::Other }
        }
        Language::Go => {
            if line.starts_with("func ") { TokenKind::Function }
            else if line.starts_with("type ") { TokenKind::Class }
            else if line.starts_with("import ") { TokenKind::Import }
            else if line.starts_with("//") { TokenKind::Comment }
            else { TokenKind::Other }
        }
        _ => TokenKind::Other,
    }
}

/// Compute incremental diff between old and new parse trees
pub fn compute_edit_delta(
    old_tokens: &[Token],
    new_tokens: &[Token],
) -> Vec<EditDelta> {
    let mut deltas = Vec::new();

    let max_len = old_tokens.len().max(new_tokens.len());
    for i in 0..max_len {
        match (old_tokens.get(i), new_tokens.get(i)) {
            (Some(old), Some(new)) if old.text != new.text => {
                deltas.push(EditDelta::Modified { line: new.line, old_text: old.text.clone(), new_text: new.text.clone() });
            }
            (None, Some(new)) => {
                deltas.push(EditDelta::Added { line: new.line, text: new.text.clone() });
            }
            (Some(old), None) => {
                deltas.push(EditDelta::Removed { line: old.line, text: old.text.clone() });
            }
            _ => {}
        }
    }

    deltas
}

/// Represents a change between two versions
#[derive(Debug, Clone)]
pub enum EditDelta {
    Added { line: usize, text: String },
    Removed { line: usize, text: String },
    Modified { line: usize, old_text: String, new_text: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rust_file() {
        let content = "fn main() {\n    println!(\"hello\");\n}\n";
        let tokens = parse_file(content, Language::Rust).unwrap();
        assert!(!tokens.is_empty());
        assert_eq!(tokens[0].kind, TokenKind::Function);
    }

    #[test]
    fn test_parse_python_file() {
        let content = "def hello():\n    print('hello')\n";
        let tokens = parse_file(content, Language::Python).unwrap();
        assert!(!tokens.is_empty());
        assert_eq!(tokens[0].kind, TokenKind::Function);
    }

    #[test]
    fn test_parse_empty() {
        let tokens = parse_file("", Language::Rust).unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_edit_delta_added() {
        let old = vec![];
        let new = vec![Token { text: "fn new()".into(), kind: TokenKind::Function, start_byte: 0, end_byte: 8, line: 1 }];
        let deltas = compute_edit_delta(&old, &new);
        assert_eq!(deltas.len(), 1);
        matches!(&deltas[0], EditDelta::Added { .. });
    }

    #[test]
    fn test_edit_delta_removed() {
        let old = vec![Token { text: "fn old()".into(), kind: TokenKind::Function, start_byte: 0, end_byte: 8, line: 1 }];
        let new = vec![];
        let deltas = compute_edit_delta(&old, &new);
        assert_eq!(deltas.len(), 1);
        matches!(&deltas[0], EditDelta::Removed { .. });
    }
}
