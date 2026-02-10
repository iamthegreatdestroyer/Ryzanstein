//! AST parsing module for σ-diff

use crate::error::{DiffError, DiffResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    Go,
    Unknown,
}

/// Simplified AST node for diff comparison
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AstNode {
    pub kind: String,
    pub name: Option<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub children: Vec<AstNode>,
    pub content_hash: u64,
}

impl AstNode {
    pub fn leaf(kind: &str, name: Option<&str>, line: usize, content: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        Self {
            kind: kind.to_string(),
            name: name.map(|s| s.to_string()),
            start_line: line,
            end_line: line,
            children: Vec::new(),
            content_hash: hasher.finish(),
        }
    }
}

/// Parse code into a simplified AST
pub fn parse_code(content: &str, lang: Language) -> DiffResult<AstNode> {
    // Simplified structural parser using line-based heuristics
    // In production this would use tree-sitter
    let lines: Vec<&str> = content.lines().collect();
    let mut children = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        let (kind, name) = classify_line(trimmed, lang);
        if let Some(kind) = kind {
            children.push(AstNode::leaf(&kind, name.as_deref(), i + 1, trimmed));
        }
    }

    Ok(AstNode {
        kind: "root".to_string(),
        name: None,
        start_line: 1,
        end_line: lines.len(),
        children,
        content_hash: 0,
    })
}

fn classify_line(line: &str, lang: Language) -> (Option<String>, Option<String>) {
    match lang {
        Language::Rust => classify_rust_line(line),
        Language::Python => classify_python_line(line),
        Language::Go => classify_go_line(line),
        Language::JavaScript => classify_js_line(line),
        Language::Unknown => (None, None),
    }
}

fn classify_rust_line(line: &str) -> (Option<String>, Option<String>) {
    if line.starts_with("fn ") || line.starts_with("pub fn ") || line.starts_with("async fn ") || line.starts_with("pub async fn ") {
        let name = extract_name_before_paren(line, "fn ");
        (Some("function".to_string()), name)
    } else if line.starts_with("struct ") || line.starts_with("pub struct ") {
        let name = extract_name_after(line, "struct ");
        (Some("struct".to_string()), name)
    } else if line.starts_with("enum ") || line.starts_with("pub enum ") {
        let name = extract_name_after(line, "enum ");
        (Some("enum".to_string()), name)
    } else if line.starts_with("use ") || line.starts_with("pub use ") {
        (Some("import".to_string()), None)
    } else if line.starts_with("//") {
        (Some("comment".to_string()), None)
    } else {
        (None, None)
    }
}

fn classify_python_line(line: &str) -> (Option<String>, Option<String>) {
    if line.starts_with("def ") || line.starts_with("async def ") {
        let name = extract_name_before_paren(line, "def ");
        (Some("function".to_string()), name)
    } else if line.starts_with("class ") {
        let name = extract_name_before_paren(line, "class ");
        (Some("struct".to_string()), name)
    } else if line.starts_with("import ") || line.starts_with("from ") {
        (Some("import".to_string()), None)
    } else if line.starts_with("#") {
        (Some("comment".to_string()), None)
    } else {
        (None, None)
    }
}

fn classify_go_line(line: &str) -> (Option<String>, Option<String>) {
    if line.starts_with("func ") {
        let name = extract_name_before_paren(line, "func ");
        (Some("function".to_string()), name)
    } else if line.starts_with("type ") && line.contains("struct") {
        let name = extract_name_after(line, "type ");
        (Some("struct".to_string()), name)
    } else if line.starts_with("import") {
        (Some("import".to_string()), None)
    } else if line.starts_with("//") {
        (Some("comment".to_string()), None)
    } else {
        (None, None)
    }
}

fn classify_js_line(line: &str) -> (Option<String>, Option<String>) {
    if line.starts_with("function ") || line.contains("=> {") || line.starts_with("async function") {
        let name = extract_name_before_paren(line, "function ");
        (Some("function".to_string()), name)
    } else if line.starts_with("class ") {
        let name = extract_name_before_paren(line, "class ");
        (Some("struct".to_string()), name)
    } else if line.starts_with("import ") || line.starts_with("const ") && line.contains("require") {
        (Some("import".to_string()), None)
    } else {
        (None, None)
    }
}

fn extract_name_before_paren(line: &str, keyword: &str) -> Option<String> {
    if let Some(pos) = line.find(keyword) {
        let after = &line[pos + keyword.len()..];
        let name: String = after.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if name.is_empty() { None } else { Some(name) }
    } else {
        None
    }
}

fn extract_name_after(line: &str, keyword: &str) -> Option<String> {
    if let Some(pos) = line.find(keyword) {
        let after = &line[pos + keyword.len()..];
        let name: String = after.trim().chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if name.is_empty() { None } else { Some(name) }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rust_code() {
        let code = "fn foo() {}\npub struct Bar {}\nuse std::io;\n";
        let ast = parse_code(code, Language::Rust).unwrap();
        assert_eq!(ast.children.len(), 3);
        assert_eq!(ast.children[0].kind, "function");
        assert_eq!(ast.children[0].name.as_deref(), Some("foo"));
        assert_eq!(ast.children[1].kind, "struct");
        assert_eq!(ast.children[2].kind, "import");
    }

    #[test]
    fn test_parse_python_code() {
        let code = "def hello():\n    pass\nclass Foo:\n    pass\nimport os\n";
        let ast = parse_code(code, Language::Python).unwrap();
        assert!(ast.children.len() >= 3);
    }

    #[test]
    fn test_parse_go_code() {
        let code = "func main() {\n}\ntype Config struct {\n}\n";
        let ast = parse_code(code, Language::Go).unwrap();
        assert!(ast.children.len() >= 2);
    }

    #[test]
    fn test_ast_node_leaf() {
        let node = AstNode::leaf("function", Some("test"), 1, "fn test() {}");
        assert_eq!(node.kind, "function");
        assert_eq!(node.name.as_deref(), Some("test"));
        assert!(node.content_hash != 0);
    }
}
