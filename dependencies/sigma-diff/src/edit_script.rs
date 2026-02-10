//! Edit script computation for structural diffs

use crate::ast::AstNode;
use crate::config::DiffConfig;
use crate::error::DiffResult;
use crate::{ChangeCategory, ChangeKind, Span, StructuralChange};
use std::collections::HashMap;

/// Compute structural diff between two ASTs
pub fn compute_structural_diff(
    old_ast: &AstNode,
    new_ast: &AstNode,
    config: &DiffConfig,
) -> DiffResult<Vec<StructuralChange>> {
    let mut changes = Vec::new();

    // Build name → node maps for quick comparison
    let old_map = build_node_map(&old_ast.children);
    let new_map = build_node_map(&new_ast.children);

    // Find removed nodes
    for (key, old_node) in &old_map {
        if !new_map.contains_key(key) {
            changes.push(StructuralChange {
                kind: ChangeKind::Removed,
                category: kind_to_category(&old_node.kind),
                old_span: Some(Span {
                    start_line: old_node.start_line,
                    end_line: old_node.end_line,
                    start_col: 0,
                    end_col: 0,
                }),
                new_span: None,
                description: format!("{} '{}' removed", old_node.kind, key),
                confidence: 0.95,
            });
        }
    }

    // Find added nodes
    for (key, new_node) in &new_map {
        if !old_map.contains_key(key) {
            // Check if it was renamed (similar hash in old)
            let possibly_renamed = old_map.values()
                .find(|o| o.kind == new_node.kind && o.content_hash == new_node.content_hash);

            if let Some(old_node) = possibly_renamed {
                changes.push(StructuralChange {
                    kind: ChangeKind::Renamed,
                    category: kind_to_category(&new_node.kind),
                    old_span: Some(Span {
                        start_line: old_node.start_line, end_line: old_node.end_line,
                        start_col: 0, end_col: 0,
                    }),
                    new_span: Some(Span {
                        start_line: new_node.start_line, end_line: new_node.end_line,
                        start_col: 0, end_col: 0,
                    }),
                    description: format!("{} renamed to '{}'", old_node.name.as_deref().unwrap_or("?"), key),
                    confidence: 0.8,
                });
            } else {
                changes.push(StructuralChange {
                    kind: ChangeKind::Added,
                    category: kind_to_category(&new_node.kind),
                    old_span: None,
                    new_span: Some(Span {
                        start_line: new_node.start_line,
                        end_line: new_node.end_line,
                        start_col: 0,
                        end_col: 0,
                    }),
                    description: format!("{} '{}' added", new_node.kind, key),
                    confidence: 0.95,
                });
            }
        }
    }

    // Find modified nodes (same name, different hash)
    for (key, old_node) in &old_map {
        if let Some(new_node) = new_map.get(key) {
            if old_node.content_hash != new_node.content_hash {
                changes.push(StructuralChange {
                    kind: ChangeKind::Modified,
                    category: kind_to_category(&old_node.kind),
                    old_span: Some(Span {
                        start_line: old_node.start_line, end_line: old_node.end_line,
                        start_col: 0, end_col: 0,
                    }),
                    new_span: Some(Span {
                        start_line: new_node.start_line, end_line: new_node.end_line,
                        start_col: 0, end_col: 0,
                    }),
                    description: format!("{} '{}' modified", old_node.kind, key),
                    confidence: 0.9,
                });
            }
        }
    }

    // Filter by confidence
    let changes: Vec<_> = changes.into_iter()
        .filter(|c| c.confidence >= config.confidence_threshold)
        .collect();

    // Filter cosmetic if disabled
    if !config.include_cosmetic {
        return Ok(changes.into_iter()
            .filter(|c| !matches!(c.category, ChangeCategory::Comment | ChangeCategory::Formatting))
            .collect());
    }

    Ok(changes)
}

fn build_node_map(nodes: &[AstNode]) -> HashMap<String, &AstNode> {
    let mut map = HashMap::new();
    for node in nodes {
        let key = node.name.clone()
            .unwrap_or_else(|| format!("{}_{}", node.kind, node.start_line));
        map.insert(key, node);
    }
    map
}

fn kind_to_category(kind: &str) -> ChangeCategory {
    match kind {
        "function" => ChangeCategory::Function,
        "struct" => ChangeCategory::Struct,
        "enum" => ChangeCategory::Enum,
        "import" => ChangeCategory::Import,
        "variable" => ChangeCategory::Variable,
        "comment" => ChangeCategory::Comment,
        _ => ChangeCategory::Logic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{parse_code, Language};

    #[test]
    fn test_no_changes() {
        let code = "fn foo() {}";
        let old = parse_code(code, Language::Rust).unwrap();
        let new = parse_code(code, Language::Rust).unwrap();
        let config = DiffConfig::default();
        let changes = compute_structural_diff(&old, &new, &config).unwrap();
        assert!(changes.is_empty());
    }

    #[test]
    fn test_added_function() {
        let old = parse_code("fn foo() {}", Language::Rust).unwrap();
        let new = parse_code("fn foo() {}\nfn bar() {}", Language::Rust).unwrap();
        let config = DiffConfig::default();
        let changes = compute_structural_diff(&old, &new, &config).unwrap();
        assert!(changes.iter().any(|c| c.kind == ChangeKind::Added));
    }

    #[test]
    fn test_removed_function() {
        let old = parse_code("fn foo() {}\nfn bar() {}", Language::Rust).unwrap();
        let new = parse_code("fn foo() {}", Language::Rust).unwrap();
        let config = DiffConfig::default();
        let changes = compute_structural_diff(&old, &new, &config).unwrap();
        assert!(changes.iter().any(|c| c.kind == ChangeKind::Removed));
    }

    #[test]
    fn test_kind_to_category() {
        assert_eq!(kind_to_category("function"), ChangeCategory::Function);
        assert_eq!(kind_to_category("struct"), ChangeCategory::Struct);
        assert_eq!(kind_to_category("unknown"), ChangeCategory::Logic);
    }
}
