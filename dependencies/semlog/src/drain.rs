use fnv::FnvHashMap;
/// Drain algorithm for log template extraction.
///
/// Drain parses log messages into templates by building a parse tree
/// from log tokens, grouping messages that share the same structure.
///
/// Reference: He et al., "Drain: An Online Log Parsing Approach with Fixed Depth Tree" (2017)
use std::collections::HashMap;

/// A log template extracted by Drain.
#[derive(Debug, Clone)]
pub struct LogTemplate {
    /// Unique template ID.
    pub id: u32,
    /// Template tokens (static text + `<*>` wildcards).
    pub tokens: Vec<String>,
    /// Number of log lines matching this template.
    pub count: u64,
}

impl LogTemplate {
    /// Render the template as a single line.
    pub fn to_string_repr(&self) -> String {
        self.tokens.join(" ")
    }
}

/// Configuration for the Drain parser.
#[derive(Debug, Clone)]
pub struct DrainConfig {
    /// Maximum parse tree depth.
    pub max_depth: usize,
    /// Similarity threshold for template matching [0, 1].
    pub sim_threshold: f64,
    /// Maximum number of children per internal node.
    pub max_children: usize,
}

impl Default for DrainConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            sim_threshold: 0.5,
            max_children: 100,
        }
    }
}

/// The Drain log parser.
pub struct DrainParser {
    config: DrainConfig,
    templates: Vec<LogTemplate>,
    /// Length → list of template indices
    length_groups: FnvHashMap<usize, Vec<usize>>,
    next_id: u32,
}

impl DrainParser {
    pub fn new(config: DrainConfig) -> Self {
        Self {
            config,
            templates: Vec::new(),
            length_groups: FnvHashMap::default(),
            next_id: 0,
        }
    }

    /// Parse a single log line and return the matched template ID.
    pub fn parse_line(&mut self, line: &str) -> u32 {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let len = tokens.len();

        // Find matching template in length group
        if let Some(group) = self.length_groups.get(&len) {
            for &idx in group {
                let template = &self.templates[idx];
                let sim = self.compute_similarity(&tokens, &template.tokens);
                if sim >= self.config.sim_threshold {
                    // Update template
                    self.templates[idx].count += 1;
                    self.merge_tokens(&tokens, idx);
                    return self.templates[idx].id;
                }
            }
        }

        // No match — create new template
        let id = self.next_id;
        self.next_id += 1;
        let template = LogTemplate {
            id,
            tokens: tokens.iter().map(|t| t.to_string()).collect(),
            count: 1,
        };
        let idx = self.templates.len();
        self.templates.push(template);
        self.length_groups.entry(len).or_default().push(idx);
        id
    }

    /// Get all discovered templates.
    pub fn templates(&self) -> &[LogTemplate] {
        &self.templates
    }

    /// Get template by ID.
    pub fn get_template(&self, id: u32) -> Option<&LogTemplate> {
        self.templates.iter().find(|t| t.id == id)
    }

    fn compute_similarity(&self, tokens: &[&str], template_tokens: &[String]) -> f64 {
        if tokens.len() != template_tokens.len() {
            return 0.0;
        }
        let matches = tokens
            .iter()
            .zip(template_tokens.iter())
            .filter(|(t, tt)| *t == tt.as_str() || tt == "<*>")
            .count();
        matches as f64 / tokens.len() as f64
    }

    fn merge_tokens(&mut self, tokens: &[&str], idx: usize) {
        let template = &mut self.templates[idx];
        for (i, token) in tokens.iter().enumerate() {
            if i < template.tokens.len()
                && template.tokens[i] != *token
                && template.tokens[i] != "<*>"
            {
                template.tokens[i] = "<*>".to_string();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drain_single_line() {
        let mut parser = DrainParser::new(DrainConfig::default());
        let id = parser.parse_line("GET /api/users 200 42ms");
        assert_eq!(id, 0);
        assert_eq!(parser.templates().len(), 1);
    }

    #[test]
    fn test_drain_same_template() {
        let mut parser = DrainParser::new(DrainConfig::default());
        let id1 = parser.parse_line("GET /api/users 200 42ms");
        let id2 = parser.parse_line("GET /api/items 200 35ms");
        // Should match same template with wildcards
        assert_eq!(id1, id2);
        assert_eq!(parser.templates()[0].count, 2);
    }

    #[test]
    fn test_drain_different_templates() {
        let mut parser = DrainParser::new(DrainConfig::default());
        let id1 = parser.parse_line("GET /api/users 200 42ms");
        let id2 = parser.parse_line("ERROR connection refused timeout after 30s retry 5");
        assert_ne!(id1, id2);
        assert_eq!(parser.templates().len(), 2);
    }

    #[test]
    fn test_drain_wildcard_creation() {
        let mut parser = DrainParser::new(DrainConfig::default());
        parser.parse_line("GET /api/users 200 42ms");
        parser.parse_line("GET /api/items 200 35ms");
        let template = &parser.templates()[0];
        assert!(template.tokens.contains(&"<*>".to_string()));
    }

    #[test]
    fn test_drain_template_to_string() {
        let t = LogTemplate {
            id: 0,
            tokens: vec!["GET".into(), "<*>".into(), "200".into()],
            count: 5,
        };
        assert_eq!(t.to_string_repr(), "GET <*> 200");
    }
}
