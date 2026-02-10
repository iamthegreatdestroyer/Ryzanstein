//! Ryzanstein integration for semantic diffs

use crate::error::{DiffError, DiffResult};
use crate::{ChangeCategory, ChangeKind, StructuralChange};

/// Client for Ryzanstein semantic diff capability
pub struct RyzansteinDiffClient {
    base_url: String,
    client: reqwest::Client,
}

impl RyzansteinDiffClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Get semantic diff between two code snippets
    pub async fn semantic_diff(
        &self,
        old_content: &str,
        new_content: &str,
    ) -> DiffResult<Vec<StructuralChange>> {
        let embeddings = self.get_embeddings(&[old_content, new_content]).await;

        match embeddings {
            Ok(embs) if embs.len() == 2 => {
                let similarity = cosine_similarity(&embs[0], &embs[1]);
                if similarity > 0.95 {
                    Ok(Vec::new()) // Too similar, no meaningful changes
                } else {
                    Ok(vec![StructuralChange {
                        kind: ChangeKind::Modified,
                        category: ChangeCategory::Logic,
                        old_span: None,
                        new_span: None,
                        description: format!("Semantic divergence: {:.4} similarity", similarity),
                        confidence: 1.0 - similarity,
                    }])
                }
            }
            _ => {
                // Fallback: simple text-based similarity
                let sim = text_similarity(old_content, new_content);
                if sim > 0.95 {
                    Ok(Vec::new())
                } else {
                    Ok(vec![StructuralChange {
                        kind: ChangeKind::Modified,
                        category: ChangeCategory::Logic,
                        old_span: None,
                        new_span: None,
                        description: format!("Text divergence: {:.4} similarity", sim),
                        confidence: 1.0 - sim,
                    }])
                }
            }
        }
    }

    async fn get_embeddings(&self, texts: &[&str]) -> DiffResult<Vec<Vec<f32>>> {
        let payload = serde_json::json!({
            "model": "sigma-lang-v1",
            "input": texts,
        });

        let resp = self.client
            .post(format!("{}/v1/embeddings", self.base_url))
            .json(&payload)
            .send()
            .await
            .map_err(|e| DiffError::RyzansteinError(e.to_string()))?;

        let body: serde_json::Value = resp.json().await
            .map_err(|e| DiffError::RyzansteinError(e.to_string()))?;

        let embeddings: Vec<Vec<f32>> = body["data"].as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|d| {
                d["embedding"].as_array().map(|arr| {
                    arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect()
                })
            })
            .collect();

        Ok(embeddings)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    (dot / (norm_a * norm_b)) as f64
}

fn text_similarity(a: &str, b: &str) -> f64 {
    if a == b { return 1.0; }
    if a.is_empty() || b.is_empty() { return 0.0; }
    let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();
    let intersection = a_words.intersection(&b_words).count();
    let union = a_words.union(&b_words).count();
    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_text_similarity_identical() {
        assert!((text_similarity("hello world", "hello world") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_text_similarity_different() {
        assert!(text_similarity("hello world", "foo bar") < 0.5);
    }

    #[test]
    fn test_text_similarity_partial() {
        let sim = text_similarity("hello world foo", "hello world bar");
        assert!(sim > 0.3 && sim < 1.0);
    }
}
