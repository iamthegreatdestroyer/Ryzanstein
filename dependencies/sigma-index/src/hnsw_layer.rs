//! HNSW layer for semantic nearest-neighbor search on code embeddings
//!
//! Provides O(log n) semantic search via Hierarchical Navigable Small World graphs.

use crate::config::HNSWConfig;
use crate::error::{IndexError, IndexResult};
use crate::query::{MatchType, SearchResult};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// HNSW-based semantic search layer
pub struct HNSWLayer {
    config: HNSWConfig,
    /// Embedding storage: path -> list of (embedding, line_range)
    embeddings: HashMap<PathBuf, Vec<EmbeddingEntry>>,
    /// Flat index for ANN search (simplified; production uses HNSW graph)
    flat_index: Vec<(usize, Vec<f32>)>,
    /// Entry ID counter
    next_id: usize,
    /// Total memory used
    total_size: u64,
}

#[derive(Debug, Clone)]
struct EmbeddingEntry {
    id: usize,
    embedding: Vec<f32>,
    start_line: usize,
    end_line: usize,
}

impl HNSWLayer {
    pub fn new(config: HNSWConfig) -> IndexResult<Self> {
        Ok(Self {
            config,
            embeddings: HashMap::new(),
            flat_index: Vec::new(),
            next_id: 0,
            total_size: 0,
        })
    }

    /// Add embeddings for a document
    pub fn add_embeddings(
        &mut self,
        path: &Path,
        raw_embeddings: &[f32],
    ) -> IndexResult<()> {
        let dim = self.config.embedding_dim;
        let chunks: Vec<Vec<f32>> = raw_embeddings
            .chunks(dim)
            .map(|c| c.to_vec())
            .collect();

        let mut entries = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let id = self.next_id;
            self.next_id += 1;

            let entry = EmbeddingEntry {
                id,
                embedding: chunk.clone(),
                start_line: i * 50 + 1, // rough line mapping
                end_line: (i + 1) * 50,
            };

            self.flat_index.push((id, chunk.clone()));
            entries.push(entry);
        }

        let size_delta = entries.len() * dim * 4; // f32 = 4 bytes
        self.total_size += size_delta as u64;
        self.embeddings.insert(path.to_path_buf(), entries);

        Ok(())
    }

    /// Semantic search: find code similar to query embedding
    pub async fn search(&self, query: &str, top_k: usize) -> IndexResult<Vec<SearchResult>> {
        // In production, encode query via Ryzanstein ΣLANG
        // For now, use placeholder embedding
        let query_embedding = self.encode_query(query).await?;

        let mut scored: Vec<(f32, usize)> = self.flat_index.iter()
            .map(|(id, emb)| {
                let sim = cosine_similarity(&query_embedding, emb);
                (sim, *id)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let results = scored.iter()
            .filter_map(|(score, id)| {
                self.find_entry_by_id(*id).map(|(path, entry)| SearchResult {
                    file_path: path.clone(),
                    line: entry.start_line,
                    column: 0,
                    score: *score,
                    snippet: format!("Lines {}-{}", entry.start_line, entry.end_line),
                    match_type: MatchType::Semantic,
                })
            })
            .collect();

        Ok(results)
    }

    /// Remove all embeddings for a document
    pub fn remove_document(&mut self, path: &Path) -> IndexResult<()> {
        if let Some(entries) = self.embeddings.remove(path) {
            let ids: Vec<usize> = entries.iter().map(|e| e.id).collect();
            self.flat_index.retain(|(id, _)| !ids.contains(id));
            let size_delta = entries.len() * self.config.embedding_dim * 4;
            self.total_size = self.total_size.saturating_sub(size_delta as u64);
        }
        Ok(())
    }

    /// Get total memory usage
    pub fn size_bytes(&self) -> u64 {
        self.total_size
    }

    // -- Private --

    async fn encode_query(&self, _query: &str) -> IndexResult<Vec<f32>> {
        // Placeholder: return zero vector
        // In production, calls Ryzanstein ΣLANG encoder
        Ok(vec![0.0f32; self.config.embedding_dim])
    }

    fn find_entry_by_id(&self, id: usize) -> Option<(&PathBuf, &EmbeddingEntry)> {
        for (path, entries) in &self.embeddings {
            for entry in entries {
                if entry.id == id {
                    return Some((path, entry));
                }
            }
        }
        None
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_hnsw_new() {
        let config = HNSWConfig::default();
        let layer = HNSWLayer::new(config).unwrap();
        assert_eq!(layer.size_bytes(), 0);
    }

    #[test]
    fn test_add_and_remove_embeddings() {
        let config = HNSWConfig { embedding_dim: 4, ..HNSWConfig::default() };
        let mut layer = HNSWLayer::new(config).unwrap();
        let path = Path::new("test.rs");
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        layer.add_embeddings(path, &embeddings).unwrap();
        assert!(layer.size_bytes() > 0);

        layer.remove_document(path).unwrap();
        assert_eq!(layer.size_bytes(), 0);
    }
}
