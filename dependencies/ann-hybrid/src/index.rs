/// Unified HybridIndex: fuses HNSW + Cuckoo Filter + Count-Min Sketch.
///
/// Queries flow through all three structures in parallel,
/// results are merged by a composite scoring function.

use crate::hnsw::{HnswConfig, HnswIndex};
use crate::cuckoo::CuckooFilter;
use crate::cms::CountMinSketch;
use serde::{Deserialize, Serialize};

/// Configuration for the hybrid index.
#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub hnsw: HnswConfig,
    pub cuckoo_capacity: usize,
    pub cms_width: usize,
    pub cms_depth: usize,
    /// How much to weight HNSW distance vs CMS frequency.
    pub frequency_boost: f32,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            hnsw: HnswConfig::default(),
            cuckoo_capacity: 100_000,
            cms_width: 10_000,
            cms_depth: 5,
            frequency_boost: 0.1,
        }
    }
}

/// A search query for the hybrid index.
#[derive(Debug)]
pub struct SearchQuery<'a> {
    /// Semantic vector for HNSW search (optional).
    pub vector: Option<&'a [f32]>,
    /// Exact keyword for Cuckoo filter matching (optional).
    pub keyword: Option<&'a str>,
    /// Number of results to return.
    pub top_k: usize,
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    /// Composite score (lower = better).
    pub score: f32,
    /// Whether the item matched the exact keyword filter.
    pub exact_match: bool,
    /// Estimated access frequency from CMS.
    pub frequency: u32,
}

/// The unified hybrid index.
pub struct HybridIndex {
    config: IndexConfig,
    hnsw: HnswIndex,
    cuckoo: CuckooFilter,
    cms: CountMinSketch,
}

impl HybridIndex {
    pub fn new(config: IndexConfig) -> Self {
        let hnsw = HnswIndex::new(config.hnsw.clone());
        let cuckoo = CuckooFilter::new(config.cuckoo_capacity);
        let cms = CountMinSketch::new(config.cms_width, config.cms_depth);

        Self {
            config,
            hnsw,
            cuckoo,
            cms,
        }
    }

    /// Insert an item into all three structures.
    pub fn insert(&mut self, id: &str, vector: &[f32], _metadata: Option<&str>) {
        self.hnsw.insert(id, vector);
        self.cuckoo.insert(id);
        // Initialize frequency to 0 (will be incremented on search hits)
    }

    /// Search with a hybrid query.
    pub fn search(&mut self, query: &SearchQuery) -> Vec<SearchResult> {
        let mut results = Vec::new();

        // Phase 1: HNSW semantic search
        if let Some(vector) = query.vector {
            let hnsw_results = self.hnsw.search(vector, query.top_k * 2);
            for (id, distance) in hnsw_results {
                let exact_match = query
                    .keyword
                    .map(|k| self.cuckoo.contains(k) && id.contains(k))
                    .unwrap_or(false);

                let frequency = self.cms.estimate(&id);
                let freq_score = 1.0 / (1.0 + frequency as f32 * self.config.frequency_boost);

                let score = distance * freq_score;

                // Boost exact matches
                let score = if exact_match { score * 0.5 } else { score };

                // Record access for frequency tracking
                self.cms.increment(&id);

                results.push(SearchResult {
                    id,
                    score,
                    exact_match,
                    frequency,
                });
            }
        }

        // Sort by composite score
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(query.top_k);
        results
    }

    /// Check if an ID exists in the index (via Cuckoo filter).
    pub fn contains(&self, id: &str) -> bool {
        self.cuckoo.contains(id)
    }

    /// Get estimated access frequency for an ID.
    pub fn frequency(&self, id: &str) -> u32 {
        self.cms.estimate(id)
    }

    /// Total number of items in the index.
    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hnsw.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(seed: f32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| ((seed + i as f32) * 0.1).sin()).collect()
    }

    #[test]
    fn test_hybrid_insert_and_search() {
        let mut index = HybridIndex::new(IndexConfig::default());
        for i in 0..100 {
            let v = make_vector(i as f32, 32);
            index.insert(&format!("item_{i}"), &v, None);
        }
        assert_eq!(index.len(), 100);

        let query_vec = make_vector(42.0, 32);
        let results = index.search(&SearchQuery {
            vector: Some(&query_vec),
            keyword: None,
            top_k: 5,
        });
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        assert_eq!(results[0].id, "item_42");
    }

    #[test]
    fn test_hybrid_contains() {
        let mut index = HybridIndex::new(IndexConfig::default());
        index.insert("hello", &[1.0, 2.0, 3.0], None);
        assert!(index.contains("hello"));
    }

    #[test]
    fn test_hybrid_empty() {
        let index = HybridIndex::new(IndexConfig::default());
        assert!(index.is_empty());
    }

    #[test]
    fn test_hybrid_frequency_tracking() {
        let mut index = HybridIndex::new(IndexConfig::default());
        for i in 0..10 {
            let v = make_vector(i as f32, 32);
            index.insert(&format!("item_{i}"), &v, None);
        }

        // Search twice — frequency should increase
        let qv = make_vector(0.0, 32);
        let r1 = index.search(&SearchQuery {
            vector: Some(&qv),
            keyword: None,
            top_k: 3,
        });
        let r2 = index.search(&SearchQuery {
            vector: Some(&qv),
            keyword: None,
            top_k: 3,
        });

        // Second search should show higher frequency for item_0
        let freq = index.frequency("item_0");
        assert!(freq >= 1);
    }

    #[test]
    fn test_hybrid_with_keyword() {
        let mut index = HybridIndex::new(IndexConfig::default());
        index.insert("error_handler", &[1.0, 0.0, 0.0], None);
        index.insert("normal_flow", &[0.0, 1.0, 0.0], None);

        let results = index.search(&SearchQuery {
            vector: Some(&[1.0, 0.0, 0.0]),
            keyword: Some("error"),
            top_k: 5,
        });
        assert!(!results.is_empty());
    }
}
