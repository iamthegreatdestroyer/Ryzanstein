/// Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor search.
///
/// Provides O(log n) semantic search over dense embedding vectors.

use rand::Rng;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// A node in the HNSW graph.
#[derive(Debug, Clone)]
pub struct HnswNode {
    pub id: String,
    pub vector: Vec<f32>,
    /// Connections per layer: layer index → neighbor node indices
    pub connections: Vec<Vec<usize>>,
    pub level: usize,
}

/// Configuration for the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per node per layer.
    pub m: usize,
    /// Size of the dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search.
    pub ef_search: usize,
    /// Normalization factor for level generation.
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (16.0_f64).ln(),
        }
    }
}

/// The HNSW index.
pub struct HnswIndex {
    config: HnswConfig,
    nodes: Vec<HnswNode>,
    entry_point: Option<usize>,
    max_level: usize,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
        }
    }

    /// Insert a vector with an ID into the index.
    pub fn insert(&mut self, id: &str, vector: &[f32]) {
        let level = self.random_level();
        let node_idx = self.nodes.len();

        let mut node = HnswNode {
            id: id.to_string(),
            vector: vector.to_vec(),
            connections: vec![Vec::new(); level + 1],
            level,
        };

        if self.nodes.is_empty() {
            self.nodes.push(node);
            self.entry_point = Some(0);
            self.max_level = level;
            return;
        }

        // Find entry point and greedily traverse from top layer
        let ep = self.entry_point.unwrap();
        let mut current = ep;

        // Traverse layers above node's level (greedy)
        for l in (level + 1..=self.max_level).rev() {
            current = self.greedy_closest(current, vector, l);
        }

        // For each layer from node's level down to 0, find and connect neighbors
        for l in (0..=level.min(self.max_level)).rev() {
            let neighbors = self.search_layer(current, vector, self.config.ef_construction, l);
            let selected: Vec<usize> = neighbors
                .into_iter()
                .take(self.config.m)
                .map(|(idx, _)| idx)
                .collect();

            node.connections[l] = selected.clone();

            // Add bidirectional connections
            for &neighbor_idx in &selected {
                if l < self.nodes[neighbor_idx].connections.len() {
                    self.nodes[neighbor_idx].connections[l].push(node_idx);
                    // Trim if over capacity
                    if self.nodes[neighbor_idx].connections[l].len() > self.config.m * 2 {
                        let nv = self.nodes[neighbor_idx].vector.clone();
                        self.nodes[neighbor_idx].connections[l]
                            .sort_by(|&a, &b| {
                                let da = cosine_distance(&nv, &self.nodes[a].vector);
                                let db = cosine_distance(&nv, &self.nodes[b].vector);
                                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        self.nodes[neighbor_idx].connections[l].truncate(self.config.m);
                    }
                }
            }

            if !selected.is_empty() {
                current = selected[0];
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(node_idx);
        }

        self.nodes.push(node);
    }

    /// Search for the top-k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let ep = self.entry_point.unwrap();
        let mut current = ep;

        // Traverse from top to layer 1 (greedy)
        for l in (1..=self.max_level).rev() {
            current = self.greedy_closest(current, query, l);
        }

        // Search layer 0 with ef_search candidates
        let candidates = self.search_layer(current, query, self.config.ef_search, 0);

        candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes[idx].id.clone(), dist))
            .collect()
    }

    /// Number of items in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut level = 0;
        while rng.gen::<f64>() < (1.0 / self.config.m as f64) && level < 16 {
            level += 1;
        }
        level
    }

    fn greedy_closest(&self, start: usize, query: &[f32], layer: usize) -> usize {
        let mut current = start;
        let mut current_dist = cosine_distance(query, &self.nodes[current].vector);

        loop {
            let mut changed = false;
            if layer < self.nodes[current].connections.len() {
                for &neighbor in &self.nodes[current].connections[layer] {
                    let dist = cosine_distance(query, &self.nodes[neighbor].vector);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    fn search_layer(
        &self,
        entry: usize,
        query: &[f32],
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = fnv::FnvHashSet::default();
        visited.insert(entry);

        let entry_dist = cosine_distance(query, &self.nodes[entry].vector);
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat, usize)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat, usize)> = BinaryHeap::new();

        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
        results.push((OrderedFloat(entry_dist), entry));

        while let Some(Reverse((OrderedFloat(c_dist), c_idx))) = candidates.pop() {
            let worst_dist = results.peek().map(|(OrderedFloat(d), _)| *d).unwrap_or(f32::MAX);
            if c_dist > worst_dist && results.len() >= ef {
                break;
            }

            if layer < self.nodes[c_idx].connections.len() {
                for &neighbor in &self.nodes[c_idx].connections[layer] {
                    if visited.insert(neighbor) {
                        let dist = cosine_distance(query, &self.nodes[neighbor].vector);
                        let worst_dist = results.peek().map(|(OrderedFloat(d), _)| *d).unwrap_or(f32::MAX);

                        if dist < worst_dist || results.len() < ef {
                            candidates.push(Reverse((OrderedFloat(dist), neighbor)));
                            results.push((OrderedFloat(dist), neighbor));
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut output: Vec<(usize, f32)> = results
            .into_iter()
            .map(|(OrderedFloat(d), idx)| (idx, d))
            .collect();
        output.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        output
    }
}

/// Cosine distance = 1 - cosine_similarity.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        return 1.0;
    }
    1.0 - (dot / denom)
}

/// Wrapper for f32 that implements Ord for use in BinaryHeap.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(seed: f32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| ((seed + i as f32) * 0.1).sin()).collect()
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let mut index = HnswIndex::new(HnswConfig::default());
        for i in 0..50 {
            let v = make_vector(i as f32, 32);
            index.insert(&format!("item_{i}"), &v);
        }
        assert_eq!(index.len(), 50);

        let query = make_vector(5.0, 32);
        let results = index.search(&query, 5);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // item_5 should be the closest to itself
        assert_eq!(results[0].0, "item_5");
    }

    #[test]
    fn test_hnsw_empty() {
        let index = HnswIndex::new(HnswConfig::default());
        assert!(index.is_empty());
        let results = index.search(&[0.1, 0.2, 0.3], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_distance(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hnsw_single_item() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("only", &[1.0, 2.0, 3.0]);
        let results = index.search(&[1.0, 2.0, 3.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "only");
    }
}
