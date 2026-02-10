/// Count-Min Sketch — sub-linear frequency estimation.
///
/// O(1) insert and query with bounded overestimation error.
/// Used by ann-hybrid for ranking results by access frequency.

use std::hash::{Hash, Hasher};
use fnv::FnvHasher;

/// A Count-Min Sketch for frequency estimation.
pub struct CountMinSketch {
    /// 2D array: depth × width
    table: Vec<Vec<u32>>,
    width: usize,
    depth: usize,
}

impl CountMinSketch {
    /// Create a new Count-Min Sketch.
    ///
    /// - `width`: number of counters per row (controls accuracy)
    /// - `depth`: number of hash functions (controls confidence)
    ///
    /// Error ≤ ε·N with probability ≥ 1-δ where:
    /// - ε = e/width
    /// - δ = 1/e^depth
    pub fn new(width: usize, depth: usize) -> Self {
        Self {
            table: vec![vec![0u32; width]; depth],
            width,
            depth,
        }
    }

    /// Create with target error rate and confidence.
    pub fn with_error_rate(epsilon: f64, delta: f64) -> Self {
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;
        Self::new(width.max(1), depth.max(1))
    }

    /// Increment the count for an item.
    pub fn increment(&mut self, item: &str) {
        self.add(item, 1);
    }

    /// Add a count for an item.
    pub fn add(&mut self, item: &str, count: u32) {
        for i in 0..self.depth {
            let idx = self.hash(item, i);
            self.table[i][idx] = self.table[i][idx].saturating_add(count);
        }
    }

    /// Estimate the count for an item (minimum across all hash functions).
    pub fn estimate(&self, item: &str) -> u32 {
        (0..self.depth)
            .map(|i| self.table[i][self.hash(item, i)])
            .min()
            .unwrap_or(0)
    }

    /// Reset all counters to zero.
    pub fn clear(&mut self) {
        for row in &mut self.table {
            for cell in row {
                *cell = 0;
            }
        }
    }

    fn hash(&self, item: &str, seed: usize) -> usize {
        let mut hasher = FnvHasher::default();
        seed.hash(&mut hasher);
        item.hash(&mut hasher);
        hasher.finish() as usize % self.width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cms_basic() {
        let mut cms = CountMinSketch::new(1000, 5);
        cms.increment("hello");
        cms.increment("hello");
        cms.increment("hello");
        cms.increment("world");

        assert_eq!(cms.estimate("hello"), 3);
        assert_eq!(cms.estimate("world"), 1);
    }

    #[test]
    fn test_cms_unseen_item() {
        let cms = CountMinSketch::new(1000, 5);
        assert_eq!(cms.estimate("never_seen"), 0);
    }

    #[test]
    fn test_cms_overestimate_only() {
        let mut cms = CountMinSketch::new(100, 4);
        for i in 0..1000 {
            cms.increment(&format!("item_{i}"));
        }
        // Count-Min Sketch can only overestimate, never underestimate
        for i in 0..1000 {
            assert!(cms.estimate(&format!("item_{i}")) >= 1);
        }
    }

    #[test]
    fn test_cms_add_count() {
        let mut cms = CountMinSketch::new(1000, 5);
        cms.add("bulk", 100);
        assert_eq!(cms.estimate("bulk"), 100);
    }

    #[test]
    fn test_cms_clear() {
        let mut cms = CountMinSketch::new(1000, 5);
        cms.increment("hello");
        assert_eq!(cms.estimate("hello"), 1);
        cms.clear();
        assert_eq!(cms.estimate("hello"), 0);
    }

    #[test]
    fn test_cms_with_error_rate() {
        let cms = CountMinSketch::with_error_rate(0.01, 0.001);
        assert!(cms.width > 0);
        assert!(cms.depth > 0);
    }
}
