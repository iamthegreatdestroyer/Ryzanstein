/// Cuckoo Filter — space-efficient probabilistic set membership.
///
/// O(1) lookup, supports deletion (unlike Bloom filters).
/// Used by ann-hybrid for exact identifier matching.

use std::hash::{Hash, Hasher};
use fnv::FnvHasher;

const BUCKET_SIZE: usize = 4;
const MAX_KICKS: usize = 500;

/// A Cuckoo Filter for set membership testing.
pub struct CuckooFilter {
    buckets: Vec<[u8; BUCKET_SIZE]>,
    num_buckets: usize,
    count: usize,
}

impl CuckooFilter {
    /// Create a new Cuckoo Filter with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let num_buckets = (capacity / BUCKET_SIZE).max(1);
        Self {
            buckets: vec![[0u8; BUCKET_SIZE]; num_buckets],
            num_buckets,
            count: 0,
        }
    }

    /// Insert an item into the filter.
    pub fn insert(&mut self, item: &str) -> bool {
        let fp = fingerprint(item);
        let i1 = self.index(item);
        let i2 = self.alt_index(i1, fp);

        // Try bucket i1
        if self.try_insert_bucket(i1, fp) {
            self.count += 1;
            return true;
        }
        // Try bucket i2
        if self.try_insert_bucket(i2, fp) {
            self.count += 1;
            return true;
        }

        // Kick existing entries
        let mut idx = if rand_bool() { i1 } else { i2 };
        let mut current_fp = fp;

        for _ in 0..MAX_KICKS {
            let slot = rand_slot();
            let evicted = self.buckets[idx][slot];
            self.buckets[idx][slot] = current_fp;
            current_fp = evicted;

            idx = self.alt_index(idx, current_fp);
            if self.try_insert_bucket(idx, current_fp) {
                self.count += 1;
                return true;
            }
        }

        false // Filter is full
    }

    /// Check if an item might be in the filter.
    pub fn contains(&self, item: &str) -> bool {
        let fp = fingerprint(item);
        let i1 = self.index(item);
        let i2 = self.alt_index(i1, fp);

        self.bucket_contains(i1, fp) || self.bucket_contains(i2, fp)
    }

    /// Delete an item from the filter.
    pub fn delete(&mut self, item: &str) -> bool {
        let fp = fingerprint(item);
        let i1 = self.index(item);
        let i2 = self.alt_index(i1, fp);

        if self.try_remove_bucket(i1, fp) {
            self.count -= 1;
            return true;
        }
        if self.try_remove_bucket(i2, fp) {
            self.count -= 1;
            return true;
        }
        false
    }

    /// Number of items inserted.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn index(&self, item: &str) -> usize {
        let mut hasher = FnvHasher::default();
        item.hash(&mut hasher);
        hasher.finish() as usize % self.num_buckets
    }

    fn alt_index(&self, idx: usize, fp: u8) -> usize {
        let mut hasher = FnvHasher::default();
        fp.hash(&mut hasher);
        (idx ^ hasher.finish() as usize) % self.num_buckets
    }

    fn try_insert_bucket(&mut self, bucket: usize, fp: u8) -> bool {
        for slot in &mut self.buckets[bucket] {
            if *slot == 0 {
                *slot = fp;
                return true;
            }
        }
        false
    }

    fn bucket_contains(&self, bucket: usize, fp: u8) -> bool {
        self.buckets[bucket].iter().any(|&s| s == fp)
    }

    fn try_remove_bucket(&mut self, bucket: usize, fp: u8) -> bool {
        for slot in &mut self.buckets[bucket] {
            if *slot == fp {
                *slot = 0;
                return true;
            }
        }
        false
    }
}

fn fingerprint(item: &str) -> u8 {
    let mut hasher = FnvHasher::default();
    item.hash(&mut hasher);
    let h = hasher.finish();
    let fp = (h >> 32) as u8;
    if fp == 0 { 1 } else { fp } // Never store 0 (empty marker)
}

fn rand_bool() -> bool {
    // Deterministic for reproducibility; in production use rand
    true
}

fn rand_slot() -> usize {
    // Deterministic; cycles through slots
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuckoo_insert_contains() {
        let mut cf = CuckooFilter::new(1000);
        assert!(cf.insert("hello"));
        assert!(cf.contains("hello"));
        assert!(!cf.contains("world"));
    }

    #[test]
    fn test_cuckoo_delete() {
        let mut cf = CuckooFilter::new(1000);
        cf.insert("hello");
        assert!(cf.contains("hello"));
        assert!(cf.delete("hello"));
        assert!(!cf.contains("hello"));
    }

    #[test]
    fn test_cuckoo_many_items() {
        let mut cf = CuckooFilter::new(10000);
        for i in 0..500 {
            cf.insert(&format!("item_{i}"));
        }
        // All inserted items should be found
        for i in 0..500 {
            assert!(cf.contains(&format!("item_{i}")), "Missing item_{i}");
        }
        assert_eq!(cf.len(), 500);
    }

    #[test]
    fn test_cuckoo_empty() {
        let cf = CuckooFilter::new(100);
        assert!(cf.is_empty());
        assert!(!cf.contains("anything"));
    }

    #[test]
    fn test_cuckoo_false_positive_rate() {
        let mut cf = CuckooFilter::new(10000);
        for i in 0..1000 {
            cf.insert(&format!("real_{i}"));
        }
        let mut false_positives = 0;
        for i in 0..1000 {
            if cf.contains(&format!("fake_{i}")) {
                false_positives += 1;
            }
        }
        // FP rate should be <5% for this configuration
        assert!(false_positives < 50, "FP rate too high: {false_positives}/1000");
    }
}
