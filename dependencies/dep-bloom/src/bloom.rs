//! Bloom filter implementation using SipHash for uniform distribution.

use bitvec::prelude::*;
use siphasher::sip::SipHasher;
use std::hash::{Hash, Hasher};

use crate::error::DepBloomError;

/// Probabilistic set membership data structure with O(1) insert/query.
#[derive(Clone)]
pub struct BloomFilter {
    bits: BitVec,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter sized for `expected_items` with `fp_rate` false positive rate.
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        let size = optimal_size(expected_items, fp_rate);
        let num_hashes = optimal_hashes(size, expected_items);
        BloomFilter {
            bits: bitvec![0; size],
            num_hashes,
            size,
            count: 0,
        }
    }

    /// Insert an item into the filter.
    pub fn insert(&mut self, item: &str) {
        for i in 0..self.num_hashes {
            let idx = self.hash_index(item, i);
            self.bits.set(idx, true);
        }
        self.count += 1;
    }

    /// Check if an item may be in the filter (probabilistic).
    pub fn may_contain(&self, item: &str) -> bool {
        (0..self.num_hashes).all(|i| {
            let idx = self.hash_index(item, i);
            self.bits[idx]
        })
    }

    /// Merge two Bloom filters (must be same size).
    pub fn union(&self, other: &BloomFilter) -> Result<BloomFilter, DepBloomError> {
        if self.size != other.size {
            return Err(DepBloomError::SizeMismatch {
                expected: self.size,
                actual: other.size,
            });
        }
        let mut merged = self.clone();
        merged.bits |= &other.bits;
        merged.count = self.count + other.count;
        Ok(merged)
    }

    /// Approximate current false positive rate.
    pub fn estimated_fp_rate(&self) -> f64 {
        let ones = self.bits.count_ones() as f64;
        let m = self.size as f64;
        let k = self.num_hashes as f64;
        (ones / m).powf(k)
    }

    /// Number of items inserted.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn hash_index(&self, item: &str, seed: usize) -> usize {
        let mut hasher = SipHasher::new_with_keys(seed as u64, 0x517cc1b727220a95);
        item.hash(&mut hasher);
        (hasher.finish() as usize) % self.size
    }
}

fn optimal_size(n: usize, fp: f64) -> usize {
    let ln2_sq = (2.0_f64.ln()).powi(2);
    (-(n as f64) * fp.ln() / ln2_sq).ceil() as usize
}

fn optimal_hashes(m: usize, n: usize) -> usize {
    let k = (m as f64 / n as f64) * 2.0_f64.ln();
    k.ceil().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_filter() {
        let bf = BloomFilter::new(100, 0.01);
        assert!(bf.is_empty());
        assert!(!bf.may_contain("anything"));
    }

    #[test]
    fn test_single_insert() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert("hello");
        assert!(bf.may_contain("hello"));
        assert_eq!(bf.len(), 1);
    }

    #[test]
    fn test_union_different_sizes_errors() {
        let bf1 = BloomFilter::new(100, 0.01);
        let bf2 = BloomFilter::new(200, 0.01);
        assert!(bf1.union(&bf2).is_err());
    }

    #[test]
    fn test_fp_rate_estimate() {
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..500 {
            bf.insert(&format!("item_{}", i));
        }
        let rate = bf.estimated_fp_rate();
        assert!(rate < 0.1);
    }
}
