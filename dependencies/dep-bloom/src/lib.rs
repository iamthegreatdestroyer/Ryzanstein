//! # dep-bloom
//!
//! Probabilistic dependency resolution using Bloom filters.
//! Provides O(1) lookup for dependency existence with configurable
//! false positive rates. Used by Ryzanstein for fast dependency
//! graph queries and conflict detection.

pub mod bloom;
pub mod config;
pub mod dependency;
pub mod error;
pub mod ryzanstein_integration;

pub use bloom::BloomFilter;
pub use config::DepBloomConfig;
pub use dependency::{DepResolver, Dependency, DependencyGraph, VersionConstraint};
pub use error::DepBloomError;

/// Convenience constructor for a dependency resolver with default config.
pub fn new_resolver() -> DepResolver {
    DepResolver::new(DepBloomConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_insert_and_query() {
        let mut bf = BloomFilter::new(1000, 0.01);
        bf.insert("tokio");
        bf.insert("serde");
        assert!(bf.may_contain("tokio"));
        assert!(bf.may_contain("serde"));
        // Should almost certainly not contain
        assert!(!bf.may_contain("nonexistent_crate_xyz"));
    }

    #[test]
    fn test_bloom_filter_false_positive_rate() {
        let mut bf = BloomFilter::new(10000, 0.01);
        for i in 0..5000 {
            bf.insert(&format!("item_{}", i));
        }
        let mut false_positives = 0;
        for i in 5000..10000 {
            if bf.may_contain(&format!("item_{}", i)) {
                false_positives += 1;
            }
        }
        let rate = false_positives as f64 / 5000.0;
        assert!(rate < 0.05, "False positive rate {:.4} exceeds 5%", rate);
    }

    #[test]
    fn test_dependency_resolver() {
        let mut resolver = new_resolver();
        resolver.add_dependency(Dependency {
            name: "tokio".into(),
            version: "1.34.0".into(),
            constraint: VersionConstraint::Compatible,
        });
        assert!(resolver.has_dependency("tokio"));
        assert!(!resolver.has_dependency("missing"));
    }

    #[test]
    fn test_dependency_graph_cycle_detection() {
        let mut graph = DependencyGraph::new();
        graph.add_edge("A", "B");
        graph.add_edge("B", "C");
        assert!(!graph.has_cycle());
        graph.add_edge("C", "A");
        assert!(graph.has_cycle());
    }

    #[test]
    fn test_dependency_graph_topological_sort() {
        let mut graph = DependencyGraph::new();
        graph.add_edge("A", "B");
        graph.add_edge("B", "C");
        graph.add_edge("A", "C");
        let order = graph.topological_sort().unwrap();
        let pos_a = order.iter().position(|x| x == "A").unwrap();
        let pos_b = order.iter().position(|x| x == "B").unwrap();
        let pos_c = order.iter().position(|x| x == "C").unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_bloom_union() {
        let mut bf1 = BloomFilter::new(100, 0.01);
        let mut bf2 = BloomFilter::new(100, 0.01);
        bf1.insert("alpha");
        bf2.insert("beta");
        let merged = bf1.union(&bf2).unwrap();
        assert!(merged.may_contain("alpha"));
        assert!(merged.may_contain("beta"));
    }

    #[test]
    fn test_resolver_conflicts() {
        let mut resolver = new_resolver();
        resolver.add_dependency(Dependency {
            name: "serde".into(),
            version: "1.0.0".into(),
            constraint: VersionConstraint::Exact,
        });
        resolver.add_dependency(Dependency {
            name: "serde".into(),
            version: "2.0.0".into(),
            constraint: VersionConstraint::Exact,
        });
        let conflicts = resolver.find_conflicts();
        assert!(!conflicts.is_empty());
    }
}
