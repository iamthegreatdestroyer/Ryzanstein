//! Dependency types and resolution using Bloom filters.

use crate::bloom::BloomFilter;
use crate::config::DepBloomConfig;
use crate::error::DepBloomError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VersionConstraint {
    /// `^version` — compatible updates
    Compatible,
    /// `=version` — exact match
    Exact,
    /// `>=version` — minimum version
    Minimum,
    /// `*` — any version
    Any,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub constraint: VersionConstraint,
}

/// Probabilistic dependency resolver backed by Bloom filters.
pub struct DepResolver {
    filter: BloomFilter,
    dependencies: Vec<Dependency>,
    config: DepBloomConfig,
}

impl DepResolver {
    pub fn new(config: DepBloomConfig) -> Self {
        DepResolver {
            filter: BloomFilter::new(config.expected_dependencies, config.false_positive_rate),
            dependencies: Vec::new(),
            config,
        }
    }

    /// Add a dependency — inserts into both the Vec and the Bloom filter.
    pub fn add_dependency(&mut self, dep: Dependency) {
        self.filter.insert(&dep.name);
        self.dependencies.push(dep);
    }

    /// Fast probabilistic check (O(1)).
    pub fn has_dependency(&self, name: &str) -> bool {
        self.filter.may_contain(name)
    }

    /// Find version conflicts (same dep, different exact versions).
    pub fn find_conflicts(&self) -> Vec<(Dependency, Dependency)> {
        let mut conflicts = Vec::new();
        let mut by_name: HashMap<&str, Vec<&Dependency>> = HashMap::new();

        for dep in &self.dependencies {
            by_name.entry(&dep.name).or_default().push(dep);
        }

        for (_name, deps) in &by_name {
            if deps.len() > 1 {
                for i in 0..deps.len() {
                    for j in (i + 1)..deps.len() {
                        if deps[i].constraint == VersionConstraint::Exact
                            && deps[j].constraint == VersionConstraint::Exact
                            && deps[i].version != deps[j].version
                        {
                            conflicts.push((deps[i].clone(), deps[j].clone()));
                        }
                    }
                }
            }
        }
        conflicts
    }

    pub fn count(&self) -> usize {
        self.dependencies.len()
    }
}

/// Directed graph for dependency ordering and cycle detection.
pub struct DependencyGraph {
    adjacency: HashMap<String, Vec<String>>,
    nodes: HashSet<String>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        DependencyGraph {
            adjacency: HashMap::new(),
            nodes: HashSet::new(),
        }
    }

    pub fn add_edge(&mut self, from: &str, to: &str) {
        self.nodes.insert(from.to_string());
        self.nodes.insert(to.to_string());
        self.adjacency
            .entry(from.to_string())
            .or_default()
            .push(to.to_string());
    }

    /// Check if the graph has a cycle using DFS.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        for node in &self.nodes {
            if self.dfs_cycle(node, &mut visited, &mut in_stack) {
                return true;
            }
        }
        false
    }

    /// Topological sort via Kahn's algorithm. Returns Err if cycle exists.
    pub fn topological_sort(&self) -> Result<Vec<String>, DepBloomError> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            in_degree.entry(node.as_str()).or_insert(0);
        }
        for (_from, tos) in &self.adjacency {
            for to in tos {
                *in_degree.entry(to.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&k, _)| k.to_string())
            .collect();

        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node.clone());
            if let Some(neighbors) = self.adjacency.get(&node) {
                for neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor.as_str()) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(DepBloomError::CyclicDependency);
        }
        Ok(order)
    }

    fn dfs_cycle(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        in_stack: &mut HashSet<String>,
    ) -> bool {
        if in_stack.contains(node) {
            return true;
        }
        if visited.contains(node) {
            return false;
        }
        visited.insert(node.to_string());
        in_stack.insert(node.to_string());

        if let Some(neighbors) = self.adjacency.get(node) {
            for neighbor in neighbors {
                if self.dfs_cycle(neighbor, visited, in_stack) {
                    return true;
                }
            }
        }
        in_stack.remove(node);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolver_count() {
        let mut r = DepResolver::new(DepBloomConfig::default());
        r.add_dependency(Dependency {
            name: "a".into(),
            version: "1.0".into(),
            constraint: VersionConstraint::Compatible,
        });
        assert_eq!(r.count(), 1);
    }

    #[test]
    fn test_graph_no_cycle() {
        let mut g = DependencyGraph::new();
        g.add_edge("A", "B");
        g.add_edge("B", "C");
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_graph_with_cycle() {
        let mut g = DependencyGraph::new();
        g.add_edge("A", "B");
        g.add_edge("B", "A");
        assert!(g.has_cycle());
    }

    #[test]
    fn test_toposort_cycle_errors() {
        let mut g = DependencyGraph::new();
        g.add_edge("X", "Y");
        g.add_edge("Y", "X");
        assert!(g.topological_sort().is_err());
    }
}
