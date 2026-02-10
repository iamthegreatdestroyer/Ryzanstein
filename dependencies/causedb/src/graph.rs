//! Graph operations for causedb using petgraph.

use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use uuid::Uuid;

/// Directed acyclic graph for causal relationships
pub struct CausalGraph {
    graph: DiGraph<Uuid, f64>,
    node_map: HashMap<Uuid, NodeIndex>,
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    /// Add a node (event) to the graph
    pub fn add_node(&mut self, id: Uuid) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(&id) {
            return idx;
        }
        let idx = self.graph.add_node(id);
        self.node_map.insert(id, idx);
        idx
    }

    /// Add a causal edge with confidence weight
    pub fn add_edge(&mut self, cause: Uuid, effect: Uuid, confidence: f64) {
        let cause_idx = self.add_node(cause);
        let effect_idx = self.add_node(effect);
        self.graph.add_edge(cause_idx, effect_idx, confidence);
    }

    /// Count nodes
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Count edges
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if path exists between two events
    pub fn has_path(&self, from: Uuid, to: Uuid) -> bool {
        let from_idx = match self.node_map.get(&from) {
            Some(idx) => *idx,
            None => return false,
        };
        let to_idx = match self.node_map.get(&to) {
            Some(idx) => *idx,
            None => return false,
        };
        petgraph::algo::has_path_connecting(&self.graph, from_idx, to_idx, None)
    }

    /// Get all descendants (effects) of a node
    pub fn descendants(&self, id: Uuid) -> Vec<Uuid> {
        let idx = match self.node_map.get(&id) {
            Some(idx) => *idx,
            None => return vec![],
        };
        let mut result = Vec::new();
        let mut dfs = petgraph::visit::Dfs::new(&self.graph, idx);
        dfs.next(&self.graph); // skip self
        while let Some(node) = dfs.next(&self.graph) {
            result.push(self.graph[node]);
        }
        result
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_nodes_and_edges() {
        let mut g = CausalGraph::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        g.add_edge(a, b, 0.9);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_has_path() {
        let mut g = CausalGraph::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        g.add_edge(a, b, 0.9);
        g.add_edge(b, c, 0.8);
        assert!(g.has_path(a, c));
        assert!(!g.has_path(c, a));
    }

    #[test]
    fn test_descendants() {
        let mut g = CausalGraph::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        g.add_edge(a, b, 0.9);
        g.add_edge(b, c, 0.8);
        let desc = g.descendants(a);
        assert_eq!(desc.len(), 2);
    }

    #[test]
    fn test_no_path_disconnected() {
        let mut g = CausalGraph::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        g.add_node(a);
        g.add_node(b);
        assert!(!g.has_path(a, b));
    }
}
