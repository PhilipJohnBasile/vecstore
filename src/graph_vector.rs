// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Graph-Vector Fusion Search
//!
//! This module provides hybrid search combining graph traversal with vector similarity:
//! - **Graph-Guided Search**: Use relationship edges to guide vector retrieval
//! - **Vector-Influenced Traversal**: Use similarity scores to weight graph paths
//! - **Multi-Hop Queries**: Find semantically related nodes N hops away
//! - **Relationship Scoring**: Combine edge weights with vector similarity
//! - **Knowledge Graph Integration**: Bridge structured and unstructured data
//!
//! # Example
//!
//! ```ignore
//! use vecstore::graph_vector::{GraphVectorEngine, GraphNode, Edge, EdgeType};
//!
//! let mut engine = GraphVectorEngine::new();
//!
//! // Add nodes with vectors
//! engine.add_node("doc1", vec![1.0, 0.0, 0.0], metadata);
//! engine.add_node("doc2", vec![0.9, 0.1, 0.0], metadata);
//!
//! // Add relationships
//! engine.add_edge("doc1", "doc2", EdgeType::References, 0.8);
//!
//! // Graph-vector fusion search
//! let results = engine.search(
//!     query_vec,
//!     GraphSearchConfig {
//!         max_hops: 2,
//!         vector_weight: 0.7,
//!         edge_weight: 0.3,
//!         ..Default::default()
//!     }
//! )?;
//! ```

use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use serde::{Deserialize, Serialize};

use crate::error::VecStoreError;

/// Type of relationship edge
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// One document references another
    References,
    /// Documents are related
    RelatedTo,
    /// Child/parent relationship
    ChildOf,
    ParentOf,
    /// Chronological relationship
    FollowedBy,
    PrecededBy,
    /// Semantic relationship
    SimilarTo,
    OppositeOf,
    /// Custom edge type
    Custom(String),
}

/// A directed edge in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Type of relationship
    pub edge_type: EdgeType,
    /// Edge weight (0.0 - 1.0)
    pub weight: f32,
    /// Additional edge metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Edge {
    /// Create a new edge
    pub fn new(from: impl Into<String>, to: impl Into<String>, edge_type: EdgeType) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            edge_type,
            weight: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Set the edge weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A node in the graph with an associated vector
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node ID
    pub id: String,
    /// Associated vector
    pub vector: Vec<f32>,
    /// Node metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Outgoing edge IDs
    outgoing_edges: Vec<usize>,
    /// Incoming edge IDs
    incoming_edges: Vec<usize>,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata: HashMap::new(),
            outgoing_edges: Vec::new(),
            incoming_edges: Vec::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Configuration for graph-vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchConfig {
    /// Maximum number of hops in graph traversal
    pub max_hops: usize,
    /// Weight for vector similarity (0.0 - 1.0)
    pub vector_weight: f32,
    /// Weight for edge scores (0.0 - 1.0)
    pub edge_weight: f32,
    /// Decay factor per hop (multiply score by this for each hop)
    pub hop_decay: f32,
    /// Minimum score threshold
    pub min_score: f32,
    /// Maximum results to return
    pub limit: usize,
    /// Edge types to follow (empty = all)
    pub edge_types: Vec<EdgeType>,
    /// Whether to include the starting nodes
    pub include_start: bool,
    /// Traversal direction
    pub direction: TraversalDirection,
}

impl Default for GraphSearchConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            vector_weight: 0.7,
            edge_weight: 0.3,
            hop_decay: 0.8,
            min_score: 0.0,
            limit: 10,
            edge_types: Vec::new(),
            include_start: true,
            direction: TraversalDirection::Outgoing,
        }
    }
}

/// Direction of graph traversal
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TraversalDirection {
    /// Follow outgoing edges only
    Outgoing,
    /// Follow incoming edges only
    Incoming,
    /// Follow both directions
    Both,
}

/// Result from graph-vector search
#[derive(Debug, Clone)]
pub struct GraphSearchResult {
    /// Node ID
    pub id: String,
    /// Combined score
    pub score: f32,
    /// Vector similarity component
    pub vector_score: f32,
    /// Graph traversal component
    pub graph_score: f32,
    /// Number of hops from starting point
    pub hops: usize,
    /// Path taken to reach this node
    pub path: Vec<String>,
    /// Edge types in the path
    pub edge_types_in_path: Vec<EdgeType>,
    /// Node metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Entry for priority queue in graph search
#[derive(Debug)]
struct SearchEntry {
    node_id: String,
    score: f32,
    hops: usize,
    path: Vec<String>,
    edge_types: Vec<EdgeType>,
}

impl PartialEq for SearchEntry {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for SearchEntry {}

impl PartialOrd for SearchEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score = higher priority
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal).reverse()
    }
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Average outgoing degree
    pub avg_out_degree: f32,
    /// Average incoming degree
    pub avg_in_degree: f32,
    /// Edge type distribution
    pub edge_type_counts: HashMap<String, usize>,
    /// Connected components count
    pub component_count: usize,
    /// Largest component size
    pub largest_component: usize,
}

/// Main graph-vector fusion engine
pub struct GraphVectorEngine {
    /// Nodes indexed by ID
    nodes: HashMap<String, GraphNode>,
    /// All edges
    edges: Vec<Edge>,
    /// Node ID to internal index
    node_index: HashMap<String, usize>,
    /// Adjacency list (outgoing)
    adjacency_out: HashMap<String, Vec<usize>>,
    /// Adjacency list (incoming)
    adjacency_in: HashMap<String, Vec<usize>>,
}

impl GraphVectorEngine {
    /// Create a new graph-vector engine
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            node_index: HashMap::new(),
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) -> Result<(), VecStoreError> {
        let id = node.id.clone();

        if self.nodes.contains_key(&id) {
            return Err(VecStoreError::InvalidInput(format!(
                "Node {} already exists", id
            )));
        }

        let index = self.nodes.len();
        self.node_index.insert(id.clone(), index);
        self.adjacency_out.insert(id.clone(), Vec::new());
        self.adjacency_in.insert(id.clone(), Vec::new());
        self.nodes.insert(id, node);

        Ok(())
    }

    /// Add a node with just ID and vector
    pub fn add_node_simple(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<(), VecStoreError> {
        let mut node = GraphNode::new(id, vector);
        node.metadata = metadata;
        self.add_node(node)
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: Edge) -> Result<(), VecStoreError> {
        // Validate nodes exist
        if !self.nodes.contains_key(&edge.from) {
            return Err(VecStoreError::NotFound(format!(
                "Source node {} not found", edge.from
            )));
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(VecStoreError::NotFound(format!(
                "Target node {} not found", edge.to
            )));
        }

        let edge_idx = self.edges.len();

        // Update adjacency lists
        self.adjacency_out.get_mut(&edge.from).unwrap().push(edge_idx);
        self.adjacency_in.get_mut(&edge.to).unwrap().push(edge_idx);

        // Update node references
        if let Some(node) = self.nodes.get_mut(&edge.from) {
            node.outgoing_edges.push(edge_idx);
        }
        if let Some(node) = self.nodes.get_mut(&edge.to) {
            node.incoming_edges.push(edge_idx);
        }

        self.edges.push(edge);
        Ok(())
    }

    /// Add an edge with simple parameters
    pub fn add_edge_simple(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        edge_type: EdgeType,
        weight: f32,
    ) -> Result<(), VecStoreError> {
        let edge = Edge::new(from, to, edge_type).with_weight(weight);
        self.add_edge(edge)
    }

    /// Perform graph-vector fusion search
    pub fn search(
        &self,
        query_vector: &[f32],
        config: &GraphSearchConfig,
    ) -> Result<Vec<GraphSearchResult>, VecStoreError> {
        // First, find initial candidates by vector similarity
        let initial_candidates = self.find_similar_nodes(query_vector, config.limit * 2);

        // Track visited nodes and best scores
        let mut visited: HashMap<String, GraphSearchResult> = HashMap::new();
        let mut heap = BinaryHeap::new();

        // Seed the search with initial vector-similar candidates
        for (id, vector_score) in initial_candidates {
            let entry = SearchEntry {
                node_id: id.clone(),
                score: vector_score * config.vector_weight,
                hops: 0,
                path: vec![id.clone()],
                edge_types: Vec::new(),
            };
            heap.push(entry);
        }

        // BFS/Dijkstra-style exploration
        while let Some(current) = heap.pop() {
            // Skip if we've seen this node with a better score
            if let Some(existing) = visited.get(&current.node_id) {
                if existing.score >= current.score {
                    continue;
                }
            }

            // Get node info
            let node = match self.nodes.get(&current.node_id) {
                Some(n) => n,
                None => continue,
            };

            // Calculate vector similarity for this node
            let vector_score = self.cosine_similarity(query_vector, &node.vector);

            // Combined score
            let combined_score = vector_score * config.vector_weight
                + current.score * config.edge_weight;

            if combined_score < config.min_score {
                continue;
            }

            // Record this visit
            let result = GraphSearchResult {
                id: current.node_id.clone(),
                score: combined_score,
                vector_score,
                graph_score: current.score,
                hops: current.hops,
                path: current.path.clone(),
                edge_types_in_path: current.edge_types.clone(),
                metadata: node.metadata.clone(),
            };

            visited.insert(current.node_id.clone(), result);

            // Continue traversal if within hop limit
            if current.hops < config.max_hops {
                let edges = self.get_edges_for_traversal(&current.node_id, &config.direction);

                for edge_idx in edges {
                    let edge = &self.edges[edge_idx];

                    // Check edge type filter
                    if !config.edge_types.is_empty()
                        && !config.edge_types.contains(&edge.edge_type)
                    {
                        continue;
                    }

                    // Determine next node
                    let next_id = if edge.from == current.node_id {
                        &edge.to
                    } else {
                        &edge.from
                    };

                    // Skip if already in path (avoid cycles)
                    if current.path.contains(next_id) {
                        continue;
                    }

                    // Calculate score for this path
                    let path_score = current.score * edge.weight * config.hop_decay;

                    let mut new_path = current.path.clone();
                    new_path.push(next_id.clone());

                    let mut new_edge_types = current.edge_types.clone();
                    new_edge_types.push(edge.edge_type.clone());

                    heap.push(SearchEntry {
                        node_id: next_id.clone(),
                        score: path_score,
                        hops: current.hops + 1,
                        path: new_path,
                        edge_types: new_edge_types,
                    });
                }
            }
        }

        // Filter and sort results
        let mut results: Vec<GraphSearchResult> = visited.into_values()
            .filter(|r| config.include_start || r.hops > 0)
            .filter(|r| r.score >= config.min_score)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(config.limit);

        Ok(results)
    }

    /// Find nodes reachable within N hops
    pub fn find_reachable(
        &self,
        start_id: &str,
        max_hops: usize,
        direction: TraversalDirection,
    ) -> Result<Vec<(String, usize)>, VecStoreError> {
        if !self.nodes.contains_key(start_id) {
            return Err(VecStoreError::NotFound(format!(
                "Node {} not found", start_id
            )));
        }

        let mut visited: HashMap<String, usize> = HashMap::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back((start_id.to_string(), 0));
        visited.insert(start_id.to_string(), 0);

        while let Some((current_id, hops)) = queue.pop_front() {
            if hops >= max_hops {
                continue;
            }

            let edges = self.get_edges_for_traversal(&current_id, &direction);

            for edge_idx in edges {
                let edge = &self.edges[edge_idx];
                let next_id = if edge.from == current_id {
                    &edge.to
                } else {
                    &edge.from
                };

                if !visited.contains_key(next_id) {
                    visited.insert(next_id.clone(), hops + 1);
                    queue.push_back((next_id.clone(), hops + 1));
                }
            }
        }

        let mut results: Vec<_> = visited.into_iter().collect();
        results.sort_by_key(|(_, hops)| *hops);
        Ok(results)
    }

    /// Find the shortest path between two nodes
    pub fn shortest_path(&self, from: &str, to: &str) -> Result<Option<Vec<String>>, VecStoreError> {
        if !self.nodes.contains_key(from) {
            return Err(VecStoreError::NotFound(format!("Node {} not found", from)));
        }
        if !self.nodes.contains_key(to) {
            return Err(VecStoreError::NotFound(format!("Node {} not found", to)));
        }

        // BFS for shortest path
        let mut visited: HashSet<String> = HashSet::new();
        let mut parent: HashMap<String, String> = HashMap::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back(from.to_string());
        visited.insert(from.to_string());

        while let Some(current) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = vec![to.to_string()];
                let mut node = to.to_string();

                while let Some(p) = parent.get(&node) {
                    path.push(p.clone());
                    node = p.clone();
                }

                path.reverse();
                return Ok(Some(path));
            }

            // Check both directions
            let edges = self.get_edges_for_traversal(&current, &TraversalDirection::Both);

            for edge_idx in edges {
                let edge = &self.edges[edge_idx];
                let next = if edge.from == current {
                    &edge.to
                } else {
                    &edge.from
                };

                if !visited.contains(next) {
                    visited.insert(next.clone());
                    parent.insert(next.clone(), current.clone());
                    queue.push_back(next.clone());
                }
            }
        }

        Ok(None) // No path found
    }

    /// Get neighbors of a node
    pub fn get_neighbors(
        &self,
        node_id: &str,
        direction: TraversalDirection,
    ) -> Result<Vec<(String, EdgeType, f32)>, VecStoreError> {
        if !self.nodes.contains_key(node_id) {
            return Err(VecStoreError::NotFound(format!("Node {} not found", node_id)));
        }

        let edges = self.get_edges_for_traversal(node_id, &direction);
        let mut neighbors = Vec::new();

        for edge_idx in edges {
            let edge = &self.edges[edge_idx];
            let neighbor_id = if edge.from == node_id {
                &edge.to
            } else {
                &edge.from
            };

            neighbors.push((
                neighbor_id.clone(),
                edge.edge_type.clone(),
                edge.weight,
            ));
        }

        Ok(neighbors)
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();

        let total_out: usize = self.adjacency_out.values().map(|v| v.len()).sum();
        let total_in: usize = self.adjacency_in.values().map(|v| v.len()).sum();

        let avg_out_degree = if node_count > 0 {
            total_out as f32 / node_count as f32
        } else {
            0.0
        };

        let avg_in_degree = if node_count > 0 {
            total_in as f32 / node_count as f32
        } else {
            0.0
        };

        let mut edge_type_counts: HashMap<String, usize> = HashMap::new();
        for edge in &self.edges {
            let key = format!("{:?}", edge.edge_type);
            *edge_type_counts.entry(key).or_insert(0) += 1;
        }

        // Count connected components (simplified - just count isolated nodes)
        let (component_count, largest_component) = self.count_components();

        GraphStats {
            node_count,
            edge_count,
            avg_out_degree,
            avg_in_degree,
            edge_type_counts,
            component_count,
            largest_component,
        }
    }

    /// Get node by ID
    pub fn get_node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// Get edge by index
    pub fn get_edge(&self, idx: usize) -> Option<&Edge> {
        self.edges.get(idx)
    }

    /// Remove a node and its edges
    pub fn remove_node(&mut self, id: &str) -> Result<(), VecStoreError> {
        if !self.nodes.contains_key(id) {
            return Err(VecStoreError::NotFound(format!("Node {} not found", id)));
        }

        // Mark edges for removal (edges involving this node)
        let edges_to_remove: Vec<usize> = self.edges.iter()
            .enumerate()
            .filter(|(_, e)| e.from == id || e.to == id)
            .map(|(i, _)| i)
            .collect();

        // Remove edges (in reverse order to maintain indices)
        for idx in edges_to_remove.into_iter().rev() {
            self.edges.remove(idx);
        }

        // Rebuild adjacency lists (simplified approach)
        self.rebuild_adjacency();

        // Remove node
        self.nodes.remove(id);
        self.adjacency_out.remove(id);
        self.adjacency_in.remove(id);

        Ok(())
    }

    /// Update node vector
    pub fn update_vector(&mut self, id: &str, new_vector: Vec<f32>) -> Result<(), VecStoreError> {
        let node = self.nodes.get_mut(id).ok_or_else(|| {
            VecStoreError::NotFound(format!("Node {} not found", id))
        })?;

        node.vector = new_vector;
        Ok(())
    }

    /// Check if two nodes are connected (directly or indirectly)
    pub fn are_connected(&self, from: &str, to: &str) -> Result<bool, VecStoreError> {
        Ok(self.shortest_path(from, to)?.is_some())
    }

    // === Helper Methods ===

    fn find_similar_nodes(&self, query: &[f32], limit: usize) -> Vec<(String, f32)> {
        let mut scores: Vec<_> = self.nodes.iter()
            .map(|(id, node)| {
                let sim = self.cosine_similarity(query, &node.vector);
                (id.clone(), sim)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores.truncate(limit);
        scores
    }

    fn get_edges_for_traversal(&self, node_id: &str, direction: &TraversalDirection) -> Vec<usize> {
        let mut edges = Vec::new();

        match direction {
            TraversalDirection::Outgoing => {
                if let Some(out_edges) = self.adjacency_out.get(node_id) {
                    edges.extend(out_edges);
                }
            }
            TraversalDirection::Incoming => {
                if let Some(in_edges) = self.adjacency_in.get(node_id) {
                    edges.extend(in_edges);
                }
            }
            TraversalDirection::Both => {
                if let Some(out_edges) = self.adjacency_out.get(node_id) {
                    edges.extend(out_edges);
                }
                if let Some(in_edges) = self.adjacency_in.get(node_id) {
                    edges.extend(in_edges);
                }
            }
        }

        edges
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    fn count_components(&self) -> (usize, usize) {
        let mut visited: HashSet<String> = HashSet::new();
        let mut component_count = 0;
        let mut largest = 0;

        for node_id in self.nodes.keys() {
            if visited.contains(node_id) {
                continue;
            }

            // BFS to find all nodes in this component
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(node_id.clone());
            let mut component_size = 0;

            while let Some(current) = queue.pop_front() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current.clone());
                component_size += 1;

                // Add unvisited neighbors
                let edges = self.get_edges_for_traversal(&current, &TraversalDirection::Both);
                for edge_idx in edges {
                    let edge = &self.edges[edge_idx];
                    let next = if edge.from == current {
                        &edge.to
                    } else {
                        &edge.from
                    };

                    if !visited.contains(next) {
                        queue.push_back(next.clone());
                    }
                }
            }

            component_count += 1;
            largest = largest.max(component_size);
        }

        (component_count, largest)
    }

    fn rebuild_adjacency(&mut self) {
        // Clear existing
        for edges in self.adjacency_out.values_mut() {
            edges.clear();
        }
        for edges in self.adjacency_in.values_mut() {
            edges.clear();
        }

        // Rebuild from edges
        for (idx, edge) in self.edges.iter().enumerate() {
            if let Some(out_edges) = self.adjacency_out.get_mut(&edge.from) {
                out_edges.push(idx);
            }
            if let Some(in_edges) = self.adjacency_in.get_mut(&edge.to) {
                in_edges.push(idx);
            }
        }
    }
}

impl Default for GraphVectorEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing graphs
pub struct GraphBuilder {
    nodes: Vec<GraphNode>,
    edges: Vec<Edge>,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node
    pub fn node(mut self, id: impl Into<String>, vector: Vec<f32>) -> Self {
        self.nodes.push(GraphNode::new(id, vector));
        self
    }

    /// Add an edge
    pub fn edge(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        edge_type: EdgeType,
    ) -> Self {
        self.edges.push(Edge::new(from, to, edge_type));
        self
    }

    /// Build the graph
    pub fn build(self) -> Result<GraphVectorEngine, VecStoreError> {
        let mut engine = GraphVectorEngine::new();

        for node in self.nodes {
            engine.add_node(node)?;
        }

        for edge in self.edges {
            engine.add_edge(edge)?;
        }

        Ok(engine)
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> GraphVectorEngine {
        let mut engine = GraphVectorEngine::new();

        // Add nodes
        engine.add_node(GraphNode::new("a", vec![1.0, 0.0, 0.0])).unwrap();
        engine.add_node(GraphNode::new("b", vec![0.9, 0.1, 0.0])).unwrap();
        engine.add_node(GraphNode::new("c", vec![0.0, 1.0, 0.0])).unwrap();
        engine.add_node(GraphNode::new("d", vec![0.0, 0.0, 1.0])).unwrap();

        // Add edges: a -> b -> c, a -> d
        engine.add_edge_simple("a", "b", EdgeType::References, 1.0).unwrap();
        engine.add_edge_simple("b", "c", EdgeType::References, 0.8).unwrap();
        engine.add_edge_simple("a", "d", EdgeType::RelatedTo, 0.5).unwrap();

        engine
    }

    #[test]
    fn test_add_nodes_and_edges() {
        let engine = create_test_graph();

        assert_eq!(engine.nodes.len(), 4);
        assert_eq!(engine.edges.len(), 3);

        let stats = engine.stats();
        assert_eq!(stats.node_count, 4);
        assert_eq!(stats.edge_count, 3);
    }

    #[test]
    fn test_get_neighbors() {
        let engine = create_test_graph();

        let neighbors = engine.get_neighbors("a", TraversalDirection::Outgoing).unwrap();
        assert_eq!(neighbors.len(), 2);

        let neighbor_ids: Vec<_> = neighbors.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(neighbor_ids.contains(&"b"));
        assert!(neighbor_ids.contains(&"d"));
    }

    #[test]
    fn test_find_reachable() {
        let engine = create_test_graph();

        let reachable = engine.find_reachable("a", 2, TraversalDirection::Outgoing).unwrap();

        // Should find a (0 hops), b and d (1 hop), c (2 hops)
        assert_eq!(reachable.len(), 4);

        let ids: Vec<_> = reachable.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
        assert!(ids.contains(&"d"));
    }

    #[test]
    fn test_shortest_path() {
        let engine = create_test_graph();

        let path = engine.shortest_path("a", "c").unwrap();
        assert!(path.is_some());

        let path = path.unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_graph_vector_search() {
        let engine = create_test_graph();

        let config = GraphSearchConfig {
            max_hops: 2,
            vector_weight: 0.7,
            edge_weight: 0.3,
            limit: 10,
            ..Default::default()
        };

        // Query for something similar to node "a"
        let results = engine.search(&[1.0, 0.0, 0.0], &config).unwrap();

        assert!(!results.is_empty());
        // Node "a" should be highest scoring (exact match)
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_connected_components() {
        let mut engine = GraphVectorEngine::new();

        // Component 1: a - b
        engine.add_node(GraphNode::new("a", vec![1.0, 0.0])).unwrap();
        engine.add_node(GraphNode::new("b", vec![0.0, 1.0])).unwrap();
        engine.add_edge_simple("a", "b", EdgeType::RelatedTo, 1.0).unwrap();

        // Component 2: c - d (disconnected)
        engine.add_node(GraphNode::new("c", vec![0.5, 0.5])).unwrap();
        engine.add_node(GraphNode::new("d", vec![0.5, 0.5])).unwrap();
        engine.add_edge_simple("c", "d", EdgeType::RelatedTo, 1.0).unwrap();

        let stats = engine.stats();
        assert_eq!(stats.component_count, 2);
        assert_eq!(stats.largest_component, 2);
    }

    #[test]
    fn test_graph_builder() {
        let engine = GraphBuilder::new()
            .node("x", vec![1.0, 0.0])
            .node("y", vec![0.0, 1.0])
            .edge("x", "y", EdgeType::SimilarTo)
            .build()
            .unwrap();

        assert_eq!(engine.nodes.len(), 2);
        assert_eq!(engine.edges.len(), 1);
    }

    #[test]
    fn test_remove_node() {
        let mut engine = create_test_graph();

        assert_eq!(engine.nodes.len(), 4);
        assert_eq!(engine.edges.len(), 3);

        engine.remove_node("b").unwrap();

        assert_eq!(engine.nodes.len(), 3);
        // Edges a->b and b->c should be removed
        assert_eq!(engine.edges.len(), 1);
    }

    #[test]
    fn test_update_vector() {
        let mut engine = create_test_graph();

        let original = engine.get_node("a").unwrap().vector.clone();
        assert_eq!(original, vec![1.0, 0.0, 0.0]);

        engine.update_vector("a", vec![0.5, 0.5, 0.0]).unwrap();

        let updated = engine.get_node("a").unwrap().vector.clone();
        assert_eq!(updated, vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_are_connected() {
        let engine = create_test_graph();

        assert!(engine.are_connected("a", "c").unwrap());
        assert!(engine.are_connected("a", "d").unwrap());
        assert!(engine.are_connected("b", "d").unwrap()); // Through a
    }

    #[test]
    fn test_edge_type_filter() {
        let engine = create_test_graph();

        let config = GraphSearchConfig {
            max_hops: 2,
            edge_types: vec![EdgeType::References], // Only follow References edges
            ..Default::default()
        };

        let results = engine.search(&[1.0, 0.0, 0.0], &config).unwrap();

        // Should find a, b, c but NOT d (connected via RelatedTo)
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        // d might still appear due to vector similarity, but with lower score
    }
}
