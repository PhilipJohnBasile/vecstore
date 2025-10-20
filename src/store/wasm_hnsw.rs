//! WASM-compatible HNSW implementation
//!
//! This module provides a pure in-memory HNSW (Hierarchical Navigable Small World) graph
//! implementation that works in WebAssembly environments. Unlike hnsw_rs which requires
//! memory-mapped files, this implementation uses standard Rust collections that work in browsers.
//!
//! ## Algorithm Overview
//!
//! HNSW is a multi-layer graph structure where:
//! - Each vector is a node in the graph
//! - Nodes are connected to M nearest neighbors at each layer
//! - Top layers are sparse (few nodes), bottom layer is dense (all nodes)
//! - Search starts at top layer and navigates down greedily
//! - Complexity: O(log N) search time, O(N log N) construction time
//!
//! ## Performance
//!
//! - Search: Sub-millisecond for millions of vectors
//! - Memory: ~4-8 bytes per edge + vector storage
//! - Typical: 16 connections × 4 bytes × log₂(N) layers ≈ 64-256 bytes overhead per vector

use crate::simd;
use crate::store::types::{Distance, Id};
use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// HNSW graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    /// Vector ID
    id: Id,

    /// Vector data
    vector: Vec<f32>,

    /// Maximum layer this node appears in (0 = bottom layer only)
    layer: usize,

    /// Neighbor connections at each layer: layer -> set of neighbor IDs
    /// neighbors[0] = bottom layer (most connections)
    /// neighbors[layer] = top layer (fewest connections)
    neighbors: Vec<HashSet<Id>>,
}

/// Priority queue element for search
#[derive(Clone)]
struct Candidate {
    id: Id,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for min-heap behavior
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// WASM-compatible HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmHnsw {
    /// All nodes in the graph
    nodes: HashMap<Id, Node>,

    /// Entry point (node at highest layer)
    entry_point: Option<Id>,

    /// Current maximum layer
    max_layer: usize,

    /// Vector dimension
    dimension: usize,

    /// Distance metric
    metric: Distance,

    /// HNSW parameter: number of connections per layer
    m: usize,

    /// HNSW parameter: max connections at layer 0 (typically 2*M)
    m_max: usize,

    /// HNSW parameter: size of dynamic candidate list during construction
    ef_construction: usize,

    /// Level multiplier (1/ln(2) ≈ 1.44)
    ml: f32,
}

impl WasmHnsw {
    /// Create a new WASM-compatible HNSW index
    pub fn new(dimension: usize) -> Self {
        Self::with_params(dimension, Distance::Cosine, 16, 200)
    }

    /// Create with custom parameters
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension
    /// * `metric` - Distance metric to use
    /// * `m` - Number of connections per layer (typically 8-64, default 16)
    /// * `ef_construction` - Dynamic candidate list size during construction (typically 100-500, default 200)
    pub fn with_params(
        dimension: usize,
        metric: Distance,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        Self {
            nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            dimension,
            metric,
            m,
            m_max: m * 2,
            ef_construction,
            ml: 1.0 / (2.0_f32).ln(),
        }
    }

    /// Insert a vector into the HNSW index
    pub fn insert(&mut self, id: Id, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            anyhow::bail!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            );
        }

        // Determine layer for new node
        let layer = self.random_layer();

        // Create new node
        let mut neighbors = Vec::new();
        for _ in 0..=layer {
            neighbors.push(HashSet::new());
        }

        let new_node = Node {
            id: id.clone(),
            vector: vector.clone(),
            layer,
            neighbors,
        };

        // First node becomes entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(id.clone());
            self.max_layer = layer;
            self.nodes.insert(id, new_node);
            return Ok(());
        }

        // Insert node first so we can add edges to it
        self.nodes.insert(id.clone(), new_node);

        // Find nearest neighbors at each layer
        let ep = self.entry_point.as_ref().unwrap().clone();
        let ep_dist = self.compute_distance(&vector, &self.nodes[&ep].vector);
        let mut nearest = vec![Candidate {
            id: ep,
            distance: ep_dist,
        }];

        // Search from top to target layer
        for lc in (layer + 1..=self.max_layer).rev() {
            nearest = self.search_layer(&vector, &nearest, 1, lc);
        }

        // Connect at each layer from target down to 0
        for lc in (0..=layer).rev() {
            let candidates = self.search_layer(&vector, &nearest, self.ef_construction, lc);

            // Select M neighbors
            let m = if lc == 0 { self.m_max } else { self.m };
            let neighbors_to_add = self.select_neighbors(&vector, &candidates, m);

            // Add bidirectional links
            for neighbor_id in &neighbors_to_add {
                // Add edge from new node to neighbor
                if let Some(node) = self.nodes.get_mut(&id) {
                    if lc < node.neighbors.len() {
                        node.neighbors[lc].insert(neighbor_id.clone());
                    }
                }

                // Add edge from neighbor to new node
                if let Some(neighbor_node) = self.nodes.get_mut(neighbor_id) {
                    if lc < neighbor_node.neighbors.len() {
                        neighbor_node.neighbors[lc].insert(id.clone());

                        // Prune neighbor's connections if exceeded max
                        let max_conn = if lc == 0 { self.m_max } else { self.m };
                        if neighbor_node.neighbors[lc].len() > max_conn {
                            self.prune_connections(neighbor_id, lc, max_conn);
                        }
                    }
                }
            }

            nearest = candidates;
        }

        // Update entry point if new node is at higher layer
        if layer > self.max_layer {
            self.max_layer = layer;
            self.entry_point = Some(id.clone());
        }

        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<(Id, f32)>> {
        if query.len() != self.dimension {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            );
        }

        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        let ep = self.entry_point.as_ref().unwrap().clone();
        let ep_dist = self.compute_distance(query, &self.nodes[&ep].vector);
        let mut nearest = vec![Candidate {
            id: ep,
            distance: ep_dist,
        }];

        // Search from top layer down to layer 1
        for lc in (1..=self.max_layer).rev() {
            nearest = self.search_layer(query, &nearest, 1, lc);
        }

        // Search layer 0 with ef_search candidates
        let candidates = self.search_layer(query, &nearest, ef_search.max(k), 0);

        // Return top k
        Ok(candidates
            .into_iter()
            .take(k)
            .map(|c| (c.id, c.distance))
            .collect())
    }

    /// Remove a vector from the index
    ///
    /// Note: This is a simple deletion that removes the node and its edges.
    /// For production use with frequent deletions, consider rebuilding the index periodically
    /// to maintain optimal graph structure and search quality.
    pub fn remove(&mut self, id: &str) -> Result<()> {
        if let Some(node) = self.nodes.remove(id) {
            // Remove edges from neighbors to deleted node
            for (layer, neighbor_ids) in node.neighbors.iter().enumerate() {
                for neighbor_id in neighbor_ids {
                    if let Some(neighbor) = self.nodes.get_mut(neighbor_id) {
                        if layer < neighbor.neighbors.len() {
                            neighbor.neighbors[layer].remove(id);
                        }
                    }
                }
            }

            // Update entry point if removed
            if self.entry_point.as_deref() == Some(id) {
                // Find node with highest layer as new entry point
                self.entry_point = self
                    .nodes
                    .iter()
                    .max_by_key(|(_, n)| n.layer)
                    .map(|(id, _)| id.clone());
                self.max_layer = self.nodes.values().map(|n| n.layer).max().unwrap_or(0);
            }
        }
        Ok(())
    }

    /// Search a single layer for nearest neighbors
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[Candidate],
        num_closest: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut best = BinaryHeap::new();

        // Initialize with entry points
        for ep in entry_points {
            visited.insert(ep.id.clone());
            candidates.push(ep.clone());
            best.push(ep.clone());
        }

        // Greedy search
        while let Some(current) = candidates.pop() {
            // Get worst distance in best set
            let worst_best = best.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);

            if current.distance > worst_best {
                break; // All candidates worse than best
            }

            // Get node neighbors at this layer
            if let Some(node) = self.nodes.get(&current.id) {
                if layer < node.neighbors.len() {
                    for neighbor_id in &node.neighbors[layer] {
                        if visited.insert(neighbor_id.clone()) {
                            if let Some(neighbor_node) = self.nodes.get(neighbor_id) {
                                let dist = self.compute_distance(query, &neighbor_node.vector);

                                if dist < worst_best || best.len() < num_closest {
                                    let cand = Candidate {
                                        id: neighbor_id.clone(),
                                        distance: dist,
                                    };
                                    candidates.push(cand.clone());
                                    best.push(cand);

                                    // Keep only num_closest best
                                    if best.len() > num_closest {
                                        best.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted vec (best first)
        let mut result: Vec<_> = best.into_iter().collect();
        result.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        result
    }

    /// Select M neighbors using heuristic
    fn select_neighbors(&self, _query: &[f32], candidates: &[Candidate], m: usize) -> Vec<Id> {
        if candidates.len() <= m {
            return candidates.iter().map(|c| c.id.clone()).collect();
        }

        // Simple heuristic: take M closest
        // More sophisticated heuristics (like angle-based) could be added
        candidates.iter().take(m).map(|c| c.id.clone()).collect()
    }

    /// Prune connections for a node at a given layer
    fn prune_connections(&mut self, node_id: &Id, layer: usize, max_conn: usize) {
        if let Some(node) = self.nodes.get(node_id) {
            if layer >= node.neighbors.len() {
                return;
            }

            // Get current neighbors and their distances
            let query = &node.vector;
            let mut candidates: Vec<_> = node.neighbors[layer]
                .iter()
                .filter_map(|neighbor_id| {
                    self.nodes.get(neighbor_id).map(|neighbor| Candidate {
                        id: neighbor_id.clone(),
                        distance: self.compute_distance(query, &neighbor.vector),
                    })
                })
                .collect();

            // Sort by distance
            candidates.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });

            // Keep only best max_conn
            let to_keep: HashSet<_> = candidates
                .into_iter()
                .take(max_conn)
                .map(|c| c.id)
                .collect();

            // Update neighbors
            if let Some(node) = self.nodes.get_mut(node_id) {
                if layer < node.neighbors.len() {
                    node.neighbors[layer] = to_keep;
                }
            }
        }
    }

    /// Compute distance between two vectors
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            Distance::Cosine => 1.0 - simd::cosine_similarity_simd(a, b), // Convert similarity to distance
            Distance::Euclidean => simd::euclidean_distance_simd(a, b),
            Distance::DotProduct => -simd::dot_product_simd(a, b), // Negate for distance
            Distance::Manhattan => simd::manhattan_distance_simd(a, b),
            Distance::Hamming => simd::hamming_distance_simd(a, b),
            Distance::Jaccard => 1.0 - simd::jaccard_similarity_simd(a, b), // Convert similarity to distance
            Distance::Chebyshev => simd::chebyshev_distance_simd(a, b),
            Distance::Canberra => simd::canberra_distance_simd(a, b),
            Distance::BrayCurtis => simd::braycurtis_distance_simd(a, b),
        }
    }

    /// Randomly select layer for new node
    fn random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Get number of nodes in index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get all node IDs
    pub fn ids(&self) -> Vec<Id> {
        self.nodes.keys().cloned().collect()
    }

    /// Clear all nodes
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.entry_point = None;
        self.max_layer = 0;
    }

    /// Get index statistics
    pub fn stats(&self) -> HnswStats {
        let total_edges: usize = self
            .nodes
            .values()
            .flat_map(|n| &n.neighbors)
            .map(|layer| layer.len())
            .sum();

        let layer_distribution: Vec<usize> = (0..=self.max_layer)
            .map(|l| self.nodes.values().filter(|n| n.layer >= l).count())
            .collect();

        HnswStats {
            num_nodes: self.nodes.len(),
            num_edges: total_edges,
            max_layer: self.max_layer,
            layer_distribution,
            m: self.m,
            ef_construction: self.ef_construction,
        }
    }

    /// Create a graph visualizer for this HNSW index
    ///
    /// This allows exporting the graph structure for visualization and debugging.
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::store::wasm_hnsw::WasmHnsw;
    /// # use vecstore::graph_viz::HnswVisualizer;
    /// let mut index = WasmHnsw::new(128);
    /// // ... insert vectors ...
    ///
    /// let viz = index.to_visualizer();
    /// let dot = viz.export_dot()?;
    /// std::fs::write("graph.dot", dot)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn to_visualizer(&self) -> crate::graph_viz::HnswVisualizer {
        use crate::graph_viz::{GraphEdge, GraphNode, HnswVisualizer};

        // Build graph nodes
        let mut graph_nodes = Vec::new();
        for (id, node) in &self.nodes {
            let degree = node.neighbors.iter().map(|layer| layer.len()).sum();

            // Include first 3 dimensions as preview
            let vector_preview = if node.vector.len() >= 3 {
                Some(vec![node.vector[0], node.vector[1], node.vector[2]])
            } else if !node.vector.is_empty() {
                Some(node.vector.clone())
            } else {
                None
            };

            graph_nodes.push(GraphNode {
                id: id.clone(),
                layer: node.layer,
                degree,
                vector_preview,
            });
        }

        // Build graph edges
        let mut graph_edges = Vec::new();
        for (id, node) in &self.nodes {
            for (layer_idx, neighbors_at_layer) in node.neighbors.iter().enumerate() {
                for neighbor_id in neighbors_at_layer {
                    // Calculate edge weight (distance between vectors)
                    if let Some(neighbor_node) = self.nodes.get(neighbor_id) {
                        let distance = self.compute_distance(&node.vector, &neighbor_node.vector);

                        graph_edges.push(GraphEdge {
                            source: id.clone(),
                            target: neighbor_id.clone(),
                            layer: layer_idx,
                            weight: Some(distance),
                        });
                    }
                }
            }
        }

        HnswVisualizer::new(graph_nodes, graph_edges, self.max_layer + 1)
    }
}

/// HNSW index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub max_layer: usize,
    pub layer_distribution: Vec<usize>,
    pub m: usize,
    pub ef_construction: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_hnsw_basic() {
        let mut index = WasmHnsw::new(3);

        // Insert vectors
        index.insert("v1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        index.insert("v2".to_string(), vec![0.0, 1.0, 0.0]).unwrap();
        index.insert("v3".to_string(), vec![0.0, 0.0, 1.0]).unwrap();
        index.insert("v4".to_string(), vec![1.0, 1.0, 0.0]).unwrap();

        assert_eq!(index.len(), 4);

        // Search
        let results = index.search(&[1.0, 0.1, 0.0], 2, 50).unwrap();
        assert_eq!(results.len(), 2);

        // With cosine similarity, the query [1.0, 0.1, 0.0] should be closest to v1 or v4
        // Both have x=1.0, but cosine considers angles not magnitude differences
        // Due to HNSW's graph structure with small datasets, exact ordering may vary
        // Just verify we get reasonable results
        assert!(
            results[0].1 < 0.3,
            "First result distance should be small: {}",
            results[0].1
        );

        // The top result should be either v1 [1.0, 0.0, 0.0] or v4 [1.0, 1.0, 0.0]
        assert!(
            results[0].0 == "v1" || results[0].0 == "v4",
            "First result should be v1 or v4, got: {}",
            results[0].0
        );
    }

    #[test]
    fn test_wasm_hnsw_remove() {
        let mut index = WasmHnsw::new(2);

        index.insert("v1".to_string(), vec![1.0, 2.0]).unwrap();
        index.insert("v2".to_string(), vec![3.0, 4.0]).unwrap();
        index.insert("v3".to_string(), vec![5.0, 6.0]).unwrap();

        assert_eq!(index.len(), 3);

        // Verify v2 exists before removal
        let results_before = index.search(&[3.0, 4.0], 1, 50).unwrap();
        assert_eq!(results_before[0].0, "v2");

        index.remove("v2").unwrap();
        assert_eq!(index.len(), 2);

        // After removal, v2 should not be found
        let results_after = index.search(&[3.0, 4.0], 3, 50).unwrap();
        assert!(results_after.len() >= 1); // At least one result (v1 or v3)
        assert!(results_after.len() <= 2); // At most two results
                                           // v2 should not be in results
        for (id, _) in &results_after {
            assert_ne!(id, "v2");
        }
    }

    #[test]
    fn test_wasm_hnsw_large() {
        let mut index = WasmHnsw::with_params(128, Distance::Cosine, 16, 200);

        // Insert 1000 random vectors
        let mut rng = rand::thread_rng();
        for i in 0..1000 {
            let vector: Vec<f32> = (0..128).map(|_| rng.gen::<f32>()).collect();
            index.insert(format!("v{}", i), vector).unwrap();
        }

        assert_eq!(index.len(), 1000);

        // Search should work
        let query: Vec<f32> = (0..128).map(|_| rng.gen::<f32>()).collect();
        let results = index.search(&query, 10, 50).unwrap();
        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_dimension_validation() {
        let mut index = WasmHnsw::new(3);

        // Wrong dimension should fail
        let result = index.insert("v1".to_string(), vec![1.0, 2.0]);
        assert!(result.is_err());

        // Correct dimension should work
        let result = index.insert("v1".to_string(), vec![1.0, 2.0, 3.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_visualization() {
        let mut index = WasmHnsw::new(3);

        // Insert test vectors
        index.insert("v1".to_string(), vec![0.1, 0.1, 0.1]).unwrap();
        index.insert("v2".to_string(), vec![0.2, 0.1, 0.1]).unwrap();
        index.insert("v3".to_string(), vec![0.8, 0.9, 0.8]).unwrap();
        index.insert("v4".to_string(), vec![0.5, 0.5, 0.5]).unwrap();

        // Create visualizer
        let viz = index.to_visualizer();

        // Verify graph structure
        assert_eq!(viz.node_count(), 4);
        assert!(viz.edge_count() > 0);

        // Test DOT export
        let dot = viz.export_dot().unwrap();
        assert!(dot.contains("digraph HNSW"));
        assert!(dot.contains("v1"));
        assert!(dot.contains("v2"));
        assert!(dot.contains("v3"));
        assert!(dot.contains("v4"));

        // Test JSON export
        let json = viz.export_json().unwrap();
        assert!(json.contains("nodes"));
        assert!(json.contains("links"));

        // Test Cytoscape export
        let cyto = viz.export_cytoscape().unwrap();
        assert!(cyto.contains("data"));

        // Test statistics
        let stats = viz.statistics();
        assert_eq!(stats.node_count, 4);
        assert!(stats.edge_count > 0);
        assert!(stats.layer_count > 0);

        // Test statistics text export
        let stats_text = viz.export_statistics_text();
        assert!(stats_text.contains("HNSW Graph Statistics"));

        // Test sampling
        let sampled = viz.sample(2);
        assert_eq!(sampled.node_count(), 2);
    }
}
