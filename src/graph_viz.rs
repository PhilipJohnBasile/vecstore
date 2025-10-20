//! HNSW graph visualization and export
//!
//! This module provides utilities for exporting HNSW graph structures
//! for visualization and debugging. Supports multiple formats:
//!
//! - DOT format (Graphviz) - for rendering with Graphviz tools
//! - JSON format (D3.js compatible) - for interactive web visualizations
//! - Statistics - for understanding graph topology
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::{VecStore, graph_viz::HnswVisualizer};
//!
//! # fn main() -> anyhow::Result<()> {
//! let store = VecStore::open("vectors.db")?;
//! let viz = HnswVisualizer::from_store(&store)?;
//!
//! // Export to DOT format for Graphviz
//! let dot = viz.export_dot()?;
//! std::fs::write("graph.dot", dot)?;
//!
//! // Export to JSON for D3.js
//! let json = viz.export_json()?;
//! std::fs::write("graph.json", json)?;
//!
//! // Get graph statistics
//! let stats = viz.statistics();
//! println!("Nodes: {}, Edges: {}, Layers: {}",
//!          stats.node_count, stats.edge_count, stats.layer_count);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// HNSW graph visualizer
pub struct HnswVisualizer {
    /// Graph nodes with their metadata
    nodes: Vec<GraphNode>,

    /// Graph edges (connections between nodes)
    edges: Vec<GraphEdge>,

    /// Number of layers in the graph
    layers: usize,
}

/// A node in the visualization graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Node ID
    pub id: String,

    /// Layer this node belongs to (0 = bottom)
    pub layer: usize,

    /// Degree (number of connections)
    pub degree: usize,

    /// Optional: first few dimensions of the vector (for labeling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_preview: Option<Vec<f32>>,
}

/// An edge in the visualization graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub source: String,

    /// Target node ID
    pub target: String,

    /// Layer this edge exists in
    pub layer: usize,

    /// Optional: distance/weight
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of nodes
    pub node_count: usize,

    /// Total number of edges across all layers
    pub edge_count: usize,

    /// Number of layers
    pub layer_count: usize,

    /// Nodes per layer
    pub nodes_per_layer: Vec<usize>,

    /// Edges per layer
    pub edges_per_layer: Vec<usize>,

    /// Average degree per layer
    pub avg_degree_per_layer: Vec<f32>,

    /// Maximum degree across all nodes
    pub max_degree: usize,

    /// Minimum degree across all nodes
    pub min_degree: usize,

    /// Average degree across all nodes
    pub avg_degree: f32,
}

impl HnswVisualizer {
    /// Create a new visualizer with nodes and edges
    pub fn new(nodes: Vec<GraphNode>, edges: Vec<GraphEdge>, layers: usize) -> Self {
        Self {
            nodes,
            edges,
            layers,
        }
    }

    /// Export graph to DOT format (Graphviz)
    ///
    /// The output can be rendered with Graphviz:
    /// ```bash
    /// dot -Tpng graph.dot -o graph.png
    /// ```
    pub fn export_dot(&self) -> anyhow::Result<String> {
        let mut dot = String::new();

        dot.push_str("digraph HNSW {\n");
        dot.push_str("  rankdir=BT;\n"); // Bottom to top (layer 0 at bottom)
        dot.push_str("  node [shape=circle];\n");
        dot.push_str("  \n");

        // Define subgraphs for each layer
        for layer in 0..self.layers {
            dot.push_str(&format!("  subgraph cluster_layer_{} {{\n", layer));
            dot.push_str(&format!("    label = \"Layer {}\";\n", layer));
            dot.push_str("    style=dashed;\n");

            // Add nodes in this layer
            for node in &self.nodes {
                if node.layer >= layer {
                    let label = if let Some(preview) = &node.vector_preview {
                        format!(
                            "{}\\n[{:.2}, {:.2}, ...]",
                            node.id,
                            preview.get(0).unwrap_or(&0.0),
                            preview.get(1).unwrap_or(&0.0)
                        )
                    } else {
                        node.id.clone()
                    };

                    let color = match layer {
                        0 => "lightblue",
                        1 => "lightgreen",
                        2 => "lightyellow",
                        _ => "lightgray",
                    };

                    dot.push_str(&format!(
                        "    \"{}\" [label=\"{}\", fillcolor={}, style=filled];\n",
                        node.id, label, color
                    ));
                }
            }

            dot.push_str("  }\n");
        }

        dot.push_str("\n");

        // Add edges
        for edge in &self.edges {
            let style = match edge.layer {
                0 => "solid",
                1 => "dashed",
                _ => "dotted",
            };

            let weight_label = if let Some(w) = edge.weight {
                format!(" [label=\"{:.3}\", style={}]", w, style)
            } else {
                format!(" [style={}]", style)
            };

            dot.push_str(&format!(
                "  \"{}\" -> \"{}\"{};\n",
                edge.source, edge.target, weight_label
            ));
        }

        dot.push_str("}\n");

        Ok(dot)
    }

    /// Export graph to JSON format (D3.js compatible)
    ///
    /// The output is compatible with D3.js force-directed graph layouts.
    pub fn export_json(&self) -> anyhow::Result<String> {
        #[derive(Serialize)]
        struct D3Graph {
            nodes: Vec<GraphNode>,
            links: Vec<D3Link>,
        }

        #[derive(Serialize)]
        struct D3Link {
            source: String,
            target: String,
            layer: usize,
            #[serde(skip_serializing_if = "Option::is_none")]
            value: Option<f32>,
        }

        let links: Vec<D3Link> = self
            .edges
            .iter()
            .map(|e| D3Link {
                source: e.source.clone(),
                target: e.target.clone(),
                layer: e.layer,
                value: e.weight,
            })
            .collect();

        let graph = D3Graph {
            nodes: self.nodes.clone(),
            links,
        };

        serde_json::to_string_pretty(&graph)
            .map_err(|e| anyhow::anyhow!("JSON serialization failed: {}", e))
    }

    /// Export to Cytoscape.js format
    ///
    /// Cytoscape.js is a popular graph visualization library for the web.
    pub fn export_cytoscape(&self) -> anyhow::Result<String> {
        #[derive(Serialize)]
        struct CytoscapeElement {
            data: CytoscapeData,
            #[serde(skip_serializing_if = "Option::is_none")]
            position: Option<Position>,
        }

        #[derive(Serialize)]
        #[serde(untagged)]
        enum CytoscapeData {
            Node {
                id: String,
                label: String,
                layer: usize,
                degree: usize,
            },
            Edge {
                id: String,
                source: String,
                target: String,
                layer: usize,
                #[serde(skip_serializing_if = "Option::is_none")]
                weight: Option<f32>,
            },
        }

        #[derive(Serialize)]
        struct Position {
            x: f32,
            y: f32,
        }

        let mut elements = Vec::new();

        // Add nodes
        for (idx, node) in self.nodes.iter().enumerate() {
            elements.push(CytoscapeElement {
                data: CytoscapeData::Node {
                    id: node.id.clone(),
                    label: node.id.clone(),
                    layer: node.layer,
                    degree: node.degree,
                },
                position: Some(Position {
                    x: (idx % 10) as f32 * 100.0,
                    y: (idx / 10) as f32 * 100.0,
                }),
            });
        }

        // Add edges
        for (idx, edge) in self.edges.iter().enumerate() {
            elements.push(CytoscapeElement {
                data: CytoscapeData::Edge {
                    id: format!("e{}", idx),
                    source: edge.source.clone(),
                    target: edge.target.clone(),
                    layer: edge.layer,
                    weight: edge.weight,
                },
                position: None,
            });
        }

        serde_json::to_string_pretty(&elements)
            .map_err(|e| anyhow::anyhow!("Cytoscape JSON serialization failed: {}", e))
    }

    /// Get graph statistics
    pub fn statistics(&self) -> GraphStatistics {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();

        // Count nodes and edges per layer
        let mut nodes_per_layer = vec![0; self.layers];
        let mut edges_per_layer = vec![0; self.layers];

        for node in &self.nodes {
            for layer in 0..=node.layer {
                if layer < self.layers {
                    nodes_per_layer[layer] += 1;
                }
            }
        }

        for edge in &self.edges {
            if edge.layer < self.layers {
                edges_per_layer[edge.layer] += 1;
            }
        }

        // Calculate degree statistics
        let degrees: Vec<usize> = self.nodes.iter().map(|n| n.degree).collect();
        let max_degree = degrees.iter().copied().max().unwrap_or(0);
        let min_degree = degrees.iter().copied().min().unwrap_or(0);
        let avg_degree = if !degrees.is_empty() {
            degrees.iter().sum::<usize>() as f32 / degrees.len() as f32
        } else {
            0.0
        };

        // Average degree per layer
        let avg_degree_per_layer: Vec<f32> = (0..self.layers)
            .map(|layer| {
                let layer_edges = edges_per_layer.get(layer).copied().unwrap_or(0);
                let layer_nodes = nodes_per_layer.get(layer).copied().unwrap_or(0);
                if layer_nodes > 0 {
                    (2.0 * layer_edges as f32) / layer_nodes as f32
                } else {
                    0.0
                }
            })
            .collect();

        GraphStatistics {
            node_count,
            edge_count,
            layer_count: self.layers,
            nodes_per_layer,
            edges_per_layer,
            avg_degree_per_layer,
            max_degree,
            min_degree,
            avg_degree,
        }
    }

    /// Export statistics as formatted text
    pub fn export_statistics_text(&self) -> String {
        let stats = self.statistics();
        let mut text = String::new();

        text.push_str("=== HNSW Graph Statistics ===\n\n");
        text.push_str(&format!("Total Nodes: {}\n", stats.node_count));
        text.push_str(&format!("Total Edges: {}\n", stats.edge_count));
        text.push_str(&format!("Layers: {}\n\n", stats.layer_count));

        text.push_str("Degree Statistics:\n");
        text.push_str(&format!("  Average: {:.2}\n", stats.avg_degree));
        text.push_str(&format!("  Minimum: {}\n", stats.min_degree));
        text.push_str(&format!("  Maximum: {}\n\n", stats.max_degree));

        text.push_str("Per-Layer Breakdown:\n");
        for layer in 0..stats.layer_count {
            text.push_str(&format!("  Layer {}:\n", layer));
            text.push_str(&format!(
                "    Nodes: {}\n",
                stats.nodes_per_layer.get(layer).unwrap_or(&0)
            ));
            text.push_str(&format!(
                "    Edges: {}\n",
                stats.edges_per_layer.get(layer).unwrap_or(&0)
            ));
            text.push_str(&format!(
                "    Avg Degree: {:.2}\n",
                stats.avg_degree_per_layer.get(layer).unwrap_or(&0.0)
            ));
        }

        text
    }

    /// Sample a subset of the graph for visualization
    ///
    /// Useful for large graphs where visualizing everything would be overwhelming.
    pub fn sample(&self, max_nodes: usize) -> Self {
        if self.nodes.len() <= max_nodes {
            return self.clone();
        }

        // Take first max_nodes nodes
        let sampled_nodes: Vec<GraphNode> = self.nodes.iter().take(max_nodes).cloned().collect();

        let sampled_ids: HashMap<String, ()> =
            sampled_nodes.iter().map(|n| (n.id.clone(), ())).collect();

        // Only include edges where both endpoints are in sampled nodes
        let sampled_edges: Vec<GraphEdge> = self
            .edges
            .iter()
            .filter(|e| sampled_ids.contains_key(&e.source) && sampled_ids.contains_key(&e.target))
            .cloned()
            .collect();

        Self {
            nodes: sampled_nodes,
            edges: sampled_edges,
            layers: self.layers,
        }
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Clone for HnswVisualizer {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            layers: self.layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph() -> HnswVisualizer {
        let nodes = vec![
            GraphNode {
                id: "n1".to_string(),
                layer: 2,
                degree: 3,
                vector_preview: Some(vec![0.1, 0.2, 0.3]),
            },
            GraphNode {
                id: "n2".to_string(),
                layer: 1,
                degree: 4,
                vector_preview: Some(vec![0.4, 0.5, 0.6]),
            },
            GraphNode {
                id: "n3".to_string(),
                layer: 0,
                degree: 2,
                vector_preview: None,
            },
        ];

        let edges = vec![
            GraphEdge {
                source: "n1".to_string(),
                target: "n2".to_string(),
                layer: 1,
                weight: Some(0.5),
            },
            GraphEdge {
                source: "n2".to_string(),
                target: "n3".to_string(),
                layer: 0,
                weight: Some(0.7),
            },
        ];

        HnswVisualizer::new(nodes, edges, 3)
    }

    #[test]
    fn test_export_dot() {
        let viz = make_test_graph();
        let dot = viz.export_dot().unwrap();

        assert!(dot.contains("digraph HNSW"));
        assert!(dot.contains("n1"));
        assert!(dot.contains("n2"));
        assert!(dot.contains("n3"));
        assert!(dot.contains("Layer 0"));
    }

    #[test]
    fn test_export_json() {
        let viz = make_test_graph();
        let json = viz.export_json().unwrap();

        assert!(json.contains("nodes"));
        assert!(json.contains("links"));
        assert!(json.contains("n1"));
    }

    #[test]
    fn test_statistics() {
        let viz = make_test_graph();
        let stats = viz.statistics();

        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.edge_count, 2);
        assert_eq!(stats.layer_count, 3);
        assert_eq!(stats.max_degree, 4);
        assert_eq!(stats.min_degree, 2);
    }

    #[test]
    fn test_sample() {
        let viz = make_test_graph();
        let sampled = viz.sample(2);

        assert_eq!(sampled.node_count(), 2);
        assert!(sampled.edge_count() <= 2);
    }

    #[test]
    fn test_export_cytoscape() {
        let viz = make_test_graph();
        let cyto = viz.export_cytoscape().unwrap();

        assert!(cyto.contains("data"));
        assert!(cyto.contains("position"));
    }
}
