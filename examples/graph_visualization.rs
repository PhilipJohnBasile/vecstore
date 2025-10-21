//! HNSW Graph Visualization Example
//!
//! This example demonstrates how to visualize the HNSW graph structure
//! used by vecstore for fast approximate nearest neighbor search.
//!
//! ## What is HNSW?
//!
//! HNSW (Hierarchical Navigable Small World) is a graph-based algorithm that
//! organizes vectors in a multi-layer structure for efficient similarity search.
//!
//! ## Visualization Formats
//!
//! This example shows how to export the HNSW graph to three formats:
//! - **DOT (Graphviz)**: For rendering static graph images
//! - **JSON (D3.js)**: For interactive web visualizations
//! - **Cytoscape.js**: For web-based graph analysis tools
//!
//! ## Running
//!
//! Note: Graph visualization only works with WASM builds. For native builds,
//! compile with --target wasm32-unknown-unknown.
//!
//! ```bash
//! cargo run --example graph_visualization
//! ```
//!
//! To render the DOT file with Graphviz:
//! ```bash
//! dot -Tpng graph.dot -o graph.png
//! open graph.png
//! ```

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{Metadata, VecStore};

fn main() -> Result<()> {
    println!("🔍 HNSW Graph Visualization Example\n");

    // Create a temporary store
    let temp_dir = tempfile::tempdir()?;
    let mut store = VecStore::open(temp_dir.path())?;

    // Insert some sample vectors in a 3D space
    println!("📊 Inserting sample vectors...");

    let vectors = vec![
        // Cluster 1: Near origin
        ("v1", vec![0.1, 0.1, 0.1]),
        ("v2", vec![0.2, 0.1, 0.1]),
        ("v3", vec![0.1, 0.2, 0.1]),
        // Cluster 2: Upper region
        ("v4", vec![0.8, 0.9, 0.8]),
        ("v5", vec![0.9, 0.8, 0.9]),
        ("v6", vec![0.85, 0.85, 0.85]),
        // Cluster 3: Middle region
        ("v7", vec![0.5, 0.5, 0.5]),
        ("v8", vec![0.6, 0.5, 0.5]),
        ("v9", vec![0.5, 0.6, 0.5]),
        // Outliers
        ("v10", vec![0.3, 0.7, 0.2]),
        ("v11", vec![0.7, 0.3, 0.8]),
    ];

    for (id, vector) in vectors {
        store.upsert(
            id.to_string(),
            vector,
            Metadata {
                fields: HashMap::new(),
            },
        )?;
    }

    println!("✓ Inserted {} vectors\n", store.len());

    // Create visualizer
    println!("🎨 Creating graph visualizer...");

    match store.visualizer() {
        Ok(viz) => {
            // Export to different formats
            println!("✓ Visualizer created successfully!\n");

            // 1. Export to DOT format (Graphviz)
            println!("📄 Exporting to DOT format...");
            let dot = viz.export_dot()?;
            std::fs::write("graph.dot", &dot)?;
            println!("   → Saved to graph.dot");
            println!("   → Render with: dot -Tpng graph.dot -o graph.png\n");

            // 2. Export to JSON format (D3.js)
            println!("📄 Exporting to D3.js JSON format...");
            let json = viz.export_json()?;
            std::fs::write("graph.json", &json)?;
            println!("   → Saved to graph.json\n");

            // 3. Export to Cytoscape.js format
            println!("📄 Exporting to Cytoscape.js format...");
            let cyto = viz.export_cytoscape()?;
            std::fs::write("graph_cytoscape.json", &cyto)?;
            println!("   → Saved to graph_cytoscape.json\n");

            // 4. Get and display statistics
            println!("📊 Graph Statistics:");
            let stats = viz.statistics();
            println!("   Total Nodes: {}", stats.node_count);
            println!("   Total Edges: {}", stats.edge_count);
            println!("   Layers: {}", stats.layer_count);
            println!("   Avg Degree: {:.2}", stats.avg_degree);
            println!("   Min Degree: {}", stats.min_degree);
            println!("   Max Degree: {}", stats.max_degree);

            println!("\n   Per-Layer Breakdown:");
            for layer in 0..stats.layer_count {
                println!(
                    "   Layer {}: {} nodes, {} edges, {:.2} avg degree",
                    layer,
                    stats.nodes_per_layer.get(layer).unwrap_or(&0),
                    stats.edges_per_layer.get(layer).unwrap_or(&0),
                    stats.avg_degree_per_layer.get(layer).unwrap_or(&0.0)
                );
            }

            // 5. Export statistics as text
            println!("\n📄 Exporting statistics...");
            let stats_text = viz.export_statistics_text();
            std::fs::write("graph_stats.txt", &stats_text)?;
            println!("   → Saved to graph_stats.txt\n");

            // 6. Sample graph for large datasets
            if viz.node_count() > 100 {
                println!("📄 Creating sampled graph (first 50 nodes)...");
                let sampled = viz.sample(50);
                let sampled_dot = sampled.export_dot()?;
                std::fs::write("graph_sampled.dot", &sampled_dot)?;
                println!("   → Saved to graph_sampled.dot\n");
            }

            println!("✅ Graph visualization complete!");
            println!("\n🎯 Next Steps:");
            println!("   1. Render DOT file: dot -Tpng graph.dot -o graph.png");
            println!("   2. Open graph.json in a D3.js visualization tool");
            println!("   3. Load graph_cytoscape.json in Cytoscape.js");
            println!("   4. Read graph_stats.txt for detailed metrics");
        }
        Err(e) => {
            println!("⚠️  Graph visualization not available: {}", e);
            println!("\nNote: Graph visualization is currently only supported for WASM builds.");
            println!("To use visualization:");
            println!("  1. Build with: cargo build --target wasm32-unknown-unknown");
            println!("  2. Or use the WASM-based backend");
        }
    }

    Ok(())
}
