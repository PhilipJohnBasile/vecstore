//! Distance Metrics Demo
//!
//! This example demonstrates VecStore's support for multiple distance metrics:
//! - Cosine (default) - Best for text embeddings
//! - Euclidean (L2) - Best for spatial data
//! - Manhattan (L1) - Robust to outliers
//! - Hamming - For binary vectors
//! - Jaccard - For sparse vectors and sets
//! - DotProduct - When magnitude matters
//!
//! Run with: cargo run --example distance_metrics_demo

use std::collections::HashMap;
use vecstore::{
    hamming_distance_simd, jaccard_distance_simd, manhattan_distance_simd, Distance, Metadata,
    VecStore,
};

fn main() -> anyhow::Result<()> {
    println!("=== VecStore Distance Metrics Demo ===\n");

    // Clean up any previous test data
    let _ = std::fs::remove_dir_all("./distance_demo");

    // ========== 1. Cosine Distance (Default) ==========
    println!("1. Cosine Distance (Default - Best for text embeddings)");
    println!("   Range: [-1, 1], higher is more similar\n");

    let mut store_cosine = VecStore::open("./distance_demo/cosine")?;

    let mut meta1 = Metadata {
        fields: HashMap::new(),
    };
    meta1
        .fields
        .insert("doc".into(), serde_json::json!("Document 1"));

    let mut meta2 = Metadata {
        fields: HashMap::new(),
    };
    meta2
        .fields
        .insert("doc".into(), serde_json::json!("Document 2"));

    store_cosine.upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta1.clone())?;
    store_cosine.upsert("doc2".into(), vec![1.0, 0.5, 0.0], meta2.clone())?;

    let query = vecstore::Query {
        vector: vec![1.0, 0.25, 0.0],
        k: 2,
        filter: None,
    };

    let results = store_cosine.query(query)?;
    println!("   Query: [1.0, 0.25, 0.0]");
    for result in results {
        println!("   - {}: score = {:.4}", result.id, result.score);
    }
    println!();

    // ========== 2. Manhattan Distance (L1) ==========
    println!("2. Manhattan Distance (L1 - Robust to outliers)");
    println!("   Range: [0, ∞), lower is more similar\n");

    let store_manhattan = VecStore::builder("./distance_demo/manhattan")
        .distance(Distance::Manhattan)
        .build()?;

    println!("   Direct SIMD calculation:");
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let dist = manhattan_distance_simd(&a, &b);
    println!("   manhattan_distance([1, 2, 3], [4, 5, 6]) = {:.2}", dist);
    println!("   |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9.0\n");

    // ========== 3. Hamming Distance ==========
    println!("3. Hamming Distance (Binary vectors)");
    println!("   Range: [0, n], lower is more similar\n");

    let _store_hamming = VecStore::builder("./distance_demo/hamming")
        .distance(Distance::Hamming)
        .build()?;

    println!("   Direct SIMD calculation:");
    // Binary vectors (values > 0.5 treated as 1, <= 0.5 treated as 0)
    let bin_a = vec![1.0, 0.0, 1.0, 1.0]; // Binary: [1,0,1,1]
    let bin_b = vec![1.0, 1.0, 1.0, 0.0]; // Binary: [1,1,1,0]
    let hamming_dist = hamming_distance_simd(&bin_a, &bin_b);
    println!(
        "   hamming_distance([1,0,1,1], [1,1,1,0]) = {:.0}",
        hamming_dist
    );
    println!("   Positions differ at indices 1 and 3 = 2\n");

    // ========== 4. Jaccard Distance ==========
    println!("4. Jaccard Distance (Set similarity, sparse vectors)");
    println!("   Range: [0, 1], lower is more similar\n");

    let _store_jaccard = VecStore::builder("./distance_demo/jaccard")
        .distance(Distance::Jaccard)
        .build()?;

    println!("   Direct SIMD calculation:");
    let set_a = vec![1.0, 1.0, 0.0, 0.0]; // Set: {0, 1}
    let set_b = vec![1.0, 0.0, 1.0, 0.0]; // Set: {0, 2}
    let jaccard_dist = jaccard_distance_simd(&set_a, &set_b);
    println!(
        "   jaccard_distance([1,1,0,0], [1,0,1,0]) = {:.4}",
        jaccard_dist
    );
    println!("   Intersection: 1 (index 0), Union: 3 (indices 0,1,2)");
    println!("   Distance: 1 - (1/3) = {:.4}\n", jaccard_dist);

    // ========== 5. Builder Pattern Demo ==========
    println!("5. Builder Pattern (Custom configuration)");
    println!();

    let store_custom = VecStore::builder("./distance_demo/custom")
        .distance(Distance::Manhattan)
        .hnsw_m(32) // More connections = better recall
        .hnsw_ef_construction(400) // Higher quality index
        .build()?;

    println!("   Created store with:");
    println!(
        "   - Distance metric: {}",
        store_custom.distance_metric().name()
    );
    println!("   - HNSW M: {}", store_custom.config().hnsw_m);
    println!(
        "   - HNSW ef_construction: {}",
        store_custom.config().hnsw_ef_construction
    );
    println!();

    // ========== 6. Distance Metric Information ==========
    println!("6. All Available Distance Metrics");
    println!();

    let metrics = [
        Distance::Cosine,
        Distance::Euclidean,
        Distance::Manhattan,
        Distance::Hamming,
        Distance::Jaccard,
        Distance::DotProduct,
    ];

    for metric in metrics {
        println!("   {} - {}", metric.name(), metric.description());
    }
    println!();

    // ========== 7. Parsing Distance from String ==========
    println!("7. Parsing Distance Metrics from Strings");
    println!();

    let examples = vec![
        ("cosine", Distance::Cosine),
        ("manhattan", Distance::Manhattan),
        ("l1", Distance::Manhattan),
        ("hamming", Distance::Hamming),
        ("jaccard", Distance::Jaccard),
    ];

    for (string, expected) in examples {
        let parsed = Distance::from_str(string)?;
        println!("   '{}' -> {} ✓", string, parsed.name());
        assert_eq!(parsed, expected);
    }
    println!();

    // ========== Summary ==========
    println!("=== Summary ===");
    println!();
    println!("VecStore now supports 6 distance metrics:");
    println!("  ✓ Cosine - Default, best for text embeddings");
    println!("  ✓ Euclidean - Straight-line distance");
    println!("  ✓ Manhattan - Robust to outliers");
    println!("  ✓ Hamming - Binary vectors");
    println!("  ✓ Jaccard - Sparse vectors, sets");
    println!("  ✓ DotProduct - When magnitude matters");
    println!();
    println!("All metrics are SIMD-accelerated for 4-8x performance!");
    println!();
    println!("Use the builder pattern to choose:");
    println!("  VecStore::builder(path)");
    println!("    .distance(Distance::Manhattan)");
    println!("    .build()?");

    // Cleanup
    let _ = std::fs::remove_dir_all("./distance_demo");

    Ok(())
}
