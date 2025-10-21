//! Vector Deduplication Demo
//!
//! Demonstrates finding and removing duplicate/near-duplicate vectors.

use anyhow::Result;
use std::collections::HashMap;
use tempfile::TempDir;
use vecstore::*;

fn main() -> Result<()> {
    println!("\n🔍 VecStore Deduplication Demo\n");
    println!("{}", "=".repeat(60));

    let temp_dir = TempDir::new()?;
    let mut store = VecStore::open(temp_dir.path().join("test.db"))?;

    // Create test data with duplicates
    println!("\n📝 Creating test dataset with duplicates...");

    // Original documents
    let mut meta1 = Metadata {
        fields: HashMap::new(),
    };
    meta1
        .fields
        .insert("title".to_string(), serde_json::json!("Document 1"));
    store.upsert("doc1".to_string(), vec![1.0, 0.0, 0.0], meta1.clone())?;

    // Exact duplicate
    let mut meta2 = Metadata {
        fields: HashMap::new(),
    };
    meta2
        .fields
        .insert("title".to_string(), serde_json::json!("Document 1 Copy"));
    store.upsert("doc1_dup".to_string(), vec![1.0, 0.0, 0.0], meta2)?;

    // Near duplicate (99% similar)
    let mut meta3 = Metadata {
        fields: HashMap::new(),
    };
    meta3
        .fields
        .insert("title".to_string(), serde_json::json!("Document 1 Similar"));
    store.upsert("doc1_near".to_string(), vec![0.99, 0.01, 0.0], meta3)?;

    // Different document
    let mut meta4 = Metadata {
        fields: HashMap::new(),
    };
    meta4
        .fields
        .insert("title".to_string(), serde_json::json!("Document 2"));
    store.upsert("doc2".to_string(), vec![0.0, 1.0, 0.0], meta4.clone())?;

    // Another exact duplicate of doc2
    let mut meta5 = Metadata {
        fields: HashMap::new(),
    };
    meta5
        .fields
        .insert("title".to_string(), serde_json::json!("Document 2 Copy"));
    store.upsert("doc2_dup".to_string(), vec![0.0, 1.0, 0.0], meta5)?;

    println!("✓ Created 5 vectors (3 duplicates expected)");

    // Test 1: Find exact duplicates
    println!("\n[1/4] Finding Exact Duplicates");
    println!("{}", "-".repeat(60));

    let config_exact = DeduplicationConfig {
        similarity_threshold: 1.0, // Only exact matches
        strategy: DeduplicationStrategy::KeepFirst,
        batch_size: 100,
        use_cosine: true,
    };

    let dedup_exact = Deduplicator::new(config_exact);
    let exact_groups = dedup_exact.find_exact_duplicates(&store)?;

    println!("Found {} groups of exact duplicates:", exact_groups.len());
    for (i, group) in exact_groups.iter().enumerate() {
        println!("  Group {}: {} vectors", i + 1, group.duplicates.len());
        for id in &group.duplicates {
            println!("    - {}", id);
        }
    }

    // Test 2: Find near-duplicates
    println!("\n[2/4] Finding Near-Duplicates (99% threshold)");
    println!("{}", "-".repeat(60));

    let config_near = DeduplicationConfig {
        similarity_threshold: 0.99,
        strategy: DeduplicationStrategy::KeepFirst,
        batch_size: 100,
        use_cosine: true,
    };

    let dedup_near = Deduplicator::new(config_near);
    let near_groups = dedup_near.find_duplicates(&store)?;

    println!("Found {} groups of near-duplicates:", near_groups.len());
    for (i, group) in near_groups.iter().enumerate() {
        println!(
            "  Group {}: {} vectors (avg similarity: {:.4})",
            i + 1,
            group.duplicates.len(),
            group.avg_similarity
        );
        for (j, id) in group.duplicates.iter().enumerate() {
            println!("    - {} (similarity: {:.4})", id, group.scores[j]);
        }
    }

    // Test 3: Analyze duplication without removing
    println!("\n[3/4] Duplication Analysis");
    println!("{}", "-".repeat(60));

    let stats = dedup_near.analyze_duplication(&store)?;

    println!("Duplication Statistics:");
    println!("  Total vectors:        {}", stats.total_vectors);
    println!("  Duplicate groups:     {}", stats.duplicate_groups);
    println!("  Total duplicates:     {}", stats.total_duplicates);
    println!("  Would remove:         {}", stats.removed_count);
    println!("  Would keep:           {}", stats.kept_count);
    println!("  Storage saved:        {} bytes", stats.storage_saved);
    println!(
        "  Duplication ratio:    {:.1}%",
        stats.duplication_ratio * 100.0
    );

    // Test 4: Remove duplicates
    println!("\n[4/4] Removing Duplicates (Keep First Strategy)");
    println!("{}", "-".repeat(60));

    let removal_stats = dedup_near.remove_duplicates(&mut store)?;

    println!("Deduplication Complete:");
    println!("  Removed {} vectors", removal_stats.removed_count);
    println!("  Kept {} vectors", removal_stats.kept_count);
    println!("  Storage saved: {} bytes", removal_stats.storage_saved);
    println!("  Final vector count: {}", store.len());

    // Test 5: Different strategies
    println!("\n[5/5] Testing Different Strategies");
    println!("{}", "-".repeat(60));

    let strategies = [
        (DeduplicationStrategy::KeepFirst, "Keep First"),
        (DeduplicationStrategy::KeepLast, "Keep Last"),
        (
            DeduplicationStrategy::KeepMostMetadata,
            "Keep Most Metadata",
        ),
        (
            DeduplicationStrategy::KeepHighestQuality,
            "Keep Highest Quality",
        ),
    ];

    for (strategy, name) in &strategies {
        println!("\nStrategy: {}", name);
        println!("  Description: {:?}", strategy);
    }

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("📊 Demo Complete!");
    println!("{}", "=".repeat(60));

    println!("\n✨ Key Features Demonstrated:");
    println!("  ✓ Exact duplicate detection");
    println!("  ✓ Near-duplicate detection with threshold");
    println!("  ✓ Similarity scoring");
    println!("  ✓ Duplication analysis (without modification)");
    println!("  ✓ Automatic deduplication with strategies");
    println!("  ✓ Storage savings calculation");

    println!("\n💡 Use Cases:");
    println!("  • Data cleaning and quality management");
    println!("  • Finding plagiarized or similar content");
    println!("  • Reducing storage costs");
    println!("  • Improving search relevance");

    println!();

    Ok(())
}
