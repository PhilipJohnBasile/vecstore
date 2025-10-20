//! Vector partitioning demonstration
//!
//! Shows how to partition vectors for multi-tenancy:
//! - Tenant-based data isolation
//! - Per-partition queries
//! - Cross-partition search
//! - Partition management
//! - Statistics and monitoring

use anyhow::Result;
use std::collections::HashMap;
use tempfile::TempDir;
use vecstore::{Metadata, PartitionConfig, PartitionedStore, Query};

fn main() -> Result<()> {
    println!("🗂️  VecStore Partitioning Demo\n");
    println!("{}", "=".repeat(80));

    let temp_dir = TempDir::new()?;

    // Step 1: Create partitioned store
    println!("\n[1/5] Creating partitioned store...");
    let config = PartitionConfig {
        partition_field: "partition".to_string(),
        auto_create: true,
        max_vectors_per_partition: Some(1000),
    };

    let mut store = PartitionedStore::new(temp_dir.path().join("partitions"), config)?;
    println!("   ✓ Partitioned store created");

    // Step 2: Insert vectors into different partitions (tenants)
    println!("\n[2/5] Inserting vectors into partitions...");

    // Tenant A: Tech company vectors
    for i in 0..5 {
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("tenant".to_string(), serde_json::json!("A"));
        metadata
            .fields
            .insert("category".to_string(), serde_json::json!("technology"));
        metadata.fields.insert(
            "title".to_string(),
            serde_json::json!(format!("Tech Document {}", i)),
        );

        store.insert(
            "tenant_a",
            format!("tech_{}", i),
            vec![0.1 + i as f32 * 0.1, 0.2, 0.3, 0.4],
            metadata,
        )?;
    }
    println!("   ✓ Inserted 5 vectors for Tenant A");

    // Tenant B: Science company vectors
    for i in 0..5 {
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("tenant".to_string(), serde_json::json!("B"));
        metadata
            .fields
            .insert("category".to_string(), serde_json::json!("science"));
        metadata.fields.insert(
            "title".to_string(),
            serde_json::json!(format!("Science Document {}", i)),
        );

        store.insert(
            "tenant_b",
            format!("science_{}", i),
            vec![0.9 - i as f32 * 0.1, 0.8, 0.7, 0.6],
            metadata,
        )?;
    }
    println!("   ✓ Inserted 5 vectors for Tenant B");

    // Tenant C: Healthcare company vectors
    for i in 0..3 {
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("tenant".to_string(), serde_json::json!("C"));
        metadata
            .fields
            .insert("category".to_string(), serde_json::json!("healthcare"));
        metadata.fields.insert(
            "title".to_string(),
            serde_json::json!(format!("Healthcare Document {}", i)),
        );

        store.insert(
            "tenant_c",
            format!("health_{}", i),
            vec![0.5, 0.5 + i as f32 * 0.1, 0.4, 0.3],
            metadata,
        )?;
    }
    println!("   ✓ Inserted 3 vectors for Tenant C");

    // Step 3: Query within a specific partition
    println!("\n[3/5] Querying within Tenant A partition...");
    let query = Query::new(vec![0.15, 0.2, 0.3, 0.4]).with_limit(3);
    let results = store.query_partition("tenant_a", query)?;

    println!("   Found {} results in Tenant A:", results.len());
    for (i, result) in results.iter().enumerate() {
        println!(
            "   {}. ID: {} | Score: {:.4}",
            i + 1,
            result.id,
            result.score
        );
    }

    // Verify data isolation
    assert!(results.iter().all(|r| r.id.starts_with("tech_")));
    println!("   ✓ Data isolation verified (only Tenant A data returned)");

    // Step 4: Cross-partition search
    println!("\n[4/5] Searching across all partitions...");
    let query = Query::new(vec![0.5, 0.5, 0.4, 0.3]).with_limit(5);
    let all_results = store.query_all(query, 5)?;

    println!("   Found {} results across all tenants:", all_results.len());
    for (i, result) in all_results.iter().enumerate() {
        let tenant = if result.id.starts_with("tech_") {
            "A"
        } else if result.id.starts_with("science_") {
            "B"
        } else {
            "C"
        };
        println!(
            "   {}. Tenant {} | ID: {} | Score: {:.4}",
            i + 1,
            tenant,
            result.id,
            result.score
        );
    }

    // Step 5: Partition statistics
    println!("\n[5/5] Partition statistics...");
    let partitions = store.list_partitions();

    println!("   Total partitions: {}", partitions.len());
    for partition in &partitions {
        println!("\n   Partition: {}", partition.id);
        println!("   • Vectors: {}", partition.vector_count);
        println!("   • Path: {:?}", partition.path);
        println!("   • Size: {} bytes", partition.size_bytes);
        println!("   • Created: {:?}", partition.created_at);
        println!("   • Modified: {:?}", partition.modified_at);
    }

    // Get aggregate stats
    let stats = store.partition_stats();
    println!("\n   Aggregate Statistics:");
    println!("   • Total partitions: {}", stats.total_partitions);
    println!("   • Total vectors: {}", stats.total_vectors);
    println!(
        "   • Avg vectors/partition: {:.1}",
        stats.avg_vectors_per_partition
    );
    if let Some(ref largest) = stats.largest_partition {
        println!(
            "   • Largest partition: {} ({} vectors)",
            largest, stats.max_partition_size
        );
    }
    if let Some(ref smallest) = stats.smallest_partition {
        println!(
            "   • Smallest partition: {} ({} vectors)",
            smallest, stats.min_partition_size
        );
    }

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("📊 Summary");
    println!("{}", "=".repeat(80));

    println!("\n✅ Partitioning working!");

    println!("\n💡 Use Cases:");
    println!("   • Multi-tenant SaaS applications");
    println!("   • Data isolation and compliance (GDPR, HIPAA)");
    println!("   • Department/team-level data separation");
    println!("   • Customer-specific vector stores");
    println!("   • Geographic data partitioning");

    println!("\n🚀 Features:");
    println!("   • Automatic partition creation");
    println!("   • Per-partition size limits");
    println!("   • Isolated queries (fast, tenant-specific)");
    println!("   • Cross-partition search (comprehensive)");
    println!("   • Partition-level statistics");
    println!("   • Delete entire partitions");
    println!("   • Lazy loading (only load when accessed)");

    println!("\n🔒 Data Isolation Benefits:");
    println!("   • No cross-tenant data leakage");
    println!("   • Separate backup/restore per tenant");
    println!("   • Independent scaling per partition");
    println!("   • Easier compliance auditing");
    println!("   • Tenant-specific retention policies");

    println!("\n⚡ Performance:");
    println!("   • Partition queries are faster (smaller index)");
    println!("   • Lazy loading reduces memory usage");
    println!("   • Parallel cross-partition search possible");
    println!("   • Better cache locality per tenant");

    println!("\n🔧 Management Operations:");
    println!("   # Delete tenant data");
    println!("   store.delete_partition(\"tenant_a\")?;");
    println!();
    println!("   # Compact partition (remove deleted vectors)");
    println!("   store.compact_partition(\"tenant_b\")?;");
    println!();
    println!("   # Get partition info");
    println!("   let info = store.get_partition_info(\"tenant_c\")?;");

    Ok(())
}
