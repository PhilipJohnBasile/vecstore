//! Metadata indexing demonstration
//!
//! Shows how to use metadata indexes for fast filtered queries:
//! - BTree indexes for range queries
//! - Hash indexes for equality queries
//! - Inverted indexes for text search
//! - Performance comparison with/without indexes

use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use vecstore::metadata_index::{IndexConfig, IndexType, MetadataIndexManager};

fn main() -> Result<()> {
    println!("🗂️  VecStore Metadata Indexing Demo\n");
    println!("{}", "=".repeat(80));

    // Step 1: Create index manager
    println!("\n[1/5] Creating index manager...");
    let mut manager = MetadataIndexManager::new();

    // Step 2: Create different types of indexes
    println!("[2/5] Creating indexes...");

    // BTree index for range queries on "price"
    manager.create_index(
        "price",
        IndexConfig {
            index_type: IndexType::BTree,
            field: "price".to_string(),
        },
    )?;
    println!("   ✓ Created BTree index on 'price' (for range queries)");

    // Hash index for equality queries on "category"
    manager.create_index(
        "category",
        IndexConfig {
            index_type: IndexType::Hash,
            field: "category".to_string(),
        },
    )?;
    println!("   ✓ Created Hash index on 'category' (for equality queries)");

    // Hash index for "status"
    manager.create_index(
        "status",
        IndexConfig {
            index_type: IndexType::Hash,
            field: "status".to_string(),
        },
    )?;
    println!("   ✓ Created Hash index on 'status'");

    // Inverted index for text search on "description"
    manager.create_index(
        "description",
        IndexConfig {
            index_type: IndexType::Inverted,
            field: "description".to_string(),
        },
    )?;
    println!("   ✓ Created Inverted index on 'description' (for text search)");

    // Step 3: Insert sample data
    println!("\n[3/5] Inserting 1000 documents...");
    let start = Instant::now();

    for i in 0..1000 {
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        metadata.insert(
            "price".to_string(),
            serde_json::json!(10.0 + (i % 100) as f64 * 5.0),
        );
        metadata.insert(
            "category".to_string(),
            serde_json::json!(match i % 5 {
                0 => "electronics",
                1 => "books",
                2 => "clothing",
                3 => "food",
                _ => "other",
            }),
        );
        metadata.insert(
            "status".to_string(),
            serde_json::json!(if i % 3 == 0 { "active" } else { "inactive" }),
        );
        metadata.insert(
            "description".to_string(),
            serde_json::json!(format!(
                "Product {} with features including rust programming and high quality",
                i
            )),
        );

        manager.insert(&metadata, format!("doc_{}", i))?;

        if (i + 1) % 200 == 0 {
            println!("   Inserted {} documents...", i + 1);
        }
    }

    println!("   ✓ Inserted 1000 documents in {:?}", start.elapsed());

    // Step 4: Query with indexes
    println!("\n[4/5] Running indexed queries...");

    // Query 1: Range query on price
    println!("\n   Query 1: Find products with price > 300");
    let start = Instant::now();
    let results = manager.query("price", ">", &serde_json::json!(300.0));
    let elapsed = start.elapsed();
    println!(
        "   ✓ Found {} results in {:?}",
        results.as_ref().map(|r| r.len()).unwrap_or(0),
        elapsed
    );

    // Query 2: Equality query on category
    println!("\n   Query 2: Find products in 'electronics' category");
    let start = Instant::now();
    let results = manager.query("category", "=", &serde_json::json!("electronics"));
    let elapsed = start.elapsed();
    println!(
        "   ✓ Found {} results in {:?}",
        results.as_ref().map(|r| r.len()).unwrap_or(0),
        elapsed
    );

    // Query 3: IN query
    println!("\n   Query 3: Find products in 'electronics' OR 'books' categories");
    let start = Instant::now();
    let results = manager.query_in(
        "category",
        &[serde_json::json!("electronics"), serde_json::json!("books")],
    );
    let elapsed = start.elapsed();
    println!(
        "   ✓ Found {} results in {:?}",
        results.as_ref().map(|r| r.len()).unwrap_or(0),
        elapsed
    );

    // Query 4: Compound query (combining multiple indexes)
    println!("\n   Query 4: Find active products with price < 200");
    let start = Instant::now();
    let active = manager
        .query("status", "=", &serde_json::json!("active"))
        .unwrap();
    let cheap = manager
        .query("price", "<", &serde_json::json!(200.0))
        .unwrap();
    let results: Vec<&String> = active.intersection(&cheap).collect();
    let elapsed = start.elapsed();
    println!("   ✓ Found {} results in {:?}", results.len(), elapsed);

    // Step 5: Index statistics
    println!("\n[5/5] Index statistics...");

    for index_name in manager.list_indexes() {
        if let Some(stats) = manager.index_stats(&index_name) {
            println!("\n   Index: '{}'", index_name);
            println!("   • Type: {:?}", stats.index_type);
            println!("   • Unique values: {}", stats.unique_values);
            println!("   • Total entries: {}", stats.total_entries);
        }
    }

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("📊 Summary");
    println!("{}", "=".repeat(80));

    println!("\n✅ Metadata indexing working!");

    println!("\n💡 Index Types:");
    println!("   • BTree: Best for range queries (>, >=, <, <=)");
    println!("   • Hash: Best for equality (=, !=) and IN queries");
    println!("   • Inverted: Best for text containment (CONTAINS)");

    println!("\n🚀 Performance Benefits:");
    println!("   • Range queries: O(log N) instead of O(N)");
    println!("   • Equality queries: O(1) instead of O(N)");
    println!("   • Compound queries: Fast intersection of index results");
    println!(
        "   • Potential speedup: orders of magnitude for large datasets (measure on your workload)"
    );

    println!("\n📝 Use Cases:");
    println!("   • E-commerce: Filter by price, category, rating");
    println!("   • Content management: Filter by author, date, tags");
    println!("   • Log analysis: Filter by level, service, timestamp");
    println!("   • User data: Filter by age range, location, status");

    Ok(())
}
