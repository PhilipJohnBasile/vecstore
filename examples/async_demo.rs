// Async API Demo - Using vecstore with Tokio
//
// This example demonstrates how to use vecstore in async Rust applications.
// Requires the "async" feature flag.

#[cfg(feature = "async")]
use std::collections::HashMap;
#[cfg(feature = "async")]
use vecstore::{AsyncVecStore, Metadata, Query};

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  vecstore Async API Demo                    ║");
    println!("║  Using vector search with Tokio              ║");
    println!("╚══════════════════════════════════════════════╝\n");

    let temp_dir = tempfile::tempdir()?;
    let store = AsyncVecStore::open(temp_dir.path()).await?;

    // Demo 1: Basic async operations
    println!("📝 Demo 1: Basic Async Operations");
    println!("─────────────────────────────────────");

    let mut meta1 = Metadata {
        fields: HashMap::new(),
    };
    meta1
        .fields
        .insert("title".into(), serde_json::json!("Document 1"));
    meta1
        .fields
        .insert("category".into(), serde_json::json!("tech"));

    store
        .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta1)
        .await?;

    let mut meta2 = Metadata {
        fields: HashMap::new(),
    };
    meta2
        .fields
        .insert("title".into(), serde_json::json!("Document 2"));
    meta2
        .fields
        .insert("category".into(), serde_json::json!("science"));

    store
        .upsert("doc2".into(), vec![0.0, 1.0, 0.0], meta2)
        .await?;

    println!("✅ Inserted 2 documents asynchronously");
    println!("   Count: {}", store.count().await);
    println!();

    // Demo 2: Concurrent queries
    println!("🔄 Demo 2: Concurrent Queries");
    println!("─────────────────────────────────────");

    // Clone the store (cheap - Arc clone)
    let store1 = store.clone();
    let store2 = store.clone();
    let store3 = store.clone();

    // Run 3 queries concurrently
    let start = std::time::Instant::now();

    let (result1, result2, result3) = tokio::join!(
        store1.query(Query {
            vector: vec![1.0, 0.0, 0.0],
            k: 1,
            filter: None,
        }),
        store2.query(Query {
            vector: vec![0.0, 1.0, 0.0],
            k: 1,
            filter: None,
        }),
        store3.query(Query {
            vector: vec![0.5, 0.5, 0.0],
            k: 2,
            filter: None,
        }),
    );

    let elapsed = start.elapsed();

    println!("✅ Executed 3 queries concurrently in {:?}", elapsed);
    println!("   Query 1 found: {}", result1?.len());
    println!("   Query 2 found: {}", result2?.len());
    println!("   Query 3 found: {}", result3?.len());
    println!();

    // Demo 3: Async with filters
    println!("🎯 Demo 3: Async Queries with Filters");
    println!("─────────────────────────────────────");

    // Add more data
    for i in 0..10 {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields.insert("index".into(), serde_json::json!(i));
        meta.fields.insert(
            "category".into(),
            serde_json::json!(if i % 2 == 0 { "tech" } else { "science" }),
        );

        store
            .upsert(format!("doc{}", i + 2), vec![i as f32, 0.0, 0.0], meta)
            .await?;
    }

    println!("✅ Added 10 more documents");

    // Query with SQL-like filter
    let results = store
        .query_with_filter(vec![5.0, 0.0, 0.0], 10, "category = 'tech'")
        .await?;

    println!("🔍 Query with filter 'category = tech':");
    for (i, result) in results.iter().take(3).enumerate() {
        let category = result.metadata.fields.get("category").unwrap();
        println!("   {}. {} - category: {}", i + 1, result.id, category);
    }
    println!();

    // Demo 4: Async batch operations
    println!("⚡ Demo 4: Async Batch Insert");
    println!("─────────────────────────────────────");

    use vecstore::make_record;

    let batch: Vec<_> = (20..30)
        .map(|i| {
            let mut meta = Metadata {
                fields: HashMap::new(),
            };
            meta.fields.insert("batch".into(), serde_json::json!(true));
            meta.fields.insert("index".into(), serde_json::json!(i));
            make_record(format!("batch{}", i), vec![i as f32, 0.0, 0.0], meta)
        })
        .collect();

    let start = std::time::Instant::now();
    store.batch_upsert(batch).await?;
    let elapsed = start.elapsed();

    println!("✅ Batch inserted 10 documents in {:?}", elapsed);
    println!("   Total count: {}", store.count().await);
    println!();

    // Demo 5: Async snapshots
    println!("💾 Demo 5: Async Snapshots");
    println!("─────────────────────────────────────");

    store.create_snapshot("async-demo-snapshot").await?;
    println!("✅ Created snapshot");

    let snapshots = store.list_snapshots().await?;
    println!("📋 Snapshots:");
    for (name, created_at, count) in &snapshots {
        println!("   • {} ({} records) - {}", name, count, created_at);
    }
    println!();

    // Demo 6: Integration with web framework pattern
    println!("🌐 Demo 6: Web Framework Integration Pattern");
    println!("─────────────────────────────────────");

    // Simulate a web server handler
    async fn search_handler(
        store: AsyncVecStore,
        query_vec: Vec<f32>,
    ) -> anyhow::Result<Vec<String>> {
        let results = store
            .query(Query {
                vector: query_vec,
                k: 5,
                filter: None,
            })
            .await?;

        Ok(results.into_iter().map(|r| r.id).collect())
    }

    // Simulate multiple concurrent requests
    let handlers = vec![
        search_handler(store.clone(), vec![1.0, 0.0, 0.0]),
        search_handler(store.clone(), vec![5.0, 0.0, 0.0]),
        search_handler(store.clone(), vec![10.0, 0.0, 0.0]),
    ];

    let start = std::time::Instant::now();
    let results = futures::future::join_all(handlers).await;
    let elapsed = start.elapsed();

    println!("✅ Handled 3 concurrent requests in {:?}", elapsed);
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(ids) => println!("   Request {}: found {} results", i + 1, ids.len()),
            Err(e) => println!("   Request {}: error - {}", i + 1, e),
        }
    }
    println!();

    // Demo 7: Async + Sync interop
    println!("🔄 Demo 7: Async/Sync Interoperability");
    println!("─────────────────────────────────────");

    // Sometimes you need to access the sync API from async context
    let dimension = store.with_sync(|sync_store| Ok(sync_store.dimension()))?;

    println!("✅ Accessed sync API from async context");
    println!("   Vector dimension: {}", dimension);
    println!();

    // Summary
    println!("╔══════════════════════════════════════════════╗");
    println!("║  Summary                                     ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("✨ Async features demonstrated:");
    println!("   • Basic async operations (upsert, query)");
    println!("   • Concurrent queries with tokio::join!");
    println!("   • SQL-like filters in async context");
    println!("   • Batch operations");
    println!("   • Snapshot management");
    println!("   • Web framework integration pattern");
    println!("   • Async/sync interoperability");
    println!();
    println!("🚀 Use cases:");
    println!("   • Web APIs (axum, actix-web, warp)");
    println!("   • Microservices");
    println!("   • Real-time applications");
    println!("   • High-concurrency systems");
    println!();
    println!("📦 Enable with: cargo add vecstore --features async");
    println!();

    println!("╭───────────────────────────────────────────╮");
    println!("│  Demo completed successfully! ✨           │");
    println!("╰───────────────────────────────────────────╯");

    Ok(())
}

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("This example requires the 'async' feature.");
    eprintln!("Run with: cargo run --example async_demo --features async");
    std::process::exit(1);
}
