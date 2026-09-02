// Streaming/Incremental Indexing Demo
//
// This example demonstrates vecstore's ability to add vectors incrementally
// without rebuilding the entire index. This is crucial for real-time applications!

use std::collections::HashMap;
use std::time::Instant;
use vecstore::{Metadata, VecStore, make_record};

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                                                               ║");
    println!("║     vecstore Streaming/Incremental Indexing Demo             ║");
    println!("║                                                               ║");
    println!("║         Add vectors in real-time - No rebuild needed!        ║");
    println!("║                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let temp_dir = tempfile::tempdir()?;
    let mut store = VecStore::open(temp_dir.path())?;

    // Demo 1: Streaming Insertion (one at a time)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 1: Streaming Insertion");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Adding vectors one at a time - perfect for real-time systems!\n");

    let start = Instant::now();
    for i in 0..1000 {
        let id = format!("stream_{}", i);
        let vector: Vec<f32> = (0..128).map(|j| ((i + j) as f32) / 1000.0).collect();
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields
            .insert("type".into(), serde_json::json!("streaming"));
        let record = make_record(&id, vector, meta);
        store.upsert(id, record.vector, record.metadata)?;
    }
    let elapsed = start.elapsed();

    println!("✅ Added 1,000 vectors incrementally");
    println!(
        "   Time: {:.2}ms ({:.2}μs per vector)",
        elapsed.as_millis(),
        elapsed.as_micros() as f64 / 1000.0
    );
    println!("   • No index rebuild required!");
    println!("   • Each vector is immediately searchable");
    println!();

    // Verify vectors are searchable immediately
    let query_vec: Vec<f32> = (0..128).map(|i| (i as f32) / 1000.0).collect();
    let results = store.query(vecstore::Query {
        vector: query_vec.clone(),
        k: 5,
        filter: None,
    })?;

    println!("🔍 Immediate search works:");
    for (i, result) in results.iter().take(3).enumerate() {
        println!("   {}. {} (score: {:.3})", i + 1, result.id, result.score);
    }
    println!();

    // Demo 2: Batch Insertion (optimized for bulk adds)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 2: Batch Insertion (Optimized)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Adding vectors in batches - uses parallel processing!\n");

    let batch_records: Vec<_> = (0..1000)
        .map(|i| {
            let id = format!("batch_{}", i);
            let vector: Vec<f32> = (0..128).map(|j| ((i + j + 1000) as f32) / 1000.0).collect();
            let mut meta = Metadata {
                fields: HashMap::new(),
            };
            meta.fields
                .insert("type".into(), serde_json::json!("batch"));
            make_record(&id, vector, meta)
        })
        .collect();

    let start = Instant::now();
    store.batch_upsert(batch_records)?;
    let elapsed = start.elapsed();

    println!("✅ Added 1,000 vectors in batch");
    println!(
        "   Time: {:.2}ms ({:.2}μs per vector)",
        elapsed.as_millis(),
        elapsed.as_micros() as f64 / 1000.0
    );
    println!("   • Uses parallel processing (rayon)");
    println!("   • Much faster than streaming for bulk inserts");
    println!();

    // Demo 3: Incremental Updates (upsert behavior)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 3: Incremental Updates");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Updating existing vectors without rebuilding the index\n");

    let start = Instant::now();
    for i in 0..100 {
        let id = format!("stream_{}", i);
        let vector: Vec<f32> = (0..128).map(|j| ((i + j + 5000) as f32) / 1000.0).collect();
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields
            .insert("type".into(), serde_json::json!("updated"));
        let record = make_record(&id, vector, meta);
        store.upsert(id, record.vector, record.metadata)?;
    }
    let elapsed = start.elapsed();

    println!("✅ Updated 100 existing vectors");
    println!("   Time: {:.2}ms", elapsed.as_millis());
    println!("   • Old vectors automatically replaced");
    println!("   • No index rebuild required");
    println!();

    // Demo 4: Deletions and Optimization
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 4: Deletions and Index Optimization");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Removing vectors and optimizing the index\n");

    // Remove some vectors
    let start = Instant::now();
    for i in 0..200 {
        let id = format!("stream_{}", i);
        let _ = store.remove(&id); // Ignore errors if already removed
    }
    let elapsed = start.elapsed();

    println!("✅ Removed 200 vectors");
    println!("   Time: {:.2}ms", elapsed.as_millis());
    println!("   ⚠️  Note: HNSW index still contains 'ghost' entries");
    println!();

    // Optimize to clean up ghost entries
    println!("🔧 Optimizing index to remove ghost entries...");
    let start = Instant::now();
    let removed = store.optimize()?;
    let elapsed = start.elapsed();

    println!("✅ Optimization complete");
    println!("   Time: {:.2}ms", elapsed.as_millis());
    println!("   Ghost entries removed: {}", removed);
    println!("   • Index is now compact and efficient");
    println!("   • Search performance improved");
    println!();

    // Demo 5: Real-time Use Case
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 5: Real-time Use Case Simulation");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Simulating a real-time document indexing system\n");

    println!("Scenario: Indexing incoming documents as they arrive");
    println!();

    for batch in 0..5 {
        let doc_id = format!("document_{}", batch);
        let vector: Vec<f32> = (0..128).map(|i| ((batch + i) as f32) / 100.0).collect();
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields
            .insert("type".into(), serde_json::json!("document"));
        meta.fields.insert("batch".into(), serde_json::json!(batch));
        let record = make_record(&doc_id, vector, meta);

        let start = Instant::now();
        store.upsert(doc_id.clone(), record.vector, record.metadata)?;
        let elapsed = start.elapsed();

        println!(
            "   📄 {} indexed in {:.2}ms - immediately searchable!",
            doc_id,
            elapsed.as_millis() as f64
        );

        // Simulate some processing time
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    println!();
    println!("✅ All documents indexed incrementally");
    println!("   • No downtime for index rebuilds");
    println!("   • Each document is searchable immediately");
    println!("   • Well-suited for real-time RAG systems");
    println!();

    // Summary
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                  Summary & Best Practices                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("✨ Key Features:");
    println!("   • True incremental indexing - no rebuild needed");
    println!("   • Vectors are immediately searchable after insert");
    println!("   • Batch insert for optimal bulk loading performance");
    println!("   • Optimize() to clean up after deletions");
    println!();
    println!("🎯 When to use each method:");
    println!("   • upsert() - Real-time, one-at-a-time indexing");
    println!("   • batch_upsert() - Bulk loading, initial index creation");
    println!("   • optimize() - Periodically after many deletions");
    println!();
    println!("📊 Performance Tips:");
    println!("   • Use batch_upsert() for initial data loading");
    println!("   • Use upsert() for real-time incremental updates");
    println!("   • Run optimize() when >20% of data is deleted");
    println!("   • Batch size sweet spot: 1,000-10,000 vectors");
    println!();
    println!("🔥 Why This Matters:");
    println!("   1. No downtime for index maintenance");
    println!("   2. Real-time RAG applications can update continuously");
    println!("   3. Better than systems requiring full rebuilds");
    println!("   4. Memory efficient with periodic optimization");
    println!();
    println!("💡 Example Use Cases:");
    println!("   • Real-time document indexing");
    println!("   • Streaming log analysis");
    println!("   • Live chatbot knowledge base updates");
    println!("   • Continuous embedding updates from LLMs");
    println!();

    println!("╭───────────────────────────────────────────────────────────╮");
    println!("│  Streaming Indexing: Making Real-time RAG Possible! ✨   │");
    println!("╰───────────────────────────────────────────────────────────╯");

    Ok(())
}
