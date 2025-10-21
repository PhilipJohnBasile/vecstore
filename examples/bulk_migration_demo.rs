//! Bulk migration demonstration
//!
//! Shows how to migrate data from other vector databases:
//! - Import from Pinecone exports
//! - Import from Qdrant snapshots
//! - Import from ChromaDB exports
//! - Progress tracking
//! - Performance metrics

use anyhow::Result;
use std::fs;
use tempfile::TempDir;
use vecstore::bulk_migration::{ChromaDBMigration, MigrationConfig, PineconeMigration};
use vecstore::VecStore;

fn main() -> Result<()> {
    println!("🔄 VecStore Bulk Migration Demo\n");
    println!("{}", "=".repeat(80));

    let temp_dir = TempDir::new()?;

    // Step 1: Create test Pinecone export file
    println!("\n[1/4] Creating test Pinecone export...");
    let pinecone_data = serde_json::json!({
        "vectors": [
            {
                "id": "doc1",
                "values": [0.1, 0.2, 0.3, 0.4],
                "metadata": {"title": "Document 1", "category": "tech"}
            },
            {
                "id": "doc2",
                "values": [0.5, 0.6, 0.7, 0.8],
                "metadata": {"title": "Document 2", "category": "science"}
            },
            {
                "id": "doc3",
                "values": [0.2, 0.3, 0.4, 0.5],
                "metadata": {"title": "Document 3", "category": "tech"}
            }
        ]
    });

    let pinecone_file = temp_dir.path().join("pinecone_export.json");
    fs::write(&pinecone_file, pinecone_data.to_string())?;
    println!("   ✓ Created Pinecone export with 3 vectors");

    // Step 2: Migrate from Pinecone
    println!("\n[2/4] Migrating from Pinecone...");
    let mut store = VecStore::open(temp_dir.path().join("migrated.db"))?;

    let config = MigrationConfig {
        batch_size: 100,
        validate: true,
        resume_from: None,
    };

    let migration = PineconeMigration::new(config.clone()).with_progress(|current, total| {
        if current % 1 == 0 {
            println!("   Progress: {}/{} vectors", current, total);
        }
    });

    let stats = migration.import_from_file(pinecone_file.to_str().unwrap(), &mut store)?;

    println!("\n   Migration complete!");
    println!("   • Total vectors: {}", stats.total_vectors);
    println!("   • Errors: {}", stats.errors);
    println!("   • Duration: {:?}", stats.duration);
    println!("   • Throughput: {:.0} vectors/sec", stats.throughput);
    println!("   • Bytes processed: {} bytes", stats.bytes_processed);

    // Verify migration
    assert_eq!(store.len(), 3);
    println!("   ✓ Verification passed");

    // Step 3: Create test ChromaDB export
    println!("\n[3/4] Creating test ChromaDB export...");
    let chroma_data = serde_json::json!({
        "ids": ["chroma1", "chroma2"],
        "embeddings": [[0.9, 0.8, 0.7, 0.6], [0.3, 0.4, 0.5, 0.6]],
        "metadatas": [
            {"source": "chroma", "type": "embedding"},
            {"source": "chroma", "type": "embedding"}
        ]
    });

    let chroma_file = temp_dir.path().join("chroma_export.json");
    fs::write(&chroma_file, chroma_data.to_string())?;
    println!("   ✓ Created ChromaDB export with 2 vectors");

    // Step 4: Migrate from ChromaDB
    println!("\n[4/4] Migrating from ChromaDB...");
    let chroma_migration = ChromaDBMigration::new(config);
    let chroma_stats =
        chroma_migration.import_from_file(chroma_file.to_str().unwrap(), &mut store)?;

    println!("\n   Migration complete!");
    println!("   • Total vectors: {}", chroma_stats.total_vectors);
    println!("   • Duration: {:?}", chroma_stats.duration);
    println!(
        "   • Throughput: {:.0} vectors/sec",
        chroma_stats.throughput
    );

    // Verify total
    assert_eq!(store.len(), 5);
    println!("   ✓ Total vectors in store: {}", store.len());

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("📊 Summary");
    println!("{}", "=".repeat(80));

    println!("\n✅ Bulk migration working!");

    println!("\n💡 Supported Formats:");
    println!("   • Pinecone: JSON export with 'vectors' array");
    println!("   • Qdrant: JSONL snapshot with 'points'");
    println!("   • ChromaDB: JSON export with ids/embeddings/metadatas");
    println!("   • Weaviate: JSON export (coming soon)");
    println!("   • Milvus: JSON export (coming soon)");

    println!("\n🚀 Features:");
    println!("   • Batch processing for efficiency");
    println!("   • Progress tracking with callbacks");
    println!("   • Validation of input data");
    println!("   • Resume capability for large migrations");
    println!("   • Detailed statistics (throughput, errors, bytes)");
    println!("   • Format converters for standardization");

    println!("\n📝 Migration Best Practices:");
    println!("   1. Test migration on a sample first");
    println!("   2. Use appropriate batch sizes (1000-10000)");
    println!("   3. Enable validation for production data");
    println!("   4. Monitor memory usage for large datasets");
    println!("   5. Keep original data until verification complete");
    println!("   6. Use resume_from for interrupted migrations");

    println!("\n🔧 Example Commands:");
    println!("   # Convert Pinecone export to universal JSONL");
    println!("   FormatConverter::pinecone_to_jsonl(\"export.json\", \"universal.jsonl\")");
    println!();
    println!("   # Migrate with progress tracking");
    println!("   let migration = PineconeMigration::new(config)");
    println!("       .with_progress(|current, total| /* track progress */);");

    Ok(())
}
