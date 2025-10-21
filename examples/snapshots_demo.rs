// Snapshot Demo - showing backup and restore functionality

use std::collections::HashMap;
use vecstore::{Metadata, VecStore};

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  vecstore Snapshot & Backup Demo            ║");
    println!("║  Save and restore your vector database      ║");
    println!("╚══════════════════════════════════════════════╝\n");

    let temp_dir = tempfile::tempdir()?;
    let mut store = VecStore::open(temp_dir.path())?;

    // Phase 1: Insert initial data
    println!("📝 Phase 1: Creating initial dataset");
    println!("─────────────────────────────────────");

    for i in 0..10 {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields.insert("version".into(), serde_json::json!(1));
        meta.fields.insert("index".into(), serde_json::json!(i));
        meta.fields
            .insert("name".into(), serde_json::json!(format!("doc{}", i)));

        store.upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta)?;
    }

    println!("✅ Inserted {} documents (version 1)", store.count());
    println!();

    // Create first snapshot
    println!("💾 Creating snapshot 'version-1'");
    println!("─────────────────────────────────────");
    store.create_snapshot("version-1")?;
    println!("✅ Snapshot created!");
    println!();

    // Phase 2: Modify data
    println!("✏️  Phase 2: Modifying dataset");
    println!("─────────────────────────────────────");

    // Update existing records
    for i in 0..5 {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields.insert("version".into(), serde_json::json!(2));
        meta.fields.insert("index".into(), serde_json::json!(i));
        meta.fields
            .insert("name".into(), serde_json::json!(format!("doc{}_v2", i)));
        meta.fields
            .insert("updated".into(), serde_json::json!(true));

        store.upsert(format!("doc{}", i), vec![i as f32 + 0.5, 0.1, 0.0], meta)?;
    }

    // Add new records
    for i in 10..15 {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields.insert("version".into(), serde_json::json!(2));
        meta.fields.insert("index".into(), serde_json::json!(i));
        meta.fields
            .insert("name".into(), serde_json::json!(format!("doc{}", i)));

        store.upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta)?;
    }

    println!("✅ Modified 5 documents");
    println!("✅ Added 5 new documents");
    println!("Current count: {}", store.count());
    println!();

    // Create second snapshot
    println!("💾 Creating snapshot 'version-2'");
    println!("─────────────────────────────────────");
    store.create_snapshot("version-2")?;
    println!("✅ Snapshot created!");
    println!();

    // Phase 3: More modifications
    println!("✏️  Phase 3: Further modifications");
    println!("─────────────────────────────────────");

    for i in 15..20 {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields.insert("version".into(), serde_json::json!(3));
        meta.fields.insert("index".into(), serde_json::json!(i));
        meta.fields
            .insert("name".into(), serde_json::json!(format!("doc{}", i)));

        store.upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta)?;
    }

    println!("✅ Added 5 more documents");
    println!("Current count: {}", store.count());
    println!();

    // List all snapshots
    println!("📋 Listing all snapshots");
    println!("─────────────────────────────────────");

    let snapshots = store.list_snapshots()?;
    println!("Found {} snapshots:\n", snapshots.len());

    for (name, created_at, count) in &snapshots {
        println!("  📸 {}", name);
        println!("     Created: {}", created_at);
        println!("     Records: {}", count);
        println!();
    }

    // Demonstrate restore
    println!("⏪ Restoring to 'version-1'");
    println!("─────────────────────────────────────");

    store.restore_snapshot("version-1")?;

    println!("✅ Restored!");
    println!("Current count: {} (should be 10)", store.count());
    println!();

    // Verify restoration
    println!("🔍 Verifying restoration");
    println!("─────────────────────────────────────");

    let results = store.query(vecstore::Query {
        vector: vec![5.0, 0.0, 0.0],
        k: 3,
        filter: None,
    })?;

    println!("Sample records after restore:");
    for result in results.iter().take(3) {
        let version = result.metadata.fields.get("version").unwrap();
        let name = result.metadata.fields.get("name").unwrap();
        println!("  - {} (version: {}, name: {})", result.id, version, name);
    }
    println!();

    // Restore to version 2
    println!("⏪ Now restoring to 'version-2'");
    println!("─────────────────────────────────────");

    store.restore_snapshot("version-2")?;

    println!("✅ Restored!");
    println!("Current count: {} (should be 15)", store.count());
    println!();

    // Clean up - delete old snapshot
    println!("🗑️  Deleting 'version-1' snapshot");
    println!("─────────────────────────────────────");

    store.delete_snapshot("version-1")?;

    let snapshots = store.list_snapshots()?;
    println!("✅ Deleted!");
    println!("Remaining snapshots: {}", snapshots.len());
    println!();

    // Summary
    println!("╔══════════════════════════════════════════════╗");
    println!("║  Summary                                     ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("✨ Snapshot features demonstrated:");
    println!("   • create_snapshot()  - Create named backups");
    println!("   • list_snapshots()   - List all snapshots");
    println!("   • restore_snapshot() - Restore from backup");
    println!("   • delete_snapshot()  - Clean up old backups");
    println!();
    println!("📚 Use cases:");
    println!("   • Version control for embeddings");
    println!("   • Rollback after failed updates");
    println!("   • A/B testing different datasets");
    println!("   • Safe experimentation");
    println!();

    println!("╭───────────────────────────────────────────╮");
    println!("│  Demo completed successfully! ✨           │");
    println!("╰───────────────────────────────────────────╯");

    Ok(())
}
