//! Vector versioning demonstration

use anyhow::Result;
use std::collections::HashMap;
use tempfile::TempDir;
use vecstore::versioning::VersionedStore;
use vecstore::Metadata;

fn create_metadata(desc: &str) -> Metadata {
    let mut fields = HashMap::new();
    fields.insert("description".to_string(), serde_json::json!(desc));
    Metadata { fields }
}

fn main() -> Result<()> {
    println!("📜 VecStore Versioning Demo\n");
    println!("{}", "=".repeat(80));

    let temp_dir = TempDir::new()?;

    println!("\n[1/4] Creating versioned store and tracking changes...");
    let mut store = VersionedStore::new(temp_dir.path().join("versioned.db"))?;

    // Insert creates version 1
    println!("   Creating document with initial embedding...");
    let v1 = store.insert(
        "doc1",
        vec![1.0, 0.5, 0.2],
        create_metadata("Initial embedding"),
    )?;
    println!("   ✓ Version {}: Initial embedding", v1);

    // Update creates version 2
    println!("   Updating with improved model...");
    let v2 = store.update(
        "doc1",
        vec![1.1, 0.6, 0.3],
        create_metadata("Updated model"),
        Some("Improved embedding model".to_string()),
    )?;
    println!("   ✓ Version {}: Improved model", v2);

    // Another update creates version 3
    println!("   Fine-tuning embedding...");
    let v3 = store.update(
        "doc1",
        vec![1.2, 0.7, 0.4],
        create_metadata("Fine-tuned"),
        Some("Fine-tuned on domain data".to_string()),
    )?;
    println!("   ✓ Version {}: Fine-tuned", v3);

    println!("\n[2/4] Viewing version history...");
    let history = store.get_history("doc1").unwrap();
    println!("   Total versions: {}", history.versions.len());
    println!("   Current version: {}", history.current_version);
    println!("\n   Version timeline:");
    for version in &history.versions {
        println!(
            "   • V{}: {:?} - {}",
            version.version,
            version.vector,
            version.description.as_deref().unwrap_or("No description")
        );
    }

    println!("\n[3/4] Rolling back to previous version...");
    println!("   Current version: V{}", history.current_version);
    store.rollback("doc1", 1)?;
    let current = store.get_current_version("doc1").unwrap();
    println!("   ✓ Rolled back to V{}", current.version);
    println!("   Vector: {:?}", current.vector);

    println!("\n[4/4] Snapshot management...");
    store.create_snapshot("baseline", Some("Before experiment".to_string()))?;
    println!("   ✓ Created snapshot: 'baseline'");

    // Make changes
    store.update(
        "doc1",
        vec![2.0, 1.0, 0.5],
        create_metadata("Experimental"),
        Some("Testing new approach".to_string()),
    )?;
    println!("   Made experimental changes...");

    // Restore snapshot
    store.restore_snapshot("baseline")?;
    println!("   ✓ Restored snapshot: 'baseline'");

    let stats = store.stats();
    println!("\n   Versioning Statistics:");
    println!("   • Total vectors: {}", stats.total_vectors);
    println!("   • Total versions: {}", stats.total_versions);
    println!("   • Snapshots: {}", stats.total_snapshots);
    println!(
        "   • Avg versions/vector: {:.1}",
        stats.avg_versions_per_vector
    );

    println!("\n{}", "=".repeat(80));
    println!("✅ Versioning complete!");
    println!("\n💡 Use Cases: Audit trails, A/B testing, safe experimentation");
    println!("🎯 Features: Full history, rollback, snapshots, persistence");

    Ok(())
}
