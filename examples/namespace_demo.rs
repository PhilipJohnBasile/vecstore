//! Multi-Tenant Namespace Demo
//!
//! Demonstrates namespace isolation, quota enforcement, and multi-tenant patterns.
//!
//! Run with:
//! ```bash
//! cargo run --example namespace_demo --features server
//! ```

use std::collections::HashMap;
use vecstore::{Metadata, NamespaceManager, NamespaceQuotas, NamespaceStatus, Query};

fn main() -> anyhow::Result<()> {
    println!("🦀 VecStore Multi-Tenant Namespace Demo\n");
    println!("========================================\n");

    // Create a temporary directory for this demo
    let temp_dir = tempfile::TempDir::new()?;
    let root_path = temp_dir.path();

    // Create namespace manager
    println!("1️⃣  Creating NamespaceManager...");
    let manager = NamespaceManager::new(root_path)?;
    println!("✅ NamespaceManager created at {:?}\n", root_path);

    // Demo 1: Create namespaces with different tiers
    println!("2️⃣  Creating namespaces with different quota tiers...\n");

    // Free tier namespace
    manager.create_namespace(
        "free-customer".to_string(),
        "Free Tier Customer".to_string(),
        Some(NamespaceQuotas::free_tier()),
    )?;
    println!("✅ Created 'free-customer' with free tier quotas:");
    println!("   - Max vectors: 10,000");
    println!("   - Max storage: 100 MB");
    println!("   - Max requests/sec: 10");
    println!("   - Max concurrent queries: 2\n");

    // Pro tier namespace
    manager.create_namespace(
        "pro-customer".to_string(),
        "Pro Tier Customer".to_string(),
        Some(NamespaceQuotas::pro_tier()),
    )?;
    println!("✅ Created 'pro-customer' with pro tier quotas:");
    println!("   - Max vectors: 1,000,000");
    println!("   - Max storage: 10 GB");
    println!("   - Max requests/sec: 100");
    println!("   - Max concurrent queries: 20\n");

    // Unlimited tier namespace
    manager.create_namespace(
        "enterprise-customer".to_string(),
        "Enterprise Customer".to_string(),
        Some(NamespaceQuotas::unlimited()),
    )?;
    println!("✅ Created 'enterprise-customer' with unlimited quotas\n");

    // Demo 2: Insert vectors into different namespaces
    println!("3️⃣  Inserting vectors into different namespaces...\n");

    let metadata = Metadata {
        fields: {
            let mut fields = HashMap::new();
            fields.insert("category".to_string(), serde_json::json!("tech"));
            fields.insert("title".to_string(), serde_json::json!("Document 1"));
            fields
        },
    };

    // Insert into free tier
    manager.upsert(
        &"free-customer".to_string(),
        "doc1".to_string(),
        vec![0.1, 0.2, 0.3, 0.4],
        metadata.clone(),
    )?;
    println!("✅ Inserted vector into 'free-customer' namespace");

    // Insert into pro tier
    manager.upsert(
        &"pro-customer".to_string(),
        "doc1".to_string(),
        vec![0.5, 0.6, 0.7, 0.8],
        metadata.clone(),
    )?;
    println!("✅ Inserted vector into 'pro-customer' namespace\n");

    // Demo 3: Query isolation - vectors are isolated per namespace
    println!("4️⃣  Demonstrating namespace isolation...\n");

    let query = Query {
        vector: vec![0.1, 0.2, 0.3, 0.4],
        k: 10,
        filter: None,
    };

    // Query free-customer namespace
    let results = manager.query(&"free-customer".to_string(), query.clone())?;
    println!("🔍 Query in 'free-customer' namespace:");
    println!("   Found {} vectors", results.len());
    for result in &results {
        println!("   - {}: score = {:.4}", result.id, result.score);
    }
    println!();

    // Query pro-customer namespace - won't find free-customer's vectors
    let results = manager.query(&"pro-customer".to_string(), query.clone())?;
    println!("🔍 Query in 'pro-customer' namespace:");
    println!("   Found {} vectors", results.len());
    for result in &results {
        println!("   - {}: score = {:.4}", result.id, result.score);
    }
    println!("   ✅ Namespaces are completely isolated!\n");

    // Demo 4: Get namespace statistics
    println!("5️⃣  Namespace statistics...\n");

    for namespace_id in &["free-customer", "pro-customer", "enterprise-customer"] {
        let stats = manager.get_stats(&namespace_id.to_string())?;
        let namespace = manager.get_namespace(&namespace_id.to_string())?;

        println!("📊 Namespace: {}", namespace.name);
        println!("   ID: {}", stats.namespace_id);
        println!("   Vectors: {}", stats.vector_count);
        println!("   Active: {}", stats.active_count);
        println!("   Deleted: {}", stats.deleted_count);
        println!("   Dimension: {}", stats.dimension);
        println!(
            "   Quota utilization: {:.1}%",
            stats.quota_utilization * 100.0
        );
        println!("   Total requests: {}", stats.total_requests);
        println!("   Total queries: {}", stats.total_queries);
        println!("   Total upserts: {}", stats.total_upserts);
        println!("   Status: {:?}", stats.status);
        println!();
    }

    // Demo 5: Quota enforcement
    println!("6️⃣  Testing quota enforcement...\n");

    // Try to exceed rate limit
    println!("⏱️  Testing rate limit (free tier allows 10 req/sec)...");
    let mut rate_limit_hit = false;

    for i in 0..15 {
        let result = manager.upsert(
            &"free-customer".to_string(),
            format!("rate-test-{}", i),
            vec![0.1, 0.2, 0.3, 0.4],
            metadata.clone(),
        );

        if let Err(e) = result {
            if e.to_string().contains("Rate limit exceeded") {
                println!("   ✅ Rate limit triggered at request #{}: {}", i + 1, e);
                rate_limit_hit = true;
                break;
            }
        }
    }

    if !rate_limit_hit {
        println!("   ⚠️  Rate limit not triggered (requests may have been too slow)");
    }
    println!();

    // Demo 6: Update namespace quotas
    println!("7️⃣  Upgrading customer tier...\n");

    println!("📈 Upgrading 'free-customer' to pro tier...");
    manager.update_quotas(&"free-customer".to_string(), NamespaceQuotas::pro_tier())?;

    let namespace = manager.get_namespace(&"free-customer".to_string())?;
    println!("✅ Quota updated:");
    println!("   - Max vectors: {:?}", namespace.quotas.max_vectors);
    println!("   - Max storage: {:?}", namespace.quotas.max_storage_bytes);
    println!(
        "   - Max requests/sec: {:?}",
        namespace.quotas.max_requests_per_second
    );
    println!();

    // Demo 7: Namespace status management
    println!("8️⃣  Managing namespace status...\n");

    // Suspend a namespace
    println!("🚫 Suspending 'pro-customer' namespace...");
    manager.update_status(&"pro-customer".to_string(), NamespaceStatus::Suspended)?;

    let result = manager.upsert(
        &"pro-customer".to_string(),
        "doc2".to_string(),
        vec![0.1, 0.2, 0.3, 0.4],
        metadata.clone(),
    );

    match result {
        Err(e) if e.to_string().contains("not active") => {
            println!("   ✅ Write rejected: {}", e);
        },
        _ => println!("   ⚠️  Expected error but got: {:?}", result),
    }

    // Restore to active
    println!("\n✅ Restoring 'pro-customer' to active status...");
    manager.update_status(&"pro-customer".to_string(), NamespaceStatus::Active)?;

    // Verify write works again
    manager.upsert(
        &"pro-customer".to_string(),
        "doc2".to_string(),
        vec![0.1, 0.2, 0.3, 0.4],
        metadata.clone(),
    )?;
    println!("   ✅ Write succeeded after restoration\n");

    // Demo 8: Aggregate statistics
    println!("9️⃣  Aggregate statistics across all namespaces...\n");

    let agg_stats = manager.get_aggregate_stats()?;
    println!("📊 Global Statistics:");
    println!("   Total namespaces: {}", agg_stats.total_namespaces);
    println!("   Active namespaces: {}", agg_stats.active_namespaces);
    println!("   Total vectors: {}", agg_stats.total_vectors);
    println!("   Total requests: {}", agg_stats.total_requests);
    println!();

    // Demo 9: List all namespaces
    println!("🔟 Listing all namespaces...\n");

    let namespaces = manager.list_namespaces();
    for ns in namespaces {
        println!("📦 {}", ns.name);
        println!("   ID: {}", ns.id);
        println!("   Status: {:?}", ns.status);
        println!("   Created: {}", ns.created_at);
        println!(
            "   Quota utilization: {:.1}%",
            ns.quota_utilization() * 100.0
        );
        if ns.is_near_quota() {
            println!("   ⚠️  WARNING: Near quota limit!");
        }
        println!();
    }

    // Demo 10: Persistence
    println!("1️⃣1️⃣  Demonstrating persistence...\n");

    println!("💾 Saving all namespace metadata...");
    manager.save_all()?;
    println!("✅ Metadata saved to disk\n");

    println!("📂 Directory structure:");
    for entry in std::fs::read_dir(root_path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let namespace_id = entry.file_name().to_string_lossy().to_string();
            println!("   /{}/", namespace_id);

            for file in std::fs::read_dir(entry.path())? {
                let file = file?;
                let metadata = file.metadata()?;
                println!(
                    "      ├── {} ({} bytes)",
                    file.file_name().to_string_lossy(),
                    metadata.len()
                );
            }
        }
    }
    println!();

    // Demo 11: Loading namespaces
    println!("1️⃣2️⃣  Loading namespaces from disk...\n");

    let manager2 = NamespaceManager::new(root_path)?;
    let loaded = manager2.load_namespaces()?;
    println!("✅ Loaded {} namespaces from disk:", loaded.len());
    for ns_id in loaded {
        let stats = manager2.get_stats(&ns_id)?;
        println!("   - {}: {} vectors", ns_id, stats.vector_count);
    }
    println!();

    // Demo 12: Delete namespace
    println!("1️⃣3️⃣  Deleting a namespace...\n");

    println!("🗑️  Deleting 'enterprise-customer' namespace...");
    manager.delete_namespace(&"enterprise-customer".to_string())?;

    let remaining = manager.list_namespaces();
    println!(
        "✅ Namespace deleted. Remaining namespaces: {}",
        remaining.len()
    );
    for ns in remaining {
        println!("   - {}", ns.name);
    }
    println!();

    // Summary
    println!("========================================");
    println!("✅ Demo completed successfully!");
    println!();
    println!("Key Takeaways:");
    println!("  ✓ Namespaces provide complete data isolation");
    println!("  ✓ Quotas are enforced automatically (vectors, storage, rate limits)");
    println!("  ✓ Different quota tiers support various customer segments");
    println!("  ✓ Namespace status controls (Active/Suspended/ReadOnly)");
    println!("  ✓ Statistics tracking per namespace and aggregate");
    println!("  ✓ Persistence and recovery from disk");
    println!();
    println!("📚 For more information, see:");
    println!("   - NAMESPACES.md - Complete namespace guide");
    println!("   - SERVER.md - Multi-tenant server deployment");
    println!("   - DEVELOPER_GUIDE.md - Architecture and internals");

    Ok(())
}
