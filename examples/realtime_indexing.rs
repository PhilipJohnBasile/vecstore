//! Real-time Indexing Example
//!
//! This example demonstrates real-time index updates that allow continuous
//! data ingestion without blocking queries.
//!
//! ## Features Demonstrated
//!
//! - Non-blocking inserts and updates
//! - Concurrent reads during writes
//! - Write buffering and batching
//! - Soft deletes for fast deletion
//! - Background workers for async processing
//! - Metrics and monitoring
//! - Snapshot isolation
//!
//! ## Running
//!
//! ```bash
//! # Sync version
//! cargo run --example realtime_indexing
//!
//! # Async version (requires async feature)
//! cargo run --example realtime_indexing --features async
//! ```

use std::time::Instant;
use vecstore::realtime::{RealtimeConfig, RealtimeIndex, RealtimeMetrics, Snapshot, WriteBuffer};

fn main() {
    println!("⚡ Real-time Indexing Example\n");

    // ============================================================
    // 1. Configuration
    // ============================================================
    println!("⚙️  Configuration:");
    println!("═════════════════\n");

    let config = RealtimeConfig::default()
        .with_buffer_size(100)
        .with_compaction_interval_secs(60)
        .with_compaction_threshold(0.15)
        .with_wal(true)
        .with_sync_wal(false); // Async for performance

    println!("Buffer size: {}", config.buffer_size);
    println!("Compaction interval: {}s", config.compaction_interval_secs);
    println!(
        "Compaction threshold: {:.0}%",
        config.compaction_threshold * 100.0
    );
    println!("WAL enabled: {}", config.enable_wal);
    println!("Sync WAL: {}", config.sync_wal);

    // ============================================================
    // 2. Write Buffer
    // ============================================================
    println!("\n\n📝 Write Buffer (Batching):");
    println!("═══════════════════════════\n");

    let mut buffer = WriteBuffer::new();

    println!("Buffering 5 inserts...");
    buffer.insert("doc1".to_string(), vec![0.1, 0.2, 0.3]);
    buffer.insert("doc2".to_string(), vec![0.2, 0.3, 0.4]);
    buffer.insert("doc3".to_string(), vec![0.3, 0.4, 0.5]);
    buffer.insert("doc4".to_string(), vec![0.4, 0.5, 0.6]);
    buffer.insert("doc5".to_string(), vec![0.5, 0.6, 0.7]);

    println!("  Buffer size: {}", buffer.len());
    println!("  Deleted count: {}", buffer.deleted_count());

    println!("\nBuffering 2 updates...");
    buffer.update("doc1".to_string(), vec![0.15, 0.25, 0.35]);
    buffer.update("doc2".to_string(), vec![0.25, 0.35, 0.45]);

    println!("  Buffer size: {}", buffer.len());

    println!("\nBuffering 1 delete...");
    buffer.delete("doc3".to_string());

    println!("  Buffer size: {}", buffer.len());
    println!("  Is doc3 deleted? {}", buffer.is_deleted("doc3"));
    println!("  Is doc1 deleted? {}", buffer.is_deleted("doc1"));

    println!("\nDraining buffer...");
    let entries = buffer.drain();
    println!("  Drained {} entries", entries.len());
    println!("  Buffer now empty: {}", buffer.is_empty());

    // ============================================================
    // 3. Snapshot Isolation
    // ============================================================
    println!("\n\n📸 Snapshot Isolation:");
    println!("══════════════════════\n");

    let mut snapshot = Snapshot::new(1000);

    println!("Snapshot timestamp: {}", snapshot.timestamp);
    println!("\nBefore deletions:");
    println!("  doc1 visible? {}", snapshot.is_visible("doc1"));
    println!("  doc2 visible? {}", snapshot.is_visible("doc2"));

    println!("\nMarking doc1 as deleted...");
    snapshot.mark_deleted("doc1".to_string());

    println!("\nAfter deletions:");
    println!("  doc1 visible? {}", snapshot.is_visible("doc1"));
    println!("  doc2 visible? {}", snapshot.is_visible("doc2"));

    println!("\n💡 Queries using this snapshot won't see doc1");

    // ============================================================
    // 4. Real-time Index (Sync)
    // ============================================================
    println!("\n\n🔄 Real-time Index (Synchronous):");
    println!("═══════════════════════════════════\n");

    let config = RealtimeConfig::default().with_buffer_size(5); // Small buffer for demo

    let mut index = RealtimeIndex::open("realtime.db", config).unwrap();

    println!("Inserting 10 vectors...");
    let start = Instant::now();

    for i in 0..10 {
        let id = format!("doc{}", i);
        let vector = vec![i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3];
        index.insert(&id, vector).unwrap();
    }

    let elapsed = start.elapsed();
    println!("  Inserted 10 vectors in {:?}", elapsed);
    println!(
        "  Average latency: {:.2}ms",
        elapsed.as_micros() as f64 / 10.0 / 1000.0
    );

    let metrics = index.metrics();
    println!("\nMetrics:");
    println!("  Total inserts: {}", metrics.total_inserts);
    println!("  Buffer flushes: {}", metrics.buffer_flushes);
    println!("  Buffer size: {}", metrics.buffer_size);

    println!("\nUpdating 3 vectors...");
    for i in 0..3 {
        let id = format!("doc{}", i);
        let vector = vec![i as f32 * 0.15, i as f32 * 0.25, i as f32 * 0.35];
        index.update(&id, vector).unwrap();
    }

    println!("  Total updates: {}", index.metrics().total_updates);

    println!("\nDeleting 2 vectors (soft delete)...");
    index.delete("doc5").unwrap();
    index.delete("doc7").unwrap();

    let metrics = index.metrics();
    println!("  Total deletes: {}", metrics.total_deletes);
    println!("  Deleted count: {}", metrics.deleted_count);

    println!("\nManual flush...");
    index.flush().unwrap();

    let snapshot = index.snapshot();
    println!("  doc5 visible? {}", snapshot.is_visible("doc5"));
    println!("  doc7 visible? {}", snapshot.is_visible("doc7"));
    println!("  doc0 visible? {}", snapshot.is_visible("doc0"));

    // ============================================================
    // 5. Compaction
    // ============================================================
    println!("\n\n🗜️  Compaction:");
    println!("══════════════\n");

    let metrics = index.metrics();
    println!("Before compaction:");
    println!("  Index size: {}", metrics.index_size);
    println!("  Deleted count: {}", metrics.deleted_count);
    println!("  Deletion ratio: {:.1}%", metrics.deletion_ratio() * 100.0);
    println!(
        "  Needs compaction (>10%)? {}",
        metrics.needs_compaction(0.1)
    );

    println!("\nRunning compaction...");
    let start = Instant::now();
    let stats = index.compact().unwrap();
    let elapsed = start.elapsed();

    println!("\nCompaction results:");
    println!("  Items removed: {}", stats.items_removed);
    println!("  Items reindexed: {}", stats.items_reindexed);
    println!("  Duration: {:?}", elapsed);
    println!(
        "  Bytes freed: {} bytes ({:.1} KB)",
        stats.bytes_freed,
        stats.bytes_freed as f32 / 1024.0
    );

    // ============================================================
    // 6. Metrics and Monitoring
    // ============================================================
    println!("\n\n📊 Metrics and Monitoring:");
    println!("═══════════════════════════\n");

    let mut metrics = RealtimeMetrics::default();
    metrics.total_inserts = 10000;
    metrics.total_updates = 500;
    metrics.total_deletes = 200;
    metrics.total_queries = 50000;
    metrics.buffer_flushes = 100;
    metrics.compactions = 5;
    metrics.avg_insert_latency_ms = 0.5;
    metrics.avg_query_latency_ms = 1.2;
    metrics.buffer_size = 50;
    metrics.index_size = 9800;
    metrics.deleted_count = 200;

    println!("Operations:");
    println!("  Total inserts: {}", metrics.total_inserts);
    println!("  Total updates: {}", metrics.total_updates);
    println!("  Total deletes: {}", metrics.total_deletes);
    println!("  Total queries: {}", metrics.total_queries);

    println!("\nPerformance:");
    println!(
        "  Avg insert latency: {:.2}ms",
        metrics.avg_insert_latency_ms
    );
    println!("  Avg query latency: {:.2}ms", metrics.avg_query_latency_ms);
    println!(
        "  Inserts/sec: {:.0}",
        1000.0 / metrics.avg_insert_latency_ms
    );
    println!(
        "  Queries/sec: {:.0}",
        1000.0 / metrics.avg_query_latency_ms
    );

    println!("\nMaintenance:");
    println!("  Buffer flushes: {}", metrics.buffer_flushes);
    println!("  Compactions: {}", metrics.compactions);

    println!("\nIndex state:");
    println!("  Buffer size: {}", metrics.buffer_size);
    println!("  Index size: {}", metrics.index_size);
    println!("  Deleted count: {}", metrics.deleted_count);
    println!("  Deletion ratio: {:.1}%", metrics.deletion_ratio() * 100.0);

    // ============================================================
    // 7. Update Strategies
    // ============================================================
    println!("\n\n🎯 Update Strategies:");
    println!("═════════════════════\n");

    println!("1. Blocking:");
    println!("   - Acquires write lock");
    println!("   - Blocks until index is updated");
    println!("   - Use case: Low write volume, consistency critical\n");

    println!("2. TryLock:");
    println!("   - Tries to acquire write lock");
    println!("   - Buffers if locked");
    println!("   - Use case: Moderate write volume, balanced\n");

    println!("3. AlwaysBuffer:");
    println!("   - Never blocks");
    println!("   - Always buffers writes");
    println!("   - Use case: High write volume, latency critical");

    // ============================================================
    // 8. Real-world Scenarios
    // ============================================================
    println!("\n\n🌍 Real-world Scenarios:");
    println!("═════════════════════════\n");

    // Scenario 1: E-commerce product catalog
    println!("1. E-commerce Product Catalog:");
    println!("   - 1M products, 1000 updates/sec");
    println!("   - Config:");
    println!("     • Buffer size: 10,000");
    println!("     • Flush interval: 100ms");
    println!("     • Strategy: AlwaysBuffer");
    println!("     • Compaction: Every 5 minutes");
    println!("   - Target: single-digit millisecond insert/query latency (hardware dependent)\n");

    // Scenario 2: Document search
    println!("2. Document Search Engine:");
    println!("   - 100K documents, 50 updates/sec");
    println!("   - Config:");
    println!("     • Buffer size: 1,000");
    println!("     • Flush interval: 1s");
    println!("     • Strategy: TryLock");
    println!("     • Compaction: Every 10 minutes");
    println!("   - Target: low-single-digit millisecond insert/query latency\n");

    // Scenario 3: Log analytics
    println!("3. Log Analytics:");
    println!("   - 10M logs, 10,000 inserts/sec");
    println!("   - Config:");
    println!("     • Buffer size: 100,000");
    println!("     • Flush interval: 5s");
    println!("     • Strategy: AlwaysBuffer");
    println!("     • Compaction: Every hour");
    println!("   - Expected: <0.1ms insert latency, <10ms query latency");

    // ============================================================
    // 9. Best Practices
    // ============================================================
    println!("\n\n💡 Best Practices:");
    println!("══════════════════\n");

    println!("1. Buffer sizing:");
    println!("   - Small buffer (<100): Low latency inserts, frequent flushes");
    println!("   - Medium buffer (100-1000): Balanced");
    println!("   - Large buffer (>1000): High throughput, higher latency\n");

    println!("2. Compaction:");
    println!("   - Trigger at 10-20% deletion ratio");
    println!("   - Run during low-traffic periods");
    println!("   - Monitor duration and adjust threshold\n");

    println!("3. Write-Ahead Log:");
    println!("   - Enable for durability");
    println!("   - Disable sync for performance (async flush)");
    println!("   - Trade-off: Durability vs. throughput\n");

    println!("4. Update strategy:");
    println!("   - Low writes: Blocking");
    println!("   - Medium writes: TryLock");
    println!("   - High writes: AlwaysBuffer\n");

    println!("5. Monitoring:");
    println!("   - Track deletion ratio");
    println!("   - Monitor buffer size");
    println!("   - Alert on high latency");
    println!("   - Dashboard for metrics");

    println!("\n✅ Real-time indexing example complete!\n");

    println!("🎯 Key Takeaways:");
    println!("  - Write buffering enables high throughput");
    println!("  - Snapshot isolation provides consistent reads");
    println!("  - Soft deletes avoid expensive index rebuilds");
    println!("  - Background workers handle async processing");
    println!("  - Compaction maintains index efficiency");
    println!("  - Metrics are crucial for monitoring");
}
