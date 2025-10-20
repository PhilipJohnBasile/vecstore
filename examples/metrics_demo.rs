use std::time::{Duration, Instant};
use vecstore::metrics::{Metrics, MetricsConfig};

fn main() -> anyhow::Result<()> {
    println!("=== VecStore Metrics Demo ===\n");

    // Create metrics collector with default configuration
    let metrics = Metrics::new(MetricsConfig::default());

    // Simulate some operations
    println!("Simulating 100 queries...");
    for i in 0..100 {
        let start = Instant::now();

        // Simulate query work
        std::thread::sleep(Duration::from_micros(500 + (i % 10) * 100));

        let latency = start.elapsed();
        let cache_hit = i % 3 == 0; // 33% cache hit rate

        metrics.record_query(latency, cache_hit);
    }

    println!("Simulating 10 query errors...");
    for _ in 0..10 {
        metrics.record_query_error();
    }

    println!("Simulating insert/update/delete operations...");
    for _ in 0..50 {
        metrics.record_insert();
    }
    for _ in 0..20 {
        metrics.record_update();
    }
    for _ in 0..10 {
        metrics.record_delete();
    }

    println!("Simulating HNSW graph traversal...");
    for _ in 0..100 {
        metrics.record_hnsw_stats(125, 45); // avg comparisons and visits per query
    }

    println!("\n");

    // Get and print metrics snapshot
    let snapshot = metrics.snapshot();
    snapshot.print_summary();

    println!("\n=== Prometheus Format ===\n");
    println!("{}", metrics.export_prometheus());

    Ok(())
}
