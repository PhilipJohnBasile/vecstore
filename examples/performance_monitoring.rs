//! Performance monitoring and benchmarking example
//!
//! Demonstrates:
//! - Comprehensive benchmarking
//! - Health monitoring
//! - Performance tracking
//! - Resource utilization

use anyhow::Result;
use vecstore::{print_health_report, BenchmarkConfig, Benchmarker, HealthChecker, VecStore};

fn main() -> Result<()> {
    println!("📊 VecStore Performance Monitoring Example\n");
    println!("{}", "=".repeat(80));

    // Step 1: Run comprehensive benchmarks
    println!("\n[1/3] Running comprehensive benchmarks...");
    println!("This may take a minute...\n");

    let config = BenchmarkConfig {
        num_vectors: 1_000,
        dimension: 128,
        num_queries: 50,
        k_values: vec![1, 10, 50],
        test_filters: true,
        test_concurrent: false,
        num_threads: 4,
        test_indexing_strategies: false, // Skip to make it faster
        test_quantization: true,
    };

    let benchmarker = Benchmarker::new(config);
    let results = benchmarker.run()?;

    // Print results
    Benchmarker::print_results(&results);

    // Step 2: Create a test database for health monitoring
    println!("\n[2/3] Creating test database...");
    let store = VecStore::open("data/perf_test.db")?;

    // Step 3: Run health check
    println!("\n[3/3] Running health check...");
    let checker = HealthChecker::default();
    let health_report = checker.check(&store)?;

    print_health_report(&health_report);

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("📈 Performance Summary");
    println!("{}", "=".repeat(80));

    println!("\n✅ Benchmark completed successfully!");
    println!("\n💡 Key Metrics:");
    println!(
        "   • Insert throughput: {:.0} vectors/sec",
        results.insert.batch_throughput
    );
    println!(
        "   • Avg query latency: {:.2} μs",
        results.query.by_k.get(&10).map(|s| s.avg_us).unwrap_or(0.0)
    );

    if let Some(ref quant) = results.quantization {
        if let Some(ref sq8) = quant.sq8 {
            println!("   • SQ8 compression: {:.1}x", sq8.compression_ratio);
        }
    }

    println!("\n📝 Recommendations:");
    if results.insert.batch_throughput < 1000.0 {
        println!("   • Consider tuning HNSW parameters for better insert performance");
    }
    if let Some(stats) = results.query.by_k.get(&10) {
        if stats.p95_us > 1000.0 {
            println!("   • Query latency is high - consider using quantization or IVF-PQ");
        }
    }
    println!("   • Monitor health metrics regularly in production");
    println!("   • Use compaction to reclaim deleted vector space");

    Ok(())
}
