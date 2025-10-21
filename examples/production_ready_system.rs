//! Production-Ready Vector Search System
//!
//! A complete example showing best practices for production deployment:
//! - Health monitoring
//! - Performance tracking
//! - Error handling
//! - Resource management
//! - Automatic compaction
//! - Metrics collection

use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use vecstore::{
    BenchmarkConfig, Benchmarker, HealthCheckConfig, HealthChecker, HealthStatus, Metadata, Query,
    VecStore, VecStoreBuilder,
};

/// Production vector search service
struct VectorSearchService {
    store: VecStore,
    health_checker: HealthChecker,
    last_compaction: Option<Instant>,
    compaction_interval: Duration,
    metrics: ServiceMetrics,
}

#[derive(Debug, Default)]
struct ServiceMetrics {
    total_queries: u64,
    total_inserts: u64,
    total_errors: u64,
    avg_query_latency_ms: f64,
}

impl VectorSearchService {
    /// Create a demo service struct for the alpha server
    fn new(db_path: &str) -> Result<Self> {
        println!("🚀 Initializing production vector search service...");

        // Configure health check thresholds
        let health_config = HealthCheckConfig {
            deletion_ratio_warning: 0.2,
            deletion_ratio_critical: 0.4,
            fragmentation_warning: 40.0,
            memory_warning: 75.0,
            latency_warning_ms: 50.0,
            min_performance_score: 80.0,
        };

        // Open database
        let store = VecStore::open(db_path)?;

        println!("   ✓ Database opened: {}", db_path);

        Ok(Self {
            store,
            health_checker: HealthChecker::new(health_config),
            last_compaction: None,
            compaction_interval: Duration::from_secs(3600), // 1 hour
            metrics: ServiceMetrics::default(),
        })
    }

    /// Insert a vector with automatic health management
    fn insert(&mut self, id: String, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        let start = Instant::now();

        self.store.upsert(id, vector, metadata)?;
        self.metrics.total_inserts += 1;

        // Auto-compact if needed
        self.maybe_compact()?;

        Ok(())
    }

    /// Query vectors with monitoring
    fn query(&mut self, vector: Vec<f32>, k: usize, filter: Option<&str>) -> Result<Vec<String>> {
        let start = Instant::now();

        let mut query = Query::new(vector).with_limit(k);
        if let Some(f) = filter {
            query = query.with_filter(f);
        }

        let results = self.store.query(query)?;

        // Update metrics
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.total_queries += 1;

        // Update rolling average
        let alpha = 0.1; // Exponential moving average factor
        self.metrics.avg_query_latency_ms =
            alpha * latency_ms + (1.0 - alpha) * self.metrics.avg_query_latency_ms;

        Ok(results.into_iter().map(|r| r.id).collect())
    }

    /// Perform health check
    fn health_check(&self) -> Result<HealthStatus> {
        let report = self.health_checker.check(&self.store)?;

        // Log alerts
        for alert in &report.alerts {
            match alert.severity {
                vecstore::AlertSeverity::Critical => {
                    eprintln!("❌ CRITICAL: {}", alert.message);
                    if let Some(rec) = &alert.recommendation {
                        eprintln!("   → {}", rec);
                    }
                }
                vecstore::AlertSeverity::Warning => {
                    println!("⚠️  WARNING: {}", alert.message);
                }
                vecstore::AlertSeverity::Info => {
                    println!("ℹ️  INFO: {}", alert.message);
                }
            }
        }

        Ok(report.status)
    }

    /// Maybe run compaction based on schedule and health
    fn maybe_compact(&mut self) -> Result<()> {
        let should_compact = if let Some(last) = self.last_compaction {
            last.elapsed() > self.compaction_interval
        } else {
            true
        };

        if should_compact {
            let deletion_ratio = self.store.deleted_count() as f64
                / (self.store.len() + self.store.deleted_count()) as f64;

            if deletion_ratio > 0.2 {
                println!(
                    "🧹 Running compaction (deletion ratio: {:.1}%)...",
                    deletion_ratio * 100.0
                );
                let removed = self.store.compact()?;
                println!("   ✓ Compacted {} deleted vectors", removed);
                self.last_compaction = Some(Instant::now());
            }
        }

        Ok(())
    }

    /// Get current metrics
    fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }

    /// Run startup benchmark
    fn startup_benchmark(&self) -> Result<()> {
        println!("\n📊 Running startup benchmark...");

        let config = BenchmarkConfig {
            num_vectors: 500,
            dimension: 128,
            num_queries: 20,
            k_values: vec![10],
            test_filters: false,
            test_concurrent: false,
            num_threads: 1,
            test_indexing_strategies: false,
            test_quantization: false,
        };

        let benchmarker = Benchmarker::new(config);
        let results = benchmarker.run()?;

        println!("\n   Baseline Performance:");
        println!(
            "   • Insert: {:.0} vectors/sec",
            results.insert.batch_throughput
        );
        println!(
            "   • Query: {:.2} μs (avg)",
            results.query.by_k.get(&10).map(|s| s.avg_us).unwrap_or(0.0)
        );

        Ok(())
    }
}

fn main() -> Result<()> {
    println!("{}", "=".repeat(80));
    println!("Production-Ready Vector Search System");
    println!("{}", "=".repeat(80));

    // Initialize service
    let mut service = VectorSearchService::new("data/production.db")?;

    // Run startup benchmark
    service.startup_benchmark()?;

    // Simulate production workload
    println!("\n{}", "=".repeat(80));
    println!("Simulating Production Workload");
    println!("{}", "=".repeat(80));

    println!("\n[1/4] Inserting documents...");
    for i in 0..100 {
        let vector: Vec<f32> = (0..128)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let mut fields = HashMap::new();
        fields.insert("index".to_string(), serde_json::json!(i));
        fields.insert("category".to_string(), serde_json::json!(i % 5));

        service.insert(format!("doc_{}", i), vector, Metadata { fields })?;

        if (i + 1) % 20 == 0 {
            println!("   Inserted {} documents...", i + 1);
        }
    }

    println!("\n[2/4] Running queries...");
    for i in 0..10 {
        let query_vec: Vec<f32> = (0..128)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let results = service.query(query_vec, 5, None)?;
        println!("   Query {}: found {} results", i + 1, results.len());
    }

    println!("\n[3/4] Checking system health...");
    let health = service.health_check()?;
    println!("   Health status: {:?}", health);

    println!("\n[4/4] Service metrics:");
    let metrics = service.metrics();
    println!("   • Total inserts: {}", metrics.total_inserts);
    println!("   • Total queries: {}", metrics.total_queries);
    println!(
        "   • Avg query latency: {:.2} ms",
        metrics.avg_query_latency_ms
    );
    println!("   • Total errors: {}", metrics.total_errors);

    // Final health report
    println!("\n{}", "=".repeat(80));
    println!("Final Health Report");
    println!("{}", "=".repeat(80));

    let final_health = service.health_check()?;

    println!("\n✅ Production system check complete!");

    println!("\n💡 Best Practices Demonstrated:");
    println!("   • Automatic health monitoring");
    println!("   • Scheduled compaction");
    println!("   • Performance metrics tracking");
    println!("   • Error handling");
    println!("   • Resource management");
    println!("   • Startup benchmarking");

    println!("\n📝 Production Deployment Tips:");
    println!("   1. Monitor health status regularly");
    println!("   2. Set up alerts for critical conditions");
    println!("   3. Schedule compaction during low-traffic periods");
    println!("   4. Track query latency trends");
    println!("   5. Benchmark after configuration changes");
    println!("   6. Use filters for large datasets");
    println!("   7. Consider quantization for memory efficiency");

    Ok(())
}
