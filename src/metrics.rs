//! Observability and metrics for production monitoring
//!
//! This module provides:
//! - Prometheus metrics for query latency, throughput, cache hit rate
//! - OpenTelemetry integration for distributed tracing
//! - Performance counters for operations
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::metrics::{Metrics, MetricsConfig};
//!
//! # fn main() -> anyhow::Result<()> {
//! let config = MetricsConfig::default();
//! let metrics = Metrics::new(config);
//!
//! // Record query operation
//! let start = std::time::Instant::now();
//! // ... perform query ...
//! metrics.record_query(start.elapsed(), true); // cache_hit = true
//!
//! // Get metrics snapshot
//! let snapshot = metrics.snapshot();
//! println!("Queries/sec: {}", snapshot.queries_per_sec);
//! println!("Cache hit rate: {:.2}%", snapshot.cache_hit_rate * 100.0);
//! # Ok(())
//! # }
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for metrics collection
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,

    /// Histogram bucket boundaries for latency (in milliseconds)
    pub latency_buckets: Vec<f64>,

    /// Window size for throughput calculation (seconds)
    pub throughput_window_secs: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            latency_buckets: vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0],
            throughput_window_secs: 60,
        }
    }
}

/// Metrics collector for vecstore operations
#[derive(Clone)]
pub struct Metrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    config: MetricsConfig,

    // Query metrics
    total_queries: AtomicU64,
    query_errors: AtomicU64,
    query_latency_sum_micros: AtomicU64,

    // Cache metrics
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,

    // Insert/update metrics
    total_inserts: AtomicU64,
    total_updates: AtomicU64,
    total_deletes: AtomicU64,

    // HNSW metrics
    hnsw_comparisons: AtomicU64,
    hnsw_node_visits: AtomicU64,

    // Start time for throughput calculation
    start_time: Instant,
}

impl Metrics {
    /// Create a new metrics collector
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            inner: Arc::new(MetricsInner {
                config,
                total_queries: AtomicU64::new(0),
                query_errors: AtomicU64::new(0),
                query_latency_sum_micros: AtomicU64::new(0),
                cache_hits: AtomicU64::new(0),
                cache_misses: AtomicU64::new(0),
                total_inserts: AtomicU64::new(0),
                total_updates: AtomicU64::new(0),
                total_deletes: AtomicU64::new(0),
                hnsw_comparisons: AtomicU64::new(0),
                hnsw_node_visits: AtomicU64::new(0),
                start_time: Instant::now(),
            }),
        }
    }

    /// Record a query operation
    pub fn record_query(&self, latency: Duration, cache_hit: bool) {
        if !self.inner.config.enabled {
            return;
        }

        self.inner.total_queries.fetch_add(1, Ordering::Relaxed);
        self.inner
            .query_latency_sum_micros
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);

        if cache_hit {
            self.inner.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.inner.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a query error
    pub fn record_query_error(&self) {
        if !self.inner.config.enabled {
            return;
        }
        self.inner.query_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an insert operation
    pub fn record_insert(&self) {
        if !self.inner.config.enabled {
            return;
        }
        self.inner.total_inserts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an update operation
    pub fn record_update(&self) {
        if !self.inner.config.enabled {
            return;
        }
        self.inner.total_updates.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a delete operation
    pub fn record_delete(&self) {
        if !self.inner.config.enabled {
            return;
        }
        self.inner.total_deletes.fetch_add(1, Ordering::Relaxed);
    }

    /// Record HNSW graph traversal statistics
    pub fn record_hnsw_stats(&self, comparisons: u64, node_visits: u64) {
        if !self.inner.config.enabled {
            return;
        }
        self.inner
            .hnsw_comparisons
            .fetch_add(comparisons, Ordering::Relaxed);
        self.inner
            .hnsw_node_visits
            .fetch_add(node_visits, Ordering::Relaxed);
    }

    /// Get a snapshot of current metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        let total_queries = self.inner.total_queries.load(Ordering::Relaxed);
        let cache_hits = self.inner.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.inner.cache_misses.load(Ordering::Relaxed);
        let total_cache_lookups = cache_hits + cache_misses;

        let cache_hit_rate = if total_cache_lookups > 0 {
            cache_hits as f64 / total_cache_lookups as f64
        } else {
            0.0
        };

        let avg_query_latency_micros = if total_queries > 0 {
            self.inner.query_latency_sum_micros.load(Ordering::Relaxed) as f64
                / total_queries as f64
        } else {
            0.0
        };

        let uptime_secs = self.inner.start_time.elapsed().as_secs_f64();
        let queries_per_sec = if uptime_secs > 0.0 {
            total_queries as f64 / uptime_secs
        } else {
            0.0
        };

        MetricsSnapshot {
            total_queries,
            query_errors: self.inner.query_errors.load(Ordering::Relaxed),
            avg_query_latency_micros,
            cache_hit_rate,
            cache_hits,
            cache_misses,
            total_inserts: self.inner.total_inserts.load(Ordering::Relaxed),
            total_updates: self.inner.total_updates.load(Ordering::Relaxed),
            total_deletes: self.inner.total_deletes.load(Ordering::Relaxed),
            hnsw_comparisons: self.inner.hnsw_comparisons.load(Ordering::Relaxed),
            hnsw_node_visits: self.inner.hnsw_node_visits.load(Ordering::Relaxed),
            queries_per_sec,
            uptime_secs,
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.inner.total_queries.store(0, Ordering::Relaxed);
        self.inner.query_errors.store(0, Ordering::Relaxed);
        self.inner
            .query_latency_sum_micros
            .store(0, Ordering::Relaxed);
        self.inner.cache_hits.store(0, Ordering::Relaxed);
        self.inner.cache_misses.store(0, Ordering::Relaxed);
        self.inner.total_inserts.store(0, Ordering::Relaxed);
        self.inner.total_updates.store(0, Ordering::Relaxed);
        self.inner.total_deletes.store(0, Ordering::Relaxed);
        self.inner.hnsw_comparisons.store(0, Ordering::Relaxed);
        self.inner.hnsw_node_visits.store(0, Ordering::Relaxed);
    }

    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> String {
        let snapshot = self.snapshot();

        format!(
            "# HELP vecstore_queries_total Total number of queries executed\n\
             # TYPE vecstore_queries_total counter\n\
             vecstore_queries_total {}\n\
             \n\
             # HELP vecstore_query_errors_total Total number of query errors\n\
             # TYPE vecstore_query_errors_total counter\n\
             vecstore_query_errors_total {}\n\
             \n\
             # HELP vecstore_query_latency_microseconds Average query latency in microseconds\n\
             # TYPE vecstore_query_latency_microseconds gauge\n\
             vecstore_query_latency_microseconds {:.2}\n\
             \n\
             # HELP vecstore_cache_hit_rate Cache hit rate (0.0 to 1.0)\n\
             # TYPE vecstore_cache_hit_rate gauge\n\
             vecstore_cache_hit_rate {:.4}\n\
             \n\
             # HELP vecstore_cache_hits_total Total cache hits\n\
             # TYPE vecstore_cache_hits_total counter\n\
             vecstore_cache_hits_total {}\n\
             \n\
             # HELP vecstore_cache_misses_total Total cache misses\n\
             # TYPE vecstore_cache_misses_total counter\n\
             vecstore_cache_misses_total {}\n\
             \n\
             # HELP vecstore_inserts_total Total insert operations\n\
             # TYPE vecstore_inserts_total counter\n\
             vecstore_inserts_total {}\n\
             \n\
             # HELP vecstore_updates_total Total update operations\n\
             # TYPE vecstore_updates_total counter\n\
             vecstore_updates_total {}\n\
             \n\
             # HELP vecstore_deletes_total Total delete operations\n\
             # TYPE vecstore_deletes_total counter\n\
             vecstore_deletes_total {}\n\
             \n\
             # HELP vecstore_queries_per_second Current query throughput\n\
             # TYPE vecstore_queries_per_second gauge\n\
             vecstore_queries_per_second {:.2}\n\
             \n\
             # HELP vecstore_uptime_seconds Uptime in seconds\n\
             # TYPE vecstore_uptime_seconds counter\n\
             vecstore_uptime_seconds {:.2}\n\
             \n\
             # HELP vecstore_hnsw_comparisons_total Total HNSW distance comparisons\n\
             # TYPE vecstore_hnsw_comparisons_total counter\n\
             vecstore_hnsw_comparisons_total {}\n\
             \n\
             # HELP vecstore_hnsw_node_visits_total Total HNSW node visits\n\
             # TYPE vecstore_hnsw_node_visits_total counter\n\
             vecstore_hnsw_node_visits_total {}\n",
            snapshot.total_queries,
            snapshot.query_errors,
            snapshot.avg_query_latency_micros,
            snapshot.cache_hit_rate,
            snapshot.cache_hits,
            snapshot.cache_misses,
            snapshot.total_inserts,
            snapshot.total_updates,
            snapshot.total_deletes,
            snapshot.queries_per_sec,
            snapshot.uptime_secs,
            snapshot.hnsw_comparisons,
            snapshot.hnsw_node_visits,
        )
    }
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub total_queries: u64,
    pub query_errors: u64,
    pub avg_query_latency_micros: f64,
    pub cache_hit_rate: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_inserts: u64,
    pub total_updates: u64,
    pub total_deletes: u64,
    pub hnsw_comparisons: u64,
    pub hnsw_node_visits: u64,
    pub queries_per_sec: f64,
    pub uptime_secs: f64,
}

impl MetricsSnapshot {
    /// Print a human-readable metrics summary
    pub fn print_summary(&self) {
        println!("=== VecStore Metrics ===");
        println!("Uptime: {:.2}s", self.uptime_secs);
        println!();

        println!("Queries:");
        println!("  Total: {}", self.total_queries);
        println!("  Errors: {}", self.query_errors);
        println!("  Throughput: {:.2} queries/sec", self.queries_per_sec);
        println!(
            "  Avg Latency: {:.2}ms",
            self.avg_query_latency_micros / 1000.0
        );
        println!();

        println!("Cache:");
        println!("  Hit Rate: {:.2}%", self.cache_hit_rate * 100.0);
        println!("  Hits: {}", self.cache_hits);
        println!("  Misses: {}", self.cache_misses);
        println!();

        println!("Operations:");
        println!("  Inserts: {}", self.total_inserts);
        println!("  Updates: {}", self.total_updates);
        println!("  Deletes: {}", self.total_deletes);
        println!();

        if self.total_queries > 0 {
            let avg_comparisons = self.hnsw_comparisons as f64 / self.total_queries as f64;
            let avg_visits = self.hnsw_node_visits as f64 / self.total_queries as f64;

            println!("HNSW Graph:");
            println!("  Total Comparisons: {}", self.hnsw_comparisons);
            println!("  Total Node Visits: {}", self.hnsw_node_visits);
            println!("  Avg Comparisons/Query: {:.1}", avg_comparisons);
            println!("  Avg Visits/Query: {:.1}", avg_visits);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_metrics_creation() {
        let config = MetricsConfig::default();
        let metrics = Metrics::new(config);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_queries, 0);
        assert_eq!(snapshot.cache_hits, 0);
    }

    #[test]
    fn test_record_query() {
        let metrics = Metrics::new(MetricsConfig::default());

        metrics.record_query(Duration::from_millis(10), true);
        metrics.record_query(Duration::from_millis(20), false);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_queries, 2);
        assert_eq!(snapshot.cache_hits, 1);
        assert_eq!(snapshot.cache_misses, 1);
        assert_eq!(snapshot.cache_hit_rate, 0.5);
    }

    #[test]
    fn test_record_operations() {
        let metrics = Metrics::new(MetricsConfig::default());

        metrics.record_insert();
        metrics.record_insert();
        metrics.record_update();
        metrics.record_delete();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_inserts, 2);
        assert_eq!(snapshot.total_updates, 1);
        assert_eq!(snapshot.total_deletes, 1);
    }

    #[test]
    fn test_avg_latency() {
        let metrics = Metrics::new(MetricsConfig::default());

        metrics.record_query(Duration::from_millis(10), false);
        metrics.record_query(Duration::from_millis(20), false);
        metrics.record_query(Duration::from_millis(30), false);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_queries, 3);
        assert!((snapshot.avg_query_latency_micros - 20000.0).abs() < 1.0);
    }

    #[test]
    fn test_throughput_calculation() {
        let metrics = Metrics::new(MetricsConfig::default());

        sleep(Duration::from_millis(100));

        for _ in 0..10 {
            metrics.record_query(Duration::from_millis(1), false);
        }

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_queries, 10);
        assert!(snapshot.queries_per_sec > 0.0);
        assert!(snapshot.uptime_secs >= 0.1);
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = Metrics::new(MetricsConfig::default());

        metrics.record_query(Duration::from_millis(10), true);
        metrics.record_insert();

        let snapshot1 = metrics.snapshot();
        assert_eq!(snapshot1.total_queries, 1);
        assert_eq!(snapshot1.total_inserts, 1);

        metrics.reset();

        let snapshot2 = metrics.snapshot();
        assert_eq!(snapshot2.total_queries, 0);
        assert_eq!(snapshot2.total_inserts, 0);
    }

    #[test]
    fn test_prometheus_export() {
        let metrics = Metrics::new(MetricsConfig::default());

        metrics.record_query(Duration::from_millis(10), true);
        metrics.record_insert();

        let prometheus_output = metrics.export_prometheus();

        assert!(prometheus_output.contains("vecstore_queries_total 1"));
        assert!(prometheus_output.contains("vecstore_cache_hits_total 1"));
        assert!(prometheus_output.contains("vecstore_inserts_total 1"));
        assert!(prometheus_output.contains("# HELP"));
        assert!(prometheus_output.contains("# TYPE"));
    }

    #[test]
    fn test_disabled_metrics() {
        let config = MetricsConfig {
            enabled: false,
            ..Default::default()
        };
        let metrics = Metrics::new(config);

        metrics.record_query(Duration::from_millis(10), true);
        metrics.record_insert();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_queries, 0);
        assert_eq!(snapshot.total_inserts, 0);
    }

    #[test]
    fn test_hnsw_stats() {
        let metrics = Metrics::new(MetricsConfig::default());

        metrics.record_hnsw_stats(100, 50);
        metrics.record_hnsw_stats(200, 75);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.hnsw_comparisons, 300);
        assert_eq!(snapshot.hnsw_node_visits, 125);
    }
}
