//! Observability and metrics for production monitoring
//!
//! This module provides:
//! - Prometheus metrics for query latency, throughput, cache hit rate
//! - Optional integration with the `prometheus` crate for embedded HTTP endpoints
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
//!
//! ## Prometheus Integration (Optional)
//!
//! When the `prometheus` feature is enabled (via server mode), you can expose metrics
//! at an HTTP endpoint for Prometheus scraping:
//!
//! ```ignore
//! use vecstore::metrics::PrometheusMetrics;
//!
//! let prom = PrometheusMetrics::new();
//! prom.record_query("vector_search", 0.005, 10);
//!
//! // Expose at /metrics endpoint
//! let metrics_text = prom.encode()?;
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

// ============================================================================
// Embedded Metrics Registry
// ============================================================================

/// A simple embedded metrics registry for applications that want
/// Prometheus-compatible metrics without running a full server.
///
/// This allows embedding vecstore in applications while still
/// exposing metrics via a simple HTTP endpoint or pushgateway.
#[derive(Clone)]
pub struct EmbeddedMetrics {
    inner: Arc<EmbeddedMetricsInner>,
}

struct EmbeddedMetricsInner {
    /// Core metrics
    core: Metrics,

    /// Latency histogram buckets (in milliseconds)
    latency_buckets: Vec<f64>,

    /// Latency counts per bucket
    latency_bucket_counts: Vec<AtomicU64>,

    /// Quantization metrics
    quantized_vectors: AtomicU64,
    quantization_bits: AtomicU64,

    /// Memory metrics
    estimated_memory_bytes: AtomicU64,

    /// Index-specific metrics
    hnsw_layers: AtomicU64,
    hnsw_connections_per_layer: AtomicU64,

    /// Custom labels for this instance
    labels: std::sync::RwLock<std::collections::HashMap<String, String>>,
}

impl EmbeddedMetrics {
    /// Create a new embedded metrics registry
    pub fn new() -> Self {
        Self::with_buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0])
    }

    /// Create with custom latency buckets (in milliseconds)
    pub fn with_buckets(buckets: Vec<f64>) -> Self {
        let bucket_counts: Vec<AtomicU64> = buckets.iter().map(|_| AtomicU64::new(0)).collect();

        Self {
            inner: Arc::new(EmbeddedMetricsInner {
                core: Metrics::new(MetricsConfig::default()),
                latency_buckets: buckets,
                latency_bucket_counts: bucket_counts,
                quantized_vectors: AtomicU64::new(0),
                quantization_bits: AtomicU64::new(0),
                estimated_memory_bytes: AtomicU64::new(0),
                hnsw_layers: AtomicU64::new(0),
                hnsw_connections_per_layer: AtomicU64::new(0),
                labels: std::sync::RwLock::new(std::collections::HashMap::new()),
            }),
        }
    }

    /// Add a custom label to all metrics
    pub fn add_label(&self, key: impl Into<String>, value: impl Into<String>) {
        if let Ok(mut labels) = self.inner.labels.write() {
            labels.insert(key.into(), value.into());
        }
    }

    /// Record a query with latency in milliseconds
    pub fn record_query_ms(&self, latency_ms: f64, cache_hit: bool) {
        self.inner.core.record_query(
            Duration::from_micros((latency_ms * 1000.0) as u64),
            cache_hit,
        );

        // Update histogram
        for (i, &bucket) in self.inner.latency_buckets.iter().enumerate() {
            if latency_ms <= bucket {
                self.inner.latency_bucket_counts[i].fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    /// Record HNSW stats
    pub fn record_hnsw_stats(&self, comparisons: u64, node_visits: u64) {
        self.inner.core.record_hnsw_stats(comparisons, node_visits);
    }

    /// Set the estimated memory usage
    pub fn set_memory_bytes(&self, bytes: u64) {
        self.inner.estimated_memory_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Set HNSW index stats
    pub fn set_hnsw_stats(&self, layers: u64, connections_per_layer: u64) {
        self.inner.hnsw_layers.store(layers, Ordering::Relaxed);
        self.inner.hnsw_connections_per_layer.store(connections_per_layer, Ordering::Relaxed);
    }

    /// Set quantization stats
    pub fn set_quantization_stats(&self, quantized_count: u64, bits: u64) {
        self.inner.quantized_vectors.store(quantized_count, Ordering::Relaxed);
        self.inner.quantization_bits.store(bits, Ordering::Relaxed);
    }

    /// Record insert/update/delete
    pub fn record_insert(&self) {
        self.inner.core.record_insert();
    }

    pub fn record_update(&self) {
        self.inner.core.record_update();
    }

    pub fn record_delete(&self) {
        self.inner.core.record_delete();
    }

    /// Get the core snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        self.inner.core.snapshot()
    }

    /// Export metrics in Prometheus text format with histograms
    pub fn export_prometheus(&self) -> String {
        let snapshot = self.inner.core.snapshot();
        let labels = self.inner.labels.read().ok();
        let label_str = if let Some(ref l) = labels {
            if l.is_empty() {
                String::new()
            } else {
                let pairs: Vec<String> = l.iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect();
                format!("{{{}}}", pairs.join(","))
            }
        } else {
            String::new()
        };

        let mut output = String::new();

        // Core counters
        output.push_str(&format!(
            "# HELP vecstore_queries_total Total number of queries executed\n\
             # TYPE vecstore_queries_total counter\n\
             vecstore_queries_total{} {}\n\n",
            label_str, snapshot.total_queries
        ));

        output.push_str(&format!(
            "# HELP vecstore_query_errors_total Total number of query errors\n\
             # TYPE vecstore_query_errors_total counter\n\
             vecstore_query_errors_total{} {}\n\n",
            label_str, snapshot.query_errors
        ));

        output.push_str(&format!(
            "# HELP vecstore_inserts_total Total insert operations\n\
             # TYPE vecstore_inserts_total counter\n\
             vecstore_inserts_total{} {}\n\n",
            label_str, snapshot.total_inserts
        ));

        output.push_str(&format!(
            "# HELP vecstore_updates_total Total update operations\n\
             # TYPE vecstore_updates_total counter\n\
             vecstore_updates_total{} {}\n\n",
            label_str, snapshot.total_updates
        ));

        output.push_str(&format!(
            "# HELP vecstore_deletes_total Total delete operations\n\
             # TYPE vecstore_deletes_total counter\n\
             vecstore_deletes_total{} {}\n\n",
            label_str, snapshot.total_deletes
        ));

        // Latency histogram
        output.push_str("# HELP vecstore_query_latency_ms Query latency in milliseconds\n");
        output.push_str("# TYPE vecstore_query_latency_ms histogram\n");

        let mut cumulative = 0u64;
        for (i, &bucket) in self.inner.latency_buckets.iter().enumerate() {
            cumulative += self.inner.latency_bucket_counts[i].load(Ordering::Relaxed);
            let bucket_label = if label_str.is_empty() {
                format!("{{le=\"{}\"}}", bucket)
            } else {
                let label_inner = label_str.trim_matches(|c| c == '{' || c == '}');
                format!("{{le=\"{}\",{}}}", bucket, label_inner)
            };
            output.push_str(&format!(
                "vecstore_query_latency_ms_bucket{} {}\n",
                bucket_label, cumulative
            ));
        }

        let inf_label = if label_str.is_empty() {
            "{le=\"+Inf\"}".to_string()
        } else {
            let label_inner = label_str.trim_matches(|c| c == '{' || c == '}');
            format!("{{le=\"+Inf\",{}}}", label_inner)
        };
        output.push_str(&format!(
            "vecstore_query_latency_ms_bucket{} {}\n",
            inf_label, snapshot.total_queries
        ));
        output.push_str(&format!(
            "vecstore_query_latency_ms_sum{} {:.2}\n",
            label_str, snapshot.avg_query_latency_micros * snapshot.total_queries as f64 / 1000.0
        ));
        output.push_str(&format!(
            "vecstore_query_latency_ms_count{} {}\n\n",
            label_str, snapshot.total_queries
        ));

        // Cache metrics
        output.push_str(&format!(
            "# HELP vecstore_cache_hit_rate Cache hit rate (0.0 to 1.0)\n\
             # TYPE vecstore_cache_hit_rate gauge\n\
             vecstore_cache_hit_rate{} {:.4}\n\n",
            label_str, snapshot.cache_hit_rate
        ));

        output.push_str(&format!(
            "# HELP vecstore_cache_hits_total Total cache hits\n\
             # TYPE vecstore_cache_hits_total counter\n\
             vecstore_cache_hits_total{} {}\n\n",
            label_str, snapshot.cache_hits
        ));

        output.push_str(&format!(
            "# HELP vecstore_cache_misses_total Total cache misses\n\
             # TYPE vecstore_cache_misses_total counter\n\
             vecstore_cache_misses_total{} {}\n\n",
            label_str, snapshot.cache_misses
        ));

        // Throughput
        output.push_str(&format!(
            "# HELP vecstore_queries_per_second Current query throughput\n\
             # TYPE vecstore_queries_per_second gauge\n\
             vecstore_queries_per_second{} {:.2}\n\n",
            label_str, snapshot.queries_per_sec
        ));

        // HNSW metrics
        output.push_str(&format!(
            "# HELP vecstore_hnsw_comparisons_total Total HNSW distance comparisons\n\
             # TYPE vecstore_hnsw_comparisons_total counter\n\
             vecstore_hnsw_comparisons_total{} {}\n\n",
            label_str, snapshot.hnsw_comparisons
        ));

        output.push_str(&format!(
            "# HELP vecstore_hnsw_node_visits_total Total HNSW node visits\n\
             # TYPE vecstore_hnsw_node_visits_total counter\n\
             vecstore_hnsw_node_visits_total{} {}\n\n",
            label_str, snapshot.hnsw_node_visits
        ));

        let layers = self.inner.hnsw_layers.load(Ordering::Relaxed);
        let conns = self.inner.hnsw_connections_per_layer.load(Ordering::Relaxed);
        output.push_str(&format!(
            "# HELP vecstore_hnsw_layers Number of HNSW layers\n\
             # TYPE vecstore_hnsw_layers gauge\n\
             vecstore_hnsw_layers{} {}\n\n",
            label_str, layers
        ));

        output.push_str(&format!(
            "# HELP vecstore_hnsw_connections_per_layer Average connections per layer\n\
             # TYPE vecstore_hnsw_connections_per_layer gauge\n\
             vecstore_hnsw_connections_per_layer{} {}\n\n",
            label_str, conns
        ));

        // Memory metrics
        let memory = self.inner.estimated_memory_bytes.load(Ordering::Relaxed);
        output.push_str(&format!(
            "# HELP vecstore_memory_bytes Estimated memory usage in bytes\n\
             # TYPE vecstore_memory_bytes gauge\n\
             vecstore_memory_bytes{} {}\n\n",
            label_str, memory
        ));

        // Quantization metrics
        let quant_count = self.inner.quantized_vectors.load(Ordering::Relaxed);
        let quant_bits = self.inner.quantization_bits.load(Ordering::Relaxed);
        output.push_str(&format!(
            "# HELP vecstore_quantized_vectors Number of quantized vectors\n\
             # TYPE vecstore_quantized_vectors gauge\n\
             vecstore_quantized_vectors{} {}\n\n",
            label_str, quant_count
        ));

        output.push_str(&format!(
            "# HELP vecstore_quantization_bits Bits used for quantization\n\
             # TYPE vecstore_quantization_bits gauge\n\
             vecstore_quantization_bits{} {}\n\n",
            label_str, quant_bits
        ));

        // Uptime
        output.push_str(&format!(
            "# HELP vecstore_uptime_seconds Uptime in seconds\n\
             # TYPE vecstore_uptime_seconds counter\n\
             vecstore_uptime_seconds{} {:.2}\n",
            label_str, snapshot.uptime_secs
        ));

        output
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.inner.core.reset();
        for count in &self.inner.latency_bucket_counts {
            count.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for EmbeddedMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod embedded_tests {
    use super::*;

    #[test]
    fn test_embedded_metrics() {
        let metrics = EmbeddedMetrics::new();

        metrics.record_query_ms(5.0, true);
        metrics.record_query_ms(15.0, false);
        metrics.record_insert();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_queries, 2);
        assert_eq!(snapshot.cache_hits, 1);
        assert_eq!(snapshot.total_inserts, 1);
    }

    #[test]
    fn test_prometheus_histogram() {
        let metrics = EmbeddedMetrics::new();
        metrics.add_label("instance", "test");

        metrics.record_query_ms(3.0, true);
        metrics.record_query_ms(50.0, true);
        metrics.record_query_ms(200.0, false);

        let output = metrics.export_prometheus();
        assert!(output.contains("vecstore_query_latency_ms_bucket"));
        assert!(output.contains("le=\"5\""));
        assert!(output.contains("le=\"50\""));
        assert!(output.contains("instance=\"test\""));
    }

    #[test]
    fn test_custom_labels() {
        let metrics = EmbeddedMetrics::new();
        metrics.add_label("app", "myapp");
        metrics.add_label("env", "production");

        metrics.record_query_ms(10.0, true);

        let output = metrics.export_prometheus();
        assert!(output.contains("app=\"myapp\""));
        assert!(output.contains("env=\"production\""));
    }

    #[test]
    fn test_memory_and_quantization() {
        let metrics = EmbeddedMetrics::new();
        metrics.set_memory_bytes(1_000_000);
        metrics.set_quantization_stats(5000, 8);
        metrics.set_hnsw_stats(4, 16);

        let output = metrics.export_prometheus();
        assert!(output.contains("vecstore_memory_bytes 1000000"));
        assert!(output.contains("vecstore_quantized_vectors 5000"));
        assert!(output.contains("vecstore_quantization_bits 8"));
        assert!(output.contains("vecstore_hnsw_layers 4"));
    }
}
