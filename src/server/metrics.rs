//! Prometheus metrics for VecStore server
//!
//! Exposes metrics at /metrics endpoint for Prometheus scraping.

use lazy_static::lazy_static;
use prometheus::{
    register_counter_vec, register_gauge, register_histogram_vec, CounterVec, Encoder, Gauge,
    HistogramVec, TextEncoder,
};

lazy_static! {
    /// Total number of requests by endpoint
    pub static ref REQUEST_COUNTER: CounterVec = register_counter_vec!(
        "vecstore_requests_total",
        "Total number of requests by endpoint",
        &["endpoint", "method"]
    )
    .unwrap();

    /// Request duration in seconds
    pub static ref REQUEST_DURATION: HistogramVec = register_histogram_vec!(
        "vecstore_request_duration_seconds",
        "Request duration in seconds",
        &["endpoint", "method"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    .unwrap();

    /// Query operations
    pub static ref QUERY_COUNTER: CounterVec = register_counter_vec!(
        "vecstore_queries_total",
        "Total number of query operations",
        &["type"]
    )
    .unwrap();

    /// Query result counts
    pub static ref QUERY_RESULTS: HistogramVec = register_histogram_vec!(
        "vecstore_query_results",
        "Number of results returned per query",
        &["type"],
        vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    )
    .unwrap();

    /// Upsert operations
    pub static ref UPSERT_COUNTER: CounterVec = register_counter_vec!(
        "vecstore_upserts_total",
        "Total number of upsert operations",
        &["batch"]
    )
    .unwrap();

    /// Delete operations
    pub static ref DELETE_COUNTER: CounterVec = register_counter_vec!(
        "vecstore_deletes_total",
        "Total number of delete operations",
        &["type"]
    )
    .unwrap();

    /// Total vectors in database
    pub static ref TOTAL_VECTORS: Gauge = register_gauge!(
        "vecstore_vectors_total",
        "Total number of vectors in the database"
    )
    .unwrap();

    /// Active vectors (not deleted)
    pub static ref ACTIVE_VECTORS: Gauge = register_gauge!(
        "vecstore_vectors_active",
        "Number of active (non-deleted) vectors"
    )
    .unwrap();

    /// Deleted vectors
    pub static ref DELETED_VECTORS: Gauge = register_gauge!(
        "vecstore_vectors_deleted",
        "Number of soft-deleted vectors"
    )
    .unwrap();

    /// Database dimension
    pub static ref DIMENSION: Gauge = register_gauge!(
        "vecstore_dimension",
        "Vector dimension of the database"
    )
    .unwrap();

    /// Errors by type
    pub static ref ERROR_COUNTER: CounterVec = register_counter_vec!(
        "vecstore_errors_total",
        "Total number of errors by type",
        &["error_type"]
    )
    .unwrap();

    /// Cache operations (if semantic cache is enabled)
    pub static ref CACHE_HITS: CounterVec = register_counter_vec!(
        "vecstore_cache_hits_total",
        "Total number of cache hits",
        &["type"]
    )
    .unwrap();

    pub static ref CACHE_MISSES: CounterVec = register_counter_vec!(
        "vecstore_cache_misses_total",
        "Total number of cache misses",
        &["type"]
    )
    .unwrap();

    /// Snapshot operations
    pub static ref SNAPSHOT_COUNTER: CounterVec = register_counter_vec!(
        "vecstore_snapshots_total",
        "Total number of snapshot operations",
        &["operation"]
    )
    .unwrap();

    /// Compact operations
    pub static ref COMPACT_COUNTER: CounterVec = register_counter_vec!(
        "vecstore_compacts_total",
        "Total number of compact operations",
        &["status"]
    )
    .unwrap();

    /// WebSocket connections
    pub static ref WEBSOCKET_CONNECTIONS: Gauge = register_gauge!(
        "vecstore_websocket_connections",
        "Current number of active WebSocket connections"
    )
    .unwrap();
}

/// Encode metrics in Prometheus format
pub fn encode_metrics() -> Result<String, prometheus::Error> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer)?;
    Ok(String::from_utf8(buffer).unwrap())
}

/// Update database statistics
pub fn update_db_stats(total: usize, active: usize, deleted: usize, dimension: usize) {
    TOTAL_VECTORS.set(total as f64);
    ACTIVE_VECTORS.set(active as f64);
    DELETED_VECTORS.set(deleted as f64);
    DIMENSION.set(dimension as f64);
}

/// Record a request
pub fn record_request(endpoint: &str, method: &str, duration: f64) {
    REQUEST_COUNTER.with_label_values(&[endpoint, method]).inc();
    REQUEST_DURATION
        .with_label_values(&[endpoint, method])
        .observe(duration);
}

/// Record a query
pub fn record_query(query_type: &str, result_count: usize, _duration: f64) {
    QUERY_COUNTER.with_label_values(&[query_type]).inc();
    QUERY_RESULTS
        .with_label_values(&[query_type])
        .observe(result_count as f64);
}

/// Record an upsert
pub fn record_upsert(is_batch: bool) {
    let batch_label = if is_batch { "batch" } else { "single" };
    UPSERT_COUNTER.with_label_values(&[batch_label]).inc();
}

/// Record a delete
pub fn record_delete(delete_type: &str) {
    DELETE_COUNTER.with_label_values(&[delete_type]).inc();
}

/// Record an error
pub fn record_error(error_type: &str) {
    ERROR_COUNTER.with_label_values(&[error_type]).inc();
}

/// Record a cache operation
pub fn record_cache_hit(cache_type: &str) {
    CACHE_HITS.with_label_values(&[cache_type]).inc();
}

pub fn record_cache_miss(cache_type: &str) {
    CACHE_MISSES.with_label_values(&[cache_type]).inc();
}

/// Record a snapshot operation
pub fn record_snapshot(operation: &str) {
    SNAPSHOT_COUNTER.with_label_values(&[operation]).inc();
}

/// Record a compact operation
pub fn record_compact(status: &str) {
    COMPACT_COUNTER.with_label_values(&[status]).inc();
}

/// Increment WebSocket connections
pub fn websocket_connected() {
    WEBSOCKET_CONNECTIONS.inc();
}

/// Decrement WebSocket connections
pub fn websocket_disconnected() {
    WEBSOCKET_CONNECTIONS.dec();
}
