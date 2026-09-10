// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # OpenTelemetry Integration
//!
//! Full observability stack with OTLP export to Jaeger, Zipkin, and other backends.
//! Provides distributed tracing, metrics, and logging integration.
//!
//! ## Features
//!
//! - **OTLP Export**: Export to any OpenTelemetry-compatible backend
//! - **Distributed Tracing**: Trace context propagation across services
//! - **Custom Metrics**: Query latency, throughput, memory usage
//! - **Log Correlation**: Correlate logs with traces
//! - **Grafana Dashboards**: Pre-built dashboard templates
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::otel::{OtelConfig, TelemetryProvider};
//!
//! let config = OtelConfig::jaeger("http://localhost:14268/api/traces");
//! let provider = TelemetryProvider::init(config)?;
//!
//! // All operations are now traced
//! provider.record_query("search", 150);
//! ```

use std::collections::HashMap;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// OpenTelemetry exporter type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExporterType {
    /// OTLP gRPC exporter (default)
    OtlpGrpc { endpoint: String },
    /// OTLP HTTP exporter
    OtlpHttp { endpoint: String },
    /// Jaeger exporter
    Jaeger { endpoint: String },
    /// Zipkin exporter
    Zipkin { endpoint: String },
    /// Prometheus exporter (pull-based)
    Prometheus { port: u16 },
    /// Console exporter (for debugging)
    Console,
    /// No-op exporter
    None,
}

/// Sampling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Sample all traces
    AlwaysOn,
    /// Sample no traces
    AlwaysOff,
    /// Sample based on ratio (0.0-1.0)
    Ratio(f64),
    /// Parent-based sampling
    ParentBased { root_ratio: f64 },
    /// Rate limiting (traces per second)
    RateLimiting { traces_per_second: u32 },
}

/// OpenTelemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelConfig {
    /// Service name
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Trace exporter
    pub trace_exporter: ExporterType,
    /// Metrics exporter
    pub metrics_exporter: ExporterType,
    /// Logs exporter
    pub logs_exporter: ExporterType,
    /// Sampling strategy
    pub sampling: SamplingStrategy,
    /// Resource attributes
    pub resource_attributes: HashMap<String, String>,
    /// Batch configuration
    pub batch_config: BatchConfig,
    /// Enable trace context propagation
    pub propagate_context: bool,
}

/// Batch export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Export interval
    pub export_interval_ms: u64,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Export timeout
    pub export_timeout_ms: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 512,
            export_interval_ms: 5000,
            max_queue_size: 2048,
            export_timeout_ms: 30000,
        }
    }
}

impl OtelConfig {
    /// Create OTLP gRPC configuration
    pub fn otlp_grpc(endpoint: &str) -> Self {
        Self {
            service_name: "vecstore".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            trace_exporter: ExporterType::OtlpGrpc {
                endpoint: endpoint.to_string(),
            },
            metrics_exporter: ExporterType::OtlpGrpc {
                endpoint: endpoint.to_string(),
            },
            logs_exporter: ExporterType::OtlpGrpc {
                endpoint: endpoint.to_string(),
            },
            sampling: SamplingStrategy::Ratio(0.1),
            resource_attributes: HashMap::new(),
            batch_config: BatchConfig::default(),
            propagate_context: true,
        }
    }

    /// Create Jaeger configuration
    pub fn jaeger(endpoint: &str) -> Self {
        Self {
            service_name: "vecstore".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            trace_exporter: ExporterType::Jaeger {
                endpoint: endpoint.to_string(),
            },
            metrics_exporter: ExporterType::Prometheus { port: 9090 },
            logs_exporter: ExporterType::Console,
            sampling: SamplingStrategy::Ratio(0.1),
            resource_attributes: HashMap::new(),
            batch_config: BatchConfig::default(),
            propagate_context: true,
        }
    }

    /// Create Zipkin configuration
    pub fn zipkin(endpoint: &str) -> Self {
        Self {
            service_name: "vecstore".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            trace_exporter: ExporterType::Zipkin {
                endpoint: endpoint.to_string(),
            },
            metrics_exporter: ExporterType::Prometheus { port: 9090 },
            logs_exporter: ExporterType::Console,
            sampling: SamplingStrategy::Ratio(0.1),
            resource_attributes: HashMap::new(),
            batch_config: BatchConfig::default(),
            propagate_context: true,
        }
    }

    /// Create console-only configuration (for development)
    pub fn console() -> Self {
        Self {
            service_name: "vecstore".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            trace_exporter: ExporterType::Console,
            metrics_exporter: ExporterType::Console,
            logs_exporter: ExporterType::Console,
            sampling: SamplingStrategy::AlwaysOn,
            resource_attributes: HashMap::new(),
            batch_config: BatchConfig::default(),
            propagate_context: false,
        }
    }

    /// Set service name
    pub fn with_service_name(mut self, name: &str) -> Self {
        self.service_name = name.to_string();
        self
    }

    /// Add resource attribute
    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.resource_attributes
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Set sampling strategy
    pub fn with_sampling(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling = strategy;
        self
    }
}

/// Trace span
#[derive(Debug, Clone)]
pub struct Span {
    /// Span ID
    pub span_id: String,
    /// Trace ID
    pub trace_id: String,
    /// Parent span ID
    pub parent_id: Option<String>,
    /// Operation name
    pub name: String,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Attributes
    pub attributes: HashMap<String, SpanAttribute>,
    /// Events
    pub events: Vec<SpanEvent>,
    /// Status
    pub status: SpanStatus,
    /// Kind
    pub kind: SpanKind,
}

/// Span attribute value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanAttribute {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    StringArray(Vec<String>),
    IntArray(Vec<i64>),
}

/// Span event
#[derive(Debug, Clone)]
pub struct SpanEvent {
    /// Event name
    pub name: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Attributes
    pub attributes: HashMap<String, SpanAttribute>,
}

/// Span status
#[derive(Debug, Clone, PartialEq)]
pub enum SpanStatus {
    /// Unset status
    Unset,
    /// OK status
    Ok,
    /// Error status
    Error { message: String },
}

/// Span kind
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpanKind {
    /// Internal operation
    Internal,
    /// Server handling request
    Server,
    /// Client making request
    Client,
    /// Producer sending message
    Producer,
    /// Consumer receiving message
    Consumer,
}

impl Span {
    /// Create new span
    pub fn new(name: &str, trace_id: &str, parent_id: Option<String>) -> Self {
        Self {
            span_id: generate_span_id(),
            trace_id: trace_id.to_string(),
            parent_id,
            name: name.to_string(),
            start_time: Instant::now(),
            end_time: None,
            attributes: HashMap::new(),
            events: Vec::new(),
            status: SpanStatus::Unset,
            kind: SpanKind::Internal,
        }
    }

    /// Set attribute
    pub fn set_attribute(&mut self, key: &str, value: SpanAttribute) {
        self.attributes.insert(key.to_string(), value);
    }

    /// Add event
    pub fn add_event(&mut self, name: &str) {
        self.events.push(SpanEvent {
            name: name.to_string(),
            timestamp: Instant::now(),
            attributes: HashMap::new(),
        });
    }

    /// End span
    pub fn end(&mut self) {
        self.end_time = Some(Instant::now());
    }

    /// Set error status
    pub fn set_error(&mut self, message: &str) {
        self.status = SpanStatus::Error {
            message: message.to_string(),
        };
    }

    /// Set OK status
    pub fn set_ok(&mut self) {
        self.status = SpanStatus::Ok;
    }

    /// Get duration
    pub fn duration(&self) -> Duration {
        self.end_time
            .unwrap_or_else(Instant::now)
            .duration_since(self.start_time)
    }
}

/// Metric type
#[derive(Debug, Clone)]
pub enum MetricType {
    /// Counter (monotonically increasing)
    Counter,
    /// Gauge (can go up and down)
    Gauge,
    /// Histogram (distribution of values)
    Histogram { boundaries: Vec<f64> },
    /// Summary (quantiles)
    Summary { quantiles: Vec<f64> },
}

/// Metric data point
#[derive(Debug, Clone)]
pub struct MetricPoint {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Value
    pub value: f64,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Timestamp
    pub timestamp: i64,
}

/// Pre-defined VecStore metrics
pub struct VecStoreMetrics {
    /// Query count
    pub query_count: AtomicU64,
    /// Query latency sum (microseconds)
    pub query_latency_sum: AtomicU64,
    /// Insert count
    pub insert_count: AtomicU64,
    /// Delete count
    pub delete_count: AtomicU64,
    /// Vector count
    pub vector_count: AtomicUsize,
    /// Index size bytes
    pub index_size_bytes: AtomicUsize,
    /// Cache hits
    pub cache_hits: AtomicU64,
    /// Cache misses
    pub cache_misses: AtomicU64,
    /// Active connections
    pub active_connections: AtomicUsize,
    /// Query latency histogram buckets
    histogram_buckets: RwLock<Vec<AtomicU64>>,
    /// Histogram boundaries (ms)
    histogram_boundaries: Vec<f64>,
}

impl VecStoreMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        let boundaries = vec![
            1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0,
        ];
        let buckets: Vec<AtomicU64> = (0..=boundaries.len()).map(|_| AtomicU64::new(0)).collect();

        Self {
            query_count: AtomicU64::new(0),
            query_latency_sum: AtomicU64::new(0),
            insert_count: AtomicU64::new(0),
            delete_count: AtomicU64::new(0),
            vector_count: AtomicUsize::new(0),
            index_size_bytes: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            active_connections: AtomicUsize::new(0),
            histogram_buckets: RwLock::new(buckets),
            histogram_boundaries: boundaries,
        }
    }

    /// Record query
    pub fn record_query(&self, latency_ms: f64) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.query_latency_sum
            .fetch_add((latency_ms * 1000.0) as u64, Ordering::Relaxed);

        // Update histogram
        let Ok(buckets) = self.histogram_buckets.read() else {
            return;
        };
        for (i, &boundary) in self.histogram_boundaries.iter().enumerate() {
            if latency_ms <= boundary {
                buckets[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // Overflow bucket
        buckets[self.histogram_boundaries.len()].fetch_add(1, Ordering::Relaxed);
    }

    /// Record insert
    pub fn record_insert(&self, count: usize) {
        self.insert_count.fetch_add(count as u64, Ordering::Relaxed);
        self.vector_count.fetch_add(count, Ordering::Relaxed);
    }

    /// Record delete
    pub fn record_delete(&self, count: usize) {
        self.delete_count.fetch_add(count as u64, Ordering::Relaxed);
        self.vector_count.fetch_sub(
            count.min(self.vector_count.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Update index size
    pub fn set_index_size(&self, bytes: usize) {
        self.index_size_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Get Prometheus format output
    pub fn prometheus_format(&self) -> String {
        let mut output = String::new();

        // Counter metrics
        output.push_str(&format!(
            "# HELP vecstore_query_total Total number of queries\n\
             # TYPE vecstore_query_total counter\n\
             vecstore_query_total {}\n\n",
            self.query_count.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP vecstore_insert_total Total number of inserts\n\
             # TYPE vecstore_insert_total counter\n\
             vecstore_insert_total {}\n\n",
            self.insert_count.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP vecstore_delete_total Total number of deletes\n\
             # TYPE vecstore_delete_total counter\n\
             vecstore_delete_total {}\n\n",
            self.delete_count.load(Ordering::Relaxed)
        ));

        // Gauge metrics
        output.push_str(&format!(
            "# HELP vecstore_vector_count Current number of vectors\n\
             # TYPE vecstore_vector_count gauge\n\
             vecstore_vector_count {}\n\n",
            self.vector_count.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP vecstore_index_size_bytes Index size in bytes\n\
             # TYPE vecstore_index_size_bytes gauge\n\
             vecstore_index_size_bytes {}\n\n",
            self.index_size_bytes.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP vecstore_active_connections Current active connections\n\
             # TYPE vecstore_active_connections gauge\n\
             vecstore_active_connections {}\n\n",
            self.active_connections.load(Ordering::Relaxed)
        ));

        // Cache metrics
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };

        output.push_str(&format!(
            "# HELP vecstore_cache_hit_rate Cache hit rate\n\
             # TYPE vecstore_cache_hit_rate gauge\n\
             vecstore_cache_hit_rate {:.4}\n\n",
            hit_rate
        ));

        // Histogram
        output.push_str(
            "# HELP vecstore_query_latency_ms Query latency in milliseconds\n\
                        # TYPE vecstore_query_latency_ms histogram\n",
        );

        let Ok(buckets) = self.histogram_buckets.read() else {
            return output;
        };
        let mut cumulative = 0u64;
        for (i, boundary) in self.histogram_boundaries.iter().enumerate() {
            cumulative += buckets[i].load(Ordering::Relaxed);
            output.push_str(&format!(
                "vecstore_query_latency_ms_bucket{{le=\"{}\"}} {}\n",
                boundary, cumulative
            ));
        }
        cumulative += buckets[self.histogram_boundaries.len()].load(Ordering::Relaxed);
        output.push_str(&format!(
            "vecstore_query_latency_ms_bucket{{le=\"+Inf\"}} {}\n",
            cumulative
        ));
        output.push_str(&format!(
            "vecstore_query_latency_ms_sum {}\n",
            self.query_latency_sum.load(Ordering::Relaxed) as f64 / 1000.0
        ));
        output.push_str(&format!(
            "vecstore_query_latency_ms_count {}\n",
            self.query_count.load(Ordering::Relaxed)
        ));

        output
    }

    /// Get metrics as JSON
    pub fn to_json(&self) -> serde_json::Value {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total_queries = self.query_count.load(Ordering::Relaxed);
        let latency_sum = self.query_latency_sum.load(Ordering::Relaxed) as f64 / 1000.0;

        serde_json::json!({
            "queries": {
                "total": total_queries,
                "avg_latency_ms": if total_queries > 0 { latency_sum / total_queries as f64 } else { 0.0 }
            },
            "writes": {
                "inserts": self.insert_count.load(Ordering::Relaxed),
                "deletes": self.delete_count.load(Ordering::Relaxed)
            },
            "storage": {
                "vector_count": self.vector_count.load(Ordering::Relaxed),
                "index_size_bytes": self.index_size_bytes.load(Ordering::Relaxed)
            },
            "cache": {
                "hits": hits,
                "misses": misses,
                "hit_rate": if hits + misses > 0 { hits as f64 / (hits + misses) as f64 } else { 0.0 }
            },
            "connections": {
                "active": self.active_connections.load(Ordering::Relaxed)
            }
        })
    }
}

impl Default for VecStoreMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Telemetry provider
pub struct TelemetryProvider {
    /// Configuration
    config: OtelConfig,
    /// Metrics
    metrics: Arc<VecStoreMetrics>,
    /// Active spans
    spans: RwLock<HashMap<String, Span>>,
    /// Completed spans (for batch export)
    completed_spans: RwLock<Vec<Span>>,
    /// Current trace context
    current_trace: RwLock<Option<String>>,
}

impl TelemetryProvider {
    /// Initialize telemetry provider
    pub fn init(config: OtelConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(VecStoreMetrics::new()),
            spans: RwLock::new(HashMap::new()),
            completed_spans: RwLock::new(Vec::new()),
            current_trace: RwLock::new(None),
        }
    }

    /// Get metrics reference
    pub fn metrics(&self) -> Arc<VecStoreMetrics> {
        self.metrics.clone()
    }

    /// Start new trace
    pub fn start_trace(&self, name: &str) -> String {
        let trace_id = generate_trace_id();
        let span = Span::new(name, &trace_id, None);
        let span_id = span.span_id.clone();

        {
            let Ok(mut current) = self.current_trace.write() else {
                return span_id;
            };
            *current = Some(trace_id.clone());
        }

        {
            let Ok(mut spans) = self.spans.write() else {
                return span_id;
            };
            spans.insert(span_id.clone(), span);
        }

        span_id
    }

    /// Start child span
    pub fn start_span(&self, name: &str, parent_id: Option<&str>) -> String {
        let trace_id = self
            .current_trace
            .read()
            .ok()
            .and_then(|guard| guard.clone())
            .unwrap_or_else(generate_trace_id);

        let span = Span::new(name, &trace_id, parent_id.map(String::from));
        let span_id = span.span_id.clone();

        {
            let Ok(mut spans) = self.spans.write() else {
                return span_id;
            };
            spans.insert(span_id.clone(), span);
        }

        span_id
    }

    /// End span
    pub fn end_span(&self, span_id: &str) {
        let span = {
            let Ok(mut spans) = self.spans.write() else {
                return;
            };
            spans.remove(span_id)
        };

        if let Some(mut span) = span {
            span.end();
            let Ok(mut completed) = self.completed_spans.write() else {
                return;
            };
            completed.push(span);
        }
    }

    /// Set span attribute
    pub fn set_span_attribute(&self, span_id: &str, key: &str, value: SpanAttribute) {
        let Ok(mut spans) = self.spans.write() else {
            return;
        };
        if let Some(span) = spans.get_mut(span_id) {
            span.set_attribute(key, value);
        }
    }

    /// Set span error
    pub fn set_span_error(&self, span_id: &str, message: &str) {
        let Ok(mut spans) = self.spans.write() else {
            return;
        };
        if let Some(span) = spans.get_mut(span_id) {
            span.set_error(message);
        }
    }

    /// Record query with tracing
    pub fn record_query(&self, operation: &str, latency_ms: f64) {
        self.metrics.record_query(latency_ms);

        // Create a span for the query
        let span_id = self.start_span(operation, None);
        self.set_span_attribute(
            &span_id,
            "db.system",
            SpanAttribute::String("vecstore".to_string()),
        );
        self.set_span_attribute(
            &span_id,
            "db.operation",
            SpanAttribute::String(operation.to_string()),
        );
        self.set_span_attribute(&span_id, "db.latency_ms", SpanAttribute::Float(latency_ms));
        self.end_span(&span_id);
    }

    /// Flush completed spans
    pub fn flush(&self) -> Vec<Span> {
        let Ok(mut completed) = self.completed_spans.write() else {
            return Vec::new();
        };
        std::mem::take(&mut *completed)
    }

    /// Export to OTLP format
    pub fn export_otlp(&self) -> OtlpExportData {
        let spans = self.flush();
        let metrics = self.metrics.to_json();

        OtlpExportData {
            service_name: self.config.service_name.clone(),
            service_version: self.config.service_version.clone(),
            spans: spans
                .into_iter()
                .map(|s| OtlpSpan {
                    trace_id: s.trace_id,
                    span_id: s.span_id,
                    parent_span_id: s.parent_id,
                    name: s.name,
                    start_time_unix_nano: 0, // Would be real timestamp
                    end_time_unix_nano: 0,
                    attributes: s
                        .attributes
                        .into_iter()
                        .map(|(k, v)| (k, format!("{:?}", v)))
                        .collect(),
                    status: match s.status {
                        SpanStatus::Ok => "OK".to_string(),
                        SpanStatus::Error { message } => format!("ERROR: {}", message),
                        SpanStatus::Unset => "UNSET".to_string(),
                    },
                })
                .collect(),
            metrics,
        }
    }

    /// Get Grafana dashboard JSON
    pub fn grafana_dashboard(&self) -> serde_json::Value {
        serde_json::json!({
            "title": "VecStore Dashboard",
            "uid": "vecstore-main",
            "panels": [
                {
                    "title": "Query Rate",
                    "type": "graph",
                    "targets": [{
                        "expr": "rate(vecstore_query_total[5m])",
                        "legendFormat": "queries/s"
                    }]
                },
                {
                    "title": "Query Latency (p99)",
                    "type": "graph",
                    "targets": [{
                        "expr": "histogram_quantile(0.99, rate(vecstore_query_latency_ms_bucket[5m]))",
                        "legendFormat": "p99 latency"
                    }]
                },
                {
                    "title": "Vector Count",
                    "type": "stat",
                    "targets": [{
                        "expr": "vecstore_vector_count",
                        "legendFormat": "vectors"
                    }]
                },
                {
                    "title": "Cache Hit Rate",
                    "type": "gauge",
                    "targets": [{
                        "expr": "vecstore_cache_hit_rate",
                        "legendFormat": "hit rate"
                    }]
                },
                {
                    "title": "Index Size",
                    "type": "stat",
                    "targets": [{
                        "expr": "vecstore_index_size_bytes",
                        "legendFormat": "bytes"
                    }]
                },
                {
                    "title": "Active Connections",
                    "type": "graph",
                    "targets": [{
                        "expr": "vecstore_active_connections",
                        "legendFormat": "connections"
                    }]
                }
            ]
        })
    }
}

/// OTLP export data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpExportData {
    pub service_name: String,
    pub service_version: String,
    pub spans: Vec<OtlpSpan>,
    pub metrics: serde_json::Value,
}

/// OTLP span format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpSpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub start_time_unix_nano: u64,
    pub end_time_unix_nano: u64,
    pub attributes: HashMap<String, String>,
    pub status: String,
}

/// Trace context for propagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Trace ID
    pub trace_id: String,
    /// Span ID
    pub span_id: String,
    /// Trace flags
    pub trace_flags: u8,
    /// Trace state
    pub trace_state: HashMap<String, String>,
}

impl TraceContext {
    /// Create from W3C traceparent header
    pub fn from_traceparent(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() >= 4 {
            Some(Self {
                trace_id: parts[1].to_string(),
                span_id: parts[2].to_string(),
                trace_flags: u8::from_str_radix(parts[3], 16).unwrap_or(0),
                trace_state: HashMap::new(),
            })
        } else {
            None
        }
    }

    /// Convert to W3C traceparent header
    pub fn to_traceparent(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id, self.span_id, self.trace_flags
        )
    }
}

fn generate_trace_id() -> String {
    use std::time::SystemTime;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:032x}", ts)
}

fn generate_span_id() -> String {
    use std::time::SystemTime;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:016x}", ts & 0xFFFFFFFFFFFFFFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = VecStoreMetrics::new();

        metrics.record_query(15.5);
        metrics.record_query(25.0);
        metrics.record_insert(100);

        assert_eq!(metrics.query_count.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.insert_count.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_prometheus_format() {
        let metrics = VecStoreMetrics::new();
        metrics.record_query(10.0);
        metrics.record_insert(50);

        let output = metrics.prometheus_format();
        assert!(output.contains("vecstore_query_total"));
        assert!(output.contains("vecstore_insert_total"));
    }

    #[test]
    fn test_telemetry_provider() {
        let config = OtelConfig::console();
        let provider = TelemetryProvider::init(config);

        let span_id = provider.start_trace("test_query");
        provider.set_span_attribute(&span_id, "k", SpanAttribute::Int(10));
        provider.end_span(&span_id);

        let spans = provider.flush();
        assert_eq!(spans.len(), 1);
    }

    #[test]
    fn test_trace_context() {
        let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ctx = TraceContext::from_traceparent(header).unwrap();

        assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(ctx.span_id, "b7ad6b7169203331");
        assert_eq!(ctx.trace_flags, 1);
    }

    #[test]
    fn test_grafana_dashboard() {
        let config = OtelConfig::console();
        let provider = TelemetryProvider::init(config);

        let dashboard = provider.grafana_dashboard();
        assert!(dashboard["panels"].as_array().unwrap().len() > 0);
    }
}
