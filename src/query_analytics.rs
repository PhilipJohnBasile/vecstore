// Query Analytics Dashboard - Comprehensive analytics for vector search operations
// Query patterns, performance insights, and usage analytics

use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Sample rate for detailed logging (0.0-1.0)
    pub sample_rate: f64,
    /// Retention period for analytics
    pub retention: Duration,
    /// Maximum events to store
    pub max_events: usize,
    /// Enable real-time streaming
    pub realtime_streaming: bool,
    /// Aggregation intervals
    pub aggregation_intervals: Vec<Duration>,
    /// Enable query logging
    pub log_queries: bool,
    /// Privacy mode (hash sensitive data)
    pub privacy_mode: bool,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            sample_rate: 0.1,
            retention: Duration::from_secs(86400 * 7), // 7 days
            max_events: 100000,
            realtime_streaming: false,
            aggregation_intervals: vec![
                Duration::from_secs(60),      // 1 minute
                Duration::from_secs(3600),    // 1 hour
                Duration::from_secs(86400),   // 1 day
            ],
            log_queries: true,
            privacy_mode: false,
        }
    }
}

/// Query analytics engine
pub struct QueryAnalytics {
    config: AnalyticsConfig,
    events: RwLock<VecDeque<QueryEvent>>,
    aggregates: RwLock<HashMap<String, AggregateStats>>,
    patterns: RwLock<PatternAnalyzer>,
    stats: AnalyticsStats,
    timeseries: RwLock<TimeSeriesStore>,
}

/// Query event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEvent {
    /// Event ID
    pub id: u64,
    /// Event type
    pub event_type: EventType,
    /// Timestamp
    pub timestamp: u64,
    /// Collection name
    pub collection: String,
    /// Query latency (ms)
    pub latency_ms: f64,
    /// Number of results
    pub result_count: usize,
    /// Filters applied
    pub filter_count: usize,
    /// Query dimensions
    pub dimensions: usize,
    /// K (top-k parameter)
    pub top_k: usize,
    /// Index type used
    pub index_type: String,
    /// Cache hit
    pub cache_hit: bool,
    /// User/client ID (hashed if privacy mode)
    pub client_id: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    Search,
    Insert,
    Update,
    Delete,
    BatchInsert,
    BatchSearch,
    IndexBuild,
    IndexOptimize,
}

/// Aggregate statistics
#[derive(Debug, Clone, Serialize, Default)]
pub struct AggregateStats {
    pub count: u64,
    pub total_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub total_results: u64,
    pub avg_results: f64,
    pub cache_hits: u64,
    pub cache_hit_rate: f64,
    pub error_count: u64,
    pub error_rate: f64,
}

/// Pattern analyzer for query patterns
struct PatternAnalyzer {
    frequent_patterns: HashMap<String, u64>,
    temporal_patterns: BTreeMap<u64, Vec<String>>,
    dimension_distribution: HashMap<usize, u64>,
    k_distribution: HashMap<usize, u64>,
}

/// Analytics statistics
struct AnalyticsStats {
    total_events: AtomicU64,
    search_events: AtomicU64,
    insert_events: AtomicU64,
    update_events: AtomicU64,
    delete_events: AtomicU64,
    avg_latency_ms: RwLock<f64>,
    started_at: Instant,
}

/// Time series store for trends
struct TimeSeriesStore {
    series: HashMap<String, Vec<TimeSeriesPoint>>,
    granularity: Duration,
}

/// Time series point
#[derive(Debug, Clone, Serialize)]
pub struct TimeSeriesPoint {
    pub timestamp: u64,
    pub value: f64,
    pub count: u64,
}

/// Dashboard data
#[derive(Debug, Clone, Serialize)]
pub struct DashboardData {
    pub summary: DashboardSummary,
    pub recent_queries: Vec<QuerySummary>,
    pub performance_trends: Vec<TrendData>,
    pub top_collections: Vec<CollectionStats>,
    pub error_summary: ErrorSummary,
    pub alerts: Vec<Alert>,
}

/// Dashboard summary
#[derive(Debug, Clone, Serialize)]
pub struct DashboardSummary {
    pub total_queries: u64,
    pub queries_per_second: f64,
    pub avg_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub active_collections: usize,
    pub uptime: Duration,
}

/// Query summary for display
#[derive(Debug, Clone, Serialize)]
pub struct QuerySummary {
    pub timestamp: u64,
    pub collection: String,
    pub latency_ms: f64,
    pub result_count: usize,
    pub event_type: String,
}

/// Trend data for charts
#[derive(Debug, Clone, Serialize)]
pub struct TrendData {
    pub metric_name: String,
    pub data_points: Vec<TimeSeriesPoint>,
    pub trend_direction: TrendDirection,
    pub change_percent: f64,
}

/// Trend direction
#[derive(Debug, Clone, Serialize)]
pub enum TrendDirection {
    Up,
    Down,
    Stable,
}

/// Collection statistics
#[derive(Debug, Clone, Serialize)]
pub struct CollectionStats {
    pub name: String,
    pub query_count: u64,
    pub avg_latency_ms: f64,
    pub total_results: u64,
    pub error_rate: f64,
}

/// Error summary
#[derive(Debug, Clone, Serialize)]
pub struct ErrorSummary {
    pub total_errors: u64,
    pub error_rate: f64,
    pub top_errors: Vec<ErrorInfo>,
}

/// Error information
#[derive(Debug, Clone, Serialize)]
pub struct ErrorInfo {
    pub error_type: String,
    pub count: u64,
    pub last_occurred: u64,
}

/// Alert
#[derive(Debug, Clone, Serialize)]
pub struct Alert {
    pub severity: AlertSeverity,
    pub message: String,
    pub metric: String,
    pub threshold: f64,
    pub current_value: f64,
    pub triggered_at: u64,
}

/// Alert severity
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Report configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub time_range: TimeRange,
    pub metrics: Vec<String>,
    pub group_by: Option<String>,
    pub format: ReportFormat,
}

/// Time range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: u64,
    pub end: u64,
}

/// Report format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Csv,
    Html,
    Prometheus,
}

impl QueryAnalytics {
    /// Create a new analytics engine
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            config,
            events: RwLock::new(VecDeque::new()),
            aggregates: RwLock::new(HashMap::new()),
            patterns: RwLock::new(PatternAnalyzer {
                frequent_patterns: HashMap::new(),
                temporal_patterns: BTreeMap::new(),
                dimension_distribution: HashMap::new(),
                k_distribution: HashMap::new(),
            }),
            stats: AnalyticsStats {
                total_events: AtomicU64::new(0),
                search_events: AtomicU64::new(0),
                insert_events: AtomicU64::new(0),
                update_events: AtomicU64::new(0),
                delete_events: AtomicU64::new(0),
                avg_latency_ms: RwLock::new(0.0),
                started_at: Instant::now(),
            },
            timeseries: RwLock::new(TimeSeriesStore {
                series: HashMap::new(),
                granularity: Duration::from_secs(60),
            }),
        }
    }

    /// Record a query event
    pub fn record(&self, event: QueryEvent) {
        // Sample based on rate
        if !self.should_sample() {
            return;
        }

        let event_id = self.stats.total_events.fetch_add(1, Ordering::Relaxed);

        // Update event type counters
        match event.event_type {
            EventType::Search | EventType::BatchSearch => {
                self.stats.search_events.fetch_add(1, Ordering::Relaxed);
            }
            EventType::Insert | EventType::BatchInsert => {
                self.stats.insert_events.fetch_add(1, Ordering::Relaxed);
            }
            EventType::Update => {
                self.stats.update_events.fetch_add(1, Ordering::Relaxed);
            }
            EventType::Delete => {
                self.stats.delete_events.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        // Update running average latency
        {
            let Ok(mut avg) = self.stats.avg_latency_ms.write() else { return; };
            let count = self.stats.total_events.load(Ordering::Relaxed) as f64;
            *avg = (*avg * (count - 1.0) + event.latency_ms) / count;
        }

        // Store event
        {
            let Ok(mut events) = self.events.write() else { return; };
            if events.len() >= self.config.max_events {
                events.pop_front();
            }
            events.push_back(QueryEvent { id: event_id, ..event.clone() });
        }

        // Update aggregates
        self.update_aggregates(&event);

        // Update patterns
        self.update_patterns(&event);

        // Update time series
        self.update_timeseries(&event);
    }

    fn should_sample(&self) -> bool {
        if self.config.sample_rate >= 1.0 {
            return true;
        }
        if self.config.sample_rate <= 0.0 {
            return false;
        }

        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        Instant::now().hash(&mut hasher);
        (hasher.finish() as f64 / u64::MAX as f64) < self.config.sample_rate
    }

    fn update_aggregates(&self, event: &QueryEvent) {
        let Ok(mut aggregates) = self.aggregates.write() else { return; };

        // Overall aggregate
        let overall = aggregates.entry("overall".to_string())
            .or_insert_with(|| AggregateStats {
                min_latency_ms: f64::INFINITY,
                ..Default::default()
            });

        overall.count += 1;
        overall.total_latency_ms += event.latency_ms;
        overall.min_latency_ms = overall.min_latency_ms.min(event.latency_ms);
        overall.max_latency_ms = overall.max_latency_ms.max(event.latency_ms);
        overall.avg_latency_ms = overall.total_latency_ms / overall.count as f64;
        overall.total_results += event.result_count as u64;
        overall.avg_results = overall.total_results as f64 / overall.count as f64;
        if event.cache_hit {
            overall.cache_hits += 1;
        }
        overall.cache_hit_rate = overall.cache_hits as f64 / overall.count as f64;

        // Per-collection aggregate
        let collection = aggregates.entry(format!("collection:{}", event.collection))
            .or_insert_with(|| AggregateStats {
                min_latency_ms: f64::INFINITY,
                ..Default::default()
            });

        collection.count += 1;
        collection.total_latency_ms += event.latency_ms;
        collection.min_latency_ms = collection.min_latency_ms.min(event.latency_ms);
        collection.max_latency_ms = collection.max_latency_ms.max(event.latency_ms);
        collection.avg_latency_ms = collection.total_latency_ms / collection.count as f64;
        collection.total_results += event.result_count as u64;
        collection.avg_results = collection.total_results as f64 / collection.count as f64;
        if event.cache_hit {
            collection.cache_hits += 1;
        }
        collection.cache_hit_rate = collection.cache_hits as f64 / collection.count as f64;
    }

    fn update_patterns(&self, event: &QueryEvent) {
        let Ok(mut patterns) = self.patterns.write() else { return; };

        // Track dimension distribution
        *patterns.dimension_distribution
            .entry(event.dimensions)
            .or_insert(0) += 1;

        // Track k distribution
        *patterns.k_distribution
            .entry(event.top_k)
            .or_insert(0) += 1;

        // Track query patterns (collection + filter count)
        let pattern_key = format!("{}:filters={}", event.collection, event.filter_count);
        *patterns.frequent_patterns
            .entry(pattern_key)
            .or_insert(0) += 1;

        // Track temporal patterns (hourly)
        let hour = event.timestamp / 3600;
        patterns.temporal_patterns
            .entry(hour)
            .or_insert_with(Vec::new)
            .push(event.collection.clone());
    }

    fn update_timeseries(&self, event: &QueryEvent) {
        let Ok(mut ts) = self.timeseries.write() else { return; };
        let bucket = event.timestamp / 60; // Minute buckets

        // Latency series
        let latency_series = ts.series
            .entry("latency".to_string())
            .or_insert_with(Vec::new);

        if let Some(last) = latency_series.last_mut() {
            if last.timestamp == bucket {
                // Update existing bucket
                let new_count = last.count + 1;
                last.value = (last.value * last.count as f64 + event.latency_ms) / new_count as f64;
                last.count = new_count;
            } else {
                latency_series.push(TimeSeriesPoint {
                    timestamp: bucket,
                    value: event.latency_ms,
                    count: 1,
                });
            }
        } else {
            latency_series.push(TimeSeriesPoint {
                timestamp: bucket,
                value: event.latency_ms,
                count: 1,
            });
        }

        // Throughput series
        let throughput_series = ts.series
            .entry("throughput".to_string())
            .or_insert_with(Vec::new);

        if let Some(last) = throughput_series.last_mut() {
            if last.timestamp == bucket {
                last.count += 1;
                last.value = last.count as f64;
            } else {
                throughput_series.push(TimeSeriesPoint {
                    timestamp: bucket,
                    value: 1.0,
                    count: 1,
                });
            }
        } else {
            throughput_series.push(TimeSeriesPoint {
                timestamp: bucket,
                value: 1.0,
                count: 1,
            });
        }

        // Prune old data
        let cutoff = current_timestamp() / 60 - (self.config.retention.as_secs() / 60);
        for series in ts.series.values_mut() {
            series.retain(|p| p.timestamp >= cutoff);
        }
    }

    /// Get dashboard data
    pub fn get_dashboard(&self) -> DashboardData {
        let Ok(events) = self.events.read() else {
            return self.empty_dashboard();
        };
        let Ok(aggregates) = self.aggregates.read() else {
            return self.empty_dashboard();
        };

        let overall = aggregates.get("overall").cloned().unwrap_or_default();

        let uptime = self.stats.started_at.elapsed();
        let qps = self.stats.total_events.load(Ordering::Relaxed) as f64 / uptime.as_secs_f64().max(1.0);

        let summary = DashboardSummary {
            total_queries: overall.count,
            queries_per_second: qps,
            avg_latency_ms: overall.avg_latency_ms,
            p99_latency_ms: overall.p99_latency_ms,
            cache_hit_rate: overall.cache_hit_rate,
            error_rate: overall.error_rate,
            active_collections: aggregates.keys()
                .filter(|k| k.starts_with("collection:"))
                .count(),
            uptime,
        };

        let recent_queries: Vec<QuerySummary> = events.iter()
            .rev()
            .take(20)
            .map(|e| QuerySummary {
                timestamp: e.timestamp,
                collection: e.collection.clone(),
                latency_ms: e.latency_ms,
                result_count: e.result_count,
                event_type: format!("{:?}", e.event_type),
            })
            .collect();

        let performance_trends = self.get_performance_trends();

        let mut top_collections: Vec<CollectionStats> = aggregates.iter()
            .filter(|(k, _)| k.starts_with("collection:"))
            .map(|(k, stats)| CollectionStats {
                name: k.replace("collection:", ""),
                query_count: stats.count,
                avg_latency_ms: stats.avg_latency_ms,
                total_results: stats.total_results,
                error_rate: stats.error_rate,
            })
            .collect();

        top_collections.sort_by(|a, b| b.query_count.cmp(&a.query_count));
        top_collections.truncate(10);

        let alerts = self.check_alerts(&overall);

        DashboardData {
            summary,
            recent_queries,
            performance_trends,
            top_collections,
            error_summary: ErrorSummary {
                total_errors: overall.error_count,
                error_rate: overall.error_rate,
                top_errors: vec![],
            },
            alerts,
        }
    }

    fn empty_dashboard(&self) -> DashboardData {
        DashboardData {
            summary: DashboardSummary {
                total_queries: 0,
                queries_per_second: 0.0,
                avg_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                cache_hit_rate: 0.0,
                error_rate: 0.0,
                active_collections: 0,
                uptime: self.stats.started_at.elapsed(),
            },
            recent_queries: vec![],
            performance_trends: vec![],
            top_collections: vec![],
            error_summary: ErrorSummary {
                total_errors: 0,
                error_rate: 0.0,
                top_errors: vec![],
            },
            alerts: vec![],
        }
    }

    fn get_performance_trends(&self) -> Vec<TrendData> {
        let Ok(ts) = self.timeseries.read() else { return vec![]; };
        let mut trends = Vec::new();

        for (metric_name, points) in &ts.series {
            if points.len() < 2 {
                continue;
            }

            let recent: Vec<_> = points.iter().rev().take(60).collect();
            let old_avg: f64 = recent.iter().skip(30).map(|p| p.value).sum::<f64>()
                / recent.iter().skip(30).count().max(1) as f64;
            let new_avg: f64 = recent.iter().take(30).map(|p| p.value).sum::<f64>()
                / recent.iter().take(30).count().max(1) as f64;

            let change_percent = if old_avg > 0.0 {
                ((new_avg - old_avg) / old_avg) * 100.0
            } else {
                0.0
            };

            let trend_direction = if change_percent > 5.0 {
                TrendDirection::Up
            } else if change_percent < -5.0 {
                TrendDirection::Down
            } else {
                TrendDirection::Stable
            };

            trends.push(TrendData {
                metric_name: metric_name.clone(),
                data_points: points.clone(),
                trend_direction,
                change_percent,
            });
        }

        trends
    }

    fn check_alerts(&self, stats: &AggregateStats) -> Vec<Alert> {
        let mut alerts = Vec::new();
        let now = current_timestamp();

        // High latency alert
        if stats.p99_latency_ms > 1000.0 {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                message: "High P99 latency detected".to_string(),
                metric: "p99_latency_ms".to_string(),
                threshold: 1000.0,
                current_value: stats.p99_latency_ms,
                triggered_at: now,
            });
        }

        // Low cache hit rate
        if stats.cache_hit_rate < 0.5 && stats.count > 100 {
            alerts.push(Alert {
                severity: AlertSeverity::Info,
                message: "Low cache hit rate".to_string(),
                metric: "cache_hit_rate".to_string(),
                threshold: 0.5,
                current_value: stats.cache_hit_rate,
                triggered_at: now,
            });
        }

        // High error rate
        if stats.error_rate > 0.01 && stats.count > 100 {
            alerts.push(Alert {
                severity: AlertSeverity::Error,
                message: "High error rate detected".to_string(),
                metric: "error_rate".to_string(),
                threshold: 0.01,
                current_value: stats.error_rate,
                triggered_at: now,
            });
        }

        alerts
    }

    /// Get aggregate statistics
    pub fn get_aggregates(&self) -> HashMap<String, AggregateStats> {
        let Ok(aggregates) = self.aggregates.read() else { return HashMap::new(); };
        aggregates.clone()
    }

    /// Get time series data
    pub fn get_timeseries(&self, metric: &str, last_n_minutes: usize) -> Vec<TimeSeriesPoint> {
        let Ok(ts) = self.timeseries.read() else { return vec![]; };
        let cutoff = current_timestamp() / 60 - last_n_minutes as u64;

        ts.series.get(metric)
            .map(|points| {
                points.iter()
                    .filter(|p| p.timestamp >= cutoff)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Generate a report
    pub fn generate_report(&self, config: &ReportConfig) -> Result<String> {
        let events = self.events.read()
            .map_err(|_| VecStoreError::LockError("events lock poisoned".into()))?;

        let filtered: Vec<_> = events.iter()
            .filter(|e| e.timestamp >= config.time_range.start && e.timestamp <= config.time_range.end)
            .collect();

        match config.format {
            ReportFormat::Json => {
                let summary: HashMap<String, serde_json::Value> = HashMap::new();
                Ok(serde_json::to_string_pretty(&summary).unwrap_or_default())
            }
            ReportFormat::Csv => {
                let mut csv = String::from("timestamp,collection,latency_ms,result_count,event_type\n");
                for e in filtered {
                    csv.push_str(&format!(
                        "{},{},{},{},{:?}\n",
                        e.timestamp, e.collection, e.latency_ms, e.result_count, e.event_type
                    ));
                }
                Ok(csv)
            }
            ReportFormat::Html => {
                let html = format!(
                    r#"<!DOCTYPE html>
<html>
<head><title>Query Analytics Report</title></head>
<body>
<h1>Query Analytics Report</h1>
<p>Time range: {} - {}</p>
<p>Total events: {}</p>
</body>
</html>"#,
                    config.time_range.start,
                    config.time_range.end,
                    filtered.len()
                );
                Ok(html)
            }
            ReportFormat::Prometheus => {
                let aggregates = self.aggregates.read()
                    .map_err(|_| VecStoreError::LockError("aggregates lock poisoned".into()))?;
                let mut output = String::new();

                if let Some(overall) = aggregates.get("overall") {
                    output.push_str(&format!(
                        "# HELP vecstore_query_total Total number of queries\n\
                         # TYPE vecstore_query_total counter\n\
                         vecstore_query_total {}\n\n\
                         # HELP vecstore_query_latency_ms Average query latency\n\
                         # TYPE vecstore_query_latency_ms gauge\n\
                         vecstore_query_latency_ms {:.2}\n\n\
                         # HELP vecstore_cache_hit_rate Cache hit rate\n\
                         # TYPE vecstore_cache_hit_rate gauge\n\
                         vecstore_cache_hit_rate {:.4}\n",
                        overall.count,
                        overall.avg_latency_ms,
                        overall.cache_hit_rate
                    ));
                }

                Ok(output)
            }
        }
    }

    /// Get query patterns
    pub fn get_patterns(&self) -> QueryPatterns {
        let Ok(patterns) = self.patterns.read() else {
            return QueryPatterns {
                frequent_patterns: vec![],
                dimension_distribution: vec![],
                k_distribution: vec![],
            };
        };

        let mut frequent: Vec<(String, u64)> = patterns.frequent_patterns.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        frequent.sort_by(|a, b| b.1.cmp(&a.1));
        frequent.truncate(10);

        let mut dimensions: Vec<(usize, u64)> = patterns.dimension_distribution.iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        dimensions.sort_by(|a, b| b.1.cmp(&a.1));

        let mut k_values: Vec<(usize, u64)> = patterns.k_distribution.iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        k_values.sort_by(|a, b| b.1.cmp(&a.1));

        QueryPatterns {
            frequent_patterns: frequent,
            dimension_distribution: dimensions,
            k_distribution: k_values,
        }
    }

    /// Export metrics for monitoring
    pub fn export_metrics(&self) -> Metrics {
        let Ok(aggregates) = self.aggregates.read() else {
            return Metrics {
                total_queries: 0,
                avg_latency_ms: 0.0,
                p50_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                cache_hit_rate: 0.0,
                error_rate: 0.0,
                queries_per_second: 0.0,
            };
        };
        let overall = aggregates.get("overall").cloned().unwrap_or_default();

        Metrics {
            total_queries: overall.count,
            avg_latency_ms: overall.avg_latency_ms,
            p50_latency_ms: overall.p50_latency_ms,
            p95_latency_ms: overall.p95_latency_ms,
            p99_latency_ms: overall.p99_latency_ms,
            cache_hit_rate: overall.cache_hit_rate,
            error_rate: overall.error_rate,
            queries_per_second: {
                let uptime = self.stats.started_at.elapsed().as_secs_f64();
                if uptime > 0.0 { overall.count as f64 / uptime } else { 0.0 }
            },
        }
    }

    /// Reset analytics
    pub fn reset(&self) {
        let Ok(mut events) = self.events.write() else { return; };
        events.clear();
        drop(events);

        let Ok(mut aggregates) = self.aggregates.write() else { return; };
        aggregates.clear();
        drop(aggregates);

        let Ok(mut patterns) = self.patterns.write() else { return; };
        *patterns = PatternAnalyzer {
            frequent_patterns: HashMap::new(),
            temporal_patterns: BTreeMap::new(),
            dimension_distribution: HashMap::new(),
            k_distribution: HashMap::new(),
        };
        drop(patterns);

        self.stats.total_events.store(0, Ordering::Relaxed);
        self.stats.search_events.store(0, Ordering::Relaxed);
        self.stats.insert_events.store(0, Ordering::Relaxed);
        self.stats.update_events.store(0, Ordering::Relaxed);
        self.stats.delete_events.store(0, Ordering::Relaxed);

        let Ok(mut avg) = self.stats.avg_latency_ms.write() else { return; };
        *avg = 0.0;
    }
}

/// Query patterns
#[derive(Debug, Clone, Serialize)]
pub struct QueryPatterns {
    pub frequent_patterns: Vec<(String, u64)>,
    pub dimension_distribution: Vec<(usize, u64)>,
    pub k_distribution: Vec<(usize, u64)>,
}

/// Exported metrics
#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub total_queries: u64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub queries_per_second: f64,
}

/// Histogram for latency tracking
pub struct LatencyHistogram {
    buckets: Vec<(f64, u64)>, // (threshold_ms, count)
    total: u64,
    sum: f64,
}

impl LatencyHistogram {
    pub fn new(bucket_boundaries: &[f64]) -> Self {
        let buckets = bucket_boundaries.iter()
            .map(|&b| (b, 0))
            .collect();

        Self {
            buckets,
            total: 0,
            sum: 0.0,
        }
    }

    pub fn observe(&mut self, value: f64) {
        self.total += 1;
        self.sum += value;

        for (threshold, count) in &mut self.buckets {
            if value <= *threshold {
                *count += 1;
                break;
            }
        }
    }

    pub fn percentile(&self, p: f64) -> f64 {
        let target = (self.total as f64 * p / 100.0) as u64;
        let mut cumulative = 0;

        for (threshold, count) in &self.buckets {
            cumulative += count;
            if cumulative >= target {
                return *threshold;
            }
        }

        self.buckets.last().map(|(t, _)| *t).unwrap_or(0.0)
    }

    pub fn mean(&self) -> f64 {
        if self.total > 0 {
            self.sum / self.total as f64
        } else {
            0.0
        }
    }
}

/// Query analyzer for recommendations
pub struct QueryRecommender {
    analytics: std::sync::Arc<QueryAnalytics>,
}

impl QueryRecommender {
    pub fn new(analytics: std::sync::Arc<QueryAnalytics>) -> Self {
        Self { analytics }
    }

    /// Get recommendations based on query patterns
    pub fn get_recommendations(&self) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();
        let patterns = self.analytics.get_patterns();
        let metrics = self.analytics.export_metrics();

        // Check if cache hit rate is low
        if metrics.cache_hit_rate < 0.5 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Performance,
                title: "Low Cache Hit Rate".to_string(),
                description: "Consider increasing cache size or adjusting cache policy".to_string(),
                impact: Impact::High,
                effort: Impact::Low,
            });
        }

        // Check if average latency is high
        if metrics.avg_latency_ms > 100.0 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Performance,
                title: "High Average Latency".to_string(),
                description: "Consider optimizing index parameters or adding more replicas".to_string(),
                impact: Impact::High,
                effort: Impact::Medium,
            });
        }

        // Check dimension distribution for index optimization
        if let Some((most_common_dim, _)) = patterns.dimension_distribution.first() {
            if *most_common_dim > 512 {
                recommendations.push(Recommendation {
                    category: RecommendationCategory::IndexOptimization,
                    title: "High Dimensional Vectors".to_string(),
                    description: format!(
                        "Most queries use {}-dimensional vectors. Consider PQ or dimension reduction.",
                        most_common_dim
                    ),
                    impact: Impact::Medium,
                    effort: Impact::High,
                });
            }
        }

        // Check k distribution
        if let Some((most_common_k, _)) = patterns.k_distribution.first() {
            if *most_common_k > 100 {
                recommendations.push(Recommendation {
                    category: RecommendationCategory::QueryOptimization,
                    title: "Large Top-K Values".to_string(),
                    description: format!(
                        "Many queries request top-{}. Consider pagination instead.",
                        most_common_k
                    ),
                    impact: Impact::Medium,
                    effort: Impact::Low,
                });
            }
        }

        recommendations
    }
}

/// Recommendation
#[derive(Debug, Clone, Serialize)]
pub struct Recommendation {
    pub category: RecommendationCategory,
    pub title: String,
    pub description: String,
    pub impact: Impact,
    pub effort: Impact,
}

/// Recommendation category
#[derive(Debug, Clone, Serialize)]
pub enum RecommendationCategory {
    Performance,
    IndexOptimization,
    QueryOptimization,
    CostReduction,
    Reliability,
}

/// Impact level
#[derive(Debug, Clone, Serialize)]
pub enum Impact {
    Low,
    Medium,
    High,
}

/// Builder for QueryAnalytics
#[must_use = "builders do nothing unless built"]
pub struct QueryAnalyticsBuilder {
    config: AnalyticsConfig,
}

impl QueryAnalyticsBuilder {
    pub fn new() -> Self {
        Self {
            config: AnalyticsConfig::default(),
        }
    }

    #[inline]
    pub fn sample_rate(mut self, rate: f64) -> Self {
        self.config.sample_rate = rate;
        self
    }

    #[inline]
    pub fn retention(mut self, retention: Duration) -> Self {
        self.config.retention = retention;
        self
    }

    #[inline]
    pub fn max_events(mut self, max: usize) -> Self {
        self.config.max_events = max;
        self
    }

    #[inline]
    pub fn privacy_mode(mut self, enabled: bool) -> Self {
        self.config.privacy_mode = enabled;
        self
    }

    pub fn build(self) -> QueryAnalytics {
        QueryAnalytics::new(self.config)
    }
}

impl Default for QueryAnalyticsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_creation() {
        let analytics = QueryAnalyticsBuilder::new()
            .sample_rate(1.0)
            .build();

        let dashboard = analytics.get_dashboard();
        assert_eq!(dashboard.summary.total_queries, 0);
    }

    #[test]
    fn test_record_event() {
        let analytics = QueryAnalyticsBuilder::new()
            .sample_rate(1.0)
            .build();

        let event = QueryEvent {
            id: 0,
            event_type: EventType::Search,
            timestamp: current_timestamp(),
            collection: "test".to_string(),
            latency_ms: 10.0,
            result_count: 5,
            filter_count: 1,
            dimensions: 128,
            top_k: 10,
            index_type: "hnsw".to_string(),
            cache_hit: true,
            client_id: None,
            metadata: HashMap::new(),
        };

        analytics.record(event);

        let dashboard = analytics.get_dashboard();
        assert_eq!(dashboard.summary.total_queries, 1);
    }

    #[test]
    fn test_timeseries() {
        let analytics = QueryAnalyticsBuilder::new()
            .sample_rate(1.0)
            .build();

        for i in 0..10 {
            let event = QueryEvent {
                id: i,
                event_type: EventType::Search,
                timestamp: current_timestamp(),
                collection: "test".to_string(),
                latency_ms: 10.0 + i as f64,
                result_count: 5,
                filter_count: 1,
                dimensions: 128,
                top_k: 10,
                index_type: "hnsw".to_string(),
                cache_hit: i % 2 == 0,
                client_id: None,
                metadata: HashMap::new(),
            };
            analytics.record(event);
        }

        let ts = analytics.get_timeseries("latency", 60);
        assert!(!ts.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let analytics = QueryAnalyticsBuilder::new()
            .sample_rate(1.0)
            .build();

        let config = ReportConfig {
            time_range: TimeRange {
                start: 0,
                end: u64::MAX,
            },
            metrics: vec!["latency".to_string()],
            group_by: None,
            format: ReportFormat::Csv,
        };

        let report = analytics.generate_report(&config);
        assert!(report.is_ok());
    }

    #[test]
    fn test_histogram() {
        let mut histogram = LatencyHistogram::new(&[1.0, 5.0, 10.0, 50.0, 100.0]);

        histogram.observe(2.0);
        histogram.observe(8.0);
        histogram.observe(45.0);

        assert!((histogram.mean() - 18.33).abs() < 1.0);
    }
}
