//! Vector Database Monitoring and Alerting
//!
//! Provides real-time monitoring of vector operations with configurable alerts
//! for data quality, performance, storage, and index health.

use crate::error::{Result, VecStoreError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Maximum number of metric samples to keep in history
    pub max_history_size: usize,

    /// Interval for collecting metrics
    pub collection_interval: Duration,

    /// Enable alert notifications
    pub enable_alerts: bool,

    /// Alert cooldown period to prevent spam
    pub alert_cooldown: Duration,

    /// Enable metric aggregation
    pub enable_aggregation: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            collection_interval: Duration::from_secs(60),
            enable_alerts: true,
            alert_cooldown: Duration::from_secs(300),
            enable_aggregation: true,
        }
    }
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertCategory {
    Performance,
    DataQuality,
    Storage,
    IndexHealth,
    Security,
}

/// Alert definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub timestamp: SystemTime,
    pub severity: AlertSeverity,
    pub category: AlertCategory,
    pub message: String,
    pub metric_name: String,
    pub metric_value: f64,
    pub threshold: f64,
}

/// Alert rule for triggering notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub metric_name: String,
    pub category: AlertCategory,
    pub severity: AlertSeverity,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub enabled: bool,
}

/// Alert condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    PercentageChange,
}

impl AlertCondition {
    fn evaluate(&self, value: f64, threshold: f64) -> bool {
        match self {
            AlertCondition::GreaterThan => value > threshold,
            AlertCondition::LessThan => value < threshold,
            AlertCondition::Equal => (value - threshold).abs() < 1e-6,
            AlertCondition::NotEqual => (value - threshold).abs() >= 1e-6,
            AlertCondition::PercentageChange => {
                (value / threshold - 1.0).abs() > 0.1 // 10% change
            }
        }
    }
}

/// Metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    // Performance metrics
    QueryLatency,
    InsertLatency,
    ThroughputQPS,

    // Data quality metrics
    VectorQuality,
    DuplicateRate,
    OutlierRate,

    // Storage metrics
    StorageUsed,
    IndexSize,
    MemoryUsage,

    // Index health metrics
    IndexFragmentation,
    CacheHitRate,
    ErrorRate,
}

impl MetricType {
    pub fn name(&self) -> &'static str {
        match self {
            MetricType::QueryLatency => "query_latency_ms",
            MetricType::InsertLatency => "insert_latency_ms",
            MetricType::ThroughputQPS => "throughput_qps",
            MetricType::VectorQuality => "vector_quality",
            MetricType::DuplicateRate => "duplicate_rate",
            MetricType::OutlierRate => "outlier_rate",
            MetricType::StorageUsed => "storage_used_mb",
            MetricType::IndexSize => "index_size_mb",
            MetricType::MemoryUsage => "memory_usage_mb",
            MetricType::IndexFragmentation => "index_fragmentation",
            MetricType::CacheHitRate => "cache_hit_rate",
            MetricType::ErrorRate => "error_rate",
        }
    }

    pub fn unit(&self) -> &'static str {
        match self {
            MetricType::QueryLatency | MetricType::InsertLatency => "ms",
            MetricType::ThroughputQPS => "qps",
            MetricType::VectorQuality
            | MetricType::DuplicateRate
            | MetricType::OutlierRate
            | MetricType::CacheHitRate
            | MetricType::ErrorRate => "ratio",
            MetricType::StorageUsed | MetricType::IndexSize | MetricType::MemoryUsage => "MB",
            MetricType::IndexFragmentation => "percent",
        }
    }
}

/// Metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub timestamp: SystemTime,
    pub value: f64,
}

/// Metric history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricHistory {
    pub metric_type: MetricType,
    pub points: VecDeque<MetricPoint>,
    pub max_size: usize,
}

impl MetricHistory {
    pub fn new(metric_type: MetricType, max_size: usize) -> Self {
        Self {
            metric_type,
            points: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub fn add(&mut self, value: f64) {
        if self.points.len() >= self.max_size {
            self.points.pop_front();
        }
        self.points.push_back(MetricPoint {
            timestamp: SystemTime::now(),
            value,
        });
    }

    pub fn latest(&self) -> Option<f64> {
        self.points.back().map(|p| p.value)
    }

    pub fn average(&self) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }
        let sum: f64 = self.points.iter().map(|p| p.value).sum();
        Some(sum / self.points.len() as f64)
    }

    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }
        let mut values: Vec<f64> = self.points.iter().map(|p| p.value).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((values.len() - 1) as f64 * p / 100.0) as usize;
        Some(values[idx])
    }
}

/// Monitoring statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStats {
    pub total_alerts: usize,
    pub alerts_by_severity: HashMap<AlertSeverity, usize>,
    pub alerts_by_category: HashMap<AlertCategory, usize>,
    pub active_rules: usize,
    pub metrics_tracked: usize,
    pub uptime: Duration,
}

/// Monitoring report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringReport {
    pub timestamp: SystemTime,
    pub metrics: HashMap<String, f64>,
    pub recent_alerts: Vec<Alert>,
    pub stats: MonitoringStats,
}

/// Vector database monitor
pub struct Monitor {
    config: MonitoringConfig,
    metrics: HashMap<MetricType, MetricHistory>,
    alert_rules: Vec<AlertRule>,
    alerts: VecDeque<Alert>,
    last_alert_time: HashMap<String, SystemTime>,
    start_time: SystemTime,
}

impl Monitor {
    /// Create a new monitor
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics: HashMap::new(),
            alert_rules: Vec::new(),
            alerts: VecDeque::new(),
            last_alert_time: HashMap::new(),
            start_time: SystemTime::now(),
        }
    }

    /// Create monitor with default configuration
    pub fn default() -> Self {
        Self::new(MonitoringConfig::default())
    }

    /// Add an alert rule
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.alert_rules.push(rule);
    }

    /// Remove an alert rule by name
    pub fn remove_rule(&mut self, name: &str) -> bool {
        if let Some(pos) = self.alert_rules.iter().position(|r| r.name == name) {
            self.alert_rules.remove(pos);
            true
        } else {
            false
        }
    }

    /// Record a metric value
    pub fn record(&mut self, metric_type: MetricType, value: f64) {
        // Get or create metric history
        let history = self
            .metrics
            .entry(metric_type)
            .or_insert_with(|| MetricHistory::new(metric_type, self.config.max_history_size));

        history.add(value);

        // Check alert rules if enabled
        if self.config.enable_alerts {
            self.check_alerts(metric_type, value);
        }
    }

    /// Check alert rules for a metric
    fn check_alerts(&mut self, metric_type: MetricType, value: f64) {
        let metric_name = metric_type.name();

        for rule in &self.alert_rules {
            if !rule.enabled || rule.metric_name != metric_name {
                continue;
            }

            // Check cooldown period
            if let Some(last_time) = self.last_alert_time.get(&rule.name) {
                if let Ok(elapsed) = SystemTime::now().duration_since(*last_time) {
                    if elapsed < self.config.alert_cooldown {
                        continue;
                    }
                }
            }

            // Evaluate condition
            if rule.condition.evaluate(value, rule.threshold) {
                let alert = Alert {
                    id: format!(
                        "{}-{}",
                        rule.name,
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    timestamp: SystemTime::now(),
                    severity: rule.severity,
                    category: rule.category,
                    message: format!(
                        "{}: {} = {:.3} {} (threshold: {:.3})",
                        rule.name,
                        metric_name,
                        value,
                        metric_type.unit(),
                        rule.threshold
                    ),
                    metric_name: metric_name.to_string(),
                    metric_value: value,
                    threshold: rule.threshold,
                };

                self.alerts.push_back(alert);
                self.last_alert_time
                    .insert(rule.name.clone(), SystemTime::now());

                // Limit alert history
                while self.alerts.len() > self.config.max_history_size {
                    self.alerts.pop_front();
                }
            }
        }
    }

    /// Get recent alerts
    pub fn get_alerts(&self, count: usize) -> Vec<Alert> {
        self.alerts.iter().rev().take(count).cloned().collect()
    }

    /// Get alerts by severity
    pub fn get_alerts_by_severity(&self, severity: AlertSeverity) -> Vec<Alert> {
        self.alerts
            .iter()
            .filter(|a| a.severity == severity)
            .cloned()
            .collect()
    }

    /// Get alerts by category
    pub fn get_alerts_by_category(&self, category: AlertCategory) -> Vec<Alert> {
        self.alerts
            .iter()
            .filter(|a| a.category == category)
            .cloned()
            .collect()
    }

    /// Clear all alerts
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
        self.last_alert_time.clear();
    }

    /// Get metric history
    pub fn get_metric(&self, metric_type: MetricType) -> Option<&MetricHistory> {
        self.metrics.get(&metric_type)
    }

    /// Get all metrics
    pub fn get_all_metrics(&self) -> &HashMap<MetricType, MetricHistory> {
        &self.metrics
    }

    /// Generate monitoring report
    pub fn generate_report(&self) -> MonitoringReport {
        // Collect current metric values
        let mut metrics = HashMap::new();
        for (metric_type, history) in &self.metrics {
            if let Some(value) = history.latest() {
                metrics.insert(metric_type.name().to_string(), value);
            }
        }

        // Get recent alerts
        let recent_alerts = self.get_alerts(10);

        // Calculate statistics
        let mut alerts_by_severity = HashMap::new();
        let mut alerts_by_category = HashMap::new();

        for alert in &self.alerts {
            *alerts_by_severity.entry(alert.severity).or_insert(0) += 1;
            *alerts_by_category.entry(alert.category).or_insert(0) += 1;
        }

        let uptime = SystemTime::now()
            .duration_since(self.start_time)
            .unwrap_or(Duration::from_secs(0));

        let stats = MonitoringStats {
            total_alerts: self.alerts.len(),
            alerts_by_severity,
            alerts_by_category,
            active_rules: self.alert_rules.iter().filter(|r| r.enabled).count(),
            metrics_tracked: self.metrics.len(),
            uptime,
        };

        MonitoringReport {
            timestamp: SystemTime::now(),
            metrics,
            recent_alerts,
            stats,
        }
    }

    /// Get monitoring statistics
    pub fn get_stats(&self) -> MonitoringStats {
        let mut alerts_by_severity = HashMap::new();
        let mut alerts_by_category = HashMap::new();

        for alert in &self.alerts {
            *alerts_by_severity.entry(alert.severity).or_insert(0) += 1;
            *alerts_by_category.entry(alert.category).or_insert(0) += 1;
        }

        let uptime = SystemTime::now()
            .duration_since(self.start_time)
            .unwrap_or(Duration::from_secs(0));

        MonitoringStats {
            total_alerts: self.alerts.len(),
            alerts_by_severity,
            alerts_by_category,
            active_rules: self.alert_rules.iter().filter(|r| r.enabled).count(),
            metrics_tracked: self.metrics.len(),
            uptime,
        }
    }

    /// Export metrics as Prometheus format
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();

        for (metric_type, history) in &self.metrics {
            if let Some(value) = history.latest() {
                output.push_str(&format!("vecstore_{} {}\n", metric_type.name(), value));
            }
        }

        output
    }
}

/// Preset alert rules for common scenarios
pub struct AlertPresets;

impl AlertPresets {
    /// High query latency alert
    pub fn high_query_latency(threshold_ms: f64) -> AlertRule {
        AlertRule {
            name: "high_query_latency".to_string(),
            metric_name: "query_latency_ms".to_string(),
            category: AlertCategory::Performance,
            severity: AlertSeverity::Warning,
            condition: AlertCondition::GreaterThan,
            threshold: threshold_ms,
            enabled: true,
        }
    }

    /// Low cache hit rate alert
    pub fn low_cache_hit_rate(threshold: f64) -> AlertRule {
        AlertRule {
            name: "low_cache_hit_rate".to_string(),
            metric_name: "cache_hit_rate".to_string(),
            category: AlertCategory::Performance,
            severity: AlertSeverity::Warning,
            condition: AlertCondition::LessThan,
            threshold,
            enabled: true,
        }
    }

    /// High error rate alert
    pub fn high_error_rate(threshold: f64) -> AlertRule {
        AlertRule {
            name: "high_error_rate".to_string(),
            metric_name: "error_rate".to_string(),
            category: AlertCategory::Performance,
            severity: AlertSeverity::Error,
            condition: AlertCondition::GreaterThan,
            threshold,
            enabled: true,
        }
    }

    /// Low vector quality alert
    pub fn low_vector_quality(threshold: f64) -> AlertRule {
        AlertRule {
            name: "low_vector_quality".to_string(),
            metric_name: "vector_quality".to_string(),
            category: AlertCategory::DataQuality,
            severity: AlertSeverity::Warning,
            condition: AlertCondition::LessThan,
            threshold,
            enabled: true,
        }
    }

    /// High storage usage alert
    pub fn high_storage_usage(threshold_mb: f64) -> AlertRule {
        AlertRule {
            name: "high_storage_usage".to_string(),
            metric_name: "storage_used_mb".to_string(),
            category: AlertCategory::Storage,
            severity: AlertSeverity::Warning,
            condition: AlertCondition::GreaterThan,
            threshold: threshold_mb,
            enabled: true,
        }
    }

    /// High memory usage alert
    pub fn high_memory_usage(threshold_mb: f64) -> AlertRule {
        AlertRule {
            name: "high_memory_usage".to_string(),
            metric_name: "memory_usage_mb".to_string(),
            category: AlertCategory::Storage,
            severity: AlertSeverity::Error,
            condition: AlertCondition::GreaterThan,
            threshold: threshold_mb,
            enabled: true,
        }
    }

    /// High index fragmentation alert
    pub fn high_index_fragmentation(threshold_pct: f64) -> AlertRule {
        AlertRule {
            name: "high_index_fragmentation".to_string(),
            metric_name: "index_fragmentation".to_string(),
            category: AlertCategory::IndexHealth,
            severity: AlertSeverity::Warning,
            condition: AlertCondition::GreaterThan,
            threshold: threshold_pct,
            enabled: true,
        }
    }

    /// Get all default rules
    pub fn default_rules() -> Vec<AlertRule> {
        vec![
            Self::high_query_latency(100.0),
            Self::low_cache_hit_rate(0.7),
            Self::high_error_rate(0.05),
            Self::low_vector_quality(0.6),
            Self::high_storage_usage(1000.0),
            Self::high_memory_usage(2000.0),
            Self::high_index_fragmentation(30.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_monitor_creation() {
        let monitor = Monitor::default();
        assert_eq!(monitor.metrics.len(), 0);
        assert_eq!(monitor.alert_rules.len(), 0);
    }

    #[test]
    fn test_record_metrics() {
        let mut monitor = Monitor::default();

        monitor.record(MetricType::QueryLatency, 50.0);
        monitor.record(MetricType::QueryLatency, 75.0);
        monitor.record(MetricType::QueryLatency, 100.0);

        let history = monitor.get_metric(MetricType::QueryLatency).unwrap();
        assert_eq!(history.points.len(), 3);
        assert_eq!(history.latest(), Some(100.0));
    }

    #[test]
    fn test_metric_statistics() {
        let mut history = MetricHistory::new(MetricType::QueryLatency, 100);

        history.add(50.0);
        history.add(100.0);
        history.add(150.0);

        assert_eq!(history.latest(), Some(150.0));
        assert_eq!(history.average(), Some(100.0));

        // Test percentile
        let p50 = history.percentile(50.0);
        assert!(p50.is_some());
        assert!((p50.unwrap() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_alert_rules() {
        let mut monitor = Monitor::default();

        // Add high latency alert rule
        monitor.add_rule(AlertPresets::high_query_latency(100.0));

        // Record values below threshold - no alerts
        monitor.record(MetricType::QueryLatency, 50.0);
        monitor.record(MetricType::QueryLatency, 75.0);
        assert_eq!(monitor.alerts.len(), 0);

        // Record value above threshold - should trigger alert
        monitor.record(MetricType::QueryLatency, 150.0);
        assert_eq!(monitor.alerts.len(), 1);

        let alert = &monitor.alerts[0];
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.category, AlertCategory::Performance);
    }

    #[test]
    fn test_alert_cooldown() {
        let config = MonitoringConfig {
            alert_cooldown: Duration::from_millis(100),
            ..Default::default()
        };

        let mut monitor = Monitor::new(config);
        monitor.add_rule(AlertPresets::high_query_latency(100.0));

        // First trigger
        monitor.record(MetricType::QueryLatency, 150.0);
        assert_eq!(monitor.alerts.len(), 1);

        // Immediate second trigger - should be suppressed
        monitor.record(MetricType::QueryLatency, 200.0);
        assert_eq!(monitor.alerts.len(), 1);

        // Wait for cooldown
        thread::sleep(Duration::from_millis(150));

        // Third trigger after cooldown - should create new alert
        monitor.record(MetricType::QueryLatency, 250.0);
        assert_eq!(monitor.alerts.len(), 2);
    }

    #[test]
    fn test_alert_filtering() {
        let mut monitor = Monitor::default();

        monitor.add_rule(AlertPresets::high_query_latency(100.0));
        monitor.add_rule(AlertPresets::high_error_rate(0.05));

        monitor.record(MetricType::QueryLatency, 150.0);
        monitor.record(MetricType::ErrorRate, 0.1);

        assert_eq!(monitor.alerts.len(), 2);

        let warnings = monitor.get_alerts_by_severity(AlertSeverity::Warning);
        assert_eq!(warnings.len(), 1);

        let errors = monitor.get_alerts_by_severity(AlertSeverity::Error);
        assert_eq!(errors.len(), 1);

        let perf_alerts = monitor.get_alerts_by_category(AlertCategory::Performance);
        assert_eq!(perf_alerts.len(), 2);
    }

    #[test]
    fn test_monitoring_report() {
        let mut monitor = Monitor::default();

        monitor.record(MetricType::QueryLatency, 50.0);
        monitor.record(MetricType::MemoryUsage, 512.0);

        monitor.add_rule(AlertPresets::high_memory_usage(1000.0));

        let report = monitor.generate_report();

        assert_eq!(report.metrics.len(), 2);
        assert!(report.metrics.contains_key("query_latency_ms"));
        assert!(report.metrics.contains_key("memory_usage_mb"));

        assert_eq!(report.stats.metrics_tracked, 2);
        assert_eq!(report.stats.active_rules, 1);
    }

    #[test]
    fn test_prometheus_export() {
        let mut monitor = Monitor::default();

        monitor.record(MetricType::QueryLatency, 50.0);
        monitor.record(MetricType::ThroughputQPS, 1000.0);

        let output = monitor.export_prometheus();

        assert!(output.contains("vecstore_query_latency_ms"));
        assert!(output.contains("vecstore_throughput_qps"));
        assert!(output.contains("50"));
        assert!(output.contains("1000"));
    }

    #[test]
    fn test_rule_management() {
        let mut monitor = Monitor::default();

        let rule = AlertPresets::high_query_latency(100.0);
        monitor.add_rule(rule);

        assert_eq!(monitor.alert_rules.len(), 1);

        let removed = monitor.remove_rule("high_query_latency");
        assert!(removed);
        assert_eq!(monitor.alert_rules.len(), 0);

        let removed_again = monitor.remove_rule("nonexistent");
        assert!(!removed_again);
    }

    #[test]
    fn test_default_presets() {
        let rules = AlertPresets::default_rules();
        assert_eq!(rules.len(), 7);

        // Verify all rules are enabled
        for rule in &rules {
            assert!(rule.enabled);
        }
    }
}
