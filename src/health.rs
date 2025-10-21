//! Health checks and diagnostics for VecStore
//!
//! Provides comprehensive health monitoring and diagnostics:
//! - Database health status
//! - Performance metrics
//! - Resource utilization
//! - Index integrity checks
//! - Alert conditions

use crate::store::VecStore;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Overall health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All systems operational
    Healthy,

    /// Minor issues, but functional
    Degraded,

    /// Critical issues affecting functionality
    Unhealthy,
}

impl HealthStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
        }
    }
}

/// Comprehensive health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall status
    pub status: HealthStatus,

    /// Timestamp of the check
    pub timestamp: SystemTime,

    /// Database statistics
    pub database: DatabaseHealth,

    /// Index health
    pub index: IndexHealth,

    /// Performance metrics
    pub performance: PerformanceHealth,

    /// Resource utilization
    pub resources: ResourceHealth,

    /// Active alerts
    pub alerts: Vec<Alert>,

    /// Uptime duration
    pub uptime: Option<Duration>,
}

/// Database health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseHealth {
    /// Total number of vectors
    pub total_vectors: usize,

    /// Active (non-deleted) vectors
    pub active_vectors: usize,

    /// Deleted vectors pending compaction
    pub deleted_vectors: usize,

    /// Vector dimension
    pub dimension: usize,

    /// Deletion ratio (deleted/total)
    pub deletion_ratio: f64,

    /// Storage efficiency score (0-100)
    pub storage_efficiency: f64,
}

/// Index health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHealth {
    /// Index type (HNSW, IVF-PQ, etc.)
    pub index_type: String,

    /// Index integrity check result
    pub integrity_ok: bool,

    /// Average degree in HNSW graph
    pub avg_degree: Option<f64>,

    /// Fragmentation score (0-100, lower is better)
    pub fragmentation: f64,

    /// Last index rebuild time
    pub last_rebuild: Option<SystemTime>,
}

/// Performance health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHealth {
    /// Average query latency (ms)
    pub avg_query_latency_ms: Option<f64>,

    /// 95th percentile query latency (ms)
    pub p95_query_latency_ms: Option<f64>,

    /// Queries per second
    pub qps: Option<f64>,

    /// Insert throughput (vectors/sec)
    pub insert_throughput: Option<f64>,

    /// Performance score (0-100)
    pub performance_score: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceHealth {
    /// Estimated memory usage (bytes)
    pub memory_bytes: usize,

    /// Disk usage (bytes)
    pub disk_bytes: usize,

    /// Memory per vector (bytes)
    pub memory_per_vector: f64,

    /// Memory utilization percentage (0-100)
    pub memory_utilization: f64,

    /// Disk utilization percentage (0-100)
    pub disk_utilization: f64,
}

/// Health alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert severity
    pub severity: AlertSeverity,

    /// Alert category
    pub category: AlertCategory,

    /// Human-readable message
    pub message: String,

    /// Optional metric value
    pub value: Option<f64>,

    /// Recommended action
    pub recommendation: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertCategory {
    Performance,
    Storage,
    Index,
    Resource,
    Capacity,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Deletion ratio threshold for warning
    pub deletion_ratio_warning: f64,

    /// Deletion ratio threshold for critical
    pub deletion_ratio_critical: f64,

    /// Fragmentation threshold for warning
    pub fragmentation_warning: f64,

    /// Memory utilization warning threshold
    pub memory_warning: f64,

    /// Query latency warning threshold (ms)
    pub latency_warning_ms: f64,

    /// Minimum performance score
    pub min_performance_score: f64,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            deletion_ratio_warning: 0.3,
            deletion_ratio_critical: 0.5,
            fragmentation_warning: 50.0,
            memory_warning: 80.0,
            latency_warning_ms: 100.0,
            min_performance_score: 70.0,
        }
    }
}

/// Health checker
pub struct HealthChecker {
    config: HealthCheckConfig,
    start_time: SystemTime,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new(config: HealthCheckConfig) -> Self {
        Self {
            config,
            start_time: SystemTime::now(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(HealthCheckConfig::default())
    }

    /// Perform a comprehensive health check
    pub fn check(&self, store: &VecStore) -> Result<HealthReport> {
        let database = self.check_database(store);
        let index = self.check_index(store);
        let performance = self.check_performance(store);
        let resources = self.check_resources(store, &database);

        let mut alerts = Vec::new();

        // Generate alerts based on metrics
        self.generate_database_alerts(&database, &mut alerts);
        self.generate_index_alerts(&index, &mut alerts);
        self.generate_performance_alerts(&performance, &mut alerts);
        self.generate_resource_alerts(&resources, &mut alerts);

        // Determine overall status
        let status = self.determine_status(&alerts);

        let uptime = SystemTime::now().duration_since(self.start_time).ok();

        Ok(HealthReport {
            status,
            timestamp: SystemTime::now(),
            database,
            index,
            performance,
            resources,
            alerts,
            uptime,
        })
    }

    fn check_database(&self, store: &VecStore) -> DatabaseHealth {
        let total_vectors = store.len() + store.deleted_count();
        let active_vectors = store.active_count();
        let deleted_vectors = store.deleted_count();
        let dimension = store.dimension();

        let deletion_ratio = if total_vectors > 0 {
            deleted_vectors as f64 / total_vectors as f64
        } else {
            0.0
        };

        // Storage efficiency: how well we're using space
        let storage_efficiency = if total_vectors > 0 {
            (active_vectors as f64 / total_vectors as f64) * 100.0
        } else {
            100.0
        };

        DatabaseHealth {
            total_vectors,
            active_vectors,
            deleted_vectors,
            dimension,
            deletion_ratio,
            storage_efficiency,
        }
    }

    fn check_index(&self, _store: &VecStore) -> IndexHealth {
        // For now, return basic HNSW index health
        IndexHealth {
            index_type: "HNSW".to_string(),
            integrity_ok: true,
            avg_degree: None,
            fragmentation: 0.0,
            last_rebuild: None,
        }
    }

    fn check_performance(&self, _store: &VecStore) -> PerformanceHealth {
        // These would typically be collected from metrics over time
        PerformanceHealth {
            avg_query_latency_ms: None,
            p95_query_latency_ms: None,
            qps: None,
            insert_throughput: None,
            performance_score: 85.0, // Placeholder
        }
    }

    fn check_resources(&self, _store: &VecStore, db: &DatabaseHealth) -> ResourceHealth {
        // Estimate memory usage
        let vector_size = db.dimension * 4; // f32 = 4 bytes
        let vectors_memory = db.active_vectors * vector_size;
        let index_overhead = db.active_vectors * 64; // Rough HNSW overhead
        let memory_bytes = vectors_memory + index_overhead;

        let memory_per_vector = if db.active_vectors > 0 {
            memory_bytes as f64 / db.active_vectors as f64
        } else {
            0.0
        };

        ResourceHealth {
            memory_bytes,
            disk_bytes: 0, // Would need actual disk measurement
            memory_per_vector,
            memory_utilization: 0.0, // Placeholder
            disk_utilization: 0.0,
        }
    }

    fn generate_database_alerts(&self, db: &DatabaseHealth, alerts: &mut Vec<Alert>) {
        // Check deletion ratio
        if db.deletion_ratio >= self.config.deletion_ratio_critical {
            alerts.push(Alert {
                severity: AlertSeverity::Critical,
                category: AlertCategory::Storage,
                message: format!(
                    "High deletion ratio: {:.1}% of vectors are deleted",
                    db.deletion_ratio * 100.0
                ),
                value: Some(db.deletion_ratio * 100.0),
                recommendation: Some("Run compaction to reclaim space".to_string()),
            });
        } else if db.deletion_ratio >= self.config.deletion_ratio_warning {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                category: AlertCategory::Storage,
                message: format!(
                    "Elevated deletion ratio: {:.1}% of vectors are deleted",
                    db.deletion_ratio * 100.0
                ),
                value: Some(db.deletion_ratio * 100.0),
                recommendation: Some("Consider running compaction soon".to_string()),
            });
        }

        // Check if database is empty
        if db.active_vectors == 0 && db.total_vectors > 0 {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                category: AlertCategory::Capacity,
                message: "Database has no active vectors".to_string(),
                value: None,
                recommendation: Some("All vectors have been deleted".to_string()),
            });
        }

        // Check storage efficiency
        if db.storage_efficiency < 50.0 && db.total_vectors > 100 {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                category: AlertCategory::Storage,
                message: format!("Low storage efficiency: {:.1}%", db.storage_efficiency),
                value: Some(db.storage_efficiency),
                recommendation: Some("Run compaction to improve efficiency".to_string()),
            });
        }
    }

    fn generate_index_alerts(&self, index: &IndexHealth, alerts: &mut Vec<Alert>) {
        if !index.integrity_ok {
            alerts.push(Alert {
                severity: AlertSeverity::Critical,
                category: AlertCategory::Index,
                message: "Index integrity check failed".to_string(),
                value: None,
                recommendation: Some("Rebuild index from scratch".to_string()),
            });
        }

        if index.fragmentation >= self.config.fragmentation_warning {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                category: AlertCategory::Index,
                message: format!("High index fragmentation: {:.1}", index.fragmentation),
                value: Some(index.fragmentation),
                recommendation: Some("Consider rebuilding index".to_string()),
            });
        }
    }

    fn generate_performance_alerts(&self, perf: &PerformanceHealth, alerts: &mut Vec<Alert>) {
        if let Some(latency) = perf.p95_query_latency_ms {
            if latency >= self.config.latency_warning_ms {
                alerts.push(Alert {
                    severity: AlertSeverity::Warning,
                    category: AlertCategory::Performance,
                    message: format!("High query latency: {:.2}ms (p95)", latency),
                    value: Some(latency),
                    recommendation: Some(
                        "Check index parameters or consider upgrading hardware".to_string(),
                    ),
                });
            }
        }

        if perf.performance_score < self.config.min_performance_score {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                category: AlertCategory::Performance,
                message: format!("Low performance score: {:.1}", perf.performance_score),
                value: Some(perf.performance_score),
                recommendation: Some("Review configuration and system resources".to_string()),
            });
        }
    }

    fn generate_resource_alerts(&self, resources: &ResourceHealth, alerts: &mut Vec<Alert>) {
        if resources.memory_utilization >= self.config.memory_warning {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                category: AlertCategory::Resource,
                message: format!(
                    "High memory utilization: {:.1}%",
                    resources.memory_utilization
                ),
                value: Some(resources.memory_utilization),
                recommendation: Some("Consider adding memory or reducing dataset size".to_string()),
            });
        }

        if resources.disk_utilization >= 90.0 {
            alerts.push(Alert {
                severity: AlertSeverity::Critical,
                category: AlertCategory::Resource,
                message: format!(
                    "Critical disk utilization: {:.1}%",
                    resources.disk_utilization
                ),
                value: Some(resources.disk_utilization),
                recommendation: Some("Free up disk space immediately".to_string()),
            });
        }
    }

    fn determine_status(&self, alerts: &[Alert]) -> HealthStatus {
        let has_critical = alerts.iter().any(|a| a.severity == AlertSeverity::Critical);
        let has_warning = alerts.iter().any(|a| a.severity == AlertSeverity::Warning);

        if has_critical {
            HealthStatus::Unhealthy
        } else if has_warning {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }
}

/// Print health report in human-readable format
pub fn print_health_report(report: &HealthReport) {
    println!("\n{}", "=".repeat(80));
    println!("VecStore Health Report");
    println!("{}", "=".repeat(80));

    // Overall status
    let status_icon = match report.status {
        HealthStatus::Healthy => "✅",
        HealthStatus::Degraded => "⚠️",
        HealthStatus::Unhealthy => "❌",
    };
    println!(
        "\n{} Overall Status: {}",
        status_icon,
        report.status.as_str().to_uppercase()
    );

    if let Some(uptime) = report.uptime {
        println!("⏱️  Uptime: {:?}", uptime);
    }

    // Database
    println!("\n📊 Database:");
    println!("  Total vectors: {}", report.database.total_vectors);
    println!("  Active vectors: {}", report.database.active_vectors);
    println!("  Deleted vectors: {}", report.database.deleted_vectors);
    println!("  Dimension: {}", report.database.dimension);
    println!(
        "  Deletion ratio: {:.1}%",
        report.database.deletion_ratio * 100.0
    );
    println!(
        "  Storage efficiency: {:.1}%",
        report.database.storage_efficiency
    );

    // Index
    println!("\n🔍 Index:");
    println!("  Type: {}", report.index.index_type);
    println!(
        "  Integrity: {}",
        if report.index.integrity_ok {
            "✓ OK"
        } else {
            "✗ Failed"
        }
    );
    println!("  Fragmentation: {:.1}", report.index.fragmentation);

    // Performance
    println!("\n⚡ Performance:");
    println!(
        "  Performance score: {:.1}/100",
        report.performance.performance_score
    );
    if let Some(latency) = report.performance.avg_query_latency_ms {
        println!("  Avg query latency: {:.2}ms", latency);
    }
    if let Some(p95) = report.performance.p95_query_latency_ms {
        println!("  P95 query latency: {:.2}ms", p95);
    }
    if let Some(qps) = report.performance.qps {
        println!("  QPS: {:.0}", qps);
    }

    // Resources
    println!("\n💾 Resources:");
    println!(
        "  Memory usage: {:.2} MB",
        report.resources.memory_bytes as f64 / 1_000_000.0
    );
    println!(
        "  Memory per vector: {:.1} bytes",
        report.resources.memory_per_vector
    );

    // Alerts
    if !report.alerts.is_empty() {
        println!("\n🚨 Alerts ({}):", report.alerts.len());
        for alert in &report.alerts {
            let icon = match alert.severity {
                AlertSeverity::Info => "ℹ️",
                AlertSeverity::Warning => "⚠️",
                AlertSeverity::Critical => "❌",
            };
            println!(
                "  {} [{}] {}",
                icon,
                format!("{:?}", alert.category),
                alert.message
            );
            if let Some(rec) = &alert.recommendation {
                println!("     → {}", rec);
            }
        }
    } else {
        println!("\n✅ No alerts");
    }

    println!("\n{}", "=".repeat(80));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_health_check() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let store = VecStore::open(temp_dir.path().join("test.db"))?;

        let checker = HealthChecker::default();
        let report = checker.check(&store)?;

        assert_eq!(report.status, HealthStatus::Healthy);
        assert_eq!(report.database.active_vectors, 0);
        assert!(report.alerts.is_empty());

        Ok(())
    }

    #[test]
    fn test_deletion_ratio_alert() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut store = VecStore::open(temp_dir.path().join("test.db"))?;

        // Insert and delete vectors to trigger alert
        for i in 0..10 {
            store.upsert(
                format!("vec_{}", i),
                vec![1.0, 2.0, 3.0],
                crate::store::Metadata {
                    fields: std::collections::HashMap::new(),
                },
            )?;
        }

        // Delete 6 out of 10 (60% deletion ratio - this exceeds critical threshold of 50%)
        for i in 0..6 {
            store.soft_delete(&format!("vec_{}", i))?;
        }

        let checker = HealthChecker::default();
        let report = checker.check(&store)?;

        // 60% deletion ratio should trigger Unhealthy status
        assert_eq!(report.status, HealthStatus::Unhealthy);
        assert!(report
            .alerts
            .iter()
            .any(|a| a.severity == AlertSeverity::Critical));

        Ok(())
    }
}
