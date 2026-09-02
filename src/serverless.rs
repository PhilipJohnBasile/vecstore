//! Serverless Auto-Scaling Infrastructure
//!
//! Provides Pinecone-like serverless capabilities with automatic scaling,
//! pay-per-query pricing model support, and zero-to-hero scaling.
//!
//! # Features
//!
//! - **Auto-Scaling**: Scale from 0 to millions of queries automatically
//! - **Pod Management**: Automatic pod provisioning and deprovisioning
//! - **Cost Optimization**: Pay only for what you use
//! - **Cold Start Optimization**: Fast warm-up from cold state
//! - **Load Balancing**: Intelligent query routing
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::serverless::{ServerlessConfig, ServerlessCluster};
//!
//! let config = ServerlessConfig::new()
//!     .with_min_replicas(0)
//!     .with_max_replicas(100)
//!     .with_scale_down_delay_secs(300);
//!
//! let cluster = ServerlessCluster::new(config)?;
//!
//! // Cluster scales automatically based on load
//! let results = cluster.query(&query_vec, 10).await?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{
    RwLock,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use crate::error::Result;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Serverless cluster configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerlessConfig {
    /// Minimum number of replicas (0 for scale-to-zero)
    pub min_replicas: usize,
    /// Maximum number of replicas
    pub max_replicas: usize,
    /// Target queries per second per replica
    pub target_qps_per_replica: f64,
    /// Scale down delay in seconds
    pub scale_down_delay_secs: u64,
    /// Cold start timeout in milliseconds
    pub cold_start_timeout_ms: u64,
    /// Enable predictive scaling
    pub predictive_scaling: bool,
    /// Cost per query (for billing)
    pub cost_per_query: f64,
    /// Cost per GB stored per month
    pub cost_per_gb_month: f64,
}

impl Default for ServerlessConfig {
    fn default() -> Self {
        Self {
            min_replicas: 0,
            max_replicas: 100,
            target_qps_per_replica: 100.0,
            scale_down_delay_secs: 300,
            cold_start_timeout_ms: 5000,
            predictive_scaling: true,
            cost_per_query: 0.00001,
            cost_per_gb_month: 0.25,
        }
    }
}

impl ServerlessConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_min_replicas(mut self, min: usize) -> Self {
        self.min_replicas = min;
        self
    }

    pub fn with_max_replicas(mut self, max: usize) -> Self {
        self.max_replicas = max;
        self
    }

    pub fn with_scale_down_delay_secs(mut self, secs: u64) -> Self {
        self.scale_down_delay_secs = secs;
        self
    }
}

// ============================================================================
// REPLICA STATE
// ============================================================================

/// State of a single replica
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaState {
    /// Replica is starting up
    Starting,
    /// Replica is ready to serve queries
    Ready,
    /// Replica is draining (no new queries)
    Draining,
    /// Replica is stopped
    Stopped,
}

/// Replica information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaInfo {
    pub id: String,
    pub state: ReplicaState,
    pub started_at: Option<u64>,
    pub queries_served: u64,
    pub avg_latency_ms: f64,
    pub current_qps: f64,
}

// ============================================================================
// AUTOSCALER
// ============================================================================

/// Autoscaling decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingDecision {
    /// No change needed
    NoChange,
    /// Scale up by N replicas
    ScaleUp(usize),
    /// Scale down by N replicas
    ScaleDown(usize),
}

/// Autoscaler for serverless cluster
pub struct Autoscaler {
    config: ServerlessConfig,
    /// Current QPS measurements
    qps_history: RwLock<Vec<(Instant, f64)>>,
    /// Last scale down time
    last_scale_down: RwLock<Option<Instant>>,
    /// Predictive model state
    hourly_patterns: RwLock<Vec<f64>>,
}

impl Autoscaler {
    pub fn new(config: ServerlessConfig) -> Self {
        Self {
            config,
            qps_history: RwLock::new(Vec::new()),
            last_scale_down: RwLock::new(None),
            hourly_patterns: RwLock::new(vec![1.0; 24]),
        }
    }

    /// Record current QPS
    pub fn record_qps(&self, qps: f64) {
        let Ok(mut history) = self.qps_history.write() else {
            return;
        };
        let now = Instant::now();

        // Keep last 5 minutes of data
        history.retain(|(t, _)| now.duration_since(*t) < Duration::from_secs(300));
        history.push((now, qps));

        // Update hourly pattern for predictive scaling
        if self.config.predictive_scaling {
            let Ok(duration) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            else {
                return;
            };
            let hour = (duration.as_secs() / 3600 % 24) as usize;

            let Ok(mut patterns) = self.hourly_patterns.write() else {
                return;
            };
            // Exponential moving average
            patterns[hour] = patterns[hour] * 0.9 + qps * 0.1;
        }
    }

    /// Get average QPS over last N seconds
    fn avg_qps(&self, seconds: u64) -> f64 {
        let Ok(history) = self.qps_history.read() else {
            return 0.0;
        };
        let now = Instant::now();
        let cutoff = Duration::from_secs(seconds);

        let recent: Vec<f64> = history
            .iter()
            .filter(|(t, _)| now.duration_since(*t) < cutoff)
            .map(|(_, q)| *q)
            .collect();

        if recent.is_empty() {
            0.0
        } else {
            recent.iter().sum::<f64>() / recent.len() as f64
        }
    }

    /// Get predicted QPS for next hour
    fn predicted_qps(&self) -> f64 {
        if !self.config.predictive_scaling {
            return self.avg_qps(60);
        }

        let Ok(duration) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        else {
            return self.avg_qps(60);
        };
        let next_hour = ((duration.as_secs() / 3600 + 1) % 24) as usize;

        let Ok(patterns) = self.hourly_patterns.read() else {
            return self.avg_qps(60);
        };
        patterns[next_hour]
    }

    /// Make scaling decision
    pub fn decide(&self, current_replicas: usize) -> ScalingDecision {
        let current_qps = self.avg_qps(60);
        let predicted_qps = self.predicted_qps();

        // Use higher of current and predicted
        let target_qps = current_qps.max(predicted_qps * 0.8);

        // Calculate desired replicas
        let desired = if target_qps == 0.0 {
            self.config.min_replicas
        } else {
            ((target_qps / self.config.target_qps_per_replica).ceil() as usize)
                .max(self.config.min_replicas)
                .min(self.config.max_replicas)
        };

        if desired > current_replicas {
            // Scale up immediately
            ScalingDecision::ScaleUp(desired - current_replicas)
        } else if desired < current_replicas {
            // Check scale down delay
            let Ok(last_scale) = self.last_scale_down.read() else {
                return ScalingDecision::NoChange;
            };
            if let Some(last) = *last_scale
                && last.elapsed() < Duration::from_secs(self.config.scale_down_delay_secs)
            {
                return ScalingDecision::NoChange;
            }
            ScalingDecision::ScaleDown(current_replicas - desired)
        } else {
            ScalingDecision::NoChange
        }
    }

    /// Mark scale down event
    pub fn mark_scale_down(&self) {
        let Ok(mut guard) = self.last_scale_down.write() else {
            return;
        };
        *guard = Some(Instant::now());
    }
}

// ============================================================================
// LOAD BALANCER
// ============================================================================

/// Load balancing strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalanceStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Weighted random
    WeightedRandom,
    /// Latency-aware
    LatencyAware,
}

/// Load balancer for query routing
pub struct LoadBalancer {
    strategy: LoadBalanceStrategy,
    /// Current replica connections
    connections: RwLock<HashMap<String, AtomicUsize>>,
    /// Replica latencies
    latencies: RwLock<HashMap<String, f64>>,
    /// Round-robin counter
    rr_counter: AtomicU64,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalanceStrategy) -> Self {
        Self {
            strategy,
            connections: RwLock::new(HashMap::new()),
            latencies: RwLock::new(HashMap::new()),
            rr_counter: AtomicU64::new(0),
        }
    }

    /// Select replica for query
    pub fn select(&self, replicas: &[ReplicaInfo]) -> Option<String> {
        if replicas.is_empty() {
            return None;
        }

        let ready: Vec<_> = replicas
            .iter()
            .filter(|r| r.state == ReplicaState::Ready)
            .collect();

        if ready.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                let idx = self.rr_counter.fetch_add(1, Ordering::SeqCst) as usize % ready.len();
                Some(ready[idx].id.clone())
            },
            LoadBalanceStrategy::LeastConnections => {
                let Ok(conns) = self.connections.read() else {
                    return None;
                };
                ready
                    .iter()
                    .min_by_key(|r| {
                        conns
                            .get(&r.id)
                            .map(|c| c.load(Ordering::SeqCst))
                            .unwrap_or(0)
                    })
                    .map(|r| r.id.clone())
            },
            LoadBalanceStrategy::LatencyAware => {
                let Ok(lats) = self.latencies.read() else {
                    return None;
                };
                ready
                    .iter()
                    .min_by(|a, b| {
                        let la = lats.get(&a.id).unwrap_or(&f64::MAX);
                        let lb = lats.get(&b.id).unwrap_or(&f64::MAX);
                        la.total_cmp(lb)
                    })
                    .map(|r| r.id.clone())
            },
            LoadBalanceStrategy::WeightedRandom => {
                // Simple random for now
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                Instant::now().hash(&mut hasher);
                let idx = hasher.finish() as usize % ready.len();
                Some(ready[idx].id.clone())
            },
        }
    }

    /// Record connection start
    pub fn connection_start(&self, replica_id: &str) {
        let Ok(mut conns) = self.connections.write() else {
            return;
        };
        conns
            .entry(replica_id.to_string())
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::SeqCst);
    }

    /// Record connection end with latency
    pub fn connection_end(&self, replica_id: &str, latency_ms: f64) {
        {
            let Ok(conns) = self.connections.read() else {
                return;
            };
            if let Some(c) = conns.get(replica_id) {
                c.fetch_sub(1, Ordering::SeqCst);
            }
        }

        // Update latency with exponential moving average
        let Ok(mut lats) = self.latencies.write() else {
            return;
        };
        let entry = lats.entry(replica_id.to_string()).or_insert(latency_ms);
        *entry = *entry * 0.9 + latency_ms * 0.1;
    }
}

// ============================================================================
// SERVERLESS CLUSTER
// ============================================================================

/// Serverless cluster manager
pub struct ServerlessCluster {
    config: ServerlessConfig,
    /// Autoscaler
    autoscaler: Autoscaler,
    /// Load balancer
    load_balancer: LoadBalancer,
    /// Current replicas
    replicas: RwLock<Vec<ReplicaInfo>>,
    /// Query counter
    query_count: AtomicU64,
    /// Total cost accumulated
    total_cost: RwLock<f64>,
    /// Storage size in bytes
    storage_bytes: AtomicU64,
}

impl ServerlessCluster {
    pub fn new(config: ServerlessConfig) -> Result<Self> {
        let autoscaler = Autoscaler::new(config.clone());
        let load_balancer = LoadBalancer::new(LoadBalanceStrategy::LatencyAware);

        Ok(Self {
            config,
            autoscaler,
            load_balancer,
            replicas: RwLock::new(Vec::new()),
            query_count: AtomicU64::new(0),
            total_cost: RwLock::new(0.0),
            storage_bytes: AtomicU64::new(0),
        })
    }

    /// Get current replica count
    pub fn replica_count(&self) -> usize {
        let Ok(replicas) = self.replicas.read() else {
            return 0;
        };
        replicas.len()
    }

    /// Scale to target replicas
    pub fn scale_to(&self, target: usize) -> Result<()> {
        let mut replicas = self.replicas.write().map_err(|_| {
            crate::error::VecStoreError::LockError("Failed to acquire replicas write lock".into())
        })?;
        let current = replicas.len();

        if target > current {
            // Scale up
            for i in 0..(target - current) {
                replicas.push(ReplicaInfo {
                    id: format!("replica-{}-{}", current + i, uuid_simple()),
                    state: ReplicaState::Starting,
                    started_at: Some(unix_timestamp()),
                    queries_served: 0,
                    avg_latency_ms: 0.0,
                    current_qps: 0.0,
                });
            }
        } else if target < current {
            // Scale down - mark for draining
            for replica in replicas.iter_mut().skip(target) {
                replica.state = ReplicaState::Draining;
            }
            self.autoscaler.mark_scale_down();
        }

        Ok(())
    }

    /// Simulate replica becoming ready
    pub fn mark_replica_ready(&self, replica_id: &str) {
        let Ok(mut replicas) = self.replicas.write() else {
            return;
        };
        if let Some(r) = replicas.iter_mut().find(|r| r.id == replica_id) {
            r.state = ReplicaState::Ready;
        }
    }

    /// Remove stopped replicas
    pub fn cleanup_stopped(&self) {
        let Ok(mut replicas) = self.replicas.write() else {
            return;
        };
        replicas.retain(|r| r.state != ReplicaState::Stopped);
    }

    /// Record a query
    pub fn record_query(&self, _latency_ms: f64) {
        self.query_count.fetch_add(1, Ordering::SeqCst);

        // Update cost
        {
            let Ok(mut cost) = self.total_cost.write() else {
                return;
            };
            *cost += self.config.cost_per_query;
        }

        // Update autoscaler
        let count = self.query_count.load(Ordering::SeqCst);
        // Approximate QPS over last second
        self.autoscaler.record_qps(count as f64 / 60.0);
    }

    /// Get scaling decision
    pub fn get_scaling_decision(&self) -> ScalingDecision {
        self.autoscaler.decide(self.replica_count())
    }

    /// Apply scaling decision
    pub fn apply_scaling(&self) -> Result<()> {
        let decision = self.get_scaling_decision();
        let current = self.replica_count();

        match decision {
            ScalingDecision::NoChange => Ok(()),
            ScalingDecision::ScaleUp(n) => self.scale_to(current + n),
            ScalingDecision::ScaleDown(n) => self.scale_to(current.saturating_sub(n)),
        }
    }

    /// Select replica for query
    pub fn select_replica(&self) -> Option<String> {
        let Ok(replicas) = self.replicas.read() else {
            return None;
        };
        self.load_balancer.select(&replicas)
    }

    /// Get cluster statistics
    pub fn stats(&self) -> ServerlessStats {
        let storage_gb =
            self.storage_bytes.load(Ordering::SeqCst) as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_queries = self.query_count.load(Ordering::SeqCst);

        let Ok(replicas) = self.replicas.read() else {
            let Ok(total_cost) = self.total_cost.read() else {
                return ServerlessStats {
                    total_replicas: 0,
                    ready_replicas: 0,
                    starting_replicas: 0,
                    draining_replicas: 0,
                    total_queries,
                    total_cost: 0.0,
                    storage_gb,
                    estimated_monthly_cost: storage_gb * self.config.cost_per_gb_month,
                };
            };
            return ServerlessStats {
                total_replicas: 0,
                ready_replicas: 0,
                starting_replicas: 0,
                draining_replicas: 0,
                total_queries,
                total_cost: *total_cost,
                storage_gb,
                estimated_monthly_cost: *total_cost * 30.0 * 24.0
                    + storage_gb * self.config.cost_per_gb_month,
            };
        };

        let ready = replicas
            .iter()
            .filter(|r| r.state == ReplicaState::Ready)
            .count();
        let Ok(total_cost) = self.total_cost.read() else {
            return ServerlessStats {
                total_replicas: replicas.len(),
                ready_replicas: ready,
                starting_replicas: replicas
                    .iter()
                    .filter(|r| r.state == ReplicaState::Starting)
                    .count(),
                draining_replicas: replicas
                    .iter()
                    .filter(|r| r.state == ReplicaState::Draining)
                    .count(),
                total_queries,
                total_cost: 0.0,
                storage_gb,
                estimated_monthly_cost: storage_gb * self.config.cost_per_gb_month,
            };
        };

        ServerlessStats {
            total_replicas: replicas.len(),
            ready_replicas: ready,
            starting_replicas: replicas
                .iter()
                .filter(|r| r.state == ReplicaState::Starting)
                .count(),
            draining_replicas: replicas
                .iter()
                .filter(|r| r.state == ReplicaState::Draining)
                .count(),
            total_queries,
            total_cost: *total_cost,
            storage_gb,
            estimated_monthly_cost: *total_cost * 30.0 * 24.0
                + storage_gb * self.config.cost_per_gb_month,
        }
    }

    /// Get billing estimate
    pub fn billing_estimate(&self, queries_per_month: u64, storage_gb: f64) -> BillingEstimate {
        let query_cost = queries_per_month as f64 * self.config.cost_per_query;
        let storage_cost = storage_gb * self.config.cost_per_gb_month;

        BillingEstimate {
            queries_per_month,
            storage_gb,
            query_cost,
            storage_cost,
            total_monthly_cost: query_cost + storage_cost,
        }
    }
}

/// Serverless cluster statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerlessStats {
    pub total_replicas: usize,
    pub ready_replicas: usize,
    pub starting_replicas: usize,
    pub draining_replicas: usize,
    pub total_queries: u64,
    pub total_cost: f64,
    pub storage_gb: f64,
    pub estimated_monthly_cost: f64,
}

/// Billing estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingEstimate {
    pub queries_per_month: u64,
    pub storage_gb: f64,
    pub query_cost: f64,
    pub storage_cost: f64,
    pub total_monthly_cost: f64,
}

// ============================================================================
// COLD START OPTIMIZER
// ============================================================================

/// Cold start optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartConfig {
    /// Pre-warm replicas during low-traffic periods
    pub pre_warm_enabled: bool,
    /// Keep minimum replicas warm
    pub keep_warm_count: usize,
    /// Use snapshot for fast restore
    pub use_snapshots: bool,
    /// Lazy load vectors on first access
    pub lazy_loading: bool,
}

impl Default for ColdStartConfig {
    fn default() -> Self {
        Self {
            pre_warm_enabled: true,
            keep_warm_count: 1,
            use_snapshots: true,
            lazy_loading: true,
        }
    }
}

/// Cold start optimizer
pub struct ColdStartOptimizer {
    config: ColdStartConfig,
    /// Snapshot data for fast restore
    snapshot_data: RwLock<Option<Vec<u8>>>,
    /// Last snapshot time
    last_snapshot: RwLock<Option<Instant>>,
}

impl ColdStartOptimizer {
    pub fn new(config: ColdStartConfig) -> Self {
        Self {
            config,
            snapshot_data: RwLock::new(None),
            last_snapshot: RwLock::new(None),
        }
    }

    /// Create snapshot for fast restore
    pub fn create_snapshot(&self, data: Vec<u8>) {
        let Ok(mut snapshot) = self.snapshot_data.write() else {
            return;
        };
        *snapshot = Some(data);
        let Ok(mut last) = self.last_snapshot.write() else {
            return;
        };
        *last = Some(Instant::now());
    }

    /// Get snapshot for restore
    pub fn get_snapshot(&self) -> Option<Vec<u8>> {
        let Ok(snapshot) = self.snapshot_data.read() else {
            return None;
        };
        snapshot.clone()
    }

    /// Check if snapshot is fresh enough
    pub fn is_snapshot_valid(&self, max_age_secs: u64) -> bool {
        let Ok(last_snapshot) = self.last_snapshot.read() else {
            return false;
        };
        if let Some(last) = *last_snapshot {
            last.elapsed() < Duration::from_secs(max_age_secs)
        } else {
            false
        }
    }
}

// ============================================================================
// HELPERS
// ============================================================================

fn uuid_simple() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    Instant::now().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoscaler() {
        let config = ServerlessConfig {
            min_replicas: 0,
            max_replicas: 10,
            target_qps_per_replica: 100.0,
            ..Default::default()
        };

        let autoscaler = Autoscaler::new(config);

        // No load -> scale to min
        assert!(matches!(
            autoscaler.decide(5),
            ScalingDecision::ScaleDown(_)
        ));

        // Record high QPS
        autoscaler.record_qps(500.0);
        autoscaler.record_qps(500.0);

        // Should want to scale up
        let decision = autoscaler.decide(2);
        assert!(matches!(decision, ScalingDecision::ScaleUp(_)));
    }

    #[test]
    fn test_billing_estimate() {
        let config = ServerlessConfig {
            cost_per_query: 0.00001,
            cost_per_gb_month: 0.25,
            ..Default::default()
        };

        let cluster = ServerlessCluster::new(config).unwrap();

        let estimate = cluster.billing_estimate(1_000_000, 10.0);
        assert_eq!(estimate.query_cost, 10.0); // 1M * $0.00001
        assert_eq!(estimate.storage_cost, 2.5); // 10GB * $0.25
        assert_eq!(estimate.total_monthly_cost, 12.5);
    }
}
