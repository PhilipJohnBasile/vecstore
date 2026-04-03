// Query Router - Intelligent query routing and load balancing
// Routes queries to optimal replicas based on load, locality, and query characteristics

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Load balancing strategy
    pub strategy: LoadBalanceStrategy,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Max retries per request
    pub max_retries: u32,
    /// Circuit breaker threshold (failures before open)
    pub circuit_breaker_threshold: u32,
    /// Circuit breaker reset timeout
    pub circuit_breaker_timeout: Duration,
    /// Enable query-aware routing
    pub query_aware_routing: bool,
    /// Enable locality-aware routing
    pub locality_aware_routing: bool,
    /// Locality zone
    pub local_zone: Option<String>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalanceStrategy::WeightedRoundRobin,
            health_check_interval: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(30),
            query_aware_routing: true,
            locality_aware_routing: true,
            local_zone: None,
        }
    }
}

/// Load balancing strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LoadBalanceStrategy {
    /// Simple round robin
    RoundRobin,
    /// Weighted round robin based on capacity
    WeightedRoundRobin,
    /// Least connections
    LeastConnections,
    /// Least latency
    LeastLatency,
    /// Random selection
    Random,
    /// Consistent hashing (for cache affinity)
    ConsistentHashing,
    /// Adaptive (combines multiple signals)
    Adaptive,
}

/// Replica node information
#[derive(Debug)]
pub struct ReplicaNode {
    /// Node ID
    pub id: String,
    /// Node address (host:port)
    pub address: String,
    /// Zone/region
    pub zone: String,
    /// Weight (higher = more traffic)
    pub weight: u32,
    /// Node capabilities
    pub capabilities: NodeCapabilities,
    /// Current state
    state: Arc<RwLock<NodeState>>,
}

/// Node capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Maximum QPS
    pub max_qps: u32,
    /// Available memory GB
    pub memory_gb: f32,
    /// Has GPU
    pub has_gpu: bool,
    /// Supported index types
    pub index_types: Vec<String>,
    /// Maximum vector dimensions
    pub max_dimensions: usize,
}

/// Node state (mutable)
#[derive(Debug, Clone)]
struct NodeState {
    /// Health status
    health: HealthStatus,
    /// Active connections
    active_connections: u32,
    /// Recent latencies (sliding window)
    latencies: VecDeque<Duration>,
    /// Recent errors
    recent_errors: u32,
    /// Last health check
    last_health_check: Instant,
    /// Circuit breaker state
    circuit_state: CircuitState,
    /// Circuit breaker opened at
    circuit_opened_at: Option<Instant>,
}

/// Health status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Circuit breaker state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Query characteristics for routing decisions
#[derive(Debug, Clone)]
pub struct QueryCharacteristics {
    /// Vector dimensions
    pub dimensions: usize,
    /// Number of results requested
    pub k: usize,
    /// Has filters
    pub has_filters: bool,
    /// Filter complexity (estimated)
    pub filter_complexity: FilterComplexity,
    /// Requires reranking
    pub requires_reranking: bool,
    /// Query vector hash (for cache affinity)
    pub query_hash: Option<u64>,
    /// Priority level
    pub priority: QueryPriority,
}

/// Filter complexity estimation
#[derive(Debug, Clone, PartialEq)]
pub enum FilterComplexity {
    None,
    Simple,
    Moderate,
    Complex,
}

/// Query priority
#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq)]
pub enum QueryPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Routing decision
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected node
    pub node_id: String,
    /// Node address
    pub address: String,
    /// Routing reason
    pub reason: RoutingReason,
    /// Fallback nodes (in order of preference)
    pub fallbacks: Vec<String>,
    /// Estimated latency
    pub estimated_latency: Option<Duration>,
}

/// Reason for routing decision
#[derive(Debug, Clone)]
pub enum RoutingReason {
    LoadBalance,
    LocalityPreference,
    CacheAffinity,
    CapabilityMatch,
    LeastLatency,
    LeastConnections,
    Random,
}

/// Query router
pub struct QueryRouter {
    config: RouterConfig,
    /// All registered nodes
    nodes: RwLock<HashMap<String, Arc<ReplicaNode>>>,
    /// Round robin counter
    rr_counter: AtomicUsize,
    /// Consistent hash ring
    hash_ring: RwLock<ConsistentHashRing>,
    /// Router metrics
    metrics: RouterMetrics,
}

/// Consistent hash ring for cache affinity
struct ConsistentHashRing {
    /// Virtual nodes per real node
    virtual_nodes: usize,
    /// Ring: hash -> node_id
    ring: Vec<(u64, String)>,
}

impl ConsistentHashRing {
    fn new(virtual_nodes: usize) -> Self {
        Self {
            virtual_nodes,
            ring: Vec::new(),
        }
    }

    fn add_node(&mut self, node_id: &str) {
        for i in 0..self.virtual_nodes {
            let key = format!("{}#{}", node_id, i);
            let hash = Self::hash(&key);
            self.ring.push((hash, node_id.to_string()));
        }
        self.ring.sort_by_key(|(h, _)| *h);
    }

    fn remove_node(&mut self, node_id: &str) {
        self.ring.retain(|(_, id)| id != node_id);
    }

    fn get_node(&self, key: u64) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }

        // Binary search for the first node with hash >= key
        let idx = match self.ring.binary_search_by_key(&key, |(h, _)| *h) {
            Ok(i) => i,
            Err(i) => i % self.ring.len(),
        };

        Some(&self.ring[idx].1)
    }

    fn hash(key: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

/// Router metrics
#[derive(Debug, Default)]
struct RouterMetrics {
    requests_total: AtomicU64,
    requests_succeeded: AtomicU64,
    requests_failed: AtomicU64,
    requests_retried: AtomicU64,
    circuit_breaks: AtomicU64,
    locality_hits: AtomicU64,
    cache_affinity_hits: AtomicU64,
}

impl QueryRouter {
    /// Create a new query router
    pub fn new(config: RouterConfig) -> Self {
        Self {
            config,
            nodes: RwLock::new(HashMap::new()),
            rr_counter: AtomicUsize::new(0),
            hash_ring: RwLock::new(ConsistentHashRing::new(150)),
            metrics: RouterMetrics::default(),
        }
    }

    /// Register a replica node
    pub fn register_node(&self, node: ReplicaNode) -> Result<()> {
        let node_id = node.id.clone();
        let node = Arc::new(node);

        let mut nodes_guard = self.nodes.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire nodes write lock".into())
        })?;
        nodes_guard.insert(node_id.clone(), node);
        drop(nodes_guard);

        let mut hash_ring_guard = self.hash_ring.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire hash_ring write lock".into())
        })?;
        hash_ring_guard.add_node(&node_id);

        Ok(())
    }

    /// Deregister a replica node
    pub fn deregister_node(&self, node_id: &str) -> Result<()> {
        let mut nodes_guard = self.nodes.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire nodes write lock".into())
        })?;
        nodes_guard.remove(node_id);
        drop(nodes_guard);

        let mut hash_ring_guard = self.hash_ring.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire hash_ring write lock".into())
        })?;
        hash_ring_guard.remove_node(node_id);
        Ok(())
    }

    /// Route a query to the best replica
    pub fn route(&self, query: &QueryCharacteristics) -> Result<RoutingDecision> {
        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

        let nodes = self.nodes.read().map_err(|_| {
            VecStoreError::Internal("Failed to acquire nodes read lock".into())
        })?;
        let available: Vec<_> = nodes.values()
            .filter(|n| self.is_available(n))
            .cloned()
            .collect();

        if available.is_empty() {
            return Err(VecStoreError::NotFound("No available nodes".into()));
        }

        // Apply routing strategy
        let (selected, reason) = match self.config.strategy {
            LoadBalanceStrategy::RoundRobin => self.round_robin(&available),
            LoadBalanceStrategy::WeightedRoundRobin => self.weighted_round_robin(&available),
            LoadBalanceStrategy::LeastConnections => self.least_connections(&available),
            LoadBalanceStrategy::LeastLatency => self.least_latency(&available),
            LoadBalanceStrategy::Random => self.random(&available),
            LoadBalanceStrategy::ConsistentHashing => self.consistent_hash(&available, query),
            LoadBalanceStrategy::Adaptive => self.adaptive(&available, query),
        };

        // Apply locality preference if enabled
        let (selected, reason) = if self.config.locality_aware_routing {
            self.apply_locality_preference(selected, reason, &available)
        } else {
            (selected, reason)
        };

        // Build fallback list
        let fallbacks = self.build_fallbacks(&selected, &available);

        let estimated_latency = if let Ok(state) = selected.state.read() {
            if !state.latencies.is_empty() {
                let avg: Duration = state.latencies.iter().sum::<Duration>()
                    / state.latencies.len() as u32;
                Some(avg)
            } else {
                None
            }
        } else {
            None
        };

        Ok(RoutingDecision {
            node_id: selected.id.clone(),
            address: selected.address.clone(),
            reason,
            fallbacks,
            estimated_latency,
        })
    }

    fn is_available(&self, node: &ReplicaNode) -> bool {
        let Ok(state) = node.state.read() else { return false; };

        // Check circuit breaker
        if state.circuit_state == CircuitState::Open
            && let Some(opened_at) = state.circuit_opened_at
                && opened_at.elapsed() < self.config.circuit_breaker_timeout {
                    return false;
                }
                // Allow half-open state

        state.health != HealthStatus::Unhealthy
    }

    fn round_robin(&self, nodes: &[Arc<ReplicaNode>]) -> (Arc<ReplicaNode>, RoutingReason) {
        let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed) % nodes.len();
        (nodes[idx].clone(), RoutingReason::LoadBalance)
    }

    fn weighted_round_robin(&self, nodes: &[Arc<ReplicaNode>]) -> (Arc<ReplicaNode>, RoutingReason) {
        let total_weight: u32 = nodes.iter().map(|n| n.weight).sum();
        let counter = self.rr_counter.fetch_add(1, Ordering::Relaxed);
        let target = (counter as u32) % total_weight;

        let mut cumulative = 0u32;
        for node in nodes {
            cumulative += node.weight;
            if target < cumulative {
                return (node.clone(), RoutingReason::LoadBalance);
            }
        }

        (nodes[0].clone(), RoutingReason::LoadBalance)
    }

    fn least_connections(&self, nodes: &[Arc<ReplicaNode>]) -> (Arc<ReplicaNode>, RoutingReason) {
        let selected = nodes.iter()
            .min_by_key(|n| {
                n.state.read().map(|s| s.active_connections).unwrap_or(u32::MAX)
            })
            .unwrap();

        (selected.clone(), RoutingReason::LeastConnections)
    }

    fn least_latency(&self, nodes: &[Arc<ReplicaNode>]) -> (Arc<ReplicaNode>, RoutingReason) {
        let selected = nodes.iter()
            .min_by(|a, b| {
                let a_lat = self.avg_latency(a);
                let b_lat = self.avg_latency(b);
                a_lat.cmp(&b_lat)
            })
            .unwrap();

        (selected.clone(), RoutingReason::LeastLatency)
    }

    fn avg_latency(&self, node: &ReplicaNode) -> Duration {
        let Ok(state) = node.state.read() else { return Duration::from_millis(100); };
        if state.latencies.is_empty() {
            Duration::from_millis(100) // Default estimate
        } else {
            state.latencies.iter().sum::<Duration>() / state.latencies.len() as u32
        }
    }

    fn random(&self, nodes: &[Arc<ReplicaNode>]) -> (Arc<ReplicaNode>, RoutingReason) {
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as usize;
        let idx = seed % nodes.len();
        (nodes[idx].clone(), RoutingReason::Random)
    }

    fn consistent_hash(&self, nodes: &[Arc<ReplicaNode>], query: &QueryCharacteristics)
        -> (Arc<ReplicaNode>, RoutingReason)
    {
        // Check for cache affinity via consistent hash (Rust 1.92 if-let chain)
        if let Some(hash) = query.query_hash
            && let Ok(ring) = self.hash_ring.read()
            && let Some(node_id) = ring.get_node(hash)
            && let Some(node) = nodes.iter().find(|n| n.id == node_id)
        {
            self.metrics.cache_affinity_hits.fetch_add(1, Ordering::Relaxed);
            return (node.clone(), RoutingReason::CacheAffinity);
        }

        // Fallback to round robin
        self.round_robin(nodes)
    }

    fn adaptive(&self, nodes: &[Arc<ReplicaNode>], query: &QueryCharacteristics)
        -> (Arc<ReplicaNode>, RoutingReason)
    {
        // Score each node based on multiple factors
        let mut scores: Vec<(Arc<ReplicaNode>, f64)> = nodes.iter()
            .map(|n| {
                let score = self.compute_adaptive_score(n, query);
                (n.clone(), score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        (scores[0].0.clone(), RoutingReason::LoadBalance)
    }

    fn compute_adaptive_score(&self, node: &ReplicaNode, query: &QueryCharacteristics) -> f64 {
        let Ok(state) = node.state.read() else { return 0.0; };
        let mut score = 100.0;

        // Penalize based on connections (0-30 points)
        let conn_ratio = state.active_connections as f64 / node.capabilities.max_qps as f64;
        score -= conn_ratio * 30.0;

        // Penalize based on latency (0-30 points)
        if !state.latencies.is_empty() {
            let avg_ms = state.latencies.iter()
                .sum::<Duration>().as_millis() as f64 / state.latencies.len() as f64;
            score -= (avg_ms / 100.0).min(30.0);
        }

        // Penalize degraded health (20 points)
        if state.health == HealthStatus::Degraded {
            score -= 20.0;
        }

        // Bonus for GPU if query requires it
        if query.requires_reranking && node.capabilities.has_gpu {
            score += 15.0;
        }

        // Bonus for high-priority queries to less loaded nodes
        if query.priority >= QueryPriority::High {
            score += (1.0 - conn_ratio) * 10.0;
        }

        // Bonus for locality
        if self.config.locality_aware_routing
            && let Some(ref local_zone) = self.config.local_zone
                && &node.zone == local_zone {
                    score += 10.0;
                }

        score.max(0.0)
    }

    fn apply_locality_preference(
        &self,
        selected: Arc<ReplicaNode>,
        reason: RoutingReason,
        all_nodes: &[Arc<ReplicaNode>],
    ) -> (Arc<ReplicaNode>, RoutingReason) {
        if let Some(ref local_zone) = self.config.local_zone {
            // Check if selected is already local
            if &selected.zone == local_zone {
                self.metrics.locality_hits.fetch_add(1, Ordering::Relaxed);
                return (selected, reason);
            }

            // Try to find a local node with acceptable load
            let local_nodes: Vec<_> = all_nodes.iter()
                .filter(|n| &n.zone == local_zone)
                .collect();

            if !local_nodes.is_empty() {
                // Pick the least loaded local node
                let local = local_nodes.iter()
                    .min_by_key(|n| {
                        n.state.read().map(|s| s.active_connections).unwrap_or(u32::MAX)
                    })
                    .unwrap();

                let local_load = local.state.read().map(|s| s.active_connections).unwrap_or(u32::MAX);
                let selected_load = selected.state.read().map(|s| s.active_connections).unwrap_or(0);

                // Only prefer local if load difference is within 20%
                if local_load as f64 <= selected_load as f64 * 1.2 {
                    self.metrics.locality_hits.fetch_add(1, Ordering::Relaxed);
                    return ((*local).clone(), RoutingReason::LocalityPreference);
                }
            }
        }

        (selected, reason)
    }

    fn build_fallbacks(&self, selected: &ReplicaNode, all_nodes: &[Arc<ReplicaNode>]) -> Vec<String> {
        let mut fallbacks: Vec<_> = all_nodes.iter()
            .filter(|n| n.id != selected.id)
            .filter_map(|n| {
                let state = n.state.read().ok()?;
                let score = -(state.active_connections as i64);
                Some((n.id.clone(), score))
            })
            .collect();

        fallbacks.sort_by_key(|(_, score)| *score);
        fallbacks.into_iter().take(3).map(|(id, _)| id).collect()
    }

    /// Record request result (for metrics and circuit breaker)
    pub fn record_result(&self, node_id: &str, success: bool, latency: Duration) {
        let Ok(nodes) = self.nodes.read() else { return; };
        if let Some(node) = nodes.get(node_id) {
            let Ok(mut state) = node.state.write() else { return; };

            // Update latency window
            state.latencies.push_back(latency);
            if state.latencies.len() > 100 {
                state.latencies.pop_front();
            }

            if success {
                self.metrics.requests_succeeded.fetch_add(1, Ordering::Relaxed);
                state.recent_errors = 0;

                // Reset circuit breaker on success in half-open state
                if state.circuit_state == CircuitState::HalfOpen {
                    state.circuit_state = CircuitState::Closed;
                    state.circuit_opened_at = None;
                }
            } else {
                self.metrics.requests_failed.fetch_add(1, Ordering::Relaxed);
                state.recent_errors += 1;

                // Check circuit breaker threshold
                if state.recent_errors >= self.config.circuit_breaker_threshold
                    && state.circuit_state == CircuitState::Closed {
                        state.circuit_state = CircuitState::Open;
                        state.circuit_opened_at = Some(Instant::now());
                        self.metrics.circuit_breaks.fetch_add(1, Ordering::Relaxed);
                    }
            }

            state.active_connections = state.active_connections.saturating_sub(1);
        }
    }

    /// Increment active connections for a node
    pub fn increment_connections(&self, node_id: &str) {
        let Ok(nodes) = self.nodes.read() else { return; };
        if let Some(node) = nodes.get(node_id) {
            let Ok(mut state) = node.state.write() else { return; };
            state.active_connections += 1;
        }
    }

    /// Update node health status
    pub fn update_health(&self, node_id: &str, health: HealthStatus) {
        let Ok(nodes) = self.nodes.read() else { return; };
        if let Some(node) = nodes.get(node_id) {
            let Ok(mut state) = node.state.write() else { return; };
            state.health = health;
            state.last_health_check = Instant::now();
        }
    }

    /// Get router statistics
    pub fn get_stats(&self) -> RouterStats {
        let node_stats: Vec<NodeStats> = if let Ok(nodes) = self.nodes.read() {
            nodes.values()
                .filter_map(|n| {
                    let state = n.state.read().ok()?;
                    Some(NodeStats {
                        node_id: n.id.clone(),
                        address: n.address.clone(),
                        zone: n.zone.clone(),
                        health: state.health.clone(),
                        active_connections: state.active_connections,
                        avg_latency_ms: if state.latencies.is_empty() {
                            0.0
                        } else {
                            state.latencies.iter().sum::<Duration>().as_millis() as f64
                                / state.latencies.len() as f64
                        },
                        circuit_state: state.circuit_state.clone(),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        RouterStats {
            total_requests: self.metrics.requests_total.load(Ordering::Relaxed),
            succeeded_requests: self.metrics.requests_succeeded.load(Ordering::Relaxed),
            failed_requests: self.metrics.requests_failed.load(Ordering::Relaxed),
            retried_requests: self.metrics.requests_retried.load(Ordering::Relaxed),
            circuit_breaks: self.metrics.circuit_breaks.load(Ordering::Relaxed),
            locality_hit_rate: {
                let total = self.metrics.requests_total.load(Ordering::Relaxed);
                let hits = self.metrics.locality_hits.load(Ordering::Relaxed);
                if total > 0 { hits as f64 / total as f64 } else { 0.0 }
            },
            cache_affinity_hit_rate: {
                let total = self.metrics.requests_total.load(Ordering::Relaxed);
                let hits = self.metrics.cache_affinity_hits.load(Ordering::Relaxed);
                if total > 0 { hits as f64 / total as f64 } else { 0.0 }
            },
            node_stats,
        }
    }
}

impl ReplicaNode {
    /// Create a new replica node
    pub fn new(id: String, address: String, zone: String, weight: u32, capabilities: NodeCapabilities) -> Self {
        Self {
            id,
            address,
            zone,
            weight,
            capabilities,
            state: Arc::new(RwLock::new(NodeState {
                health: HealthStatus::Unknown,
                active_connections: 0,
                latencies: VecDeque::with_capacity(100),
                recent_errors: 0,
                last_health_check: Instant::now(),
                circuit_state: CircuitState::Closed,
                circuit_opened_at: None,
            })),
        }
    }
}

/// Router statistics
#[derive(Debug, Clone, Serialize)]
pub struct RouterStats {
    pub total_requests: u64,
    pub succeeded_requests: u64,
    pub failed_requests: u64,
    pub retried_requests: u64,
    pub circuit_breaks: u64,
    pub locality_hit_rate: f64,
    pub cache_affinity_hit_rate: f64,
    pub node_stats: Vec<NodeStats>,
}

/// Per-node statistics
#[derive(Debug, Clone, Serialize)]
pub struct NodeStats {
    pub node_id: String,
    pub address: String,
    pub zone: String,
    pub health: HealthStatus,
    pub active_connections: u32,
    pub avg_latency_ms: f64,
    pub circuit_state: CircuitState,
}

/// Request retry handler
pub struct RetryHandler {
    config: RetryConfig,
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub backoff_multiplier: f64,
    pub retryable_errors: Vec<String>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            retryable_errors: vec![
                "timeout".to_string(),
                "connection_refused".to_string(),
                "unavailable".to_string(),
            ],
        }
    }
}

impl RetryHandler {
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Calculate backoff duration for a retry attempt
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        let backoff = self.config.initial_backoff.as_millis() as f64
            * self.config.backoff_multiplier.powi(attempt as i32);
        let capped = backoff.min(self.config.max_backoff.as_millis() as f64);
        Duration::from_millis(capped as u64)
    }

    /// Check if an error is retryable
    pub fn is_retryable(&self, error: &str) -> bool {
        self.config.retryable_errors.iter()
            .any(|e| error.to_lowercase().contains(&e.to_lowercase()))
    }

    /// Check if we should retry
    pub fn should_retry(&self, attempt: u32, error: &str) -> bool {
        attempt < self.config.max_retries && self.is_retryable(error)
    }
}

/// Request context for tracing
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Request ID
    pub request_id: String,
    /// Trace ID
    pub trace_id: Option<String>,
    /// Span ID
    pub span_id: Option<String>,
    /// Start time
    pub start_time: Instant,
    /// Attempts made
    pub attempts: u32,
    /// Nodes tried
    pub nodes_tried: Vec<String>,
}

impl RequestContext {
    pub fn new() -> Self {
        Self {
            request_id: uuid_v4(),
            trace_id: None,
            span_id: None,
            start_time: Instant::now(),
            attempts: 0,
            nodes_tried: Vec::new(),
        }
    }

    pub fn record_attempt(&mut self, node_id: &str) {
        self.attempts += 1;
        self.nodes_tried.push(node_id.to_string());
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for RequestContext {
    fn default() -> Self {
        Self::new()
    }
}

fn uuid_v4() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    format!("{:016x}-{:04x}", now.as_nanos(), rand_u16())
}

fn rand_u16() -> u16 {
    use std::time::SystemTime;
    (SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() & 0xFFFF) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node(id: &str, zone: &str, weight: u32) -> ReplicaNode {
        ReplicaNode::new(
            id.to_string(),
            format!("{}:8080", id),
            zone.to_string(),
            weight,
            NodeCapabilities {
                max_qps: 1000,
                memory_gb: 16.0,
                has_gpu: false,
                index_types: vec!["hnsw".to_string()],
                max_dimensions: 1536,
            },
        )
    }

    #[test]
    fn test_round_robin() {
        let router = QueryRouter::new(RouterConfig {
            strategy: LoadBalanceStrategy::RoundRobin,
            ..Default::default()
        });

        router.register_node(create_test_node("node1", "us-east-1", 1)).unwrap();
        router.register_node(create_test_node("node2", "us-east-1", 1)).unwrap();
        router.register_node(create_test_node("node3", "us-east-1", 1)).unwrap();

        let query = QueryCharacteristics {
            dimensions: 768,
            k: 10,
            has_filters: false,
            filter_complexity: FilterComplexity::None,
            requires_reranking: false,
            query_hash: None,
            priority: QueryPriority::Normal,
        };

        let d1 = router.route(&query).unwrap();
        let d2 = router.route(&query).unwrap();
        let d3 = router.route(&query).unwrap();
        let d4 = router.route(&query).unwrap();

        // Should cycle through nodes
        assert_ne!(d1.node_id, d2.node_id);
        assert_ne!(d2.node_id, d3.node_id);
        assert_eq!(d1.node_id, d4.node_id); // Wraps around
    }

    #[test]
    fn test_locality_preference() {
        let router = QueryRouter::new(RouterConfig {
            strategy: LoadBalanceStrategy::RoundRobin,
            locality_aware_routing: true,
            local_zone: Some("us-east-1".to_string()),
            ..Default::default()
        });

        router.register_node(create_test_node("node1", "us-west-1", 1)).unwrap();
        router.register_node(create_test_node("node2", "us-east-1", 1)).unwrap();
        router.register_node(create_test_node("node3", "eu-west-1", 1)).unwrap();

        let query = QueryCharacteristics {
            dimensions: 768,
            k: 10,
            has_filters: false,
            filter_complexity: FilterComplexity::None,
            requires_reranking: false,
            query_hash: None,
            priority: QueryPriority::Normal,
        };

        // Should prefer local zone
        let decision = router.route(&query).unwrap();
        assert_eq!(decision.node_id, "node2");
    }

    #[test]
    fn test_circuit_breaker() {
        let router = QueryRouter::new(RouterConfig {
            circuit_breaker_threshold: 3,
            ..Default::default()
        });

        router.register_node(create_test_node("node1", "us-east-1", 1)).unwrap();
        router.register_node(create_test_node("node2", "us-east-1", 1)).unwrap();

        // Simulate failures
        for _ in 0..5 {
            router.increment_connections("node1");
            router.record_result("node1", false, Duration::from_millis(100));
        }

        let query = QueryCharacteristics {
            dimensions: 768,
            k: 10,
            has_filters: false,
            filter_complexity: FilterComplexity::None,
            requires_reranking: false,
            query_hash: None,
            priority: QueryPriority::Normal,
        };

        // Should route to node2 since node1 circuit is open
        let decision = router.route(&query).unwrap();
        assert_eq!(decision.node_id, "node2");
    }

    #[test]
    fn test_retry_handler() {
        let handler = RetryHandler::new(RetryConfig::default());

        assert!(handler.should_retry(0, "connection timeout"));
        assert!(handler.should_retry(2, "unavailable"));
        assert!(!handler.should_retry(3, "timeout")); // Exceeds max
        assert!(!handler.should_retry(0, "invalid_input")); // Not retryable
    }
}
