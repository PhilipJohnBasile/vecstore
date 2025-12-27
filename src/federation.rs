// Federation - Cross-store queries and multi-cluster management
// Query multiple vector stores as a unified namespace

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Federation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// Name of this federation
    pub name: String,
    /// Query timeout per member
    pub member_timeout: Duration,
    /// Total query timeout
    pub total_timeout: Duration,
    /// Minimum members required for quorum
    pub min_quorum: usize,
    /// Result merging strategy
    pub merge_strategy: MergeStrategy,
    /// Enable parallel queries
    pub parallel_queries: bool,
    /// Maximum concurrent queries per member
    pub max_concurrent_per_member: usize,
    /// Enable query caching
    pub enable_cache: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            member_timeout: Duration::from_secs(10),
            total_timeout: Duration::from_secs(30),
            min_quorum: 1,
            merge_strategy: MergeStrategy::ScoreBased,
            parallel_queries: true,
            max_concurrent_per_member: 10,
            enable_cache: true,
            cache_ttl: Duration::from_secs(60),
        }
    }
}

/// Result merging strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MergeStrategy {
    /// Merge by score/distance (best first)
    ScoreBased,
    /// Round-robin from each member
    RoundRobin,
    /// Prioritize certain members
    Priority,
    /// Deduplicate and merge
    Deduplicate,
    /// Custom weighted merge
    Weighted,
}

/// Member store in the federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationMember {
    /// Member ID
    pub id: String,
    /// Member name
    pub name: String,
    /// Connection endpoint
    pub endpoint: String,
    /// Member type
    pub member_type: MemberType,
    /// Priority (higher = query first)
    pub priority: u32,
    /// Weight for result merging
    pub weight: f32,
    /// Enabled status
    pub enabled: bool,
    /// Collections available
    pub collections: Vec<String>,
    /// Capabilities
    pub capabilities: MemberCapabilities,
}

/// Type of federation member
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemberType {
    /// Local VecStore instance
    Local,
    /// Remote VecStore over HTTP
    RemoteHttp,
    /// Remote VecStore over gRPC
    RemoteGrpc,
    /// External service (Pinecone, Weaviate, etc.)
    External(String),
}

/// Member capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberCapabilities {
    /// Supported dimensions
    pub dimensions: Vec<usize>,
    /// Supported distance metrics
    pub distance_metrics: Vec<String>,
    /// Supports filtering
    pub filtering: bool,
    /// Supports hybrid search
    pub hybrid_search: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum k for search
    pub max_k: usize,
}

impl Default for MemberCapabilities {
    fn default() -> Self {
        Self {
            dimensions: vec![128, 256, 384, 512, 768, 1024, 1536],
            distance_metrics: vec!["cosine".to_string(), "euclidean".to_string()],
            filtering: true,
            hybrid_search: true,
            max_batch_size: 1000,
            max_k: 10000,
        }
    }
}

/// Member state (runtime)
#[derive(Debug)]
struct MemberState {
    member: FederationMember,
    health: MemberHealth,
    last_health_check: Instant,
    active_queries: AtomicU64,
    total_queries: AtomicU64,
    total_errors: AtomicU64,
    avg_latency_ms: RwLock<f64>,
}

/// Member health status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemberHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Federated query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedQuery {
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of results
    pub k: usize,
    /// Target collections (empty = all)
    pub collections: Vec<String>,
    /// Target members (empty = all)
    pub members: Vec<String>,
    /// Filter expression
    pub filter: Option<String>,
    /// Include metadata
    pub include_metadata: bool,
    /// Minimum score threshold
    pub min_score: Option<f32>,
    /// Maximum results per member
    pub max_per_member: Option<usize>,
}

/// Federation manager
pub struct Federation {
    config: FederationConfig,
    /// Registered members
    members: RwLock<HashMap<String, Arc<MemberState>>>,
    /// Query cache
    cache: RwLock<HashMap<String, CachedResult>>,
    /// Statistics
    stats: FederationStats,
}

/// Cached query result
#[derive(Debug, Clone)]
struct CachedResult {
    results: Vec<FederatedResult>,
    cached_at: Instant,
}

/// Federation statistics
#[derive(Debug, Default)]
struct FederationStats {
    queries_total: AtomicU64,
    queries_succeeded: AtomicU64,
    queries_failed: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    members_queried: AtomicU64,
}

impl Federation {
    /// Create a new federation
    pub fn new(config: FederationConfig) -> Self {
        Self {
            config,
            members: RwLock::new(HashMap::new()),
            cache: RwLock::new(HashMap::new()),
            stats: FederationStats::default(),
        }
    }

    /// Add a member to the federation
    pub fn add_member(&self, member: FederationMember) -> Result<()> {
        let state = Arc::new(MemberState {
            member: member.clone(),
            health: MemberHealth::Unknown,
            last_health_check: Instant::now(),
            active_queries: AtomicU64::new(0),
            total_queries: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            avg_latency_ms: RwLock::new(0.0),
        });

        self.members.write().unwrap().insert(member.id.clone(), state);
        Ok(())
    }

    /// Remove a member from the federation
    pub fn remove_member(&self, member_id: &str) -> Result<()> {
        self.members.write().unwrap().remove(member_id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Member {} not found", member_id)))?;
        Ok(())
    }

    /// Update member configuration
    pub fn update_member(&self, member: FederationMember) -> Result<()> {
        let mut members = self.members.write().unwrap();
        if let Some(state) = members.get_mut(&member.id) {
            // Create new state preserving runtime info
            let new_state = Arc::new(MemberState {
                member,
                health: state.health.clone(),
                last_health_check: state.last_health_check,
                active_queries: AtomicU64::new(state.active_queries.load(Ordering::Relaxed)),
                total_queries: AtomicU64::new(state.total_queries.load(Ordering::Relaxed)),
                total_errors: AtomicU64::new(state.total_errors.load(Ordering::Relaxed)),
                avg_latency_ms: RwLock::new(*state.avg_latency_ms.read().unwrap()),
            });
            members.insert(new_state.member.id.clone(), new_state);
            Ok(())
        } else {
            Err(VecStoreError::NotFound(format!("Member {} not found", member.id)))
        }
    }

    /// Execute a federated query
    pub fn query(&self, query: FederatedQuery) -> Result<FederatedQueryResult> {
        self.stats.queries_total.fetch_add(1, Ordering::Relaxed);
        let start = Instant::now();

        // Check cache
        if self.config.enable_cache {
            let cache_key = self.compute_cache_key(&query);
            if let Some(cached) = self.get_cached(&cache_key) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(FederatedQueryResult {
                    results: cached,
                    members_queried: 0,
                    members_succeeded: 0,
                    duration: start.elapsed(),
                    from_cache: true,
                });
            }
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Get eligible members
        let members = self.get_eligible_members(&query)?;

        if members.len() < self.config.min_quorum {
            self.stats.queries_failed.fetch_add(1, Ordering::Relaxed);
            return Err(VecStoreError::InvalidInput(format!(
                "Not enough healthy members: {} < {}",
                members.len(),
                self.config.min_quorum
            )));
        }

        // Execute query on each member
        let mut all_results = Vec::new();
        let mut members_succeeded = 0;
        let members_queried = members.len();

        for member_state in &members {
            self.stats.members_queried.fetch_add(1, Ordering::Relaxed);
            member_state.active_queries.fetch_add(1, Ordering::Relaxed);
            member_state.total_queries.fetch_add(1, Ordering::Relaxed);

            match self.query_member(member_state, &query) {
                Ok(results) => {
                    all_results.extend(results);
                    members_succeeded += 1;
                }
                Err(_) => {
                    member_state.total_errors.fetch_add(1, Ordering::Relaxed);
                }
            }

            member_state.active_queries.fetch_sub(1, Ordering::Relaxed);
        }

        if members_succeeded == 0 {
            self.stats.queries_failed.fetch_add(1, Ordering::Relaxed);
            return Err(VecStoreError::Internal("All member queries failed".into()));
        }

        // Merge results
        let merged = self.merge_results(all_results, &query);

        // Cache results
        if self.config.enable_cache {
            let cache_key = self.compute_cache_key(&query);
            self.cache_result(&cache_key, merged.clone());
        }

        self.stats.queries_succeeded.fetch_add(1, Ordering::Relaxed);

        Ok(FederatedQueryResult {
            results: merged,
            members_queried,
            members_succeeded,
            duration: start.elapsed(),
            from_cache: false,
        })
    }

    fn get_eligible_members(&self, query: &FederatedQuery) -> Result<Vec<Arc<MemberState>>> {
        let members = self.members.read().unwrap();
        let mut eligible = Vec::new();

        for state in members.values() {
            if !state.member.enabled {
                continue;
            }

            if state.health == MemberHealth::Unhealthy {
                continue;
            }

            // Filter by member ID if specified
            if !query.members.is_empty() && !query.members.contains(&state.member.id) {
                continue;
            }

            // Filter by collection if specified
            if !query.collections.is_empty() {
                let has_collection = query.collections.iter()
                    .any(|c| state.member.collections.contains(c));
                if !has_collection {
                    continue;
                }
            }

            eligible.push(state.clone());
        }

        // Sort by priority
        eligible.sort_by(|a, b| b.member.priority.cmp(&a.member.priority));

        Ok(eligible)
    }

    fn query_member(&self, state: &MemberState, query: &FederatedQuery) -> Result<Vec<MemberResult>> {
        // In production: make actual HTTP/gRPC call
        // For now: simulate results

        let k = query.max_per_member.unwrap_or(query.k);

        // Simulate some results
        let results: Vec<MemberResult> = (0..k.min(10))
            .map(|i| MemberResult {
                id: format!("{}_{}", state.member.id, i),
                score: 0.9 - (i as f32 * 0.05),
                vector: None,
                metadata: None,
                member_id: state.member.id.clone(),
                collection: state.member.collections.first().cloned(),
            })
            .collect();

        Ok(results)
    }

    fn merge_results(&self, mut results: Vec<MemberResult>, query: &FederatedQuery) -> Vec<FederatedResult> {
        match self.config.merge_strategy {
            MergeStrategy::ScoreBased => {
                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            }
            MergeStrategy::RoundRobin => {
                // Group by member, then interleave
                let mut by_member: HashMap<String, Vec<MemberResult>> = HashMap::new();
                for r in results {
                    by_member.entry(r.member_id.clone()).or_default().push(r);
                }

                results = Vec::new();
                let max_len = by_member.values().map(|v| v.len()).max().unwrap_or(0);
                for i in 0..max_len {
                    for group in by_member.values() {
                        if i < group.len() {
                            results.push(group[i].clone());
                        }
                    }
                }
            }
            MergeStrategy::Priority => {
                // Already sorted by member priority, just sort within each member by score
                results.sort_by(|a, b| {
                    let member_cmp = self.get_member_priority(&b.member_id)
                        .cmp(&self.get_member_priority(&a.member_id));
                    if member_cmp == std::cmp::Ordering::Equal {
                        b.score.partial_cmp(&a.score).unwrap()
                    } else {
                        member_cmp
                    }
                });
            }
            MergeStrategy::Deduplicate => {
                // Remove duplicates by ID, keeping highest score
                let mut seen: HashMap<String, MemberResult> = HashMap::new();
                for r in results {
                    seen.entry(r.id.clone())
                        .and_modify(|existing| {
                            if r.score > existing.score {
                                *existing = r.clone();
                            }
                        })
                        .or_insert(r);
                }
                results = seen.into_values().collect();
                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            }
            MergeStrategy::Weighted => {
                // Apply member weights to scores
                for r in &mut results {
                    let weight = self.get_member_weight(&r.member_id);
                    r.score *= weight;
                }
                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            }
        }

        // Apply min score filter
        if let Some(min_score) = query.min_score {
            results.retain(|r| r.score >= min_score);
        }

        // Truncate to k
        results.truncate(query.k);

        // Convert to federated results
        results.into_iter()
            .map(|r| FederatedResult {
                id: r.id,
                score: r.score,
                vector: r.vector,
                metadata: r.metadata,
                source_member: r.member_id,
                source_collection: r.collection,
            })
            .collect()
    }

    fn get_member_priority(&self, member_id: &str) -> u32 {
        self.members.read().unwrap()
            .get(member_id)
            .map(|s| s.member.priority)
            .unwrap_or(0)
    }

    fn get_member_weight(&self, member_id: &str) -> f32 {
        self.members.read().unwrap()
            .get(member_id)
            .map(|s| s.member.weight)
            .unwrap_or(1.0)
    }

    fn compute_cache_key(&self, query: &FederatedQuery) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash vector
        for v in &query.vector {
            v.to_bits().hash(&mut hasher);
        }

        query.k.hash(&mut hasher);
        query.collections.hash(&mut hasher);
        query.members.hash(&mut hasher);
        query.filter.hash(&mut hasher);

        format!("{:016x}", hasher.finish())
    }

    fn get_cached(&self, key: &str) -> Option<Vec<FederatedResult>> {
        let cache = self.cache.read().unwrap();
        if let Some(cached) = cache.get(key) {
            if cached.cached_at.elapsed() < self.config.cache_ttl {
                return Some(cached.results.clone());
            }
        }
        None
    }

    fn cache_result(&self, key: &str, results: Vec<FederatedResult>) {
        let mut cache = self.cache.write().unwrap();
        cache.insert(key.to_string(), CachedResult {
            results,
            cached_at: Instant::now(),
        });

        // Limit cache size
        if cache.len() > 10000 {
            // Remove oldest entries
            let mut entries: Vec<_> = cache.iter()
                .map(|(k, v)| (k.clone(), v.cached_at))
                .collect();
            entries.sort_by_key(|(_, t)| *t);

            for (k, _) in entries.iter().take(1000) {
                cache.remove(k);
            }
        }
    }

    /// Update member health status
    pub fn update_health(&self, member_id: &str, health: MemberHealth) -> Result<()> {
        let members = self.members.read().unwrap();
        if let Some(state) = members.get(member_id) {
            // Can't mutate through Arc, but in production we'd use interior mutability
            // For now, just validate
            Ok(())
        } else {
            Err(VecStoreError::NotFound(format!("Member {} not found", member_id)))
        }
    }

    /// Get federation status
    pub fn get_status(&self) -> FederationStatus {
        let members = self.members.read().unwrap();

        let member_statuses: Vec<MemberStatus> = members.values()
            .map(|s| MemberStatus {
                id: s.member.id.clone(),
                name: s.member.name.clone(),
                endpoint: s.member.endpoint.clone(),
                health: s.health.clone(),
                active_queries: s.active_queries.load(Ordering::Relaxed),
                total_queries: s.total_queries.load(Ordering::Relaxed),
                total_errors: s.total_errors.load(Ordering::Relaxed),
                avg_latency_ms: *s.avg_latency_ms.read().unwrap(),
                collections: s.member.collections.clone(),
            })
            .collect();

        let healthy_count = member_statuses.iter()
            .filter(|m| m.health == MemberHealth::Healthy)
            .count();

        FederationStatus {
            name: self.config.name.clone(),
            total_members: members.len(),
            healthy_members: healthy_count,
            member_statuses,
            queries_total: self.stats.queries_total.load(Ordering::Relaxed),
            queries_succeeded: self.stats.queries_succeeded.load(Ordering::Relaxed),
            queries_failed: self.stats.queries_failed.load(Ordering::Relaxed),
            cache_hit_rate: {
                let hits = self.stats.cache_hits.load(Ordering::Relaxed);
                let misses = self.stats.cache_misses.load(Ordering::Relaxed);
                if hits + misses > 0 {
                    hits as f64 / (hits + misses) as f64
                } else {
                    0.0
                }
            },
        }
    }

    /// List all members
    pub fn list_members(&self) -> Vec<FederationMember> {
        self.members.read().unwrap()
            .values()
            .map(|s| s.member.clone())
            .collect()
    }

    /// Get a specific member
    pub fn get_member(&self, member_id: &str) -> Option<FederationMember> {
        self.members.read().unwrap()
            .get(member_id)
            .map(|s| s.member.clone())
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.write().unwrap().clear();
    }
}

/// Result from a single member
#[derive(Debug, Clone)]
struct MemberResult {
    id: String,
    score: f32,
    vector: Option<Vec<f32>>,
    metadata: Option<serde_json::Value>,
    member_id: String,
    collection: Option<String>,
}

/// Federated query result
#[derive(Debug, Clone, Serialize)]
pub struct FederatedResult {
    pub id: String,
    pub score: f32,
    pub vector: Option<Vec<f32>>,
    pub metadata: Option<serde_json::Value>,
    pub source_member: String,
    pub source_collection: Option<String>,
}

/// Complete federated query result
#[derive(Debug, Clone, Serialize)]
pub struct FederatedQueryResult {
    pub results: Vec<FederatedResult>,
    pub members_queried: usize,
    pub members_succeeded: usize,
    pub duration: Duration,
    pub from_cache: bool,
}

/// Member status
#[derive(Debug, Clone, Serialize)]
pub struct MemberStatus {
    pub id: String,
    pub name: String,
    pub endpoint: String,
    pub health: MemberHealth,
    pub active_queries: u64,
    pub total_queries: u64,
    pub total_errors: u64,
    pub avg_latency_ms: f64,
    pub collections: Vec<String>,
}

/// Federation status
#[derive(Debug, Clone, Serialize)]
pub struct FederationStatus {
    pub name: String,
    pub total_members: usize,
    pub healthy_members: usize,
    pub member_statuses: Vec<MemberStatus>,
    pub queries_total: u64,
    pub queries_succeeded: u64,
    pub queries_failed: u64,
    pub cache_hit_rate: f64,
}

/// Federation builder
pub struct FederationBuilder {
    config: FederationConfig,
    members: Vec<FederationMember>,
}

impl FederationBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            config: FederationConfig {
                name: name.to_string(),
                ..Default::default()
            },
            members: Vec::new(),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.total_timeout = timeout;
        self
    }

    pub fn with_merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.config.merge_strategy = strategy;
        self
    }

    pub fn with_quorum(mut self, min_quorum: usize) -> Self {
        self.config.min_quorum = min_quorum;
        self
    }

    pub fn add_member(mut self, member: FederationMember) -> Self {
        self.members.push(member);
        self
    }

    pub fn build(self) -> Result<Federation> {
        let federation = Federation::new(self.config);

        for member in self.members {
            federation.add_member(member)?;
        }

        Ok(federation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_member(id: &str, priority: u32) -> FederationMember {
        FederationMember {
            id: id.to_string(),
            name: format!("Member {}", id),
            endpoint: format!("http://localhost:{}", 8080 + priority),
            member_type: MemberType::Local,
            priority,
            weight: 1.0,
            enabled: true,
            collections: vec!["default".to_string()],
            capabilities: MemberCapabilities::default(),
        }
    }

    #[test]
    fn test_add_member() {
        let federation = Federation::new(FederationConfig::default());

        federation.add_member(create_test_member("m1", 1)).unwrap();
        federation.add_member(create_test_member("m2", 2)).unwrap();

        let members = federation.list_members();
        assert_eq!(members.len(), 2);
    }

    #[test]
    fn test_federated_query() {
        let federation = Federation::new(FederationConfig::default());

        federation.add_member(create_test_member("m1", 1)).unwrap();
        federation.add_member(create_test_member("m2", 2)).unwrap();

        let query = FederatedQuery {
            vector: vec![0.1, 0.2, 0.3],
            k: 10,
            collections: vec![],
            members: vec![],
            filter: None,
            include_metadata: true,
            min_score: None,
            max_per_member: Some(5),
        };

        let result = federation.query(query).unwrap();
        assert!(result.members_queried > 0);
        assert!(!result.results.is_empty());
    }

    #[test]
    fn test_merge_strategies() {
        let mut config = FederationConfig::default();
        config.merge_strategy = MergeStrategy::ScoreBased;

        let federation = Federation::new(config);
        federation.add_member(create_test_member("m1", 1)).unwrap();

        let query = FederatedQuery {
            vector: vec![0.1, 0.2, 0.3],
            k: 5,
            collections: vec![],
            members: vec![],
            filter: None,
            include_metadata: false,
            min_score: None,
            max_per_member: None,
        };

        let result = federation.query(query).unwrap();

        // Results should be sorted by score
        for i in 1..result.results.len() {
            assert!(result.results[i-1].score >= result.results[i].score);
        }
    }

    #[test]
    fn test_builder() {
        let federation = FederationBuilder::new("test_federation")
            .with_timeout(Duration::from_secs(60))
            .with_merge_strategy(MergeStrategy::Deduplicate)
            .with_quorum(2)
            .add_member(create_test_member("m1", 1))
            .add_member(create_test_member("m2", 2))
            .build()
            .unwrap();

        let status = federation.get_status();
        assert_eq!(status.name, "test_federation");
        assert_eq!(status.total_members, 2);
    }

    #[test]
    fn test_caching() {
        let mut config = FederationConfig::default();
        config.enable_cache = true;

        let federation = Federation::new(config);
        federation.add_member(create_test_member("m1", 1)).unwrap();

        let query = FederatedQuery {
            vector: vec![0.1, 0.2, 0.3],
            k: 5,
            collections: vec![],
            members: vec![],
            filter: None,
            include_metadata: false,
            min_score: None,
            max_per_member: None,
        };

        // First query - cache miss
        let result1 = federation.query(query.clone()).unwrap();
        assert!(!result1.from_cache);

        // Second query - cache hit
        let result2 = federation.query(query).unwrap();
        assert!(result2.from_cache);
    }
}
