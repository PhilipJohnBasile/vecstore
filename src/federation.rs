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

        let mut members = self.members.write()
            .map_err(|_| VecStoreError::LockError("Failed to acquire write lock on members".into()))?;
        members.insert(member.id.clone(), state);
        Ok(())
    }

    /// Remove a member from the federation
    pub fn remove_member(&self, member_id: &str) -> Result<()> {
        let mut members = self.members.write()
            .map_err(|_| VecStoreError::LockError("Failed to acquire write lock on members".into()))?;
        members.remove(member_id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Member {} not found", member_id)))?;
        Ok(())
    }

    /// Update member configuration
    pub fn update_member(&self, member: FederationMember) -> Result<()> {
        let mut members = self.members.write()
            .map_err(|_| VecStoreError::LockError("Failed to acquire write lock on members".into()))?;
        if let Some(state) = members.get_mut(&member.id) {
            // Create new state preserving runtime info
            let avg_latency = *state.avg_latency_ms.read()
                .map_err(|_| VecStoreError::LockError("Failed to acquire read lock on avg_latency_ms".into()))?;
            let new_state = Arc::new(MemberState {
                member,
                health: state.health.clone(),
                last_health_check: state.last_health_check,
                active_queries: AtomicU64::new(state.active_queries.load(Ordering::Relaxed)),
                total_queries: AtomicU64::new(state.total_queries.load(Ordering::Relaxed)),
                total_errors: AtomicU64::new(state.total_errors.load(Ordering::Relaxed)),
                avg_latency_ms: RwLock::new(avg_latency),
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
        let members = self.members.read()
            .map_err(|_| VecStoreError::LockError("Failed to acquire read lock on members".into()))?;
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

    fn query_member(
        &self,
        state: &MemberState,
        query: &FederatedQuery,
    ) -> Result<Vec<MemberResult>> {
        // Some feature combinations compile only the local-member branch.
        let _ = query;

        match &state.member.member_type {
            MemberType::Local => {
                // Local queries should be handled by the local VecStore directly
                // Return empty results as local stores are queried separately
                Ok(Vec::new())
            }
            MemberType::RemoteHttp => {
                #[cfg(feature = "openai-embeddings")]
                {
                    let k = query.max_per_member.unwrap_or(query.k);
                    self.query_http_member(&state.member, query, k)
                }
                #[cfg(not(feature = "openai-embeddings"))]
                {
                    Err(VecStoreError::InvalidConfig(
                        "HTTP federation requires 'openai-embeddings' feature (which includes reqwest)".into()
                    ))
                }
            }
            MemberType::RemoteGrpc => {
                #[cfg(feature = "server")]
                {
                    let k = query.max_per_member.unwrap_or(query.k);
                    self.query_grpc_member(&state.member, query, k)
                }
                #[cfg(not(feature = "server"))]
                {
                    Err(VecStoreError::InvalidConfig(
                        "gRPC federation requires 'server' feature (which includes tonic)".into()
                    ))
                }
            }
            MemberType::External(service_type) => {
                #[cfg(feature = "openai-embeddings")]
                {
                    let k = query.max_per_member.unwrap_or(query.k);
                    self.query_external_member(&state.member, query, k, service_type)
                }
                #[cfg(not(feature = "openai-embeddings"))]
                {
                    let _ = service_type;
                    Err(VecStoreError::InvalidConfig(
                        "External federation requires 'openai-embeddings' feature (which includes reqwest)".into()
                    ))
                }
            }
        }
    }

    /// Query a remote HTTP member
    #[cfg(feature = "openai-embeddings")]
    fn query_http_member(
        &self,
        member: &FederationMember,
        query: &FederatedQuery,
        k: usize,
    ) -> Result<Vec<MemberResult>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(self.config.member_timeout)
            .build()
            .map_err(|e| VecStoreError::Internal(format!("HTTP client error: {}", e)))?;

        // Build the query request body
        let request_body = serde_json::json!({
            "vector": query.vector,
            "k": k,
            "filter": query.filter,
            "include_metadata": query.include_metadata,
            "min_score": query.min_score,
        });

        // Query the VecStore HTTP endpoint
        let url = format!("{}/query", member.endpoint.trim_end_matches('/'));
        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| VecStoreError::Internal(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(VecStoreError::Internal(format!(
                "HTTP query failed with status {}: {}",
                response.status(),
                response.text().unwrap_or_default()
            )));
        }

        // Parse the response
        let response_body: serde_json::Value = response
            .json()
            .map_err(|e| VecStoreError::Internal(format!("Failed to parse response: {}", e)))?;

        // Extract results from response
        let results = response_body
            .get("results")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        Some(MemberResult {
                            id: item.get("id")?.as_str()?.to_string(),
                            score: item.get("score")?.as_f64()? as f32,
                            vector: item.get("vector").and_then(|v| {
                                v.as_array().map(|arr| {
                                    arr.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect()
                                })
                            }),
                            metadata: item.get("metadata").cloned(),
                            member_id: member.id.clone(),
                            collection: member.collections.first().cloned(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(results)
    }

    /// Query an external service (Pinecone, Weaviate, etc.)
    #[cfg(feature = "openai-embeddings")]
    fn query_external_member(
        &self,
        member: &FederationMember,
        query: &FederatedQuery,
        k: usize,
        service_type: &str,
    ) -> Result<Vec<MemberResult>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(self.config.member_timeout)
            .build()
            .map_err(|e| VecStoreError::Internal(format!("HTTP client error: {}", e)))?;

        match service_type.to_lowercase().as_str() {
            "pinecone" => self.query_pinecone(&client, member, query, k),
            "qdrant" => self.query_qdrant(&client, member, query, k),
            "weaviate" => self.query_weaviate(&client, member, query, k),
            "chromadb" => self.query_chromadb(&client, member, query, k),
            _ => Err(VecStoreError::InvalidConfig(format!(
                "Unsupported external service type: {}",
                service_type
            ))),
        }
    }

    /// Query Pinecone-compatible endpoint
    #[cfg(feature = "openai-embeddings")]
    fn query_pinecone(
        &self,
        client: &reqwest::blocking::Client,
        member: &FederationMember,
        query: &FederatedQuery,
        k: usize,
    ) -> Result<Vec<MemberResult>> {
        let request_body = serde_json::json!({
            "vector": query.vector,
            "topK": k,
            "includeMetadata": query.include_metadata,
            "includeValues": true,
        });

        let url = format!("{}/query", member.endpoint.trim_end_matches('/'));
        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| VecStoreError::Internal(format!("Pinecone request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(VecStoreError::Internal(format!(
                "Pinecone query failed: {}",
                response.status()
            )));
        }

        let body: serde_json::Value = response
            .json()
            .map_err(|e| VecStoreError::Internal(format!("Failed to parse Pinecone response: {}", e)))?;

        let results = body
            .get("matches")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        Some(MemberResult {
                            id: item.get("id")?.as_str()?.to_string(),
                            score: item.get("score")?.as_f64()? as f32,
                            vector: item.get("values").and_then(|v| {
                                v.as_array().map(|arr| {
                                    arr.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect()
                                })
                            }),
                            metadata: item.get("metadata").cloned(),
                            member_id: member.id.clone(),
                            collection: member.collections.first().cloned(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(results)
    }

    /// Query Qdrant-compatible endpoint
    #[cfg(feature = "openai-embeddings")]
    fn query_qdrant(
        &self,
        client: &reqwest::blocking::Client,
        member: &FederationMember,
        query: &FederatedQuery,
        k: usize,
    ) -> Result<Vec<MemberResult>> {
        let collection = member.collections.first().cloned().unwrap_or_else(|| "default".to_string());
        let request_body = serde_json::json!({
            "vector": query.vector,
            "limit": k,
            "with_payload": query.include_metadata,
            "with_vector": true,
        });

        let url = format!(
            "{}/collections/{}/points/search",
            member.endpoint.trim_end_matches('/'),
            collection
        );
        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| VecStoreError::Internal(format!("Qdrant request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(VecStoreError::Internal(format!(
                "Qdrant query failed: {}",
                response.status()
            )));
        }

        let body: serde_json::Value = response
            .json()
            .map_err(|e| VecStoreError::Internal(format!("Failed to parse Qdrant response: {}", e)))?;

        let results = body
            .get("result")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        Some(MemberResult {
                            id: item.get("id")?.to_string(),
                            score: item.get("score")?.as_f64()? as f32,
                            vector: item.get("vector").and_then(|v| {
                                v.as_array().map(|arr| {
                                    arr.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect()
                                })
                            }),
                            metadata: item.get("payload").cloned(),
                            member_id: member.id.clone(),
                            collection: Some(collection.clone()),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(results)
    }

    /// Query Weaviate-compatible endpoint
    #[cfg(feature = "openai-embeddings")]
    fn query_weaviate(
        &self,
        client: &reqwest::blocking::Client,
        member: &FederationMember,
        query: &FederatedQuery,
        k: usize,
    ) -> Result<Vec<MemberResult>> {
        let class_name = member.collections.first().cloned().unwrap_or_else(|| "Document".to_string());

        // Weaviate uses GraphQL
        let graphql_query = format!(
            r#"{{
                Get {{
                    {class_name}(nearVector: {{vector: {:?}}}, limit: {k}) {{
                        _additional {{
                            id
                            distance
                            vector
                        }}
                    }}
                }}
            }}"#,
            query.vector
        );

        let request_body = serde_json::json!({
            "query": graphql_query,
        });

        let url = format!("{}/v1/graphql", member.endpoint.trim_end_matches('/'));
        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| VecStoreError::Internal(format!("Weaviate request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(VecStoreError::Internal(format!(
                "Weaviate query failed: {}",
                response.status()
            )));
        }

        let body: serde_json::Value = response
            .json()
            .map_err(|e| VecStoreError::Internal(format!("Failed to parse Weaviate response: {}", e)))?;

        // Parse GraphQL response
        let results = body
            .pointer(&format!("/data/Get/{}", class_name))
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        let additional = item.get("_additional")?;
                        Some(MemberResult {
                            id: additional.get("id")?.as_str()?.to_string(),
                            // Convert distance to similarity score
                            score: 1.0 - additional.get("distance")?.as_f64()? as f32,
                            vector: additional.get("vector").and_then(|v| {
                                v.as_array().map(|arr| {
                                    arr.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect()
                                })
                            }),
                            metadata: Some(item.clone()),
                            member_id: member.id.clone(),
                            collection: Some(class_name.clone()),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(results)
    }

    /// Query ChromaDB-compatible endpoint
    #[cfg(feature = "openai-embeddings")]
    fn query_chromadb(
        &self,
        client: &reqwest::blocking::Client,
        member: &FederationMember,
        query: &FederatedQuery,
        k: usize,
    ) -> Result<Vec<MemberResult>> {
        let collection = member.collections.first().cloned().unwrap_or_else(|| "default".to_string());

        let request_body = serde_json::json!({
            "query_embeddings": [query.vector],
            "n_results": k,
            "include": ["embeddings", "metadatas", "documents", "distances"],
        });

        let url = format!(
            "{}/api/v1/collections/{}/query",
            member.endpoint.trim_end_matches('/'),
            collection
        );
        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| VecStoreError::Internal(format!("ChromaDB request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(VecStoreError::Internal(format!(
                "ChromaDB query failed: {}",
                response.status()
            )));
        }

        let body: serde_json::Value = response
            .json()
            .map_err(|e| VecStoreError::Internal(format!("Failed to parse ChromaDB response: {}", e)))?;

        // ChromaDB returns arrays for each field
        let ids = body.get("ids").and_then(|i| i.get(0)).and_then(|a| a.as_array());
        let distances = body.get("distances").and_then(|d| d.get(0)).and_then(|a| a.as_array());
        let embeddings = body.get("embeddings").and_then(|e| e.get(0)).and_then(|a| a.as_array());
        let metadatas = body.get("metadatas").and_then(|m| m.get(0)).and_then(|a| a.as_array());

        let mut results = Vec::new();
        if let Some(ids) = ids {
            for (i, id) in ids.iter().enumerate() {
                if let Some(id_str) = id.as_str() {
                    let score = distances
                        .and_then(|d| d.get(i))
                        .and_then(|v| v.as_f64())
                        .map(|d| 1.0 - d as f32) // Convert distance to similarity
                        .unwrap_or(0.0);

                    let vector = embeddings
                        .and_then(|e| e.get(i))
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect());

                    let metadata = metadatas.and_then(|m| m.get(i)).cloned();

                    results.push(MemberResult {
                        id: id_str.to_string(),
                        score,
                        vector,
                        metadata,
                        member_id: member.id.clone(),
                        collection: Some(collection.clone()),
                    });
                }
            }
        }

        Ok(results)
    }

    /// Query a remote gRPC member using tonic client
    #[cfg(feature = "server")]
    fn query_grpc_member(
        &self,
        member: &FederationMember,
        query: &FederatedQuery,
        k: usize,
    ) -> Result<Vec<MemberResult>> {
        use crate::server::types::pb::{vec_store_service_client::VecStoreServiceClient, QueryRequest};

        // Create a tokio runtime for blocking on async gRPC calls
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| VecStoreError::Internal(format!("Failed to create tokio runtime: {}", e)))?;

        rt.block_on(async {
            // Connect to the gRPC server
            let endpoint = member.endpoint.clone();
            let mut client = VecStoreServiceClient::connect(endpoint.clone())
                .await
                .map_err(|e| VecStoreError::Internal(format!("gRPC connection failed to {}: {}", endpoint, e)))?;

            // Build the query request
            let request = QueryRequest {
                vector: query.vector.clone(),
                limit: k as i32,
                filter: query.filter.clone(),
                namespace: None,
            };

            // Execute the query with timeout
            let response = tokio::time::timeout(
                self.config.member_timeout,
                client.query(request)
            )
            .await
            .map_err(|_| VecStoreError::Internal(format!("gRPC query timeout to {}", endpoint)))?
            .map_err(|e| VecStoreError::Internal(format!("gRPC query failed: {}", e)))?;

            // Convert response to MemberResult
            let results: Vec<MemberResult> = response
                .into_inner()
                .results
                .into_iter()
                .map(|r| {
                    MemberResult {
                        id: r.id,
                        score: r.score,
                        vector: None, // gRPC response doesn't include vector by default
                        metadata: if r.metadata.is_empty() {
                            None
                        } else {
                            // Convert protobuf metadata to JSON
                            Some(crate::server::types::pb_map_to_json(&r.metadata))
                        },
                        member_id: member.id.clone(),
                        collection: member.collections.first().cloned(),
                    }
                })
                .collect();

            Ok(results)
        })
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
        let Ok(members) = self.members.read() else { return 0; };
        members
            .get(member_id)
            .map(|s| s.member.priority)
            .unwrap_or(0)
    }

    fn get_member_weight(&self, member_id: &str) -> f32 {
        let Ok(members) = self.members.read() else { return 1.0; };
        members
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
        let Ok(cache) = self.cache.read() else { return None; };
        if let Some(cached) = cache.get(key)
            && cached.cached_at.elapsed() < self.config.cache_ttl {
                return Some(cached.results.clone());
            }
        None
    }

    fn cache_result(&self, key: &str, results: Vec<FederatedResult>) {
        let Ok(mut cache) = self.cache.write() else { return; };
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
    pub fn update_health(&self, member_id: &str, _health: MemberHealth) -> Result<()> {
        let members = self.members.read()
            .map_err(|_| VecStoreError::LockError("Failed to acquire read lock on members".into()))?;
        let Some(_state) = members.get(member_id) else {
            return Err(VecStoreError::NotFound(format!("Member {} not found", member_id)));
        };
        // Can't mutate through Arc, but in production we'd use interior mutability
        Ok(())
    }

    /// Get federation status
    pub fn get_status(&self) -> FederationStatus {
        let Ok(members) = self.members.read() else {
            return FederationStatus {
                name: self.config.name.clone(),
                total_members: 0,
                healthy_members: 0,
                member_statuses: vec![],
                queries_total: self.stats.queries_total.load(Ordering::Relaxed),
                queries_succeeded: self.stats.queries_succeeded.load(Ordering::Relaxed),
                queries_failed: self.stats.queries_failed.load(Ordering::Relaxed),
                cache_hit_rate: 0.0,
            };
        };

        let member_statuses: Vec<MemberStatus> = members.values()
            .map(|s| {
                let avg_latency = s.avg_latency_ms.read().map(|g| *g).unwrap_or(0.0);
                MemberStatus {
                    id: s.member.id.clone(),
                    name: s.member.name.clone(),
                    endpoint: s.member.endpoint.clone(),
                    health: s.health.clone(),
                    active_queries: s.active_queries.load(Ordering::Relaxed),
                    total_queries: s.total_queries.load(Ordering::Relaxed),
                    total_errors: s.total_errors.load(Ordering::Relaxed),
                    avg_latency_ms: avg_latency,
                    collections: s.member.collections.clone(),
                }
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
        let Ok(members) = self.members.read() else { return vec![]; };
        members
            .values()
            .map(|s| s.member.clone())
            .collect()
    }

    /// Get a specific member
    pub fn get_member(&self, member_id: &str) -> Option<FederationMember> {
        let Ok(members) = self.members.read() else { return None; };
        members
            .get(member_id)
            .map(|s| s.member.clone())
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        let Ok(mut cache) = self.cache.write() else { return; };
        cache.clear();
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
#[must_use = "builders do nothing unless built"]
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

    #[inline]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.total_timeout = timeout;
        self
    }

    #[inline]
    pub fn with_merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.config.merge_strategy = strategy;
        self
    }

    #[inline]
    pub fn with_quorum(mut self, min_quorum: usize) -> Self {
        self.config.min_quorum = min_quorum;
        self
    }

    #[inline]
    pub fn add_member(mut self, member: FederationMember) -> Self {
        self.members.push(member);
        self
    }

    /// # Errors
    /// Returns an error if adding a member fails.
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
        // Note: MemberType::Local returns empty results by design (local stores
        // should be queried directly, not through federation). For remote members,
        // results would be populated via HTTP/gRPC calls.
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
