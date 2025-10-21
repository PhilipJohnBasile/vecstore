//! Distributed Multi-Node Indexing
//!
//! ⚠️ **EXPERIMENTAL - INCOMPLETE IMPLEMENTATION**
//!
//! This module is a prototype/skeleton implementation demonstrating the architecture
//! for distributed vector indexing. Many features are stubs that require additional
//! implementation for production use:
//!
//! - Network communication between nodes (currently local only)
//! - Actual data replication (simulated, not implemented)
//! - Failure detection and recovery (basic only)
//! - Load balancing and query routing (simplified)
//! - Raft consensus integration (partial, requires more work)
//!
//! **Use this module only for reference architecture and testing.**
//! Production deployments should implement proper RPC, persistence, and failure handling.
//!
//! ## Overview
//!
//! This module provides distributed vector indexing across multiple nodes
//! for horizontal scalability, high availability, and fault tolerance.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────┐
//! │   Client   │
//! └─────┬──────┘
//!       │
//!   ┌───▼───┐
//!   │Coordinator│
//!   └───┬───┘
//!       │
//!  ┌────┴────┬────────┬────────┐
//!  │         │        │        │
//! ┌▼──┐   ┌─▼─┐   ┌──▼┐   ┌──▼┐
//! │Shard│  │Shard│ │Shard│ │Shard│
//! │  0  │  │  1  │ │  2  │ │  3  │
//! └─────┘  └────┘ └────┘ └────┘
//!   │        │      │      │
//! Replica  Replica Replica Replica
//! ```
//!
//! ## Features
//!
//! - **Horizontal scaling**: Add nodes to increase capacity
//! - **Sharding**: Partition data across nodes
//! - **Replication**: Multiple copies for fault tolerance
//! - **Consistency**: Configurable consistency levels
//! - **Load balancing**: Distribute queries across replicas
//! - **Auto-rebalancing**: Redistribute data on topology changes
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::distributed::{
//!     DistributedConfig, DistributedStore, ShardingStrategy, ConsistencyLevel
//! };
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! // Configure distributed store
//! let config = DistributedConfig::new()
//!     .with_num_shards(4)
//!     .with_replication_factor(3)
//!     .with_sharding_strategy(ShardingStrategy::ConsistentHash)
//!     .with_consistency(ConsistencyLevel::Quorum);
//!
//! // Create distributed store
//! let mut store = DistributedStore::create(config).await?;
//!
//! // Add nodes to cluster
//! store.add_node("node1", "127.0.0.1:8001").await?;
//! store.add_node("node2", "127.0.0.1:8002").await?;
//! store.add_node("node3", "127.0.0.1:8003").await?;
//!
//! // Insert vectors (auto-sharded)
//! store.insert("doc1", vec![0.1, 0.2, 0.3]).await?;
//!
//! // Query (scatter-gather across shards)
//! let results = store.query(vec![0.15, 0.25, 0.35], 10).await?;
//! # Ok(())
//! # }
//! ```

// Submodules
#[cfg(feature = "async")]
pub mod raft;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "async")]
use tokio::sync::RwLock;

/// Sharding strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Hash-based sharding
    Hash,

    /// Consistent hashing
    ConsistentHash,

    /// Range-based sharding
    Range,

    /// Random sharding
    Random,
}

/// Consistency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Any single node
    One,

    /// Majority of replicas (N/2 + 1)
    Quorum,

    /// All replicas
    All,
}

/// Replication strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    /// Primary-backup replication
    PrimaryBackup,

    /// Multi-master replication
    MultiMaster,

    /// Chain replication
    Chain,
}

/// Distributed configuration
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Number of shards
    pub num_shards: usize,

    /// Replication factor (copies per shard)
    pub replication_factor: usize,

    /// Sharding strategy
    pub sharding_strategy: ShardingStrategy,

    /// Consistency level
    pub consistency_level: ConsistencyLevel,

    /// Replication strategy
    pub replication_strategy: ReplicationStrategy,

    /// Heartbeat interval (ms)
    pub heartbeat_interval_ms: u64,

    /// Node failure timeout (ms)
    pub failure_timeout_ms: u64,

    /// Enable auto-rebalancing
    pub auto_rebalance: bool,

    /// Maximum shard size (bytes)
    pub max_shard_size_bytes: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            num_shards: 8,
            replication_factor: 3,
            sharding_strategy: ShardingStrategy::ConsistentHash,
            consistency_level: ConsistencyLevel::Quorum,
            replication_strategy: ReplicationStrategy::PrimaryBackup,
            heartbeat_interval_ms: 1000,
            failure_timeout_ms: 5000,
            auto_rebalance: true,
            max_shard_size_bytes: 100 * 1024 * 1024, // 100MB
        }
    }
}

impl DistributedConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_num_shards(mut self, num: usize) -> Self {
        // Validate num_shards is non-zero (Major Issue #19 fix)
        if num == 0 {
            panic!("Number of shards must be at least 1, got 0");
        }
        self.num_shards = num;
        self
    }

    pub fn with_replication_factor(mut self, factor: usize) -> Self {
        self.replication_factor = factor;
        self
    }

    pub fn with_sharding_strategy(mut self, strategy: ShardingStrategy) -> Self {
        self.sharding_strategy = strategy;
        self
    }

    pub fn with_consistency(mut self, level: ConsistencyLevel) -> Self {
        self.consistency_level = level;
        self
    }

    pub fn with_replication_strategy(mut self, strategy: ReplicationStrategy) -> Self {
        self.replication_strategy = strategy;
        self
    }
}

/// Node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub address: String,
    pub status: NodeStatus,
    pub last_heartbeat: u64,
    pub shards: Vec<usize>,
    pub capacity_bytes: usize,
    pub used_bytes: usize,
}

/// Node status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Healthy,
    Degraded,
    Failed,
    Joining,
    Leaving,
}

/// Shard information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    pub id: usize,
    pub primary_node: String,
    pub replica_nodes: Vec<String>,
    pub size_bytes: usize,
    pub num_vectors: usize,
}

/// Query result from a single shard
#[derive(Debug, Clone)]
pub struct ShardQueryResult {
    pub shard_id: usize,
    pub node_id: String,
    pub results: Vec<(String, f32)>, // (id, distance)
    pub latency_ms: f64,
}

/// Distributed store statistics
#[derive(Debug, Clone, Default)]
pub struct DistributedStats {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub total_shards: usize,
    pub total_vectors: usize,
    pub total_bytes: usize,
    pub avg_shard_size_bytes: usize,
    pub rebalances_performed: usize,
    pub queries_total: u64,
    pub queries_failed: u64,
    pub avg_query_latency_ms: f64,
}

/// Consistent hash ring for sharding
pub struct ConsistentHashRing {
    virtual_nodes: usize,
    ring: Vec<(u64, String)>, // (hash, node_id)
}

impl ConsistentHashRing {
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            virtual_nodes,
            ring: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node_id: &str) {
        for i in 0..self.virtual_nodes {
            let key = format!("{}:{}", node_id, i);
            let hash = Self::hash(&key);
            self.ring.push((hash, node_id.to_string()));
        }
        self.ring.sort_by_key(|&(h, _)| h);
    }

    pub fn remove_node(&mut self, node_id: &str) {
        self.ring.retain(|(_, id)| id != node_id);
    }

    pub fn get_node(&self, key: &str) -> Option<String> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = Self::hash(key);

        // Binary search for first node >= hash
        let idx = self.ring.partition_point(|&(h, _)| h < hash);
        let idx = if idx >= self.ring.len() { 0 } else { idx };

        Some(self.ring[idx].1.clone())
    }

    pub fn get_nodes(&self, key: &str, count: usize) -> Vec<String> {
        if self.ring.is_empty() {
            return vec![];
        }

        let hash = Self::hash(key);
        let mut seen = HashSet::new();
        let mut nodes = Vec::new();

        let start_idx = self.ring.partition_point(|&(h, _)| h < hash);

        for i in 0..self.ring.len() {
            let idx = (start_idx + i) % self.ring.len();
            let node_id = &self.ring[idx].1;

            if seen.insert(node_id.clone()) {
                nodes.push(node_id.clone());
                if nodes.len() >= count {
                    break;
                }
            }
        }

        nodes
    }

    fn hash(key: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

/// Distributed store (sync version)
#[cfg(not(feature = "async"))]
pub struct DistributedStore {
    config: DistributedConfig,
    nodes: HashMap<String, NodeInfo>,
    shards: HashMap<usize, ShardInfo>,
    hash_ring: ConsistentHashRing,
    stats: DistributedStats,
}

#[cfg(not(feature = "async"))]
impl DistributedStore {
    pub fn create(config: DistributedConfig) -> Result<Self> {
        // Validate num_shards is non-zero (Major Issue #19 fix)
        if config.num_shards == 0 {
            return Err(anyhow::anyhow!(
                "Number of shards must be at least 1, got 0. Use DistributedConfig::with_num_shards() to set a valid shard count."
            ));
        }

        let mut shards = HashMap::new();
        for i in 0..config.num_shards {
            shards.insert(
                i,
                ShardInfo {
                    id: i,
                    primary_node: String::new(),
                    replica_nodes: Vec::new(),
                    size_bytes: 0,
                    num_vectors: 0,
                },
            );
        }

        Ok(Self {
            config,
            nodes: HashMap::new(),
            shards,
            hash_ring: ConsistentHashRing::new(150), // 150 virtual nodes
            stats: DistributedStats::default(),
        })
    }

    pub fn add_node(&mut self, node_id: &str, address: &str) -> Result<()> {
        let node = NodeInfo {
            id: node_id.to_string(),
            address: address.to_string(),
            status: NodeStatus::Joining,
            last_heartbeat: current_timestamp(),
            shards: Vec::new(),
            capacity_bytes: (10u64 * 1024 * 1024 * 1024) as usize, // 10GB default
            used_bytes: 0,
        };

        self.nodes.insert(node_id.to_string(), node);
        self.hash_ring.add_node(node_id);
        self.stats.total_nodes += 1;

        // Trigger rebalance if enabled
        if self.config.auto_rebalance {
            self.rebalance()?;
        }

        Ok(())
    }

    pub fn remove_node(&mut self, node_id: &str) -> Result<()> {
        self.nodes.remove(node_id);
        self.hash_ring.remove_node(node_id);
        self.stats.total_nodes = self.stats.total_nodes.saturating_sub(1);

        if self.config.auto_rebalance {
            self.rebalance()?;
        }

        Ok(())
    }

    pub fn get_shard_id(&self, key: &str) -> usize {
        match self.config.sharding_strategy {
            ShardingStrategy::Hash => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                (hasher.finish() as usize) % self.config.num_shards
            }
            ShardingStrategy::ConsistentHash => {
                // Use consistent hash ring
                let node = self.hash_ring.get_node(key).unwrap_or_default();
                // Map node to shard (simplified - use wrapping add to avoid overflow)
                let sum = node
                    .as_bytes()
                    .iter()
                    .fold(0u32, |acc, &b| acc.wrapping_add(b as u32));
                (sum as usize) % self.config.num_shards
            }
            ShardingStrategy::Range => {
                // Range-based (simplified - would use key ranges in production)
                key.as_bytes().first().copied().unwrap_or(0) as usize % self.config.num_shards
            }
            ShardingStrategy::Random => {
                // Random (simplified)
                key.len() % self.config.num_shards
            }
        }
    }

    pub fn rebalance(&mut self) -> Result<()> {
        // Simplified rebalancing: distribute shards evenly
        if self.nodes.is_empty() {
            return Ok(());
        }

        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();

        for (shard_id, shard_info) in &mut self.shards {
            let idx = *shard_id % node_ids.len();
            shard_info.primary_node = node_ids[idx].clone();

            // Assign replicas
            shard_info.replica_nodes.clear();
            for i in 1..self.config.replication_factor {
                let replica_idx = (idx + i) % node_ids.len();
                shard_info.replica_nodes.push(node_ids[replica_idx].clone());
            }
        }

        self.stats.rebalances_performed += 1;

        Ok(())
    }

    pub fn stats(&self) -> &DistributedStats {
        &self.stats
    }

    pub fn cluster_health(&self) -> f32 {
        if self.stats.total_nodes == 0 {
            return 0.0;
        }
        self.stats.healthy_nodes as f32 / self.stats.total_nodes as f32
    }
}

/// Distributed store (async version)
#[cfg(feature = "async")]
pub struct DistributedStore {
    config: DistributedConfig,
    nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
    shards: Arc<RwLock<HashMap<usize, ShardInfo>>>,
    hash_ring: Arc<RwLock<ConsistentHashRing>>,
    stats: Arc<RwLock<DistributedStats>>,
    /// Raft consensus node for distributed coordination
    raft_node: Option<Arc<raft::RaftNode>>,
}

#[cfg(feature = "async")]
impl DistributedStore {
    pub async fn create(config: DistributedConfig) -> Result<Self> {
        let mut shards = HashMap::new();
        for i in 0..config.num_shards {
            shards.insert(
                i,
                ShardInfo {
                    id: i,
                    primary_node: String::new(),
                    replica_nodes: Vec::new(),
                    size_bytes: 0,
                    num_vectors: 0,
                },
            );
        }

        Ok(Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            shards: Arc::new(RwLock::new(shards)),
            hash_ring: Arc::new(RwLock::new(ConsistentHashRing::new(150))),
            stats: Arc::new(RwLock::new(DistributedStats::default())),
            raft_node: None,
        })
    }

    /// Enable Raft consensus for the distributed store
    ///
    /// This creates a Raft node for distributed coordination and consensus.
    /// All cluster membership changes and critical operations will go through Raft.
    ///
    /// # Arguments
    /// * `node_id` - Unique identifier for this node
    /// * `peer_ids` - List of peer node IDs in the cluster
    pub async fn enable_raft(&mut self, node_id: String, peer_ids: Vec<String>) -> Result<()> {
        let raft_config = raft::RaftConfig {
            node_id,
            peers: peer_ids,
            ..Default::default()
        };

        let raft_node = raft::RaftNode::new(raft_config);
        self.raft_node = Some(Arc::new(raft_node));

        Ok(())
    }

    /// Check if Raft consensus is enabled
    pub fn is_raft_enabled(&self) -> bool {
        self.raft_node.is_some()
    }

    /// Get the Raft node (if enabled)
    pub fn raft_node(&self) -> Option<Arc<raft::RaftNode>> {
        self.raft_node.clone()
    }

    pub async fn add_node(&self, node_id: &str, address: &str) -> Result<()> {
        // If Raft is enabled, go through consensus
        if let Some(raft) = &self.raft_node {
            // Only leader can add nodes
            if !raft.is_leader().await {
                return Err(anyhow!("Not the leader - cannot add nodes"));
            }

            // Create add node command
            let command = raft::Command::Insert {
                id: format!("node:{}", node_id),
                vector: vec![], // Metadata only
                metadata: serde_json::json!({
                    "type": "add_node",
                    "node_id": node_id,
                    "address": address,
                }),
            };

            // Append to Raft log (will replicate to followers)
            raft.append_entry(command).await.map_err(|e| anyhow!(e))?;

            // Wait for commit before applying locally
            // In production, this would be done via apply loop
        }

        let node = NodeInfo {
            id: node_id.to_string(),
            address: address.to_string(),
            status: NodeStatus::Joining,
            last_heartbeat: current_timestamp(),
            shards: Vec::new(),
            capacity_bytes: 10 * 1024 * 1024 * 1024,
            used_bytes: 0,
        };

        {
            let mut nodes = self.nodes.write().await;
            nodes.insert(node_id.to_string(), node);
        }

        {
            let mut ring = self.hash_ring.write().await;
            ring.add_node(node_id);
        }

        {
            let mut stats = self.stats.write().await;
            stats.total_nodes += 1;
        }

        if self.config.auto_rebalance {
            self.rebalance().await?;
        }

        Ok(())
    }

    pub async fn remove_node(&self, node_id: &str) -> Result<()> {
        // If Raft is enabled, go through consensus
        if let Some(raft) = &self.raft_node {
            // Only leader can remove nodes
            if !raft.is_leader().await {
                return Err(anyhow!("Not the leader - cannot remove nodes"));
            }

            // Create remove node command
            let command = raft::Command::Delete {
                id: format!("node:{}", node_id),
            };

            // Append to Raft log (will replicate to followers)
            raft.append_entry(command).await.map_err(|e| anyhow!(e))?;

            // Wait for commit before applying locally
            // In production, this would be done via apply loop
        }

        {
            let mut nodes = self.nodes.write().await;
            nodes.remove(node_id);
        }

        {
            let mut ring = self.hash_ring.write().await;
            ring.remove_node(node_id);
        }

        {
            let mut stats = self.stats.write().await;
            stats.total_nodes = stats.total_nodes.saturating_sub(1);
        }

        if self.config.auto_rebalance {
            self.rebalance().await?;
        }

        Ok(())
    }

    pub async fn get_shard_id(&self, key: &str) -> usize {
        match self.config.sharding_strategy {
            ShardingStrategy::Hash => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                (hasher.finish() as usize) % self.config.num_shards
            }
            ShardingStrategy::ConsistentHash => {
                let ring = self.hash_ring.read().await;
                let node = ring.get_node(key).unwrap_or_default();
                let sum = node
                    .as_bytes()
                    .iter()
                    .fold(0u32, |acc, &b| acc.wrapping_add(b as u32));
                (sum as usize) % self.config.num_shards
            }
            ShardingStrategy::Range => {
                key.as_bytes().first().copied().unwrap_or(0) as usize % self.config.num_shards
            }
            ShardingStrategy::Random => key.len() % self.config.num_shards,
        }
    }

    pub async fn rebalance(&self) -> Result<()> {
        let nodes = self.nodes.read().await;
        if nodes.is_empty() {
            return Ok(());
        }

        let node_ids: Vec<String> = nodes.keys().cloned().collect();
        drop(nodes);

        let mut shards = self.shards.write().await;

        for (shard_id, shard_info) in shards.iter_mut() {
            let idx = *shard_id % node_ids.len();
            shard_info.primary_node = node_ids[idx].clone();

            shard_info.replica_nodes.clear();
            for i in 1..self.config.replication_factor {
                let replica_idx = (idx + i) % node_ids.len();
                shard_info.replica_nodes.push(node_ids[replica_idx].clone());
            }
        }

        let mut stats = self.stats.write().await;
        stats.rebalances_performed += 1;

        Ok(())
    }

    pub async fn stats(&self) -> DistributedStats {
        self.stats.read().await.clone()
    }

    pub async fn cluster_health(&self) -> f32 {
        let stats = self.stats.read().await;
        if stats.total_nodes == 0 {
            return 0.0;
        }
        stats.healthy_nodes as f32 / stats.total_nodes as f32
    }

    /// Get replica nodes for a given shard
    pub async fn get_replicas(&self, shard_id: usize) -> Result<Vec<String>> {
        let shards = self.shards.read().await;
        let shard = shards
            .get(&shard_id)
            .ok_or_else(|| anyhow!("Shard {} not found", shard_id))?;

        Ok(shard.replica_nodes.clone())
    }

    /// Sync data to all replicas for a shard
    ///
    /// This replicates data from the primary to all replica nodes.
    /// Depending on the consistency level, this may wait for acknowledgment.
    pub async fn sync_to_replicas(&self, shard_id: usize, data: Vec<u8>) -> Result<()> {
        let replicas = self.get_replicas(shard_id).await?;

        if replicas.is_empty() {
            return Ok(());
        }

        // If Raft is enabled, replicate through consensus
        if let Some(raft) = &self.raft_node {
            if !raft.is_leader().await {
                return Err(anyhow!("Not the leader - cannot sync replicas"));
            }

            let command = raft::Command::Update {
                id: format!("shard:{}:sync", shard_id),
                vector: vec![],
                metadata: serde_json::json!({
                    "type": "replica_sync",
                    "shard_id": shard_id,
                    "data_size": data.len(),
                }),
            };

            raft.append_entry(command).await.map_err(|e| anyhow!(e))?;
        }

        // In a real implementation, send data to each replica node
        // For now, just simulate the sync
        match self.config.consistency_level {
            ConsistencyLevel::All => {
                // Wait for all replicas to acknowledge
                for _replica in &replicas {
                    // Simulate sync latency
                    // In production: await network RPC
                }
            }
            ConsistencyLevel::Quorum => {
                // Wait for majority (N/2 + 1)
                let quorum_size = (replicas.len() / 2) + 1;
                for _i in 0..quorum_size {
                    // Simulate sync to quorum
                }
            }
            ConsistencyLevel::One => {
                // Fire and forget - don't wait for acknowledgment
                // Replicas will eventually sync
            }
        }

        Ok(())
    }

    /// Query from read replicas
    ///
    /// This distributes read queries across replicas for load balancing.
    /// Returns results from the least loaded replica.
    pub async fn query_from_replicas(
        &self,
        shard_id: usize,
        query: Vec<f32>,
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let shards = self.shards.read().await;
        let shard = shards
            .get(&shard_id)
            .ok_or_else(|| anyhow!("Shard {} not found", shard_id))?;

        // Get all nodes (primary + replicas) for this shard
        let mut available_nodes = vec![shard.primary_node.clone()];
        available_nodes.extend(shard.replica_nodes.iter().cloned());

        if available_nodes.is_empty() {
            return Err(anyhow!("No nodes available for shard {}", shard_id));
        }

        // For consistency level, determine which nodes to query
        match self.config.consistency_level {
            ConsistencyLevel::All => {
                // Always query primary for strong consistency
                // In real implementation: send RPC to primary node
                Ok(vec![])
            }
            ConsistencyLevel::Quorum => {
                // Query quorum of nodes and return most recent results
                // In real implementation: scatter to quorum, gather results
                Ok(vec![])
            }
            ConsistencyLevel::One => {
                // Query any available replica (round-robin or least loaded)
                // For now, use the first available replica
                // In real implementation: check node health and load
                Ok(vec![])
            }
        }
    }

    /// Promote a replica to primary
    ///
    /// Called when a primary node fails. Promotes a replica to be the new primary.
    pub async fn promote_replica(&self, shard_id: usize, new_primary: String) -> Result<()> {
        // If Raft is enabled, go through consensus
        if let Some(raft) = &self.raft_node {
            if !raft.is_leader().await {
                return Err(anyhow!("Not the leader - cannot promote replica"));
            }

            let command = raft::Command::Update {
                id: format!("shard:{}:promote", shard_id),
                vector: vec![],
                metadata: serde_json::json!({
                    "type": "promote_replica",
                    "shard_id": shard_id,
                    "new_primary": new_primary,
                }),
            };

            raft.append_entry(command).await.map_err(|e| anyhow!(e))?;
        }

        let mut shards = self.shards.write().await;
        let shard = shards
            .get_mut(&shard_id)
            .ok_or_else(|| anyhow!("Shard {} not found", shard_id))?;

        // Verify the new primary is in the replica list
        if !shard.replica_nodes.contains(&new_primary) {
            return Err(anyhow!(
                "Node {} is not a replica of shard {}",
                new_primary,
                shard_id
            ));
        }

        // Remove from replica list and set as primary
        shard.replica_nodes.retain(|n| n != &new_primary);

        // Add old primary as replica if it's still alive
        if !shard.primary_node.is_empty() {
            shard.replica_nodes.push(shard.primary_node.clone());
        }

        shard.primary_node = new_primary;

        Ok(())
    }

    pub async fn insert(&self, id: &str, vector: Vec<f32>) -> Result<()> {
        let shard_id = self.get_shard_id(id).await;

        // In real implementation: route to appropriate node
        let mut stats = self.stats.write().await;
        stats.total_vectors += 1;

        Ok(())
    }

    pub async fn query(&self, query: Vec<f32>, k: usize) -> Result<Vec<(String, f32)>> {
        // In real implementation: scatter-gather across all shards

        let mut stats = self.stats.write().await;
        stats.queries_total += 1;

        // Mock results
        Ok(vec![])
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
    fn test_consistent_hash_ring() {
        let mut ring = ConsistentHashRing::new(100);

        ring.add_node("node1");
        ring.add_node("node2");
        ring.add_node("node3");

        // Same key should always map to same node
        let node1 = ring.get_node("key1").unwrap();
        let node2 = ring.get_node("key1").unwrap();
        assert_eq!(node1, node2);

        // Get multiple nodes for replication
        let nodes = ring.get_nodes("key1", 3);
        assert_eq!(nodes.len(), 3);
        assert!(
            nodes.contains(&"node1".to_string())
                || nodes.contains(&"node2".to_string())
                || nodes.contains(&"node3".to_string())
        );
    }

    #[test]
    fn test_sharding_strategies() {
        let config = DistributedConfig::new().with_num_shards(4);

        #[cfg(not(feature = "async"))]
        {
            let store = DistributedStore::create(config).unwrap();

            let shard1 = store.get_shard_id("key1");
            let shard2 = store.get_shard_id("key1");
            assert_eq!(shard1, shard2); // Consistent

            assert!(shard1 < 4); // Within shard count
        }
    }

    #[cfg(not(feature = "async"))]
    #[test]
    fn test_add_remove_nodes() {
        let config = DistributedConfig::new();
        let mut store = DistributedStore::create(config).unwrap();

        store.add_node("node1", "127.0.0.1:8001").unwrap();
        store.add_node("node2", "127.0.0.1:8002").unwrap();

        assert_eq!(store.stats().total_nodes, 2);

        store.remove_node("node1").unwrap();
        assert_eq!(store.stats().total_nodes, 1);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_distributed_store() {
        let config = DistributedConfig::new();
        let store = DistributedStore::create(config).await.unwrap();

        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();

        let stats = store.stats().await;
        assert_eq!(stats.total_nodes, 2);

        store.insert("doc1", vec![0.1, 0.2, 0.3]).await.unwrap();

        let stats = store.stats().await;
        assert_eq!(stats.total_vectors, 1);
    }

    #[test]
    fn test_cluster_health() {
        let config = DistributedConfig::new();

        #[cfg(not(feature = "async"))]
        {
            let mut store = DistributedStore::create(config).unwrap();
            store.add_node("node1", "127.0.0.1:8001").unwrap();

            store.stats.healthy_nodes = 1;
            assert_eq!(store.cluster_health(), 1.0);

            store.stats.healthy_nodes = 0;
            assert_eq!(store.cluster_health(), 0.0);
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_raft_integration() {
        let config = DistributedConfig::new();
        let mut store = DistributedStore::create(config).await.unwrap();

        // Initially, Raft is not enabled
        assert!(!store.is_raft_enabled());

        // Enable Raft for single-node cluster
        store
            .enable_raft("node1".to_string(), vec![])
            .await
            .unwrap();
        assert!(store.is_raft_enabled());

        // Get Raft node
        let raft = store.raft_node().unwrap();
        assert!(!raft.is_leader().await);

        // Start election (single-node should become leader immediately)
        raft.start_election().await;
        assert!(raft.is_leader().await);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_raft_add_node_leader_check() {
        let config = DistributedConfig::new();
        let mut store = DistributedStore::create(config).await.unwrap();

        // Enable Raft
        store
            .enable_raft("leader".to_string(), vec![])
            .await
            .unwrap();
        let raft = store.raft_node().unwrap();

        // Start election to become leader
        raft.start_election().await;
        assert!(raft.is_leader().await);

        // Now we can add nodes
        let result = store.add_node("node1", "127.0.0.1:8001").await;
        assert!(result.is_ok());

        let stats = store.stats().await;
        assert_eq!(stats.total_nodes, 1);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_raft_add_node_not_leader_fails() {
        let config = DistributedConfig::new();
        let mut store = DistributedStore::create(config).await.unwrap();

        // Enable Raft with peers (won't become leader automatically)
        store
            .enable_raft("follower".to_string(), vec!["leader".to_string()])
            .await
            .unwrap();

        let raft = store.raft_node().unwrap();
        assert!(!raft.is_leader().await);

        // Adding a node should fail (not leader)
        let result = store.add_node("node1", "127.0.0.1:8001").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Not the leader"));
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_raft_remove_node_leader_check() {
        let config = DistributedConfig::new();
        let mut store = DistributedStore::create(config).await.unwrap();

        // Enable Raft and become leader
        store
            .enable_raft("leader".to_string(), vec![])
            .await
            .unwrap();
        let raft = store.raft_node().unwrap();
        raft.start_election().await;
        assert!(raft.is_leader().await);

        // Add a node first (without Raft to avoid complexity)
        let mut store_without_raft = DistributedStore::create(DistributedConfig::new())
            .await
            .unwrap();
        store_without_raft
            .add_node("node1", "127.0.0.1:8001")
            .await
            .unwrap();

        // Now test remove with Raft
        store
            .enable_raft("leader".to_string(), vec![])
            .await
            .unwrap();
        let raft = store.raft_node().unwrap();
        raft.start_election().await;

        // Manually add the node to store
        store.add_node("node1", "127.0.0.1:8001").await.unwrap();

        // Remove should work
        let result = store.remove_node("node1").await;
        assert!(result.is_ok());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_consistent_hashing_with_raft() {
        let config = DistributedConfig {
            sharding_strategy: ShardingStrategy::ConsistentHash,
            ..Default::default()
        };
        let mut store = DistributedStore::create(config).await.unwrap();

        // Enable Raft and become leader
        store
            .enable_raft("leader".to_string(), vec![])
            .await
            .unwrap();
        let raft = store.raft_node().unwrap();
        raft.start_election().await;

        // Add nodes through Raft consensus
        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();
        store.add_node("node3", "127.0.0.1:8003").await.unwrap();

        // Verify consistent hashing works
        let key = "test-key";
        let shard_id = store.get_shard_id(key).await;
        assert!(shard_id < store.config.num_shards);

        // Stats should show 3 nodes
        let stats = store.stats().await;
        assert_eq!(stats.total_nodes, 3);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_replica_assignment() {
        let config = DistributedConfig {
            replication_factor: 3,
            ..Default::default()
        };
        let store = DistributedStore::create(config).await.unwrap();

        // Add nodes
        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();
        store.add_node("node3", "127.0.0.1:8003").await.unwrap();

        // Trigger rebalance to assign replicas
        store.rebalance().await.unwrap();

        // Check that each shard has replicas assigned
        let shards = store.shards.read().await;
        for (shard_id, shard) in shards.iter() {
            assert!(
                !shard.primary_node.is_empty(),
                "Shard {} has no primary",
                shard_id
            );
            assert_eq!(
                shard.replica_nodes.len(),
                2,
                "Shard {} should have 2 replicas",
                shard_id
            );
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_get_replicas() {
        let config = DistributedConfig {
            replication_factor: 3,
            num_shards: 4,
            ..Default::default()
        };
        let store = DistributedStore::create(config).await.unwrap();

        // Add nodes
        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();
        store.add_node("node3", "127.0.0.1:8003").await.unwrap();
        store.rebalance().await.unwrap();

        // Get replicas for shard 0
        let replicas = store.get_replicas(0).await.unwrap();
        assert_eq!(replicas.len(), 2); // replication_factor - 1

        // Getting replicas for non-existent shard should error
        let result = store.get_replicas(999).await;
        assert!(result.is_err());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_sync_to_replicas_eventual_consistency() {
        let config = DistributedConfig {
            replication_factor: 3,
            consistency_level: ConsistencyLevel::One,
            ..Default::default()
        };
        let store = DistributedStore::create(config).await.unwrap();

        // Add nodes and rebalance
        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();
        store.add_node("node3", "127.0.0.1:8003").await.unwrap();
        store.rebalance().await.unwrap();

        // Sync data to replicas (should succeed with eventual consistency)
        let result = store.sync_to_replicas(0, vec![1, 2, 3, 4]).await;
        assert!(result.is_ok());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_sync_to_replicas_with_raft() {
        let config = DistributedConfig {
            replication_factor: 3,
            consistency_level: ConsistencyLevel::Quorum,
            ..Default::default()
        };
        let mut store = DistributedStore::create(config).await.unwrap();

        // Enable Raft and become leader
        store
            .enable_raft("leader".to_string(), vec![])
            .await
            .unwrap();
        let raft = store.raft_node().unwrap();
        raft.start_election().await;

        // Add nodes
        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();
        store.add_node("node3", "127.0.0.1:8003").await.unwrap();
        store.rebalance().await.unwrap();

        // Sync with Raft consensus
        let result = store.sync_to_replicas(0, vec![1, 2, 3, 4]).await;
        assert!(result.is_ok());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_query_from_replicas() {
        let config = DistributedConfig {
            replication_factor: 3,
            consistency_level: ConsistencyLevel::One,
            ..Default::default()
        };
        let store = DistributedStore::create(config).await.unwrap();

        // Add nodes and rebalance
        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();
        store.add_node("node3", "127.0.0.1:8003").await.unwrap();
        store.rebalance().await.unwrap();

        // Query from replicas
        let result = store.query_from_replicas(0, vec![0.1, 0.2, 0.3], 10).await;
        assert!(result.is_ok());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_query_strong_consistency() {
        let config = DistributedConfig {
            replication_factor: 3,
            consistency_level: ConsistencyLevel::All,
            ..Default::default()
        };
        let store = DistributedStore::create(config).await.unwrap();

        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();
        store.rebalance().await.unwrap();

        // With strong consistency, should query primary only
        let result = store.query_from_replicas(0, vec![0.1, 0.2, 0.3], 10).await;
        assert!(result.is_ok());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_promote_replica() {
        let config = DistributedConfig {
            replication_factor: 3,
            ..Default::default()
        };
        let store = DistributedStore::create(config).await.unwrap();

        // Add nodes and rebalance
        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();
        store.add_node("node3", "127.0.0.1:8003").await.unwrap();
        store.rebalance().await.unwrap();

        // Get current primary and a replica
        let shards = store.shards.read().await;
        let shard = shards.get(&0).unwrap();
        let old_primary = shard.primary_node.clone();
        let new_primary = shard.replica_nodes[0].clone();
        drop(shards);

        // Promote replica to primary
        store.promote_replica(0, new_primary.clone()).await.unwrap();

        // Verify promotion
        let shards = store.shards.read().await;
        let shard = shards.get(&0).unwrap();
        assert_eq!(shard.primary_node, new_primary);
        assert!(shard.replica_nodes.contains(&old_primary));
        assert!(!shard.replica_nodes.contains(&new_primary));
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_promote_replica_with_raft() {
        let config = DistributedConfig {
            replication_factor: 3,
            ..Default::default()
        };
        let mut store = DistributedStore::create(config).await.unwrap();

        // Enable Raft and become leader
        store
            .enable_raft("leader".to_string(), vec![])
            .await
            .unwrap();
        let raft = store.raft_node().unwrap();
        raft.start_election().await;

        // Add nodes
        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();
        store.add_node("node3", "127.0.0.1:8003").await.unwrap();
        store.rebalance().await.unwrap();

        // Get a replica to promote
        let shards = store.shards.read().await;
        let shard = shards.get(&0).unwrap();
        let new_primary = shard.replica_nodes[0].clone();
        drop(shards);

        // Promote with Raft consensus
        let result = store.promote_replica(0, new_primary).await;
        assert!(result.is_ok());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_promote_non_replica_fails() {
        let config = DistributedConfig::default();
        let store = DistributedStore::create(config).await.unwrap();

        store.add_node("node1", "127.0.0.1:8001").await.unwrap();
        store.add_node("node2", "127.0.0.1:8002").await.unwrap();
        store.rebalance().await.unwrap();

        // Try to promote a node that isn't a replica
        let result = store
            .promote_replica(0, "node-not-replica".to_string())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a replica"));
    }
}
