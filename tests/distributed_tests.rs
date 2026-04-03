// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Tests for distributed multi-node indexing

use vecstore::distributed::{
    ConsistencyLevel, DistributedConfig, DistributedStore, NodeStatus, ReplicationStrategy,
    ShardingStrategy,
};

/// Test distributed config defaults
#[test]
fn test_distributed_config_defaults() {
    let config = DistributedConfig::default();

    assert!(config.num_shards > 0);
    assert!(config.replication_factor > 0);
    assert!(config.heartbeat_interval_ms > 0);
    assert!(config.failure_timeout_ms > config.heartbeat_interval_ms);
}

/// Test distributed config builder
#[test]
fn test_distributed_config_builder() {
    let config = DistributedConfig::new()
        .with_num_shards(16)
        .with_replication_factor(5)
        .with_sharding_strategy(ShardingStrategy::ConsistentHash)
        .with_consistency(ConsistencyLevel::Quorum)
        .with_replication_strategy(ReplicationStrategy::PrimaryBackup);

    assert_eq!(config.num_shards, 16);
    assert_eq!(config.replication_factor, 5);
    assert_eq!(config.sharding_strategy, ShardingStrategy::ConsistentHash);
    assert_eq!(config.consistency_level, ConsistencyLevel::Quorum);
    assert_eq!(config.replication_strategy, ReplicationStrategy::PrimaryBackup);
}

/// Test sharding strategies
#[test]
fn test_sharding_strategies() {
    let strategies = vec![
        ShardingStrategy::Hash,
        ShardingStrategy::ConsistentHash,
        ShardingStrategy::Range,
        ShardingStrategy::Random,
    ];

    for strategy in strategies {
        let config = DistributedConfig::new()
            .with_num_shards(4)
            .with_sharding_strategy(strategy);

        assert_eq!(config.sharding_strategy, strategy);
    }
}

/// Test consistency levels
#[test]
fn test_consistency_levels() {
    let levels = vec![
        ConsistencyLevel::One,
        ConsistencyLevel::Quorum,
        ConsistencyLevel::All,
    ];

    for level in levels {
        let config = DistributedConfig::new().with_consistency(level);
        assert_eq!(config.consistency_level, level);
    }
}

/// Test replication strategies
#[test]
fn test_replication_strategies() {
    let strategies = vec![
        ReplicationStrategy::PrimaryBackup,
        ReplicationStrategy::MultiMaster,
        ReplicationStrategy::Chain,
    ];

    for strategy in strategies {
        let config = DistributedConfig::new().with_replication_strategy(strategy);
        assert_eq!(config.replication_strategy, strategy);
    }
}

/// Test distributed store creation (sync version)
#[test]
fn test_distributed_store_creation() {
    let config = DistributedConfig::new().with_num_shards(4);

    let store = DistributedStore::create(config).unwrap();

    // Initial cluster should be healthy (no nodes = 0%)
    assert_eq!(store.cluster_health(), 0.0);
}

/// Test adding nodes
#[test]
fn test_add_nodes() {
    let config = DistributedConfig::new().with_num_shards(4);

    let mut store = DistributedStore::create(config).unwrap();

    // Add nodes
    store.add_node("node1", "127.0.0.1:8001").unwrap();
    store.add_node("node2", "127.0.0.1:8002").unwrap();
    store.add_node("node3", "127.0.0.1:8003").unwrap();

    // Check stats
    let stats = store.stats();
    assert_eq!(stats.total_nodes, 3);
}

/// Test removing nodes
#[test]
fn test_remove_nodes() {
    let config = DistributedConfig::new().with_num_shards(4);

    let mut store = DistributedStore::create(config).unwrap();

    store.add_node("node1", "127.0.0.1:8001").unwrap();
    store.add_node("node2", "127.0.0.1:8002").unwrap();

    assert_eq!(store.stats().total_nodes, 2);

    store.remove_node("node1").unwrap();

    assert_eq!(store.stats().total_nodes, 1);
}

/// Test shard ID computation with hash strategy
#[test]
fn test_shard_id_hash() {
    let config = DistributedConfig::new()
        .with_num_shards(8)
        .with_sharding_strategy(ShardingStrategy::Hash);

    let store = DistributedStore::create(config).unwrap();

    // Same key should always map to same shard
    let key = "test_document_id";
    let shard1 = store.get_shard_id(key);
    let shard2 = store.get_shard_id(key);

    assert_eq!(shard1, shard2);
    assert!(shard1 < 8); // Shard ID should be < num_shards
}

/// Test shard ID distribution
#[test]
fn test_shard_distribution() {
    let num_shards = 8;
    let config = DistributedConfig::new()
        .with_num_shards(num_shards)
        .with_sharding_strategy(ShardingStrategy::Hash);

    let store = DistributedStore::create(config).unwrap();

    // Generate many keys and check distribution
    let mut shard_counts = vec![0; num_shards];
    for i in 0..1000 {
        let key = format!("key_{}", i);
        let shard = store.get_shard_id(&key);
        shard_counts[shard] += 1;
    }

    // Each shard should have some keys (rough check for distribution)
    for (shard, count) in shard_counts.iter().enumerate() {
        assert!(
            *count > 0,
            "Shard {} received no keys, distribution may be poor",
            shard
        );
    }
}

/// Test zero shards validation
#[test]
#[should_panic(expected = "Number of shards must be at least 1")]
fn test_zero_shards_panic() {
    let _config = DistributedConfig::new().with_num_shards(0);
}

/// Test node status enum
#[test]
fn test_node_status() {
    let statuses = vec![
        NodeStatus::Healthy,
        NodeStatus::Degraded,
        NodeStatus::Failed,
        NodeStatus::Joining,
        NodeStatus::Leaving,
    ];

    // Just ensure all statuses are valid and can be used
    for status in statuses {
        match status {
            NodeStatus::Healthy => {}
            NodeStatus::Degraded => {}
            NodeStatus::Failed => {}
            NodeStatus::Joining => {}
            NodeStatus::Leaving => {}
        }
    }
}

/// Test rebalancing
#[test]
fn test_rebalancing() {
    let config = DistributedConfig::new()
        .with_num_shards(4)
        .with_replication_factor(2);

    let mut store = DistributedStore::create(config).unwrap();

    // Add nodes
    store.add_node("node1", "127.0.0.1:8001").unwrap();
    store.add_node("node2", "127.0.0.1:8002").unwrap();

    // Force rebalance
    store.rebalance().unwrap();

    // Check that rebalancing happened
    assert!(store.stats().rebalances_performed >= 1);
}

/// Test auto-rebalancing configuration
#[test]
fn test_auto_rebalance_config() {
    // With auto-rebalance enabled (default)
    let config_auto = DistributedConfig::default();
    assert!(config_auto.auto_rebalance);

    // Create store and add node - should trigger rebalance
    let mut store = DistributedStore::create(config_auto).unwrap();
    store.add_node("node1", "127.0.0.1:8001").unwrap();

    // Rebalance should have been performed
    assert!(store.stats().rebalances_performed >= 1);
}

/// Test consistent hash ring
#[test]
fn test_consistent_hash_ring() {
    use vecstore::distributed::ConsistentHashRing;

    let mut ring = ConsistentHashRing::new(100); // 100 virtual nodes

    // Empty ring should return None
    assert!(ring.get_node("any_key").is_none());

    // Add nodes
    ring.add_node("node1");
    ring.add_node("node2");
    ring.add_node("node3");

    // Should now return a node
    let node = ring.get_node("test_key").unwrap();
    assert!(!node.is_empty());

    // Same key should return same node (consistency)
    let node2 = ring.get_node("test_key").unwrap();
    assert_eq!(node, node2);

    // Different keys might map to different nodes
    let nodes: Vec<String> = (0..100)
        .map(|i| ring.get_node(&format!("key_{}", i)).unwrap())
        .collect();

    // Should have some variety (not all same node)
    let unique_nodes: std::collections::HashSet<_> = nodes.iter().collect();
    assert!(unique_nodes.len() > 1, "All keys mapped to same node");
}

/// Test getting multiple nodes from consistent hash ring
#[test]
fn test_consistent_hash_ring_multiple_nodes() {
    use vecstore::distributed::ConsistentHashRing;

    let mut ring = ConsistentHashRing::new(100);

    ring.add_node("node1");
    ring.add_node("node2");
    ring.add_node("node3");

    // Get 2 nodes for replication
    let nodes = ring.get_nodes("test_key", 2);
    assert_eq!(nodes.len(), 2);

    // Nodes should be unique
    assert_ne!(nodes[0], nodes[1]);

    // Get all 3 nodes
    let all_nodes = ring.get_nodes("test_key", 3);
    assert_eq!(all_nodes.len(), 3);

    // Request more nodes than available
    let nodes = ring.get_nodes("test_key", 5);
    assert_eq!(nodes.len(), 3); // Should only return available nodes
}

/// Test removing node from consistent hash ring
#[test]
fn test_consistent_hash_ring_remove() {
    use vecstore::distributed::ConsistentHashRing;

    let mut ring = ConsistentHashRing::new(100);

    ring.add_node("node1");
    ring.add_node("node2");

    let before = ring.get_node("test_key").unwrap();

    ring.remove_node("node1");

    let after = ring.get_node("test_key").unwrap();

    // After removing node1, key might map to different node
    // or same node if it was already on node2
    assert_eq!(after, "node2"); // Only node2 remains
}

/// Test serialization of config types
#[test]
fn test_config_serialization() {
    use serde_json;

    let config = DistributedConfig::new()
        .with_num_shards(8)
        .with_replication_factor(3)
        .with_sharding_strategy(ShardingStrategy::ConsistentHash)
        .with_consistency(ConsistencyLevel::Quorum);

    // Strategies should serialize/deserialize
    let sharding_json = serde_json::to_string(&config.sharding_strategy).unwrap();
    let sharding: ShardingStrategy = serde_json::from_str(&sharding_json).unwrap();
    assert_eq!(sharding, ShardingStrategy::ConsistentHash);

    let consistency_json = serde_json::to_string(&config.consistency_level).unwrap();
    let consistency: ConsistencyLevel = serde_json::from_str(&consistency_json).unwrap();
    assert_eq!(consistency, ConsistencyLevel::Quorum);
}
