//! Distributed Multi-Node Indexing Example
//!
//! This example demonstrates distributed vector indexing across multiple nodes
//! for horizontal scalability, high availability, and fault tolerance.
//!
//! ## Features Demonstrated
//!
//! - Cluster setup and node management
//! - Sharding strategies (hash, consistent hash, range)
//! - Replication for fault tolerance
//! - Consistency levels (one, quorum, all)
//! - Auto-rebalancing
//! - Cluster health monitoring
//!
//! ## Running
//!
//! ```bash
//! # Sync version
//! cargo run --example distributed_indexing
//!
//! # Async version (requires async feature)
//! cargo run --example distributed_indexing --features async
//! ```

use vecstore::distributed::{
    ConsistencyLevel, ConsistentHashRing, DistributedConfig, DistributedStore, NodeStatus,
    ReplicationStrategy, ShardingStrategy,
};

fn main() {
    println!("🌐 Distributed Multi-Node Indexing Example\n");

    // ============================================================
    // 1. Configuration
    // ============================================================
    println!("⚙️  Configuration:");
    println!("═════════════════\n");

    let config = DistributedConfig::new()
        .with_num_shards(8)
        .with_replication_factor(3)
        .with_sharding_strategy(ShardingStrategy::ConsistentHash)
        .with_consistency(ConsistencyLevel::Quorum)
        .with_replication_strategy(ReplicationStrategy::PrimaryBackup);

    println!("Number of shards: {}", config.num_shards);
    println!("Replication factor: {}", config.replication_factor);
    println!("Sharding strategy: {:?}", config.sharding_strategy);
    println!("Consistency level: {:?}", config.consistency_level);
    println!("Replication strategy: {:?}", config.replication_strategy);
    println!("Heartbeat interval: {}ms", config.heartbeat_interval_ms);
    println!("Failure timeout: {}ms", config.failure_timeout_ms);
    println!("Auto-rebalance: {}", config.auto_rebalance);

    // ============================================================
    // 2. Cluster Setup
    // ============================================================
    println!("\n\n🔧 Cluster Setup:");
    println!("═════════════════\n");

    #[cfg(not(feature = "async"))]
    {
        let mut store = DistributedStore::create(config.clone()).unwrap();

        println!("Adding nodes to cluster...\n");

        store.add_node("node1", "127.0.0.1:8001").unwrap();
        println!("  ✓ Added node1 (127.0.0.1:8001)");

        store.add_node("node2", "127.0.0.1:8002").unwrap();
        println!("  ✓ Added node2 (127.0.0.1:8002)");

        store.add_node("node3", "127.0.0.1:8003").unwrap();
        println!("  ✓ Added node3 (127.0.0.1:8003)");

        store.add_node("node4", "127.0.0.1:8004").unwrap();
        println!("  ✓ Added node4 (127.0.0.1:8004)");

        let stats = store.stats();
        println!("\nCluster initialized:");
        println!("  Total nodes: {}", stats.total_nodes);
        println!("  Total shards: {}", config.num_shards);
        println!("  Rebalances performed: {}", stats.rebalances_performed);

        // ============================================================
        // 3. Sharding Strategies
        // ============================================================
        println!("\n\n🔀 Sharding Strategies:");
        println!("═══════════════════════\n");

        let keys = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];

        println!("Hash-based sharding:");
        for key in &keys {
            let shard = store.get_shard_id(key);
            println!("  {} → Shard {}", key, shard);
        }

        // ============================================================
        // 4. Consistent Hashing
        // ============================================================
        println!("\n\n⚖️  Consistent Hashing:");
        println!("═════════════════════\n");

        let mut ring = ConsistentHashRing::new(150);
        ring.add_node("node1");
        ring.add_node("node2");
        ring.add_node("node3");
        ring.add_node("node4");

        println!("Key distribution:");
        for key in &keys {
            let node = ring.get_node(key).unwrap();
            println!("  {} → {}", key, node);
        }

        println!("\nReplica placement (RF=3):");
        for key in &keys {
            let nodes = ring.get_nodes(key, 3);
            println!("  {} → {:?}", key, nodes);
        }

        // ============================================================
        // 5. Rebalancing
        // ============================================================
        println!("\n\n🔄 Rebalancing:");
        println!("══════════════\n");

        println!("Adding new node...");
        store.add_node("node5", "127.0.0.1:8005").unwrap();

        let stats = store.stats();
        println!("  Total nodes: {}", stats.total_nodes);
        println!("  Rebalances: {}", stats.rebalances_performed);

        println!("\nRemoving node...");
        store.remove_node("node3").unwrap();

        let stats = store.stats();
        println!("  Total nodes: {}", stats.total_nodes);
        println!("  Rebalances: {}", stats.rebalances_performed);

        // ============================================================
        // 6. Cluster Health
        // ============================================================
        println!("\n\n💚 Cluster Health:");
        println!("═════════════════\n");

        let health = store.cluster_health();
        println!("Cluster health: {:.1}%", health * 100.0);

        let stats = store.stats();
        println!("Total nodes: {}", stats.total_nodes);
        println!("Healthy nodes: {}", stats.healthy_nodes);

        if health >= 0.75 {
            println!("Status: ✓ HEALTHY");
        } else if health >= 0.5 {
            println!("Status: ⚠ DEGRADED");
        } else {
            println!("Status: ✗ CRITICAL");
        }
    }

    // ============================================================
    // 7. Consistency Levels
    // ============================================================
    println!("\n\n📊 Consistency Levels:");
    println!("═════════════════════\n");

    println!("ONE (N=1):");
    println!("  - Write: 1 replica acknowledges");
    println!("  - Read: Query 1 replica");
    println!("  - Latency: Lowest");
    println!("  - Durability: Lowest\n");

    println!("QUORUM (N/2+1):");
    println!("  - Write: Majority of replicas acknowledge");
    println!("  - Read: Query majority of replicas");
    println!("  - Latency: Medium");
    println!("  - Durability: High\n");

    println!("ALL (N):");
    println!("  - Write: All replicas acknowledge");
    println!("  - Read: Query all replicas");
    println!("  - Latency: Highest");
    println!("  - Durability: Highest");

    // ============================================================
    // 8. Replication Strategies
    // ============================================================
    println!("\n\n🔁 Replication Strategies:");
    println!("═══════════════════════════\n");

    println!("Primary-Backup:");
    println!("  - One primary, N-1 backups");
    println!("  - Writes go to primary, replicated to backups");
    println!("  - Reads can go to any replica");
    println!("  - Simple but primary is bottleneck\n");

    println!("Multi-Master:");
    println!("  - All replicas accept writes");
    println!("  - Conflicts resolved via versioning");
    println!("  - High write throughput");
    println!("  - Complex conflict resolution\n");

    println!("Chain Replication:");
    println!("  - Replicas form a chain");
    println!("  - Writes propagate through chain");
    println!("  - Strong consistency");
    println!("  - Lower write throughput");

    // ============================================================
    // 9. Shard Distribution
    // ============================================================
    println!("\n\n📦 Shard Distribution:");
    println!("═════════════════════\n");

    println!("Simulated shard placement:\n");
    println!("{:>8}  {:>12}  {:>20}", "Shard", "Primary", "Replicas");
    println!("{:-<42}", "");

    for i in 0..8 {
        let primary = format!("node{}", (i % 4) + 1);
        let replica1 = format!("node{}", ((i + 1) % 4) + 1);
        let replica2 = format!("node{}", ((i + 2) % 4) + 1);

        println!("{:>8}  {:>12}  [{}, {}]", i, primary, replica1, replica2);
    }

    // ============================================================
    // 10. Scalability Analysis
    // ============================================================
    println!("\n\n📈 Scalability Analysis:");
    println!("═══════════════════════\n");

    let scenarios = vec![(4, 3, 8), (8, 3, 16), (16, 3, 32), (32, 3, 64)];

    println!(
        "{:>10}  {:>10}  {:>10}  {:>15}",
        "Nodes", "RF", "Shards", "Total Copies"
    );
    println!("{:-<50}", "");

    for (nodes, rf, shards) in scenarios {
        let total_copies = shards * rf;
        println!(
            "{:>10}  {:>10}  {:>10}  {:>15}",
            nodes, rf, shards, total_copies
        );
    }

    println!("\nKey insights:");
    println!("  - More nodes → better parallelism");
    println!("  - Higher RF → better fault tolerance");
    println!("  - More shards → finer-grained distribution");

    // ============================================================
    // 11. Fault Tolerance
    // ============================================================
    println!("\n\n🛡️  Fault Tolerance:");
    println!("═══════════════════\n");

    let scenarios = vec![
        (4, 3, 1, true),
        (4, 3, 2, false),
        (6, 3, 2, true),
        (6, 5, 2, true),
    ];

    println!(
        "{:>10}  {:>10}  {:>15}  {:>15}",
        "Nodes", "RF", "Failed Nodes", "Available?"
    );
    println!("{:-<55}", "");

    for (nodes, rf, failed, available) in scenarios {
        let status = if available { "✓ YES" } else { "✗ NO" };
        println!("{:>10}  {:>10}  {:>15}  {:>15}", nodes, rf, failed, status);
    }

    println!("\nRule: System available if (RF - failed) > RF/2");

    // ============================================================
    // 12. Real-world Scenarios
    // ============================================================
    println!("\n\n🌍 Real-world Scenarios:");
    println!("═══════════════════════\n");

    println!("1. E-commerce Product Catalog:");
    println!("   - 100M products");
    println!("   - 32 nodes, 64 shards, RF=3");
    println!("   - ~3M vectors per shard");
    println!("   - Can tolerate 1 node failure\n");

    println!("2. Social Media Posts:");
    println!("   - 10B posts");
    println!("   - 128 nodes, 256 shards, RF=2");
    println!("   - ~39M vectors per shard");
    println!("   - Optimized for write throughput\n");

    println!("3. Research Paper Archive:");
    println!("   - 50M papers");
    println!("   - 16 nodes, 32 shards, RF=5");
    println!("   - ~1.5M vectors per shard");
    println!("   - High durability, can tolerate 2 node failures");

    // ============================================================
    // 13. Best Practices
    // ============================================================
    println!("\n\n💡 Best Practices:");
    println!("══════════════════\n");

    println!("1. Shard sizing:");
    println!("   - Target 1-10M vectors per shard");
    println!("   - More shards = better parallelism\n");

    println!("2. Replication factor:");
    println!("   - RF=2: Basic redundancy");
    println!("   - RF=3: Production standard (tolerates 1 failure)");
    println!("   - RF=5: High durability (tolerates 2 failures)\n");

    println!("3. Consistency level:");
    println!("   - ONE: Maximum performance, eventual consistency");
    println!("   - QUORUM: Balance of consistency and performance");
    println!("   - ALL: Strong consistency, lower performance\n");

    println!("4. Node sizing:");
    println!("   - Homogeneous nodes preferred");
    println!("   - 8-16GB RAM per node minimum");
    println!("   - SSD storage recommended\n");

    println!("5. Monitoring:");
    println!("   - Track cluster health");
    println!("   - Monitor shard balance");
    println!("   - Alert on node failures");
    println!("   - Track rebalance frequency");

    println!("\n✅ Distributed indexing example complete!\n");

    println!("🎯 Key Takeaways:");
    println!("  - Sharding enables horizontal scaling");
    println!("  - Replication provides fault tolerance");
    println!("  - Consistent hashing minimizes data movement");
    println!("  - Quorum consensus balances consistency and availability");
    println!("  - Auto-rebalancing maintains even distribution");
    println!("  - 100% Pure Rust implementation");
}
