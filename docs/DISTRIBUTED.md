# Distributed VecStore

VecStore supports distributed deployments for horizontal scaling, high availability, and fault tolerance. This guide covers the architecture, configuration, and operations of distributed VecStore clusters.

## Architecture Overview

### Components

1. **Nodes**: Individual VecStore instances that form the cluster
2. **Shards**: Data partitions distributed across nodes
3. **Replicas**: Copies of shards for redundancy
4. **Raft Consensus**: Distributed consensus for cluster coordination

```
┌─────────────────────────────────────────────────────────────┐
│                    VecStore Cluster                         │
├─────────────┬─────────────┬─────────────┬─────────────────-─┤
│   Node 1    │   Node 2    │   Node 3    │   Node N          │
│  (Leader)   │ (Follower)  │ (Follower)  │  (Follower)       │
├─────────────┼─────────────┼─────────────┼──────────────────-┤
│  Shard 1    │  Shard 2    │  Shard 3    │   Shard M         │
│  Replica    │  Replica    │  Replica    │   Replica         │
└─────────────┴─────────────┴─────────────┴───────────────────┘
```

## Configuration

### Basic Setup

```rust
use vecstore::distributed::{
    DistributedConfig, DistributedStore, ShardingStrategy,
    ConsistencyLevel, ReplicationStrategy,
};

// Create distributed configuration
let config = DistributedConfig::new()
    .with_num_shards(8)
    .with_replication_factor(3)
    .with_sharding_strategy(ShardingStrategy::ConsistentHash)
    .with_consistency(ConsistencyLevel::Quorum)
    .with_replication_strategy(ReplicationStrategy::PrimaryBackup);

// Create the distributed store
let store = DistributedStore::create(config)?;
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `num_shards` | 4 | Number of data partitions |
| `replication_factor` | 3 | Number of replicas per shard |
| `heartbeat_interval_ms` | 100 | Heartbeat interval in milliseconds |
| `failure_timeout_ms` | 500 | Node failure detection timeout |
| `auto_rebalance` | true | Automatic shard rebalancing |

## Sharding Strategies

### Hash Sharding

Simple modulo-based sharding:

```rust
let config = DistributedConfig::new()
    .with_sharding_strategy(ShardingStrategy::Hash);
```

### Consistent Hash

Minimizes data movement when nodes are added/removed:

```rust
let config = DistributedConfig::new()
    .with_sharding_strategy(ShardingStrategy::ConsistentHash);
```

### Range Sharding

Shards based on key ranges:

```rust
let config = DistributedConfig::new()
    .with_sharding_strategy(ShardingStrategy::Range);
```

### Random Sharding

Random assignment for uniform distribution:

```rust
let config = DistributedConfig::new()
    .with_sharding_strategy(ShardingStrategy::Random);
```

## Consistency Levels

### One

Write to one replica, read from one replica:

```rust
.with_consistency(ConsistencyLevel::One)
```

### Quorum

Majority agreement required:

```rust
.with_consistency(ConsistencyLevel::Quorum)
```

### All

All replicas must agree:

```rust
.with_consistency(ConsistencyLevel::All)
```

## Replication Strategies

### Primary-Backup

One primary handles writes, backups receive updates:

```rust
.with_replication_strategy(ReplicationStrategy::PrimaryBackup)
```

### Multi-Master

All replicas can handle writes:

```rust
.with_replication_strategy(ReplicationStrategy::MultiMaster)
```

### Chain Replication

Writes flow through a chain of replicas:

```rust
.with_replication_strategy(ReplicationStrategy::Chain)
```

## Cluster Operations

### Adding Nodes

```rust
let mut store = DistributedStore::create(config)?;

// Add nodes to the cluster
store.add_node("node1", "192.168.1.10:8001")?;
store.add_node("node2", "192.168.1.11:8001")?;
store.add_node("node3", "192.168.1.12:8001")?;
```

### Removing Nodes

```rust
// Gracefully remove a node
store.remove_node("node2")?;
```

### Rebalancing

```rust
// Force a rebalance
store.rebalance()?;

// Check rebalance status
let stats = store.stats();
println!("Rebalances performed: {}", stats.rebalances_performed);
```

### Cluster Health

```rust
// Get cluster health (0.0 - 1.0)
let health = store.cluster_health();
println!("Cluster health: {:.1}%", health * 100.0);

// Get node statuses
let stats = store.stats();
println!("Total nodes: {}", stats.total_nodes);
println!("Active nodes: {}", stats.active_nodes);
```

## Raft Consensus

VecStore uses Raft for distributed consensus, ensuring strong consistency across nodes.

### Leader Election

The cluster automatically elects a leader:

1. Nodes start as followers
2. If no heartbeat received, a node starts an election
3. Candidate requests votes from other nodes
4. Node with majority votes becomes leader
5. Leader sends heartbeats to maintain leadership

### Log Replication

All state changes are replicated through the log:

1. Client sends request to leader
2. Leader appends to local log
3. Leader replicates to followers
4. Once majority acknowledges, entry is committed
5. Leader responds to client

### Snapshots

Large logs are compacted via snapshots:

```rust
// Leader can install snapshots to catch up lagging followers
// This happens automatically during cluster operations
```

## Monitoring

### Cluster Statistics

```rust
let stats = store.stats();

println!("Cluster Statistics:");
println!("  Total nodes: {}", stats.total_nodes);
println!("  Active nodes: {}", stats.active_nodes);
println!("  Total shards: {}", stats.total_shards);
println!("  Rebalances: {}", stats.rebalances_performed);
```

### Node Status

Nodes can be in various states:

- `Healthy`: Operating normally
- `Degraded`: Partial functionality
- `Failed`: Not responding
- `Joining`: Being added to cluster
- `Leaving`: Being removed from cluster

## Best Practices

### Shard Count

- Start with 2x the number of nodes you plan to have
- More shards = better load distribution but more overhead
- Recommended: 4-16 shards for small clusters, 32-64 for large

### Replication Factor

- Minimum 3 for production (tolerates 1 failure)
- Higher = more durability but more storage and write latency
- Must be odd for proper quorum calculations

### Network Configuration

- Use low-latency connections between nodes
- Ensure proper firewall rules for cluster ports
- Consider dedicated network for cluster traffic

### Failure Handling

- Monitor cluster health continuously
- Set up alerts for node failures
- Have replacement nodes ready
- Test failure scenarios regularly

## Troubleshooting

### Split Brain

If network partitions occur:

1. Check network connectivity between nodes
2. Ensure quorum is maintainable
3. Use `ConsistencyLevel::Quorum` or higher

### Slow Rebalancing

If rebalancing is slow:

1. Check network bandwidth
2. Reduce shard size
3. Increase rebalance concurrency

### Leader Instability

If leader keeps changing:

1. Check for network issues
2. Adjust heartbeat timeout
3. Ensure nodes have stable clocks
