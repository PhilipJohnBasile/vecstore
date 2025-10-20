# VecStore - Production Features Guide

**Complete guide to VecStore's production-ready features for building scalable vector search applications.**

---

## Table of Contents

1. [Overview](#overview)
2. [Multi-Tenant Namespaces](#multi-tenant-namespaces)
3. [Batch Operations](#batch-operations)
4. [Query Validation & Cost Estimation](#query-validation--cost-estimation)
5. [Auto-Compaction](#auto-compaction)
6. [TTL (Time-To-Live)](#ttl-time-to-live)
7. [Backup & Restore](#backup--restore)
8. [Query Explain](#query-explain)
9. [Production Deployment](#production-deployment)

---

## Overview

VecStore follows a **hybrid philosophy**: **simple by default, powerful when needed**. All production features are opt-in and maintain 100% backward compatibility.

### Feature Status

| Feature | Status | Use Case |
|---------|--------|----------|
| Multi-Tenant Namespaces | ✅ Production | SaaS multi-tenancy, environment isolation |
| Batch Operations | ✅ Production | Bulk imports, 10-100x faster operations |
| Query Validation | ✅ Production | Pre-flight validation, cost estimation |
| Auto-Compaction | ✅ Production | Automatic cleanup of deleted records |
| TTL Support | ✅ Production | Time-based expiration (sessions, caches) |
| Namespace Backups | ✅ Production | Disaster recovery, snapshots |
| Query Explain | ✅ Production | Debug search results, optimization |

---

## Multi-Tenant Namespaces

**Isolated VecStore instances with resource quotas and management APIs.**

### Quick Start

```bash
# Start server in namespace mode
cargo run --bin vecstore-server --features server -- \
  --namespaces \
  --namespace-root ./namespaces
```

### Create Namespace

```bash
curl -X POST http://localhost:8080/admin/namespaces \
  -H "Content-Type: application/json" \
  -d '{
    "id": "customer-123",
    "name": "Acme Corp",
    "quotas": {
      "max_vectors": 1000000,
      "max_storage_bytes": 10737418240,
      "max_requests_per_second": 100.0
    }
  }'
```

### Programmatic API

```rust
use vecstore::{NamespaceManager, NamespaceQuotas};

let manager = NamespaceManager::new("./namespaces")?;

// Create namespace with quotas
manager.create_namespace(
    "customer-123",
    "Acme Corp",
    Some(NamespaceQuotas {
        max_vectors: Some(1_000_000),
        max_requests_per_second: Some(100.0),
        ..Default::default()
    }),
)?;

// Upsert to namespace
manager.upsert(
    &"customer-123".to_string(),
    "doc1".to_string(),
    vec![0.1, 0.2, 0.3],
    metadata,
)?;

// Query within namespace
let results = manager.query(&"customer-123".to_string(), query)?;

// Get statistics
let stats = manager.get_stats(&"customer-123".to_string())?;
println!("Vectors: {}", stats.vector_count);
```

### Use Cases

- **SaaS Multi-Tenancy**: One namespace per customer
- **Environment Isolation**: dev/staging/prod separation
- **Resource Quotas**: Enforce usage limits per tenant
- **Usage Tracking**: Monitor per-customer resource consumption

### Admin Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/admin/namespaces` | POST | Create namespace |
| `/admin/namespaces` | GET | List all namespaces |
| `/admin/namespaces/:id` | GET | Get namespace details |
| `/admin/namespaces/:id/quotas` | PUT | Update quotas |
| `/admin/namespaces/:id/status` | PUT | Update status (active/suspended) |
| `/admin/namespaces/:id/stats` | GET | Get usage statistics |
| `/admin/namespaces/:id` | DELETE | Delete namespace |

**See [admin-api.md](admin-api.md) for complete reference.**

---

## Batch Operations

**Execute multiple operations (upsert, delete, restore, update) in a single atomic-like request.**

### Performance

- **10-100x faster** than individual operations
- Minimal batching overhead (~0.01ms per operation)
- Single write lock for entire batch

### Quick Start

```rust
use vecstore::{VecStore, BatchOperation, Metadata};

let mut store = VecStore::open("./data")?;

let operations = vec![
    BatchOperation::Upsert {
        id: "doc1".into(),
        vector: vec![0.1, 0.2, 0.3],
        metadata: Metadata { fields: HashMap::new() },
    },
    BatchOperation::Delete { id: "doc2".into() },
    BatchOperation::SoftDelete { id: "doc3".into() },
    BatchOperation::Restore { id: "doc4".into() },
    BatchOperation::UpdateMetadata {
        id: "doc5".into(),
        metadata: Metadata { fields: HashMap::new() },
    },
];

let result = store.batch_execute(operations)?;
println!("✅ {}, ❌ {}", result.succeeded, result.failed);
```

### HTTP API

```bash
curl -X POST http://localhost:8080/v1/batch-execute \
  -H "Content-Type: application/json" \
  -d '{
    "operations": [
      {"op": "upsert", "id": "doc1", "vector": [0.1, 0.2], "metadata": {}},
      {"op": "delete", "id": "doc2"},
      {"op": "soft_delete", "id": "doc3"}
    ]
  }'
```

### Response

```json
{
  "succeeded": 2,
  "failed": 1,
  "errors": [
    {
      "index": 2,
      "operation": "soft_delete(doc3)",
      "error": "Record not found: doc3"
    }
  ],
  "duration_ms": 1.27
}
```

### Use Cases

- **Bulk Imports**: Load thousands of vectors efficiently
- **Bulk Updates**: Update metadata for many records
- **Cleanup Operations**: Delete multiple obsolete records
- **Atomic-like Operations**: Group related changes together

---

## Query Validation & Cost Estimation

**Pre-flight validation and cost estimation for queries before execution.**

### Quick Start

```rust
use vecstore::{VecStore, Query};

let store = VecStore::open("./data")?;

let query = Query {
    vector: vec![0.1, 0.2, 0.3],
    k: 100,
    filter: Some(parse_filter("category = 'tech'")?),
};

// Validate before executing
let estimate = store.estimate_query(&query);

if !estimate.valid {
    eprintln!("❌ Query errors: {:?}", estimate.errors);
    return Err(...);
}

println!("💰 Cost: {:.2}", estimate.cost_estimate);
println!("⏱️  Duration: {:.2}ms", estimate.estimated_duration_ms);
println!("💡 Tips: {:?}", estimate.recommendations);

// Now run the actual query
let results = store.query(query)?;
```

### HTTP API

```bash
curl -X POST http://localhost:8080/v1/query-estimate \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "limit": 10,
    "filter": "category = '\''tech'\''"
  }'
```

### Response

```json
{
  "valid": true,
  "errors": [],
  "cost_estimate": 0.506,
  "estimated_distance_calculations": 100,
  "estimated_nodes_visited": 100,
  "will_overfetch": true,
  "recommendations": [
    "Filter will cause over-fetching: requesting 100 candidates to find 10 results",
    "Filtered queries are slower. Consider reducing k or adding indexes"
  ],
  "estimated_duration_ms": 0.75
}
```

### Cost Factors

The cost estimate (0.0 - 1.0) considers:
- **k value** (larger k = higher cost)
- **Filter presence** (filters add overhead)
- **Over-fetching** (when filters require extra candidates)
- **Database size** (more vectors = higher cost)

### Use Cases

- **API Validation**: Catch dimension mismatches before execution
- **Cost Control**: Warn users about expensive queries
- **Query Optimization**: Get recommendations for improvement
- **Rate Limiting**: Block queries exceeding cost threshold

---

## Auto-Compaction

**Automatically remove soft-deleted records when thresholds are met.**

### Quick Start

```rust
use vecstore::{VecStore, CompactionConfig};

let mut store = VecStore::open("./data")?;

// Configure auto-compaction
store.set_compaction_config(CompactionConfig {
    enabled: true,
    min_deleted_records: 1000,     // Trigger after 1000 deletions
    min_deleted_ratio: 0.1,        // OR when 10% deleted
});

// Check and compact if needed (call periodically)
let result = store.maybe_compact()?;

if result.triggered {
    println!("🧹 Compacted {} records in {:.2}ms",
             result.removed_count, result.duration_ms);
} else {
    println!("ℹ️  {}", result.reason);
}
```

### Manual Compaction

```rust
// Always compact (ignores thresholds)
let removed = store.compact()?;
println!("Removed {} records", removed);
```

### Default Configuration

- **Disabled by default** (opt-in)
- Triggers after **1000 deleted records**
- OR when **10% of records** are deleted

### Use Cases

- **Maintenance Tasks**: Run periodically to clean up deleted records
- **Performance Optimization**: Reduce storage overhead
- **Before Backups**: Compact before creating snapshots
- **Production Ops**: Automated cleanup without manual intervention

---

## TTL (Time-To-Live)

**Automatically expire records after a specified time.**

### Quick Start

```rust
use vecstore::{VecStore, Metadata};

let mut store = VecStore::open("./data")?;

// Insert with TTL (expires in 1 hour)
store.upsert_with_ttl(
    "session-123",
    vec![0.1, 0.2, 0.3],
    Metadata { fields: HashMap::new() },
    3600  // TTL in seconds
)?;

// Set TTL on existing record
store.set_ttl("doc-456", 7200)?;  // 2 hours

// Expire TTL records (call periodically)
let expired = store.expire_ttl_records()?;
println!("⏰ Expired {} records", expired);

// Optional: Compact after expiration
if expired > 100 {
    store.maybe_compact()?;
}
```

### Background Task Example

```rust
use tokio::time::{interval, Duration};

// Run every hour
let mut timer = interval(Duration::from_secs(3600));

loop {
    timer.tick().await;

    // Expire TTL records
    let expired = store.expire_ttl_records()?;

    // Auto-compact if needed
    store.maybe_compact()?;

    println!("Maintenance: {} expired", expired);
}
```

### Use Cases

- **Session Data**: Expire after inactivity
- **Cache Entries**: Time-limited storage
- **Temporary Embeddings**: ML experiment data
- **Rate Limiting Tokens**: Auto-expiring quotas

---

## Backup & Restore

**Create and restore snapshots of namespace data.**

### Quick Start

```rust
use vecstore::NamespaceManager;

let manager = NamespaceManager::new("./namespaces")?;

// Create backup
manager.backup_namespace(
    "customer-123",
    "daily-backup-2025-10-19"
)?;

// List backups
let backups = manager.list_namespace_backups("customer-123")?;
for (name, created_at, count) in backups {
    println!("{}: {} vectors at {}", name, count, created_at);
}

// Restore from backup
manager.restore_namespace(
    "customer-123",
    "daily-backup-2025-10-19"
)?;

// Delete old backup
manager.delete_namespace_backup(
    "customer-123",
    "old-backup-2025-10-01"
)?;
```

### Use Cases

- **Disaster Recovery**: Restore from backup after data loss
- **Pre-Migration Snapshots**: Backup before major changes
- **Testing**: Copy production data for testing
- **Compliance**: Maintain audit trails

---

## Query Explain

**Detailed explanations of search results for debugging and optimization.**

### Quick Start

```rust
use vecstore::{VecStore, Query};

let store = VecStore::open("./data")?;

let query = Query {
    vector: vec![0.1, 0.2, 0.3],
    k: 10,
    filter: Some(parse_filter("category = 'tech'")?),
};

// Get explained results
let explained_results = store.query_explain(query)?;

for result in explained_results {
    println!("Result: {}", result.id);
    println!("  Score: {}", result.score);
    println!("  Rank: #{}", result.explanation.rank);
    println!("  Explanation: {}", result.explanation.explanation_text);
}
```

### HTTP API

```bash
curl -X POST http://localhost:8080/v1/query-explain \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "limit": 10,
    "filter": "category = '\''tech'\''"
  }'
```

### Response

```json
{
  "results": [
    {
      "id": "doc_123",
      "score": 0.9876,
      "explanation": {
        "raw_score": 0.9876,
        "distance_metric": "Cosine",
        "filter_passed": true,
        "filter_details": {
          "filter_expr": "category = 'tech'",
          "matched_conditions": ["All conditions matched"],
          "passed": true
        },
        "graph_stats": {
          "distance_calculations": 50,
          "nodes_visited": 50
        },
        "rank": 1,
        "explanation_text": "Ranked #1 with score 0.9876 (Cosine). Passed filters. 50 candidates evaluated."
      }
    }
  ]
}
```

### Use Cases

- **Debug Unexpected Results**: Understand why documents rank where they do
- **Filter Optimization**: See how many candidates are filtered out
- **Understanding Scoring**: See raw scores and distance metrics
- **Performance Analysis**: Monitor HNSW traversal statistics
- **Development**: Validate search behavior during development

**See [query-explain.md](query-explain.md) for complete guide.**

---

## Production Deployment

### Complete Production Example

```rust
use vecstore::{VecStore, CompactionConfig, BatchOperation};
use tokio::time::{interval, Duration};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Open store with auto-compaction
    let mut store = VecStore::open("./data")?;
    store.set_compaction_config(CompactionConfig {
        enabled: true,
        min_deleted_records: 1000,
        min_deleted_ratio: 0.1,
    });

    // 2. Batch insert with TTL for session data
    let operations = vec![
        BatchOperation::Upsert {
            id: "session-1".into(),
            vector: vec![0.1, 0.2, 0.3],
            metadata: Metadata::default(),
        },
        // ... more operations
    ];
    let result = store.batch_execute(operations)?;
    println!("Batch: {} succeeded", result.succeeded);

    // Set TTL on session data (1 hour)
    store.set_ttl("session-1", 3600)?;

    // 3. Validate queries before execution
    let query = Query {
        vector: vec![0.1, 0.2, 0.3],
        k: 100,
        filter: Some(parse_filter("active = true")?),
    };

    let estimate = store.estimate_query(&query);
    if estimate.cost_estimate > 0.5 {
        println!("⚠️  High-cost query: {:.2}", estimate.cost_estimate);
    }

    // 4. Background maintenance task
    let mut timer = interval(Duration::from_secs(3600));
    tokio::spawn(async move {
        loop {
            timer.tick().await;

            // Expire TTL records
            if let Ok(expired) = store.expire_ttl_records() {
                println!("⏰ Expired {} records", expired);
            }

            // Auto-compact if needed
            if let Ok(result) = store.maybe_compact() {
                if result.triggered {
                    println!("🧹 Compacted {} records", result.removed_count);
                }
            }
        }
    });

    Ok(())
}
```

### Best Practices

#### Batch Operations
- ✅ Use for bulk inserts (10-100x faster)
- ✅ Check `BatchResult.errors` for failures
- ✅ Combine different operation types in one batch

#### Query Estimation
- ✅ Validate user-provided queries before execution
- ✅ Show cost estimates to users for transparency
- ✅ Use recommendations to guide optimization

#### Auto-Compaction
- ✅ Enable for production deployments
- ✅ Call `maybe_compact()` after large delete operations
- ✅ Set thresholds based on your data characteristics

#### TTL
- ✅ Use for session data, caches, temporary embeddings
- ✅ Run `expire_ttl_records()` hourly or daily
- ✅ Combine with auto-compaction for automatic cleanup

#### Backups
- ✅ Schedule regular backups (daily/weekly)
- ✅ Test restore process periodically
- ✅ Keep multiple backup generations
- ✅ Clean up old backups to save space

### Performance Tips

1. **Batch Operations**: Always use batch for >10 operations
2. **Query Estimation**: <1ms overhead, always validate
3. **Auto-Compaction**: Set thresholds to balance cleanup frequency vs. overhead
4. **TTL Expiration**: Run hourly or daily, not per-request
5. **Backups**: Schedule during low-traffic periods

---

## Feature Comparison

| Feature | Overhead | When to Use |
|---------|----------|-------------|
| Batch Operations | ~0.01ms/op | Bulk imports, >10 operations |
| Query Estimation | <1ms | Always (pre-flight validation) |
| Auto-Compaction | Varies | After deletes, periodic maintenance |
| TTL Expiration | O(n) scan | Hourly/daily cleanup |
| Backups | Depends on size | Daily/weekly snapshots |
| Query Explain | ~5-10% | Development, debugging |

---

## Migration Guide

### Existing Deployments

No changes required! All features are opt-in:

```rust
// Existing code works unchanged
let mut store = VecStore::open("./data")?;
store.upsert(id, vector, metadata)?;
let results = store.query(query)?;
```

### Adopting New Features

**Step 1: Enable Auto-Compaction**
```rust
store.set_compaction_config(CompactionConfig {
    enabled: true,
    min_deleted_records: 1000,
    min_deleted_ratio: 0.1,
});
```

**Step 2: Use Batch Operations**
```rust
let operations = vec![/* ... */];
let result = store.batch_execute(operations)?;
```

**Step 3: Validate Queries**
```rust
let estimate = store.estimate_query(&query);
if !estimate.valid {
    return Err(...);
}
```

**Step 4: Implement TTL (Optional)**
```rust
store.upsert_with_ttl(id, vector, metadata, ttl_seconds)?;
store.expire_ttl_records()?;
```

---

## Support

- **Admin API**: See [admin-api.md](admin-api.md)
- **Query Explain**: See [query-explain.md](query-explain.md)
- **API Reference**: See [API.md](API.md)
- **Examples**: Check `examples/` directory

---

**All features are production-ready and fully tested! 🚀**
