# VecStore Production Features - Quick Reference

## 🚀 Batch Operations

### Rust API
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

---

## 🔍 Query Validation & Cost Estimation

### Rust API
```rust
use vecstore::{VecStore, Query};

let store = VecStore::open("./data")?;

let query = Query {
    vector: vec![0.1, 0.2, 0.3],
    k: 10,
    filter: None,
};

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

---

## 🧹 Auto-Compaction

### Enable Auto-Compaction
```rust
use vecstore::{VecStore, CompactionConfig};

let mut store = VecStore::open("./data")?;

// Configure thresholds
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

---

## ⏰ TTL (Time-To-Live)

### Insert with TTL
```rust
use vecstore::{VecStore, Metadata};

let mut store = VecStore::open("./data")?;

// Expires in 1 hour (3600 seconds)
store.upsert_with_ttl(
    "session-123".into(),
    vec![0.1, 0.2, 0.3],
    Metadata { fields: HashMap::new() },
    3600  // TTL in seconds
)?;
```

### Set TTL on Existing Record
```rust
// Set 2-hour TTL on existing record
store.set_ttl("doc-456", 7200)?;
```

### Expire TTL Records
```rust
// Call this periodically (e.g., hourly cron job)
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

---

## 💾 Namespace Backup/Restore

### Create Backup
```rust
use vecstore::namespace_manager::NamespaceManager;

let manager = NamespaceManager::new("./namespaces")?;

// Create backup
manager.backup_namespace(
    "customer-123",
    "daily-backup-2025-10-19"
)?;
```

### List Backups
```rust
let backups = manager.list_namespace_backups("customer-123")?;

for (name, created_at, count) in backups {
    println!("{}: {} vectors at {}", name, count, created_at);
}
```

### Restore from Backup
```rust
manager.restore_namespace(
    "customer-123",
    "daily-backup-2025-10-19"
)?;
```

### Delete Backup
```rust
manager.delete_namespace_backup(
    "customer-123",
    "old-backup-2025-10-01"
)?;
```

---

## 📊 Complete Production Example

```rust
use vecstore::{VecStore, CompactionConfig, BatchOperation, Metadata};
use tokio::time::{interval, Duration};
use std::collections::HashMap;

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
            metadata: Metadata { fields: HashMap::new() },
        },
        // ... more operations
    ];
    let result = store.batch_execute(operations)?;
    println!("Batch: {} succeeded, {} failed", result.succeeded, result.failed);

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

---

## 🎯 Best Practices

### Batch Operations
- ✅ Use for bulk inserts (10-100x faster)
- ✅ Check `BatchResult.errors` for failures
- ✅ Combine different operation types in one batch

### Query Estimation
- ✅ Validate user-provided queries before execution
- ✅ Show cost estimates to users for transparency
- ✅ Use recommendations to guide optimization

### Auto-Compaction
- ✅ Enable for production deployments
- ✅ Call `maybe_compact()` after large delete operations
- ✅ Set thresholds based on your data characteristics

### TTL
- ✅ Use for session data, caches, temporary embeddings
- ✅ Run `expire_ttl_records()` hourly or daily
- ✅ Combine with auto-compaction for automatic cleanup

### Backups
- ✅ Schedule regular backups (daily/weekly)
- ✅ Test restore process periodically
- ✅ Keep multiple backup generations
- ✅ Clean up old backups to save space

---

## 🔧 HTTP API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/batch-execute` | POST | Execute mixed batch operations |
| `/v1/query-estimate` | POST | Validate query and estimate cost |
| `/v1/upsert` | POST | Insert/update single vector |
| `/v1/query` | POST | Search for similar vectors |
| `/v1/query-explain` | POST | Query with detailed explanations |
| `/v1/compact` | POST | Manually trigger compaction |
| `/v1/stats` | GET | Get store statistics |

---

## 📈 Performance Tips

1. **Batch Operations**: Always use batch for >10 operations
2. **Query Estimation**: <1ms overhead, always validate
3. **Auto-Compaction**: Set thresholds to balance cleanup frequency vs. overhead
4. **TTL Expiration**: Run hourly or daily, not per-request
5. **Backups**: Schedule during low-traffic periods

---

## 🐛 Troubleshooting

### Batch operation fails
```rust
let result = store.batch_execute(operations)?;
for error in result.errors {
    println!("Operation {} failed: {} - {}",
             error.index, error.operation, error.error);
}
```

### Query validation fails
```rust
let estimate = store.estimate_query(&query);
if !estimate.valid {
    for error in estimate.errors {
        println!("❌ {}", error);
    }
}
```

### Compaction not triggering
```rust
// Check current stats
let deleted = store.deleted_count();
let total = store.records.len();
let ratio = deleted as f32 / total as f32;

println!("Deleted: {}/{} ({:.1}%)", deleted, total, ratio * 100.0);
println!("Config: {:?}", store.compaction_config());
```

---

## 📚 More Information

- **Full Documentation**: See `PRODUCTION-FEATURES.md`
- **Original Features**: See `CHANGELOG-new-features.md`
- **API Documentation**: Run `cargo doc --open`
- **Examples**: See `examples/` directory

---

**All features are production-ready and fully tested! 🚀**
