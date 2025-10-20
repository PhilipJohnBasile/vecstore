# Multi-Tenant Namespaces Guide

## Overview

VecStore supports **multi-tenant namespaces**, allowing you to run multiple isolated vector databases within a single server instance. Each namespace has its own:

- **Isolated vector storage** - Separate VecStore instance per namespace
- **Resource quotas** - Limits on vectors, storage, requests/second
- **Usage tracking** - Detailed metrics per namespace
- **Access control** - Independent lifecycle management

This is ideal for:
- **SaaS applications** - One namespace per customer/organization
- **Multi-project environments** - Separate namespaces for dev/staging/prod
- **Resource management** - Enforce quotas and prevent resource exhaustion

## Quick Start

### Creating a Namespace

```rust
use vecstore::{NamespaceManager, NamespaceQuotas};

// Create namespace manager
let manager = NamespaceManager::new("./data")?;

// Create a namespace with default quotas
manager.create_namespace(
    "customer-123".to_string(),
    "Customer 123 Workspace".to_string(),
    None,  // Use default quotas
)?;

// Or create with custom quotas
let quotas = NamespaceQuotas {
    max_vectors: Some(50_000),
    max_storage_bytes: Some(500 * 1024 * 1024),  // 500MB
    max_requests_per_second: Some(50.0),
    max_concurrent_queries: Some(5),
    max_dimension: Some(1536),
    max_results_per_query: Some(500),
    max_batch_size: Some(500),
};

manager.create_namespace(
    "premium-customer".to_string(),
    "Premium Customer".to_string(),
    Some(quotas),
)?;
```

### Using a Namespace

```rust
use vecstore::{Query, Metadata};
use serde_json::json;

// Upsert vectors to a namespace
manager.upsert(
    &"customer-123".to_string(),
    "doc1".to_string(),
    vec![0.1, 0.2, 0.3],
    Metadata::from(json!({
        "title": "Document 1",
        "category": "tech"
    })),
)?;

// Query within a namespace
let query = Query {
    vector: vec![0.15, 0.25, 0.35],
    k: 10,
    filter: Some("category = 'tech'".to_string()),
};

let results = manager.query(&"customer-123".to_string(), query)?;

for neighbor in results {
    println!("{}: {:.4}", neighbor.id, neighbor.distance);
}
```

## Quota Tiers

VecStore provides predefined quota tiers for common use cases:

### Free Tier

Perfect for trial accounts and small projects:

```rust
use vecstore::NamespaceQuotas;

let quotas = NamespaceQuotas::free_tier();
// max_vectors: 10,000
// max_storage_bytes: 100 MB
// max_requests_per_second: 10
// max_concurrent_queries: 2
// max_dimension: 1536 (OpenAI embedding size)
// max_results_per_query: 100
// max_batch_size: 100
```

### Pro Tier

For production workloads:

```rust
let quotas = NamespaceQuotas::pro_tier();
// max_vectors: 1,000,000
// max_storage_bytes: 10 GB
// max_requests_per_second: 100
// max_concurrent_queries: 20
// max_dimension: 4096
// max_results_per_query: 1,000
// max_batch_size: 1,000
```

### Unlimited

For enterprise or admin namespaces:

```rust
let quotas = NamespaceQuotas::unlimited();
// All quotas set to None (no limits)
```

### Custom Quotas

Mix and match based on your needs:

```rust
let quotas = NamespaceQuotas {
    max_vectors: Some(100_000),
    max_storage_bytes: Some(1024 * 1024 * 1024),  // 1GB
    max_requests_per_second: None,  // No rate limit
    max_concurrent_queries: Some(10),
    max_dimension: Some(768),  // Smaller embeddings
    max_results_per_query: Some(200),
    max_batch_size: Some(200),
};
```

## Namespace Status

Namespaces can be in one of four states:

### Active

Default state - namespace is fully operational and accepting all requests.

```rust
use vecstore::NamespaceStatus;

manager.update_status(&namespace_id, NamespaceStatus::Active)?;
```

### Suspended

Namespace is temporarily disabled - all requests will be rejected. Useful for:
- Quota violations
- Payment issues
- Administrative actions

```rust
manager.update_status(&namespace_id, NamespaceStatus::Suspended)?;
```

### ReadOnly

Namespace accepts queries but rejects writes. Useful for:
- Downgrading to free tier
- Maintenance windows
- Graceful degradation

```rust
manager.update_status(&namespace_id, NamespaceStatus::ReadOnly)?;
```

### PendingDeletion

Namespace is marked for deletion - only delete operations allowed.

```rust
manager.update_status(&namespace_id, NamespaceStatus::PendingDeletion)?;
manager.delete_namespace(&namespace_id)?;
```

## Resource Usage & Quotas

### Checking Usage

```rust
let stats = manager.get_stats(&namespace_id)?;

println!("Namespace: {}", stats.namespace_id);
println!("Vectors: {} / {:?}", stats.vector_count, namespace.quotas.max_vectors);
println!("Active: {}", stats.active_count);
println!("Deleted: {}", stats.deleted_count);
println!("Quota Utilization: {:.1}%", stats.quota_utilization * 100.0);
println!("Total Requests: {}", stats.total_requests);
println!("Total Queries: {}", stats.total_queries);
println!("Total Upserts: {}", stats.total_upserts);
println!("Total Deletes: {}", stats.total_deletes);
println!("Status: {:?}", stats.status);
```

### Quota Enforcement

Quotas are enforced automatically on every operation:

**Vector Count Limit:**
```rust
// Will fail if namespace already has max_vectors
manager.upsert(&namespace_id, id, vector, metadata)?;
// Error: "Upsert would exceed vector quota: 10000 + 1 > 10000"
```

**Storage Limit:**
```rust
// Checked during upsert based on current storage usage
// Error: "Storage quota exceeded: 105000000 / 104857600 bytes"
```

**Rate Limit:**
```rust
// Enforced per-second using sliding windows
// Error: "Rate limit exceeded: 10 requests/second"
```

**Concurrent Query Limit:**
```rust
// Prevents overload from too many simultaneous queries
// Error: "Concurrent query limit exceeded: 5 / 5"
```

**Result Size Limit:**
```rust
let query = Query {
    vector: vec![0.1, 0.2],
    k: 5000,  // Too many results
    filter: None,
};
// Error: "Query limit exceeds maximum: 5000 > 1000"
```

**Batch Size Limit:**
```rust
// When implementing batch upsert
// Error: "Batch size exceeds limit: 2000 > 1000"
```

### Quota Utilization

Quota utilization is calculated as the **maximum** utilization across all quota types:

```rust
let namespace = manager.get_namespace(&namespace_id)?;

// If vector_count is 90% of max_vectors
// and storage_bytes is 50% of max_storage_bytes
// Then quota_utilization() returns 0.9 (90%)

if namespace.is_near_quota() {
    println!("Warning: Namespace is near quota limits (>80%)");
}
```

## Admin API

### gRPC Admin Service

The Admin API provides full namespace lifecycle management:

```protobuf
service VecStoreAdminService {
  rpc CreateNamespace(CreateNamespaceRequest) returns (CreateNamespaceResponse);
  rpc ListNamespaces(ListNamespacesRequest) returns (ListNamespacesResponse);
  rpc GetNamespace(GetNamespaceRequest) returns (GetNamespaceResponse);
  rpc UpdateNamespaceQuotas(UpdateNamespaceQuotasRequest) returns (UpdateNamespaceQuotasResponse);
  rpc UpdateNamespaceStatus(UpdateNamespaceStatusRequest) returns (UpdateNamespaceStatusResponse);
  rpc DeleteNamespace(DeleteNamespaceRequest) returns (DeleteNamespaceResponse);
  rpc GetNamespaceStats(GetNamespaceStatsRequest) returns (GetNamespaceStatsResponse);
  rpc GetAggregateStats(GetAggregateStatsRequest) returns (GetAggregateStatsResponse);
}
```

### HTTP Admin Endpoints

**Create Namespace:**
```bash
curl -X POST http://localhost:8080/admin/namespaces \
  -H "Content-Type: application/json" \
  -d '{
    "id": "customer-123",
    "name": "Customer 123",
    "quotas": {
      "max_vectors": 50000,
      "max_storage_bytes": 524288000,
      "max_requests_per_second": 50.0
    }
  }'
```

**List Namespaces:**
```bash
curl http://localhost:8080/admin/namespaces
```

**Get Namespace Details:**
```bash
curl http://localhost:8080/admin/namespaces/customer-123
```

**Update Quotas:**
```bash
curl -X PUT http://localhost:8080/admin/namespaces/customer-123/quotas \
  -H "Content-Type: application/json" \
  -d '{
    "max_vectors": 100000,
    "max_storage_bytes": 1073741824
  }'
```

**Update Status:**
```bash
curl -X PUT http://localhost:8080/admin/namespaces/customer-123/status \
  -H "Content-Type: application/json" \
  -d '{"status": "suspended"}'
```

**Get Namespace Stats:**
```bash
curl http://localhost:8080/admin/namespaces/customer-123/stats
```

**Delete Namespace:**
```bash
curl -X DELETE http://localhost:8080/admin/namespaces/customer-123
```

**Get Aggregate Stats:**
```bash
curl http://localhost:8080/admin/stats
```

## Persistence & Recovery

### Directory Structure

Each namespace is stored in its own directory:

```
data/
├── customer-123/
│   ├── namespace.json          # Namespace metadata
│   ├── vectors.bin             # Vector data
│   ├── metadata.bin            # Metadata storage
│   └── hnsw_index.bin          # HNSW graph
├── customer-456/
│   ├── namespace.json
│   ├── vectors.bin
│   └── ...
└── premium-customer/
    └── ...
```

### Loading Namespaces

```rust
let manager = NamespaceManager::new("./data")?;

// Load all existing namespaces from disk
let loaded = manager.load_namespaces()?;
println!("Loaded {} namespaces", loaded.len());
```

### Saving Metadata

```rust
// Save all namespace metadata (quotas, usage, status)
manager.save_all()?;
```

This is automatically done when:
- Quotas are updated
- Status is changed
- Namespace is deleted

## Multi-Tenancy Patterns

### Pattern 1: Customer Isolation

One namespace per customer for complete data isolation:

```rust
// On customer signup
manager.create_namespace(
    customer.id.clone(),
    customer.name.clone(),
    Some(NamespaceQuotas::free_tier()),
)?;

// On customer upgrade
manager.update_quotas(&customer.id, NamespaceQuotas::pro_tier())?;

// On customer deletion
manager.delete_namespace(&customer.id)?;
```

### Pattern 2: Environment Separation

Separate namespaces for dev/staging/prod:

```rust
manager.create_namespace("dev".to_string(), "Development".to_string(), None)?;
manager.create_namespace("staging".to_string(), "Staging".to_string(), None)?;
manager.create_namespace("production".to_string(), "Production".to_string(), None)?;
```

### Pattern 3: Feature Flags

Use namespaces for A/B testing or feature rollouts:

```rust
manager.create_namespace("beta-features".to_string(), "Beta Features".to_string(), None)?;
manager.create_namespace("stable".to_string(), "Stable".to_string(), None)?;

// Route users based on feature flags
let namespace_id = if user.has_beta_access() {
    "beta-features"
} else {
    "stable"
};

manager.query(&namespace_id.to_string(), query)?;
```

### Pattern 4: Tiered Service

Different quotas based on subscription tier:

```rust
match customer.tier {
    Tier::Free => manager.update_quotas(&customer.id, NamespaceQuotas::free_tier())?,
    Tier::Pro => manager.update_quotas(&customer.id, NamespaceQuotas::pro_tier())?,
    Tier::Enterprise => manager.update_quotas(&customer.id, NamespaceQuotas::unlimited())?,
}
```

## Monitoring & Observability

### Aggregate Statistics

```rust
let stats = manager.get_aggregate_stats();

println!("Total Namespaces: {}", stats.total_namespaces);
println!("Active Namespaces: {}", stats.active_namespaces);
println!("Total Vectors: {}", stats.total_vectors);
println!("Total Requests: {}", stats.total_requests);
```

### Per-Namespace Metrics

Track usage over time for billing and capacity planning:

```rust
for namespace in manager.list_namespaces() {
    let stats = manager.get_stats(&namespace.id)?;

    println!("{}: {} vectors, {:.1}% utilization",
        namespace.name,
        stats.vector_count,
        stats.quota_utilization * 100.0
    );
}
```

### Prometheus Metrics

When running in server mode with observability enabled:

- `vecstore_namespace_vector_count{namespace="customer-123"}` - Current vector count
- `vecstore_namespace_quota_utilization{namespace="customer-123"}` - Quota utilization (0.0-1.0)
- `vecstore_namespace_requests_total{namespace="customer-123"}` - Total requests
- `vecstore_namespace_queries_total{namespace="customer-123"}` - Total queries

## Best Practices

### 1. Use Meaningful Namespace IDs

```rust
// Good - unique, stable identifiers
"customer-550e8400-e29b-41d4-a716-446655440000"
"org-acme-corp"
"project-recommendation-engine"

// Bad - non-unique or temporary
"test"
"user1"
"tmp"
```

### 2. Set Appropriate Quotas

```rust
// Start conservative, increase based on actual usage
let initial_quotas = NamespaceQuotas::free_tier();

// Monitor utilization and upgrade proactively
if namespace.quota_utilization() > 0.7 {
    // Warn customer they're approaching limits
}

if namespace.quota_utilization() > 0.9 {
    // Suggest upgrade or send notification
}
```

### 3. Handle Quota Errors Gracefully

```rust
match manager.upsert(&namespace_id, id, vector, metadata) {
    Ok(_) => println!("Vector inserted"),
    Err(e) if e.to_string().contains("quota exceeded") => {
        // Show user-friendly error
        println!("Storage limit reached. Please upgrade your plan.");
    }
    Err(e) if e.to_string().contains("Rate limit exceeded") => {
        // Implement backoff or queue
        std::thread::sleep(Duration::from_millis(100));
        // Retry...
    }
    Err(e) => return Err(e),
}
```

### 4. Implement Soft Deletes

```rust
// Instead of immediate deletion
manager.update_status(&namespace_id, NamespaceStatus::PendingDeletion)?;

// Allow grace period for recovery (e.g., 30 days)
// Then hard delete
manager.delete_namespace(&namespace_id)?;
```

### 5. Regular Backups

```rust
// Backup all namespace metadata
manager.save_all()?;

// Per-namespace snapshots using VecStore API
let stores = /* get stores */;
for (ns_id, store) in stores {
    store.create_snapshot(&format!("backup-{}-{}", ns_id, timestamp))?;
}
```

### 6. Monitor Quota Utilization

```rust
// Regular quota check job
for namespace in manager.list_namespaces() {
    if namespace.is_near_quota() {
        send_notification(&namespace, "Approaching quota limits");
    }
}
```

## Advanced Topics

### Custom Quota Validation

Extend NamespaceQuotas with business-specific rules:

```rust
impl NamespaceQuotas {
    pub fn validate_for_tier(&self, tier: &str) -> Result<()> {
        match tier {
            "free" => {
                if self.max_vectors.unwrap_or(0) > 10_000 {
                    return Err(anyhow!("Free tier limited to 10K vectors"));
                }
            }
            "pro" => {
                if self.max_vectors.unwrap_or(0) > 1_000_000 {
                    return Err(anyhow!("Pro tier limited to 1M vectors"));
                }
            }
            _ => {}
        }
        Ok(())
    }
}
```

### Dynamic Quota Adjustment

Implement auto-scaling based on usage:

```rust
fn check_and_adjust_quotas(manager: &NamespaceManager, namespace_id: &str) -> Result<()> {
    let stats = manager.get_stats(namespace_id)?;
    let mut namespace = manager.get_namespace(namespace_id)?;

    // If consistently high utilization, increase quota
    if stats.quota_utilization > 0.85 {
        let mut new_quotas = namespace.quotas.clone();
        if let Some(max_vectors) = new_quotas.max_vectors {
            new_quotas.max_vectors = Some((max_vectors as f64 * 1.5) as usize);
        }
        manager.update_quotas(namespace_id, new_quotas)?;
    }

    Ok(())
}
```

### Namespace Metadata

Store custom metadata per namespace:

```rust
let mut metadata = std::collections::HashMap::new();
metadata.insert("customer_tier".to_string(), "enterprise".to_string());
metadata.insert("region".to_string(), "us-west-2".to_string());
metadata.insert("created_by".to_string(), "admin@example.com".to_string());

// Metadata is part of Namespace struct
namespace.metadata = metadata;
```

## Troubleshooting

### Namespace Not Found

```
Error: Namespace not found: customer-123
```

**Solutions:**
- Check namespace ID spelling
- Verify namespace was created successfully
- Check if namespace was deleted
- Load namespaces from disk: `manager.load_namespaces()?`

### Quota Exceeded Errors

```
Error: Vector quota exceeded: 10000 / 10000 vectors
```

**Solutions:**
- Increase quotas: `manager.update_quotas()`
- Delete old vectors: `manager.remove()`
- Upgrade namespace tier
- Compact database to remove soft-deleted vectors

### Rate Limit Errors

```
Error: Rate limit exceeded: 100 requests/second
```

**Solutions:**
- Implement client-side rate limiting
- Increase `max_requests_per_second` quota
- Use batch operations instead of individual requests
- Distribute load across multiple namespaces

### Performance Issues

**Symptom:** Slow queries across all namespaces

**Solutions:**
- Check total vector count across all namespaces
- Monitor CPU and memory usage
- Consider Product Quantization for memory savings
- Scale horizontally by sharding namespaces across servers

## Examples

See `examples/namespace_demo.rs` for complete working examples:

```bash
cargo run --example namespace_demo --features server
```

## Next Steps

- Read [SERVER.md](SERVER.md) for multi-tenant server deployment
- Review [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for architecture details
- Check [observability/README.md](observability/README.md) for monitoring setup
- Explore the Admin API in `proto/vecstore.proto`

---

**Need Help?**
- Report issues: https://github.com/yourusername/vecstore/issues
- Contribute: See CONTRIBUTING.md
