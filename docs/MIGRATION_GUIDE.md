# VecStore Migration Guide

Complete guide for migrating from Pinecone, Qdrant, Weaviate, and other vector databases to VecStore.

## Table of Contents

- [Overview](#overview)
- [Migration from Pinecone](#migration-from-pinecone)
- [Migration from Qdrant](#migration-from-qdrant)
- [Migration from Weaviate](#migration-from-weaviate)
- [Feature Comparison](#feature-comparison)
- [Code Migration Examples](#code-migration-examples)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## Overview

VecStore provides automated migration tools and code equivalents for popular vector databases. This guide helps you:

1. **Export data** from your current database
2. **Import data** into VecStore using bulk migration tools
3. **Update application code** to use VecStore API
4. **Optimize performance** for your use case

> **Note:** VecStore 0.0.1 is an alpha release. Migration tooling and APIs are still evolving.

### Why Migrate to VecStore?

- **Embedded workflow**: Runs in-process without managing a separate service.
- **Self-hosted economics**: No per-request API fees.
- **File-based storage**: Back up or replicate directories with standard tooling.
- **Mature feature set**: HNSW indexing, metadata filters, snapshots, batch ingestion.
- **RAG support**: Hybrid search, quantization helpers, reranking utilities.

> **Caution:** The 0.0.1 alpha is best suited for prototypes and evaluation migrations. Keep your source system online until you are confident VecStore covers your workload.

---

## Migration from Pinecone

### 1. Export Data from Pinecone

```python
# Export from Pinecone to JSON
import pinecone
import json

pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
index = pinecone.Index("your-index")

# Fetch all vectors (in batches)
vectors = []
for ids in index.list():  # Get all IDs
    fetch_response = index.fetch(ids=ids)
    vectors.extend(fetch_response['vectors'].items())

# Save to JSON file
with open('pinecone_export.json', 'w') as f:
    json.dump({
        "vectors": [
            {
                "id": v[0],
                "values": v[1]['values'],
                "metadata": v[1].get('metadata', {})
            }
            for v in vectors
        ]
    }, f)
```

### 2. Import into VecStore

```rust
use vecstore::bulk_migration::{PineconeMigration, MigrationConfig};
use vecstore::VecStore;

fn main() -> anyhow::Result<()> {
    let mut store = VecStore::open("my_vectors.db")?;

    let config = MigrationConfig {
        batch_size: 1000,
        validate: true,
        resume_from: None,
    };

    let migration = PineconeMigration::new(config)
        .with_progress(|current, total| {
            println!("Migrated {}/{} vectors", current, total);
        });

    let stats = migration.import_from_file("pinecone_export.json", &mut store)?;

    println!("Migration complete!");
    println!("  Vectors: {}", stats.total_vectors);
    println!("  Duration: {:?}", stats.duration);
    println!("  Throughput: {:.0} vectors/sec", stats.throughput);

    Ok(())
}
```

### 3. Code Migration

#### Pinecone Code:

```python
import pinecone

# Initialize
pinecone.init(api_key="...", environment="...")
index = pinecone.Index("my-index")

# Upsert
index.upsert(vectors=[
    ("id1", [0.1, 0.2, 0.3], {"category": "tech"}),
    ("id2", [0.4, 0.5, 0.6], {"category": "science"})
])

# Query
results = index.query(
    vector=[0.15, 0.25, 0.35],
    top_k=10,
    filter={"category": "tech"}
)

for match in results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
```

#### VecStore Equivalent:

```rust
use vecstore::{VecStore, Query};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    // Initialize
    let mut store = VecStore::open("my-index.db")?;

    // Upsert
    store.upsert("id1", vec![0.1, 0.2, 0.3],
                 json!({"category": "tech"}))?;
    store.upsert("id2", vec![0.4, 0.5, 0.6],
                 json!({"category": "science"}))?;

    // Query
    let query = Query::new(vec![0.15, 0.25, 0.35])
        .with_limit(10)
        .with_filter("category = 'tech'");

    let results = store.query(query)?;

    for result in results {
        println!("ID: {}, Score: {:.4}", result.id, result.score);
    }

    Ok(())
}
```

### Feature Mapping: Pinecone → VecStore

| Pinecone Feature | VecStore Equivalent |
|-----------------|---------------------|
| `index.upsert()` | `store.upsert()` |
| `index.query()` | `store.query()` |
| `index.delete()` | `store.delete()` |
| `index.fetch()` | Use query with ID filter |
| `index.describe_index_stats()` | `store.len()`, `store.stats()` |
| Metadata filtering | `Query::with_filter()` |
| Sparse-dense hybrid | `HybridQuery` with BM25 |
| Namespaces | `PartitionedStore` |

---

## Migration from Qdrant

### 1. Export Data from Qdrant

```python
from qdrant_client import QdrantClient
import json

client = QdrantClient(host="localhost", port=6333)
collection_name = "my_collection"

# Get all points
offset = None
all_points = []

while True:
    points = client.scroll(
        collection_name=collection_name,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=True
    )

    all_points.extend(points[0])

    if points[1] is None:
        break
    offset = points[1]

# Save to JSONL
with open('qdrant_export.jsonl', 'w') as f:
    for point in all_points:
        f.write(json.dumps({
            "id": str(point.id),
            "vector": point.vector,
            "payload": point.payload
        }) + '\n')
```

### 2. Import into VecStore

```rust
use vecstore::bulk_migration::{QdrantMigration, MigrationConfig};
use vecstore::VecStore;

fn main() -> anyhow::Result<()> {
    let mut store = VecStore::open("my_vectors.db")?;

    let config = MigrationConfig {
        batch_size: 1000,
        validate: true,
        resume_from: None,
    };

    let migration = QdrantMigration::new(config);
    let stats = migration.import_from_file("qdrant_export.jsonl", &mut store)?;

    println!("Migrated {} vectors", stats.total_vectors);

    Ok(())
}
```

### 3. Code Migration

#### Qdrant Code:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition

client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=128, distance=Distance.COSINE)
)

# Upsert points
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(id=1, vector=[0.1, 0.2, ...], payload={"city": "NYC"}),
        PointStruct(id=2, vector=[0.3, 0.4, ...], payload={"city": "SF"})
    ]
)

# Search
results = client.search(
    collection_name="my_collection",
    query_vector=[0.2, 0.3, ...],
    limit=10,
    query_filter=Filter(
        must=[FieldCondition(key="city", match={"value": "NYC"})]
    )
)
```

#### VecStore Equivalent:

```rust
use vecstore::{VecStore, Query, Config, Distance};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    // Create store with config
    let config = Config {
        distance: Distance::Cosine,
        ..Default::default()
    };
    let mut store = VecStore::with_config("my_collection.db", config)?;

    // Upsert points
    store.upsert("1", vec![0.1, 0.2, /* ... */],
                 json!({"city": "NYC"}))?;
    store.upsert("2", vec![0.3, 0.4, /* ... */],
                 json!({"city": "SF"}))?;

    // Search
    let query = Query::new(vec![0.2, 0.3, /* ... */])
        .with_limit(10)
        .with_filter("city = 'NYC'");

    let results = store.query(query)?;

    for result in results {
        println!("ID: {}, Score: {:.4}", result.id, result.score);
    }

    Ok(())
}
```

### Feature Mapping: Qdrant → VecStore

| Qdrant Feature | VecStore Equivalent |
|----------------|---------------------|
| `client.upsert()` | `store.upsert()` |
| `client.search()` | `store.query()` |
| `client.delete()` | `store.delete()` |
| `client.retrieve()` | Query with ID filter |
| `client.scroll()` | Iterate over `store.query()` |
| Payload filtering | `Query::with_filter()` |
| Named vectors | Multiple `VecStore` instances |
| Snapshots | Built-in with `VersionedStore` |

---

## Migration from Weaviate

### 1. Export Data from Weaviate

```python
import weaviate
import json

client = weaviate.Client("http://localhost:8080")

# Query all objects
result = client.query.get("MyClass", ["_additional { id vector }"]).with_limit(10000).do()

# Save to JSON
vectors = []
for obj in result['data']['Get']['MyClass']:
    vectors.append({
        "id": obj['_additional']['id'],
        "vector": obj['_additional']['vector'],
        "properties": obj
    })

with open('weaviate_export.json', 'w') as f:
    json.dump({"vectors": vectors}, f)
```

### 2. Code Migration

#### Weaviate Code:

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Add objects
client.batch.add_data_object(
    data_object={"title": "Document 1", "content": "..."},
    class_name="Document",
    vector=[0.1, 0.2, 0.3, ...]
)

# Search
result = client.query.get(
    "Document",
    ["title", "content"]
).with_near_vector({
    "vector": [0.15, 0.25, 0.35, ...]
}).with_limit(10).with_where({
    "path": ["category"],
    "operator": "Equal",
    "valueString": "tech"
}).do()
```

#### VecStore Equivalent:

```rust
use vecstore::{VecStore, Query};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let mut store = VecStore::open("documents.db")?;

    // Add objects
    store.upsert("doc1", vec![0.1, 0.2, 0.3, /* ... */],
                 json!({
                     "title": "Document 1",
                     "content": "...",
                     "category": "tech"
                 }))?;

    // Search
    let query = Query::new(vec![0.15, 0.25, 0.35, /* ... */])
        .with_limit(10)
        .with_filter("category = 'tech'");

    let results = store.query(query)?;

    for result in results {
        if let Some(title) = result.metadata.fields.get("title") {
            println!("Title: {}", title);
        }
    }

    Ok(())
}
```

---

## Feature Comparison

### Core Features

| Feature | Pinecone | Qdrant | Weaviate | VecStore |
|---------|----------|---------|----------|----------|
| HNSW Indexing | ✅ | ✅ | ✅ | ✅ |
| Metadata Filtering | ✅ | ✅ | ✅ | ✅ |
| Hybrid Search | ✅ | ❌ | ✅ | ✅ |
| Product Quantization | ✅ | ✅ | ❌ | ✅ |
| Persistence | ☁️  Cloud | ✅ | ✅ | ✅ |
| Embedded Mode | ❌ | ❌ | ❌ | ✅ |
| Multi-tenancy | ✅ Namespaces | ✅ Collections | ✅ Tenants | ✅ Partitions |

### Advanced Features

| Feature | VecStore Implementation |
|---------|------------------------|
| Clustering | `KMeansClustering`, `DBSCAN`, `Hierarchical` |
| Anomaly Detection | `IsolationForest`, `LOF`, `ZScoreDetector` |
| Dimensionality Reduction | `PCA` with 8x compression |
| Recommendations | Content-based, Collaborative, Hybrid |
| Versioning | Full version history with rollback |
| Query Optimization | Cost estimation and hints |
| Bulk Migration | From Pinecone, Qdrant, ChromaDB |

---

## Performance Optimization

### After Migration

1. **Add Metadata Indexes** for frequently filtered fields:

```rust
use vecstore::MetadataIndexManager;

let mut index_manager = MetadataIndexManager::new();

// Create BTree index for range queries
index_manager.create_index("price", IndexType::BTree);

// Create Hash index for equality queries
index_manager.create_index("category", IndexType::Hash);

// Create Inverted index for text search
index_manager.create_index("tags", IndexType::Inverted);
```

2. **Use Query Optimizer** to analyze performance:

```rust
use vecstore::QueryOptimizer;

let optimizer = QueryOptimizer::new(&store);
let analysis = optimizer.analyze_query(&query)?;

println!("Estimated cost: {:.2}ms", analysis.estimated_cost);

for hint in analysis.hints {
    println!("Hint: {}", hint.suggestion);
}
```

3. **Enable Product Quantization** for compression:

```rust
use vecstore::{PQConfig, PQVectorStore};

let pq_config = PQConfig {
    num_subvectors: 16,
    num_centroids: 256,
    training_iterations: 20,
};

// 8-32x memory reduction
let pq_store = PQVectorStore::new(pq_config)?;
```

4. **Use Partitioning** for multi-tenant applications:

```rust
use vecstore::PartitionedStore;

let mut store = PartitionedStore::new("data", PartitionConfig::default())?;

// Each tenant gets isolated storage
store.insert("tenant_a", "doc1", vector, metadata)?;
store.insert("tenant_b", "doc2", vector, metadata)?;

// Fast tenant-specific queries
let results = store.query_partition("tenant_a", query)?;
```

---

## Troubleshooting

### Common Issues

#### 1. **Performance slower than expected**

**Solution:**
- Add metadata indexes for filtered fields
- Use Product Quantization for large datasets (>100K vectors)
- Enable partitioning for multi-tenant scenarios
- Run query optimizer to get specific hints

#### 2. **Out of memory with large datasets**

**Solution:**
- Enable Product Quantization (8-32x memory reduction)
- Use dimensionality reduction (PCA) to reduce vector dimensions
- Partition data into smaller chunks
- Use mmap-based storage for very large datasets

#### 3. **Filter queries are slow**

**Solution:**
- Create appropriate indexes:
  - BTree for range queries (age > 18, price < 100)
  - Hash for equality queries (category = 'tech')
  - Inverted for text search (tags contain 'machine learning')

```rust
let mut index_mgr = MetadataIndexManager::new();
index_mgr.create_index("category", IndexType::Hash);
```

#### 4. **Migration takes too long**

**Solution:**
- Increase batch size (default: 100, try 1000-10000)
- Disable validation for trusted data
- Use resume capability for interrupted migrations

```rust
let config = MigrationConfig {
    batch_size: 10000,  // Larger batches
    validate: false,     // Skip validation
    resume_from: Some(50000),  // Resume from vector 50000
};
```

#### 5. **Need to migrate incremental updates**

**Solution:**
- Use versioning to track changes
- Export only new/modified vectors
- Use `upsert` for incremental updates

```rust
use vecstore::VersionedStore;

let mut store = VersionedStore::new("vectors.db")?;

// Tracks all changes automatically
store.update("doc1", new_vector, metadata,
             Some("Updated from Pinecone".to_string()))?;
```

---

## Getting Help

- **Documentation**: https://docs.rs/vecstore
- **Examples**: https://github.com/yourusername/vecstore/tree/main/examples
- **Issues**: https://github.com/yourusername/vecstore/issues

## Next Steps

After migration:

1. ✅ Verify all data migrated correctly
2. ✅ Update application code to use VecStore API
3. ✅ Add metadata indexes for filtered fields
4. ✅ Run query optimizer to identify bottlenecks
5. ✅ Consider enabling Product Quantization for large datasets
6. ✅ Set up regular backups (simple file copy!)
7. ✅ Monitor performance with built-in metrics

Welcome to VecStore! 🎉
