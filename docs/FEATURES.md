# VecStore Complete Feature Reference

> ⚠️ This document is being updated to match the current implementation. For the canonical overview of what ships today, see [ARCHITECTURE.md](./ARCHITECTURE.md) and [STATUS.md](./STATUS.md). Sections marked with notes are accurate; the rest describe planned or partially implemented functionality and will be reviewed as the codebase evolves. VecStore 0.0.1 is an alpha release—APIs and file formats may change.

### TL;DR

- ✅ Shipping in 0.0.1: embedded store API, snapshots, metadata filters, batch ingestion, Python bindings.
- ⚠️ Optional / experimental: server mode, hybrid search helpers, query planner.
- 🚧 Prototypes only: distributed mode, realtime indexing, GPU backends, packaging manifests.

**Version:** 0.0.1 | **Tests:** 349 passing

---

## Table of Contents

1. [Core Vector Search](#core-vector-search)
2. [Query Planning](#query-planning)
3. [Advanced Query Features](#advanced-query-features)
4. [Hybrid Search](#hybrid-search)
5. [Metadata Filtering](#metadata-filtering)
6. [Performance Tuning](#performance-tuning)
7. [Production Features](#production-features)
8. [Server Mode](#server-mode)
9. [Multi-Tenancy](#multi-tenancy)
10. [RAG Stack](#rag-stack)
11. [Python Bindings](#python-bindings)
12. [WASM Support](#wasm-support)

---

## Core Vector Search

### Basic Operations

#### Create/Open Database

```rust
use vecstore::VecStore;

// Open or create a database
let mut store = VecStore::open("vectors.db")?;

// With custom configuration
let store = VecStore::builder("vectors.db")
    .distance(Distance::Euclidean)
    .hnsw_m(32)  // More connections = better recall
    .hnsw_ef_construction(400)  // Higher = better quality
    .build()?;
```

**Configuration Options:**
- `distance` - Distance metric (Cosine, Euclidean, or DotProduct in the default backend)
- `hnsw_m` - Number of connections per layer (default: 16)
- `hnsw_ef_construction` - Construction quality (default: 200)

---

#### Insert Vectors

```rust
use vecstore::Metadata;
use std::collections::HashMap;

// Create metadata
let mut meta = Metadata { fields: HashMap::new() };
meta.fields.insert("title".into(), serde_json::json!("Document Title"));
meta.fields.insert("category".into(), serde_json::json!("tech"));
meta.fields.insert("score".into(), serde_json::json!(0.95));

// Insert vector
store.upsert("doc1".into(), vec![0.1, 0.2, 0.3], meta)?;

// Batch insert to avoid per-call overhead
let batch = vec![
    ("doc1".into(), vec![0.1, 0.2, 0.3], meta1),
    ("doc2".into(), vec![0.2, 0.3, 0.4], meta2),
    // ... more vectors
];
store.batch_upsert(batch)?;
```

Only cosine, Euclidean, and dot product are backed by dedicated HNSW structures right now. Picking another distance variant logs a warning and reuses cosine until additional kernels are implemented.

---

#### Query Vectors

```rust
use vecstore::Query;

// Simple query
let results = store.query(Query {
    vector: vec![0.15, 0.25, 0.35],
    k: 10,
    filter: None,
})?;

// Using builder API
let results = store.query(
    Query::new(vec![0.15, 0.25, 0.35])
        .with_limit(10)
        .with_filter("category = 'tech'")
)?;

// Process results
for result in results {
    println!("{}: {:.4}", result.id, result.score);
    println!("  Title: {}", result.metadata.fields.get("title"));
}
```

---

### Distance Metrics

VecStore’s default backend has native support for three metrics. Selecting the others falls back to cosine and logs a warning.

```rust
use vecstore::Distance;

// Cosine similarity (default) - angle between vectors
// Best for: Text embeddings, normalized vectors
let store = VecStore::builder("db").distance(Distance::Cosine).build()?;

// Euclidean distance (L2) - straight-line distance
// Best for: Spatial data, unnormalized vectors
let store = VecStore::builder("db").distance(Distance::Euclidean).build()?;

// Dot product - alignment and magnitude
// Best for: Recommendation systems where magnitude matters
let store = VecStore::builder("db").distance(Distance::DotProduct).build()?;

// Hamming, Jaccard, Manhattan etc. currently fall back to cosine.
```

---

### Product Quantization

Compress vectors by 8-32x with minimal accuracy loss:

```rust
use vecstore::{ProductQuantizer, PQConfig};

// Configure quantization
let config = PQConfig {
    num_subvectors: 16,      // Divide vector into 16 parts
    num_centroids: 256,      // 256 centroids per subvector
    training_iterations: 20, // Training iterations
};

// Create quantizer
let mut pq = ProductQuantizer::new(384, config)?;  // 384-dim vectors

// Train on representative sample
let training_data: Vec<Vec<f32>> = load_training_data();
pq.train(&training_data)?;

// Encode vectors (384 floats → 16 bytes = 24x compression)
let compressed = pq.encode(&vector)?;

// Decode for approximate reconstruction
let reconstructed = pq.decode(&compressed)?;

// Use with VecStore
let pq_store = PQVectorStore::new(store, pq)?;
```

**Compression Ratios:**
- 128-dim: 128 floats (512 bytes) → 8-16 bytes = 32-64x
- 384-dim: 384 floats (1536 bytes) → 16-32 bytes = 48-96x
- 768-dim: 768 floats (3072 bytes) → 32-64 bytes = 48-96x

---

## Query Planning

VecStore includes helper methods that estimate query cost and suggest potential optimizations before you run them.

### EXPLAIN Queries

Understand how queries will execute and optimize them:

```rust
use vecstore::{VecStore, Query};

let store = VecStore::open("vectors.db")?;

// Explain a query
let query = Query::new(vec![0.5, 0.5, 0.5])
    .with_limit(10)
    .with_filter("category = 'tech' AND score > 0.9");

let plan = store.explain_query(query)?;

// View query plan
println!("Query type: {}", plan.query_type);
println!("Estimated cost: {:.2}", plan.estimated_cost);
println!("Estimated duration: {:.2}ms", plan.estimated_duration_ms);
println!("Is optimal: {}", plan.is_optimal);

// Execution steps
for step in plan.steps {
    println!("Step {}: {} (cost: {:.2})",
        step.step, step.description, step.cost);
    println!("  Input: {} candidates → Output: {} results",
        step.input_size, step.output_size);
}

// Optimization recommendations
if !plan.is_optimal {
    println!("\nOptimizations:");
    for rec in plan.recommendations {
        println!("Hint: {}", rec);
    }
}
```

_The planner uses simple heuristics; treat these numbers as guidance rather than hard guarantees._

**Example Output:**
```
Query type: Filtered Vector Search
Estimated cost: 0.35
Estimated duration: 3.25ms
Is optimal: false

Step 1: HNSW graph traversal (ef_search=50, fetch=100) (cost: 0.25)
  Input: 10000 candidates → Output: 100 results

Step 2: Apply filter (selectivity: 10.0%) (cost: 0.09)
  Input: 100 candidates → Output: 10 results

Step 3: Select top-10 results (cost: 0.05)
  Input: 10 candidates → Output: 10 results

Optimizations:
Hint: Fetching 10x more candidates than needed. Consider using filtered HNSW traversal.
Hint: Run EXPLAIN again after optimizations to see improvement.
```

**Use Cases:**
- Debug slow queries
- Optimize filter selectivity
- Understand performance bottlenecks
- Capacity planning
- Performance tuning

---

## Advanced Query Features

### Multi-Stage Prefetch Queries

Chain multiple query stages for complex retrieval (like Qdrant's prefetch API):

```rust
use vecstore::{PrefetchQuery, QueryStage};

let query = PrefetchQuery {
    stages: vec![
        // Stage 1: Broad hybrid search (100 candidates)
        QueryStage::HybridSearch {
            vector: vec![0.1, 0.2, 0.3],
            keywords: "machine learning tutorial".into(),
            k: 100,
            alpha: 0.7,  // 70% vector, 30% keywords
            filter: None,
        },

        // Stage 2: Rerank with cross-encoder (top 20)
        QueryStage::Rerank {
            k: 20,
            model: Some("cross-encoder/ms-marco-MiniLM-L-6-v2".into()),
        },

        // Stage 3: Apply MMR for diversity (final 10)
        QueryStage::MMR {
            k: 10,
            lambda: 0.7,  // 70% relevance, 30% diversity
        },

        // Stage 4: Additional filtering
        QueryStage::Filter {
            expr: FilterExpr::Cmp {
                field: "quality".into(),
                op: FilterOp::Gt,
                value: serde_json::json!(0.8),
            },
        },
    ],
};

let results = store.prefetch_query(query)?;
```

**Available Stages:**
- `VectorSearch` - Vector similarity search
- `HybridSearch` - Vector + keyword search
- `Rerank` - Rerank with cross-encoder
- `MMR` - Maximal Marginal Relevance (diversity)
- `Filter` - Additional metadata filtering

**Advanced RAG Pattern:**
```rust
// 1. Broad search → 2. Rerank → 3. Diversify → 4. Final filter
let rag_query = PrefetchQuery {
    stages: vec![
        QueryStage::HybridSearch {
            vector: query_embedding,
            keywords: user_query,
            k: 100,  // Cast wide net
            alpha: 0.8,
            filter: Some(parse_filter("source = 'docs'")?),
        },
        QueryStage::Rerank {
            k: 20,  // Rerank top 20
            model: Some("cross-encoder".into()),
        },
        QueryStage::MMR {
            k: 5,  // Select 5 diverse results
            lambda: 0.7,
        },
    ],
};
```

---

### MMR Diversity

Avoid returning too many similar results:

```rust
// MMR as a stage in prefetch query
QueryStage::MMR {
    k: 5,        // Number of diverse results
    lambda: 0.7, // Balance: 0.0 = all diversity, 1.0 = all relevance
}
```

**Lambda parameter:**
- `0.0` - Maximum diversity (results are very different)
- `0.5` - Balance relevance and diversity
- `0.7` - Mostly relevant, some diversity (recommended)
- `1.0` - Maximum relevance (no diversity, same as regular search)

**Use Case:**
```rust
// RAG: Avoid retrieving multiple similar passages
let diverse_context = store.prefetch_query(PrefetchQuery {
    stages: vec![
        QueryStage::VectorSearch {
            vector: question_embedding,
            k: 20,
            filter: None,
        },
        QueryStage::MMR {
            k: 5,        // Get 5 diverse passages
            lambda: 0.6, // Favor diversity slightly
        },
    ],
})?;
```

---

## Hybrid Search

Combine vector similarity with keyword (BM25) search:

### Basic Hybrid Search

```rust
use vecstore::HybridQuery;

// Index text for keyword search
store.index_text("doc1", "machine learning tutorial")?;
store.index_text("doc2", "deep learning fundamentals")?;

// Hybrid search: vector + keywords
let results = store.hybrid_query(HybridQuery {
    vector: vec![0.1, 0.2, 0.3],
    keywords: "machine learning".into(),
    k: 10,
    alpha: 0.7,  // 70% vector, 30% keywords
    filter: None,
})?;
```

**Alpha parameter:**
- `0.0` - Pure keyword search (BM25 only)
- `0.3` - Mostly keywords, some vector
- `0.5` - Equal weight
- `0.7` - Mostly vector, some keywords (recommended)
- `1.0` - Pure vector search

---

### Tokenizers

VecStore supports 4 pluggable tokenizers:

```rust
use vecstore::tokenizer::{Tokenizer, SimpleTokenizer, LanguageTokenizer,
                           WhitespaceTokenizer, NGramTokenizer};

// 1. Simple tokenizer (lowercase + split on non-alphanumeric)
let tokenizer = SimpleTokenizer::new();
let tokens = tokenizer.tokenize("Hello, World!");
// → ["hello", "world"]

// 2. Language tokenizer (with stopwords)
let tokenizer = LanguageTokenizer::new();
let tokens = tokenizer.tokenize("The quick brown fox");
// → ["quick", "brown", "fox"]  (removes "the")

// 3. Whitespace tokenizer (preserves case, splits on whitespace)
let tokenizer = WhitespaceTokenizer;
let tokens = tokenizer.tokenize("Hello World");
// → ["Hello", "World"]

// 4. N-gram tokenizer
let tokenizer = NGramTokenizer::new(2);  // Bigrams
let tokens = tokenizer.tokenize("hello");
// → ["he", "el", "ll", "lo"]
```

**Stopwords (60+ English):**
```rust
// LanguageTokenizer removes common words
let tokenizer = LanguageTokenizer::new();
tokenizer.tokenize("the quick brown fox jumps over the lazy dog");
// → ["quick", "brown", "fox", "jumps", "lazy", "dog"]
// Removed: "the", "over", "the"
```

---

### Phrase Matching

Position-aware phrase matching with 2x boost:

```rust
// Index with positions
store.index_text("doc1", "machine learning tutorial")?;
store.index_text("doc2", "learning about machines")?;

// Search for exact phrase
let results = store.hybrid_query(HybridQuery {
    vector: vec![0.1, 0.2, 0.3],
    keywords: "\"machine learning\"".into(),  // Quoted = phrase
    k: 10,
    alpha: 0.5,
    filter: None,
})?;

// doc1 scores 2x higher (exact phrase "machine learning")
// doc2 scores lower (has words but not as phrase)
```

**Phrase Boost:**
- Exact phrase match: 2x score
- Individual words: 1x score

---

### Fusion Strategies

8 strategies for combining vector + keyword scores:

```rust
use vecstore::vectors::{FusionStrategy, HybridSearchConfig};

let config = HybridSearchConfig {
    alpha: 0.7,
    fusion: FusionStrategy::WeightedSum,  // Default
};

// Available strategies:
// 1. WeightedSum - alpha * vector + (1-alpha) * keyword
// 2. RRF - Reciprocal Rank Fusion
// 3. Max - max(vector, keyword)
// 4. Min - min(vector, keyword)
// 5. Product - vector * keyword
// 6. Harmonic - harmonic mean
// 7. Geometric - geometric mean
// 8. Distribution - distribution-based combination
```

---

## Metadata Filtering

SQL-like filtering with 9 operators:

### Filter Syntax

```rust
// Equality
store.query(Query::new(vec).with_filter("category = 'tech'"))?;

// Comparison
store.query(Query::new(vec).with_filter("score > 0.8"))?;
store.query(Query::new(vec).with_filter("price >= 100"))?;
store.query(Query::new(vec).with_filter("age < 65"))?;

// Inequality
store.query(Query::new(vec).with_filter("status != 'archived'"))?;

// Contains (substring)
store.query(Query::new(vec).with_filter("tags CONTAINS 'urgent'"))?;

// In (array membership)
store.query(Query::new(vec).with_filter("category IN ['tech', 'science']"))?;

// Not In
store.query(Query::new(vec).with_filter("status NOT IN ['deleted', 'spam']"))?;

// Boolean logic
store.query(Query::new(vec).with_filter(
    "category = 'tech' AND score > 0.8 AND price < 500"
))?;

store.query(Query::new(vec).with_filter(
    "(category = 'tech' OR category = 'science') AND score > 0.9"
))?;

// Negation
store.query(Query::new(vec).with_filter(
    "NOT (status = 'deleted' OR status = 'spam')"
))?;
```

---

### Programmatic Filters

```rust
use vecstore::{FilterExpr, FilterOp};

// Build filters programmatically
let filter = FilterExpr::And(vec![
    FilterExpr::Cmp {
        field: "score".into(),
        op: FilterOp::Gt,
        value: serde_json::json!(0.8),
    },
    FilterExpr::Cmp {
        field: "category".into(),
        op: FilterOp::In,
        value: serde_json::json!(["tech", "science"]),
    },
    FilterExpr::Not(Box::new(FilterExpr::Cmp {
        field: "status".into(),
        op: FilterOp::Eq,
        value: serde_json::json!("deleted"),
    })),
]);

let results = store.query(Query {
    vector: vec![0.1, 0.2, 0.3],
    k: 10,
    filter: Some(filter),
})?;
```

---

## Performance Tuning

### HNSW Parameters

Control speed vs accuracy tradeoff:

```rust
use vecstore::HNSWSearchParams;

let query = Query::new(vec![0.5, 0.5, 0.5]).with_limit(10);

// Fast search (ef_search=20)
// Use when: Speed matters more than perfect recall
let results = store.query_with_params(
    query.clone(),
    HNSWSearchParams::fast(),
)?;

// Balanced search (ef_search=50) - DEFAULT
// Use when: Good balance of speed and accuracy
let results = store.query_with_params(
    query.clone(),
    HNSWSearchParams::balanced(),
)?;

// High recall search (ef_search=100)
// Use when: Accuracy matters, can tolerate slower queries
let results = store.query_with_params(
    query.clone(),
    HNSWSearchParams::high_recall(),
)?;

// Maximum recall search (ef_search=200)
// Use when: Need best possible accuracy
let results = store.query_with_params(
    query,
    HNSWSearchParams::max_recall(),
)?;

// Custom ef_search
let custom = HNSWSearchParams { ef_search: 75 };
let results = store.query_with_params(query, custom)?;
```

**Performance Impact:**
| ef_search | Speed | Recall | Use Case |
|-----------|-------|--------|----------|
| 20 | Very fast | ≈85% (heuristic) | Real-time search |
| 50 | Fast | ≈92% (heuristic) | Most applications |
| 100 | Moderate | ≈96% (heuristic) | High accuracy needed |
| 200 | Slow | ≈98% (heuristic) | Maximum accuracy |

The recall percentages are ballpark numbers drawn from local testing; confirm on your own dataset before relying on them in production charts.

---

### SIMD Acceleration

Automatic SIMD acceleration for distance calculations:

```rust
// Automatically uses SIMD when available:
// - AVX2 on x86_64
// - NEON on ARM64
// 4-8x faster than scalar code

use vecstore::simd::*;

// These are used internally, but you can call them directly:
let sim = cosine_similarity_simd(&vec1, &vec2);
let dist = euclidean_distance_simd(&vec1, &vec2);
let dot = dot_product_simd(&vec1, &vec2);
```

**Performance:**
- Cosine: SIMD significantly reduces instruction count (exact gain depends on CPU)
- Euclidean: Benefits from vectorized subtraction + multiply loops
- Dot product: Leverages fused multiply-add where available

---

## Production Features

### Write-Ahead Logging (WAL)

`wal.rs` implements an append-only write-ahead log with checkpointing and thorough tests. The default `VecStore` does not invoke it yet, so persisting via `VecStore::save`/snapshots remains the recommended path today. Integrating the WAL into the write path is on the roadmap; consult the module docs if you plan to wire it up manually.

---

### Soft Deletes & TTL

Defer cleanup and enable undo:

```rust
// Soft delete (can be undone)
store.soft_delete("doc1")?;

// Check if deleted
let is_deleted = store.is_deleted("doc1")?;  // true

// Restore
store.restore("doc1")?;

// Hard delete (permanent)
store.remove("doc1")?;

// TTL (time-to-live)
use chrono::{Utc, Duration};

let expires_at = Utc::now() + Duration::hours(24);
store.upsert_with_ttl("temp_doc", vec, meta, expires_at.timestamp())?;

// Auto-expires after 24 hours

// Manual cleanup of expired
store.cleanup_expired()?;
```

---

### Snapshots & Backups

```rust
// Create snapshot
store.create_snapshot("backup-2025-01-15")?;

// List snapshots
let snapshots = store.list_snapshots()?;
for (name, created_at, count) in snapshots {
    println!("{}: {} vectors ({})", name, count, created_at);
}

// Restore from snapshot
store.restore_snapshot("backup-2025-01-15")?;

// Delete snapshot
store.delete_snapshot("old-backup")?;
```

**Automated Backups:**
```bash
#!/bin/bash
# Daily backup cron job
DATE=$(date +%Y%m%d)
/path/to/vecstore-cli snapshot create "daily-$DATE"

# Keep last 7 days
find /data/snapshots -mtime +7 -delete
```

---

### Batch Operations

Batching reduces per-call overhead compared to issuing individual writes:

```rust
// Batch upsert
let batch = vec![
    ("doc1".into(), vec![0.1, 0.2, 0.3], meta1),
    ("doc2".into(), vec![0.2, 0.3, 0.4], meta2),
    // ... thousands more
];

let result = store.batch_upsert(batch)?;
println!("Inserted {} vectors", result.succeeded);

// Mixed batch operations
use vecstore::BatchOperation;

let operations = vec![
    BatchOperation::Upsert { id: "doc1".into(), vector: vec, metadata: meta },
    BatchOperation::Delete { id: "doc2".into() },
    BatchOperation::SoftDelete { id: "doc3".into() },
    BatchOperation::UpdateMetadata { id: "doc4".into(), metadata: new_meta },
];

let result = store.execute_batch(operations)?;
println!("Success: {}, Failed: {}", result.succeeded, result.failed);

// Check errors
for error in result.errors {
    eprintln!("Operation {} failed: {}", error.index, error.error);
}
```

---

### Metrics & Monitoring

Prometheus metrics built-in:

```rust
// Metrics are automatically collected
// Expose at /metrics endpoint (server mode)

// Available metrics:
// - vecstore_query_duration_seconds (histogram)
// - vecstore_queries_total (counter)
// - vecstore_vectors_total (gauge)
// - vecstore_index_size_bytes (gauge)
// - vecstore_cache_hits_total (counter)
// - vecstore_errors_total (counter)
```

**Grafana Dashboard:**
See `observability/grafana-dashboard.json` for pre-built dashboard.

---

## Server Mode

Run VecStore as a standalone server:

### Start Server

```bash
# Build with server feature
cargo build --release --features server

# Run server
./target/release/vecstore-server \
    --grpc-port 50051 \
    --http-port 8080 \
    --db-path /data/vectors.db
```

---

### gRPC API

```bash
# Install grpcurl
brew install grpcurl  # macOS
# or download from https://github.com/fullstorydev/grpcurl

# List services
grpcurl -plaintext localhost:50051 list

# Upsert vector
grpcurl -plaintext \
    -d '{
        "id": "doc1",
        "vector": [0.1, 0.2, 0.3],
        "metadata": {"title": "Test"}
    }' \
    localhost:50051 \
    vecstore.VecStoreService/Upsert

# Query
grpcurl -plaintext \
    -d '{
        "vector": [0.1, 0.2, 0.3],
        "k": 10
    }' \
    localhost:50051 \
    vecstore.VecStoreService/Query
```

---

### HTTP/REST API

```bash
# Health check
curl http://localhost:8080/health

# Upsert
curl -X POST http://localhost:8080/v1/vectors \
    -H "Content-Type: application/json" \
    -d '{
        "id": "doc1",
        "vector": [0.1, 0.2, 0.3],
        "metadata": {"title": "Test"}
    }'

# Query
curl -X POST http://localhost:8080/v1/query \
    -H "Content-Type: application/json" \
    -d '{
        "vector": [0.1, 0.2, 0.3],
        "k": 10
    }'

# Hybrid query
curl -X POST http://localhost:8080/v1/query/hybrid \
    -H "Content-Type: application/json" \
    -d '{
        "vector": [0.1, 0.2, 0.3],
        "keywords": "machine learning",
        "k": 10,
        "alpha": 0.7
    }'

# Metrics (Prometheus format)
curl http://localhost:8080/metrics
```

---

## Multi-Tenancy

Isolated namespaces with quotas:

```rust
use vecstore::NamespaceManager;

// Create namespace manager
let mut manager = NamespaceManager::new("/data/namespaces")?;

// Create namespace with quotas
use vecstore::NamespaceQuotas;

let quotas = NamespaceQuotas {
    max_vectors: Some(1_000_000),
    max_storage_bytes: Some(10_000_000_000),  // 10GB
    max_queries_per_second: Some(100),
    max_namespaces: None,
};

manager.create_namespace("customer_123", quotas)?;

// Get namespace
let ns = manager.get_namespace("customer_123")?;

// Use namespace
let mut store = ns.store_mut();
store.upsert("doc1", vec, meta)?;

// Check resource usage
let usage = ns.resource_usage();
println!("Vectors: {}/{}", usage.vector_count, quotas.max_vectors.unwrap());
println!("Storage: {}/{} bytes", usage.storage_bytes, quotas.max_storage_bytes.unwrap());

// List namespaces
let namespaces = manager.list_namespaces()?;
for ns_id in namespaces {
    println!("Namespace: {}", ns_id);
}

// Delete namespace
manager.delete_namespace("customer_123")?;
```

**7 Quota Types:**
1. `max_vectors` - Maximum number of vectors
2. `max_storage_bytes` - Maximum storage size
3. `max_queries_per_second` - Rate limiting
4. `max_namespaces` - Limit sub-namespaces
5. `max_dimensions` - Maximum vector dimensions
6. `max_metadata_size` - Metadata size limit
7. `max_snapshots` - Snapshot count limit

---

## RAG Stack

### Text Splitters

```rust
use vecstore::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};

// Recursive character splitter
let splitter = RecursiveCharacterTextSplitter::new(
    1000,  // chunk_size
    200,   // overlap
);

let chunks = splitter.split_text("Long document...")?;

for chunk in chunks {
    println!("Chunk: {} chars", chunk.text.len());
    println!("  Start: {}, End: {}", chunk.start_index, chunk.end_index);
}

// Other splitters:
// - TokenTextSplitter (by tokens)
// - SemanticTextSplitter (by semantic boundaries)
// - MarkdownTextSplitter (markdown-aware)
```

---

### Document Loaders

```rust
use vecstore::document_loaders::*;

// PDF
let loader = PDFLoader::new();
let docs = loader.load("document.pdf")?;

// Markdown
let loader = MarkdownLoader::new();
let docs = loader.load("README.md")?;

// HTML
let loader = HTMLLoader::new();
let docs = loader.load("page.html")?;

// JSON
let loader = JSONLoader::new();
let docs = loader.load("data.json")?;

// CSV
let loader = CSVLoader::new();
let docs = loader.load("data.csv")?;

// Supported formats:
// PDF, Markdown, HTML, JSON, CSV, Parquet, Text
```

---

### Reranking

```rust
use vecstore::reranking::{Reranker, MMRReranker};

// MMR reranking (diversity)
let reranker = MMRReranker::new(0.7);  // lambda
let reranked = reranker.rerank(&candidates, &query_vec, 10)?;

// Custom scoring reranker
// (Implement your own scoring function)
```

---

### RAG Utilities

```rust
use vecstore::rag_utils::*;

// HyDE (Hypothetical Document Embeddings)
let hypothetical_doc = generate_hyde_document(&query)?;
let hyde_embedding = embed(&hypothetical_doc)?;
let results = store.query(Query::new(hyde_embedding).with_limit(5))?;

// Multi-query fusion
let queries = expand_query(&user_question)?;
let all_results = multi_query_fusion(&store, &queries)?;

// Conversation memory
let memory = ConversationMemory::new(max_messages=10);
memory.add_message("user", "What is Rust?")?;
memory.add_message("assistant", "Rust is...")?;
let context = memory.get_context()?;
```

---

## Python Bindings

Native PyO3 bindings (not gRPC):

```python
import vecstore

# Create store
store = vecstore.VecStore("vectors.db")

# Insert
store.upsert("doc1", [0.1, 0.2, 0.3], {"title": "Test"})

# Batch insert
vectors = [
    ("doc1", [0.1, 0.2, 0.3], {"title": "Doc 1"}),
    ("doc2", [0.2, 0.3, 0.4], {"title": "Doc 2"}),
]
store.batch_upsert(vectors)

# Query
results = store.query([0.15, 0.25, 0.35], k=10)
for result in results:
    print(f"{result.id}: {result.score:.4f}")

# Hybrid query
results = store.hybrid_query(
    vector=[0.1, 0.2, 0.3],
    keywords="machine learning",
    k=10,
    alpha=0.7
)

# With filter
results = store.query(
    vector=[0.1, 0.2, 0.3],
    k=10,
    filter="category = 'tech' AND score > 0.8"
)

# Prefetch query (multi-stage)
results = store.prefetch_query({
    "stages": [
        {
            "type": "hybrid_search",
            "vector": [0.1, 0.2, 0.3],
            "keywords": "ML tutorial",
            "k": 100,
            "alpha": 0.7
        },
        {
            "type": "mmr",
            "k": 10,
            "lambda": 0.7
        }
    ]
})

# Query planning
plan = store.explain_query(
    vector=[0.1, 0.2, 0.3],
    k=10,
    filter="score > 0.8"
)
print(f"Estimated cost: {plan.estimated_cost}")
for rec in plan.recommendations:
    print(f"Hint: {rec}")
```

---

## WASM Support

Use VecStore in browsers:

```javascript
import init, { VecStore } from 'vecstore-wasm';

await init();

// Create store
const store = new VecStore('vectors.db');

// Insert
await store.upsert('doc1', [0.1, 0.2, 0.3], {title: 'Test'});

// Query
const results = await store.query([0.15, 0.25, 0.35], {k: 10});

for (const result of results) {
    console.log(`${result.id}: ${result.score}`);
}
```

---

## Performance Benchmarks

Official latency and throughput numbers are being rebuilt with a reproducible benchmarking harness. Until those results are published, please treat any historic figures you encounter as anecdotal. To measure performance locally, see `benches/` for Criterion-based suites and run them on your hardware.

---

## API Reference

For complete API documentation, run:

```bash
cargo doc --open
```

Or visit: https://docs.rs/vecstore

---

## Examples

See the `examples/` directory for complete working examples:

- `examples/basic.rs` - Basic vector search
- `examples/hybrid_search.rs` - Hybrid search
- `examples/filters.rs` - Advanced filtering
- `examples/prefetch.rs` - Multi-stage queries
- `examples/server.rs` - Server mode
- `examples/python/` - Python examples
- `examples/rag/` - RAG application

---

## Next Steps

- **[QUICKSTART.md](../QUICKSTART.md)** - Get running in 5 minutes
- **[DEPLOYMENT.md](../DEPLOYMENT.md)** - Production deployment
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribute to VecStore

---

**Core features shipped** | **349 tests passing** | **Actively maintained**
