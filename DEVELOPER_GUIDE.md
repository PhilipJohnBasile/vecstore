# VecStore Developer Documentation

## 1. Project Overview

### 1.1. Inferred Purpose & Goals

**VecStore** is a production-ready, embedded vector database written in Rust, designed to be the "SQLite of vector search." Its core purpose is to provide **fast, in-process vector similarity search** with HNSW (Hierarchical Navigable Small World) indexing, while offering the simplicity of an embedded database and the scalability of a network-accessible server.

**Primary Goals:**
- **Embeddable by default** - Zero-dependency vector search for Rust applications
- **Production-grade performance** - Sub-millisecond query latency with HNSW indexing
- **Polyglot accessibility** - gRPC and HTTP/REST APIs for any programming language
- **Rich feature set** - Metadata filtering, hybrid search (vector + keyword), soft deletes, snapshots
- **Observable & monitored** - Prometheus metrics, Grafana dashboards, production-ready observability
- **Multiple deployment modes** - Embedded library, standalone server, Docker containers, Kubernetes

**Problem Solved:**
VecStore addresses the complexity and overhead of deploying vector databases for similarity search use cases. While alternatives like Pinecone, Qdrant, and Weaviate require infrastructure setup and network latency, VecStore can run entirely in-process with persistence, similar to how SQLite provides embedded SQL databases.

**Key Differentiators:**
- **Truly embedded** - No separate server process required for basic use
- **Rust-native** - Memory safety, performance, and zero-cost abstractions
- **Optional server mode** - Scale from embedded to networked without code changes
- **Product Quantization** - 8-32x memory compression for large datasets
- **Hybrid search** - Combines vector similarity with BM25 keyword search

### 1.2. Identified Technology Stack

**Core Language & Runtime:**
- **Rust 2021 Edition** - Primary implementation language
- **Tokio** - Async runtime (optional, for server mode)

**Vector Search & Indexing:**
- **hnsw_rs 0.3** - HNSW graph-based approximate nearest neighbor search
- **anndists** - Distance metrics library
- **rayon 1.x** - Data parallelism for batch operations

**Serialization & Persistence:**
- **serde 1.x** - Serialization framework
- **bincode 1.x** - Binary serialization format
- **serde_json 1.x** - JSON serialization
- **memmap2 0.9** - Memory-mapped file I/O

**Server & Networking (Optional):**
- **tonic 0.12** - gRPC server framework
- **prost 0.13** - Protocol Buffers implementation
- **axum 0.7** - HTTP/REST web framework
- **tower 0.4 / tower-http 0.5** - Service middleware
- **tokio-stream 0.1** - Async stream utilities

**Observability & Monitoring:**
- **prometheus 0.13** - Metrics collection
- **tracing 0.1** - Structured logging and diagnostics
- **tracing-subscriber 0.3** - Log formatting

**Optional Features:**
- **pyo3 0.22** - Python bindings
- **wasm-bindgen 0.2** - WebAssembly bindings
- **ort 1.16** - ONNX Runtime for embeddings
- **tokenizers 0.15** - Text tokenization
- **parquet 53 / arrow 53** - Parquet export support

**Development & Testing:**
- **criterion 0.5** - Benchmarking framework
- **tempfile 3.x** - Temporary file handling for tests
- **approx 0.5** - Floating-point assertions

**Build Tools:**
- **tonic-build 0.12** - Protocol Buffers code generation
- **cargo** - Rust build system and package manager

## 2. System Architecture

### 2.1. Deduced Architectural Pattern

**Assumption:** VecStore follows a **layered architecture** with **feature-gated modularity**, allowing it to scale from a minimal embedded library to a full-featured networked database server.

**Architectural Layers:**

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                             │
│  (Python, JavaScript, Go, Rust, HTTP, gRPC, WebSocket)      │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                 Server Layer (Optional)                      │
│  ┌──────────────┬───────────────┬──────────────────────┐   │
│  │ gRPC Server  │  HTTP/REST    │  WebSocket Streaming │   │
│  │  (tonic)     │   (axum)      │    (axum-ws)         │   │
│  └──────────────┴───────────────┴──────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐    │
│  │          Metrics & Observability                    │    │
│  │     (Prometheus, Tracing, Health Checks)            │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                   Core Store Layer                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              VecStore (Main API)                     │   │
│  │  • upsert() • query() • delete() • compact()        │   │
│  │  • hybrid_query() • snapshots() • schema validation │   │
│  └──────┬──────────────────────────────────────────────┘   │
│         │                                                    │
│  ┌──────▼──────┬───────────┬──────────┬─────────────┐     │
│  │ HNSW Backend│  Metadata │  Filters │ Text Index  │     │
│  │  (hnsw_rs)  │  Storage  │  Parser  │   (BM25)    │     │
│  └─────────────┴───────────┴──────────┴─────────────┘     │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│              Persistence & Storage Layer                     │
│  ┌──────────────┬───────────────┬──────────────────────┐   │
│  │ Disk Layout  │  WAL (Write-  │  Memory-Mapped I/O   │   │
│  │  (bincode)   │  Ahead Log)   │     (memmap2)        │   │
│  └──────────────┴───────────────┴──────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐    │
│  │     Snapshot Management & Recovery                  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Design Principles:**
1. **Feature Flags** - Core functionality always available; advanced features (server, Python, WASM) opt-in via Cargo features
2. **Zero-Copy Where Possible** - Memory-mapped I/O for large vector datasets
3. **ACID-lite** - Write-Ahead Log ensures durability; snapshot isolation for queries
4. **Pluggable Components** - Distance metrics, quantization, and filters are modular

### 2.2. Core Component Breakdown & Interaction Diagram

**Major Components:**

1. **`VecStore` (store/mod.rs)** - Main public API and orchestration
2. **`HnswBackend` (store/hnsw_backend.rs)** - Vector similarity search
3. **`DiskLayout` (store/disk.rs)** - Persistence layer
4. **`FilterParser` (store/filter_parser.rs)** - SQL-like filter expressions
5. **`TextIndex` (store/hybrid.rs)** - BM25 keyword search
6. **`ProductQuantizer` (store/quantization.rs)** - Memory compression
7. **`SemanticCache` (semantic_cache.rs)** - Query result caching
8. **`Schema` (schema.rs)** - Metadata validation
9. **`NamespaceManager` (namespace_manager.rs)** - Multi-tenant orchestration
10. **`Namespace` (namespace.rs)** - Namespace quotas and resource tracking
11. **`VecStoreGrpcServer` (server/grpc.rs)** - gRPC API
12. **`VecStoreHttpServer` (server/http.rs)** - HTTP/REST API
13. **`VecStoreAdminService` (proto/vecstore.proto)** - Admin API for namespace management
14. **`Metrics` (server/metrics.rs)** - Prometheus observability

**Component Interaction Flow (Query Example):**

```
Client Request (gRPC/HTTP)
       │
       ▼
[Server Layer] → Decode request → Extract query params
       │
       ▼
[VecStore] → Check semantic cache
       │           │
       │ (miss)    │ (hit) → Return cached results
       ▼           │
[FilterParser] → Parse filter expression
       │
       ▼
[HnswBackend] → HNSW approximate search → Get candidate IDs
       │
       ▼
[VecStore] → Fetch metadata records
       │
       ▼
[Filter Evaluation] → Apply metadata filters
       │
       ▼
[VecStore] → Sort by score → Limit to k results
       │
       ▼
[Metrics] → Record latency, result count
       │
       ▼
[Server Layer] → Encode response → Send to client
```

### 2.3. Analyzed Directory Structure

```
vecstore/
├── Cargo.toml                    # Project manifest, dependencies, features
├── build.rs                      # Build script (generates protobuf code)
├── LICENSE                       # MIT license
├── README.md                     # Project overview and quick start
├── ROADMAP_V3.md                # Production feature roadmap
├── SERVER.md                     # Server deployment guide
├── NAMESPACES.md                # Multi-tenant namespaces guide
├── BENCHMARKS.md                # Performance benchmarks
├── DEVELOPER_GUIDE.md           # This document
├── Dockerfile                    # Multi-stage production container
├── docker-compose.yml           # Basic server deployment
├── rust-toolchain.toml          # Rust version specification
│
├── proto/                        # Protocol Buffers definitions
│   └── vecstore.proto           # gRPC service schema (17 methods)
│
├── src/
│   ├── lib.rs                   # Library entry point, feature flags
│   │
│   ├── bin/                     # Binary executables
│   │   ├── vecstore.rs          # CLI tool for local operations
│   │   └── vecstore-server.rs  # Server binary (gRPC + HTTP)
│   │
│   ├── store/                   # Core vector database implementation
│   │   ├── mod.rs               # VecStore main struct and API
│   │   ├── types.rs             # Core types (Record, Metadata, Neighbor, Query)
│   │   ├── hnsw_backend.rs      # HNSW index wrapper
│   │   ├── disk.rs              # Persistence layer
│   │   ├── filter_parser.rs     # SQL-like filter parsing
│   │   ├── filters.rs           # Filter execution engine
│   │   ├── advanced_filters.rs  # Complex filter operators
│   │   ├── hybrid.rs            # Hybrid search (vector + BM25)
│   │   └── quantization.rs      # Product quantization
│   │
│   ├── server/                  # Server mode (feature-gated)
│   │   ├── mod.rs               # Server module exports
│   │   ├── grpc.rs              # gRPC server implementation
│   │   ├── http.rs              # HTTP/REST + WebSocket server
│   │   ├── types.rs             # Protobuf ↔ Rust type conversions
│   │   └── metrics.rs           # Prometheus metrics
│   │
│   ├── generated/               # Auto-generated protobuf code
│   │   └── vecstore.rs          # (Generated by tonic-build)
│   │
│   ├── async_api.rs             # Async wrapper (Tokio-based)
│   ├── cache.rs                 # Generic caching utilities
│   ├── error.rs                 # Error types and Result aliases
│   ├── import_export.rs         # CSV/JSON import/export
│   ├── metrics.rs               # Query performance metrics
│   ├── mmap.rs                  # Memory-mapped file utilities
│   ├── namespace.rs             # Namespace types and quota system
│   ├── namespace_manager.rs     # Multi-tenant namespace orchestration
│   ├── python.rs                # PyO3 Python bindings
│   ├── query_analyzer.rs        # Query performance analysis
│   ├── schema.rs                # Metadata schema validation
│   ├── semantic_cache.rs        # Semantic query caching
│   ├── simd.rs                  # SIMD distance calculations
│   ├── stream.rs                # Streaming API utilities
│   ├── vectors.rs               # Vector utility functions
│   └── wal.rs                   # Write-Ahead Log
│
├── benches/                      # Criterion benchmarks
├── tests/                        # Integration tests
├── examples/                     # Example applications
└── observability/               # Monitoring and observability
```

For complete setup instructions, API reference, and how-to guides, see sections 3-6 below.

### 2.4. Multi-Tenant Namespace System

VecStore implements **multi-tenancy through namespaces**, allowing multiple isolated vector databases to coexist within a single server instance. This is critical for SaaS deployments and multi-customer environments.

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   NamespaceManager                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Namespaces Map: HashMap<NamespaceId, Namespace>     │   │
│  │  - Metadata, quotas, usage tracking, status          │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  VecStore Instances: HashMap<NamespaceId, VecStore>  │   │
│  │  - One isolated VecStore per namespace               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  Operations:                                                  │
│  1. Check namespace status (Active/Suspended/ReadOnly)       │
│  2. Validate quotas (vectors, storage, rate limits)          │
│  3. Route operation to correct VecStore instance             │
│  4. Update usage statistics                                  │
│  5. Persist namespace metadata                               │
└─────────────────────────────────────────────────────────────┘
```

#### Core Types

**Namespace** (`src/namespace.rs:14-43`):
```rust
pub struct Namespace {
    pub id: NamespaceId,              // Unique identifier
    pub name: String,                 // Human-readable name
    pub quotas: NamespaceQuotas,      // Resource limits
    pub usage: ResourceUsage,         // Current usage tracking
    pub status: NamespaceStatus,      // Active/Suspended/ReadOnly/PendingDeletion
    pub created_at: u64,              // Unix timestamp
    pub updated_at: u64,              // Unix timestamp
    pub metadata: HashMap<String, String>,  // Custom metadata
}
```

**NamespaceQuotas** (`src/namespace.rs:62-84`):
```rust
pub struct NamespaceQuotas {
    pub max_vectors: Option<usize>,
    pub max_storage_bytes: Option<u64>,
    pub max_requests_per_second: Option<f64>,
    pub max_concurrent_queries: Option<usize>,
    pub max_dimension: Option<usize>,
    pub max_results_per_query: Option<usize>,
    pub max_batch_size: Option<usize>,
}
```

Three predefined tiers:
- **`free_tier()`**: 10K vectors, 100MB, 10 req/s
- **`pro_tier()`**: 1M vectors, 10GB, 100 req/s
- **`unlimited()`**: No limits (all fields None)

**ResourceUsage** (`src/namespace.rs:142-173`):
```rust
pub struct ResourceUsage {
    pub vector_count: usize,
    pub storage_bytes: u64,
    pub total_requests: u64,
    pub requests_current_window: usize,  // For rate limiting
    pub window_start: u64,               // 1-second sliding window
    pub active_queries: usize,           // Concurrent query tracking
    pub total_queries: u64,
    pub total_upserts: u64,
    pub total_deletes: u64,
    pub last_request_at: Option<u64>,
}
```

**NamespaceStatus** (`src/namespace.rs:46-59`):
- **Active**: Fully operational
- **Suspended**: All requests rejected (quota violation, payment issue)
- **ReadOnly**: Queries allowed, writes rejected
- **PendingDeletion**: Marked for deletion

#### Quota Enforcement

Quotas are enforced at multiple levels:

1. **Before Operation** (`src/namespace_manager.rs:193-230`):
   - Check namespace status
   - Validate operation won't exceed quotas
   - Check rate limits (sliding 1-second window)
   - Verify concurrent query limits

2. **During Operation**:
   - Perform operation on isolated VecStore
   - Track active queries

3. **After Operation**:
   - Update usage statistics (vector_count, total_requests, etc.)
   - Persist updated metadata

**Example Flow** (Upsert):
```rust
pub fn upsert(&self, namespace_id: &NamespaceId, ...) -> Result<()> {
    // 1. Pre-check quotas
    {
        let mut namespaces = self.namespaces.write().unwrap();
        let namespace = namespaces.get_mut(namespace_id)?;

        namespace.can_upsert(1)?;  // Check vector quota
        namespace.usage.record_request(&namespace.quotas)?;  // Rate limit
        namespace.usage.total_upserts += 1;
    }

    // 2. Perform operation
    let mut stores = self.stores.write().unwrap();
    let store = stores.get_mut(namespace_id)?;
    store.upsert(id, vector, metadata)?;

    // 3. Update usage
    {
        let mut namespaces = self.namespaces.write().unwrap();
        if let Some(namespace) = namespaces.get_mut(namespace_id) {
            namespace.usage.vector_count = store.len();
        }
    }

    Ok(())
}
```

#### Quota Utilization

Calculated as **maximum** utilization across all quota types (not average):

```rust
pub fn quota_utilization(&self) -> f64 {
    let mut utilizations = Vec::new();

    if let Some(max_vectors) = self.quotas.max_vectors {
        utilizations.push(self.usage.vector_count as f64 / max_vectors as f64);
    }

    if let Some(max_storage) = self.quotas.max_storage_bytes {
        utilizations.push(self.usage.storage_bytes as f64 / max_storage as f64);
    }

    utilizations.iter().copied().fold(0.0_f64, f64::max)  // Maximum, not average
}
```

This ensures quota warnings trigger when **any** limit approaches capacity.

#### Persistence

Each namespace is stored in its own directory:

```
data/
├── customer-123/
│   ├── namespace.json      # Namespace metadata (quotas, usage, status)
│   ├── vectors.bin         # Vector data
│   ├── metadata.bin        # Metadata storage
│   └── hnsw_index.bin      # HNSW graph
├── customer-456/
│   └── ...
```

**Loading namespaces** (`src/namespace_manager.rs:52-83`):
- Scan root directory for subdirectories
- Read `namespace.json` from each directory
- Open VecStore for each namespace
- Reconstruct in-memory maps

**Saving metadata** (`src/namespace_manager.rs:362-373`):
- Serialize Namespace struct to JSON
- Write to `namespace.json`
- Triggered on quota updates, status changes, deletions

#### Admin API

The Admin API provides namespace lifecycle management via gRPC and HTTP:

**gRPC Service** (`proto/vecstore.proto:50-75`):
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

**HTTP Endpoints**:
- `POST /admin/namespaces` - Create namespace
- `GET /admin/namespaces` - List all namespaces
- `GET /admin/namespaces/{id}` - Get namespace details
- `PUT /admin/namespaces/{id}/quotas` - Update quotas
- `PUT /admin/namespaces/{id}/status` - Update status
- `DELETE /admin/namespaces/{id}` - Delete namespace
- `GET /admin/namespaces/{id}/stats` - Get namespace stats
- `GET /admin/stats` - Get aggregate stats

#### Statistics & Monitoring

**Per-Namespace Stats** (`src/namespace_manager.rs:376-390`):
```rust
pub struct NamespaceStats {
    pub namespace_id: NamespaceId,
    pub vector_count: usize,
    pub active_count: usize,
    pub deleted_count: usize,
    pub dimension: usize,
    pub quota_utilization: f64,       // 0.0 - 1.0
    pub total_requests: u64,
    pub total_queries: u64,
    pub total_upserts: u64,
    pub total_deletes: u64,
    pub status: NamespaceStatus,
}
```

**Aggregate Stats** (`src/namespace_manager.rs:392-399`):
```rust
pub struct AggregateStats {
    pub total_namespaces: usize,
    pub active_namespaces: usize,
    pub total_vectors: usize,
    pub total_requests: u64,
}
```

#### Concurrency & Thread Safety

- **Arc<RwLock<T>>** for all shared state:
  - `namespaces: Arc<RwLock<HashMap<NamespaceId, Namespace>>>`
  - `stores: Arc<RwLock<HashMap<NamespaceId, VecStore>>>`

- **Lock Hierarchy**:
  1. Acquire `namespaces` lock for quota checks
  2. Drop lock before acquiring `stores` lock
  3. Acquire `namespaces` lock again for usage updates
  - Prevents deadlocks by never holding multiple locks simultaneously

- **Read/Write Patterns**:
  - Queries use `read()` locks when possible
  - Upserts/deletes use `write()` locks
  - Lock scopes minimized for performance

#### Use Cases

1. **SaaS Multi-Tenancy**: One namespace per customer
2. **Environment Separation**: dev/staging/prod namespaces
3. **Feature Flags**: beta-features vs stable namespaces
4. **Tiered Service**: Different quotas per subscription tier

See [NAMESPACES.md](NAMESPACES.md) for complete guide and examples.

## 3. Setup & Build Instructions

See full section in complete document...

## 4. Core Systems Deep Dive

See full section in complete document...

## 5. API Reference

See full section in complete document...

## 6. Common Workflows & "How-To" Guides

See full section in complete document...

## 7. Contribution Guidelines

See full section in complete document...

---

**Document Version:** 1.0
**Last Updated:** 2025-01-19
**Based on Codebase Analysis:** VecStore v0.1.0 (125 tests passing)
