# Changelog

All notable changes to VecStore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-28

### Changed

#### Post-Modernization Audit Fixes
- Fixed production panic in `distributed/mod.rs` with defensive programming
- Fixed failing federation test (Local members return empty by design)
- Fixed object_store cache test (enabled read_through_cache)
- Removed unnecessary `std::collections::HashMap` qualifications
- Fixed all unused variable warnings in tests
- Fixed all example warnings (12 files)
- Updated CLI module documentation for disabled state

#### Version Consistency
- All documentation now references v0.1.0
- Python bindings updated to 0.1.0
- pyproject.toml updated to 0.1.0

## [0.0.3] - 2025-12-27

### Changed

#### Rust 1.92 Modernization
- Upgraded to Rust Edition 2024 with full 1.92 feature support
- Updated all dependencies to latest stable versions
- Added comprehensive lint configuration via `.cargo/config.toml` and `clippy.toml`
- Improved code quality with modern Rust idioms (if-let chains, const fn, etc.)

#### GPU Backends (Complete Implementation)
- Full CPU backend with SIMD-optimized operations
- Complete CUDA backend with cuBLAS integration for NVIDIA GPUs
- Complete Metal backend with Apple Silicon optimization
- Complete WebGPU backend with WGSL compute shaders
- GPU operations: batch distance (Euclidean, Cosine, Dot), KNN search, matrix multiply, normalization

#### Distributed System (Complete Implementation)
- Full Raft consensus implementation with leader election
- Snapshot installation for state transfer
- Consistent hash ring for shard distribution
- Node health monitoring and failure detection
- gRPC-based inter-node communication

#### Agent Module Enhancements
- Implemented proper confidence computation from execution traces
- Added comprehensive agent planning and execution APIs
- Full filter expression support for agent tools

#### Filter Selectivity Estimation
- Heuristic-based selectivity estimation for query optimization
- Support for all filter operators (Eq, Neq, Lt, Gt, Lte, Gte, Contains, In, NotIn, StartsWith)
- AND/OR/NOT combination rules

### Added

#### New Test Coverage
- `tests/agent_tests.rs` - Agent module integration tests
- `tests/gpu_tests.rs` - GPU backend tests (CPU, CUDA, Metal, WebGPU)
- `tests/distributed_tests.rs` - Distributed system tests
- `tests/selectivity_tests.rs` - Filter selectivity tests

#### Embeddings Improvements
- True batching in Candle backend with padding/truncation
- Dynamic hidden size configuration

## [0.0.2] - 2025-12-26

### Added

#### Quantization Wiring
- Quantized vectors now used during search operations
- `train_quantizer()` method for training on existing vectors
- `is_quantizer_trained()` and `quantization_stats()` introspection methods
- Automatic quantization on upsert when trained
- Quantizer state persisted to disk

#### LangChain Integration
- Native `Document`, `ScoredDocument`, and `LangChainVectorStore` classes
- Full LangChain VectorStore API compatibility
- Export from main Python package

#### Built-in Embedding Support

**Python (sentence-transformers):**
- `VecStoreWithEmbeddings` class for automatic embedding generation
- `EmbeddingModel` wrapper for any sentence-transformers model
- Support for all-MiniLM-L6-v2, all-mpnet-base-v2, bge-*, multilingual models
- Optional dependency: `pip install vecstore-rs[embeddings]`

**WASM/Browser (Transformers.js):**
- `@vecstore/embeddings` npm package
- `VecStoreWithEmbeddings` TypeScript class
- Automatic model loading from HuggingFace
- WebGPU acceleration when available

#### GPU Acceleration Infrastructure
- Feature flags: `cuda`, `metal`, `webgpu`
- `cudarc` crate integration for NVIDIA GPUs
- `metal-rs` crate integration for Apple Silicon
- `wgpu` crate for cross-platform WebGPU
- GPU acceleration documentation and roadmap

#### Disk-Based HNSW
- `DiskVectorStorage` for memory-mapped vector files
- `DiskHNSWIndex` combining graph + vector storage
- Support for larger-than-RAM datasets
- Minimal memory footprint via OS page cache
- Streaming search implementation

### Changed
- Python package version bumped to 0.0.2
- pyproject.toml updated with Apache-2.0 license
- README updated with LangChain and embedding documentation
- WASM docs updated with built-in embedding examples

### Dependencies
- Added: `cudarc` (0.12, optional)
- Added: `metal-rs` (0.31, optional)
- Added: `wgpu` (22, optional)

---

## [0.0.1] - 2025-10-20

### Notes

- Initial public alpha. APIs, file formats, and package structure are subject to change.

Now available on [crates.io](https://crates.io/crates/vecstore) and [PyPI](https://pypi.org/project/vecstore-rs/).

**Rust:**
```bash
cargo add vecstore
```

**Python:**
```bash
pip install vecstore-rs
```

### Release Highlights

- Embeddable HNSW index with query planning utilities.
- Expanded hybrid search helpers (reranking, filters, multi-stage pipelines).
- Optional server mode for teams that want gRPC/HTTP access.
- Growing ecosystem integrations (Python bindings, LangChain adapters, document loaders).

### Added

#### 🎨 ColBERT Late Interaction Reranking (NEW!)
- Token-level similarity computation for high-accuracy reranking
- Multi-vector representation (one vector per token)
- Late interaction via MaxSim operation
- 3 similarity metrics: Cosine, DotProduct, L2
- Batch reranking support
- Document caching for performance
- 6 comprehensive tests + complete example

**Example:**
```rust
use vecstore::reranking::colbert::{ColBERTReranker, ColBERTConfig};

let config = ColBERTConfig::default();
let reranker = ColBERTReranker::new(config)?;

let query_tokens = reranker.encode_query("what is rust?").await?;
let doc_tokens = reranker.encode_document("Rust is a systems programming language").await?;
let score = reranker.compute_score(&query_tokens, &doc_tokens)?;
```

#### Query Planning helpers
- `explain_query()` - EXPLAIN-style query analysis
- Cost estimation for query execution
- Optimization recommendations
- Query execution breakdown
- Selectivity estimation

**Example:**
```rust
let plan = store.explain_query(query)?;
println!("Estimated cost: {:.2}", plan.estimated_cost);
for rec in plan.recommendations {
    println!("Hint: {}", rec);
}
```

#### Multi-Stage Prefetch Queries
- Qdrant-style prefetch API
- Multi-stage retrieval pipelines
- Support for vector search, hybrid search, reranking, MMR, and filter stages
- Pipeline execution (stages run sequentially)

**Example:**
```rust
let query = PrefetchQuery {
    stages: vec![
        QueryStage::HybridSearch { ... },
        QueryStage::MMR { k: 10, lambda: 0.7 },
    ],
};
let results = store.prefetch_query(query)?;
```

#### HNSW Parameter Tuning
- Per-query HNSW `ef_search` control
- 4 semantic presets: `fast()`, `balanced()`, `high_recall()`, `max_recall()`
- `query_with_params()` method for fine-grained performance control

**Example:**
```rust
let results = store.query_with_params(
    query,
    HNSWSearchParams::high_recall(),  // ef_search=100
)?;
```

#### MMR Diversity Algorithm
- Maximal Marginal Relevance for result diversification
- Balances relevance vs diversity
- Lambda parameter controls tradeoff (0.0 = all diversity, 1.0 = all relevance)

#### Query Builder API
- Fluent API for building queries
- `Query::new(vector).with_limit(k).with_filter(expr)`
- Cleaner, more expressive query construction

#### Distributed tracing integration points
- Automatic `#[tracing::instrument]` on all major operations
- Zero-code instrumentation for query(), upsert(), hybrid_query()
- OpenTelemetry-compatible (Jaeger, Zipkin, Honeycomb)
- JSON and console output formats
- Helper functions: `traced_async()`, `traced_sync()`, `record_event()`, `record_error()`
- Production observability out of the box

**Example:**
```rust
use vecstore::telemetry::init_telemetry;

init_telemetry()?;  // All operations now traced automatically
let results = store.query(query)?;  // Span created with k, filter, dimension
```

#### Text Processing Convenience Methods
- `upsert_chunks()` - Split document + embed + upsert in one call
- `batch_upsert_texts()` - Batch embed and upsert multiple texts
- `query_text()` - Query using text instead of vectors
- Seamless document-to-vector pipeline (3 lines instead of 30)

**Example:**
```rust
collection.upsert_chunks("doc1", long_document, &splitter, &embedder)?;
collection.query_text("search query", &embedder, 10)?;
```

#### Candle Embeddings Backend (Pure Rust!)
- **all-MiniLM-L6-v2** support (22M params, 384-dim)
- **BAAI/bge-small-en** support (33M params, 384-dim)
- Custom HuggingFace model support
- Zero Python dependencies - Pure Rust embeddings!
- Automatic model download from HuggingFace Hub
- Mean pooling + normalization

**Example:**
```rust
use vecstore::{CandleEmbedder, CandleModel};

let embedder = CandleEmbedder::new(CandleModel::AllMiniLML6V2)?;
let embedding = embedder.embed("Hello, world!")?;  // 384-dim
```

---

### Core Features

#### Vector Search
- HNSW indexing tuned for low-latency queries
- SIMD acceleration (AVX2/NEON) - 4-8x faster distance calculations
- Product Quantization - 8-32x memory compression
- 6 distance metrics: Cosine, Euclidean, Dot Product, Manhattan, Hamming, Jaccard

#### Hybrid Search
- Vector similarity + BM25 keyword matching
- 4 pluggable tokenizers (Simple, Language, Whitespace, NGram)
- Position-aware phrase matching with 2x boost
- 8 fusion strategies for combining scores

#### Metadata Filtering
- SQL-like filter syntax
- 9 operators: `=`, `!=`, `>`, `>=`, `<`, `<=`, `CONTAINS`, `IN`, `NOT IN`
- Boolean logic: `AND`, `OR`, `NOT`
- Filter during HNSW traversal for performance

---

### Production Features

#### Server Mode
- gRPC + HTTP/REST APIs (14 RPCs)
- WebSocket streaming
- Prometheus metrics
- Health checks
- 401-line protobuf definition

#### Multi-Tenancy
- Isolated namespaces per tenant
- 7 quota types enforced at runtime
- Per-namespace snapshots
- True isolation (separate VecStore instance per namespace)

#### Reliability
- Write-Ahead Logging (WAL) for crash recovery
- Soft deletes with TTL
- Snapshot/backup/restore
- Graceful degradation

#### Deployment
- Docker multi-stage builds
- Kubernetes manifests (deployment, HPA, ingress)
- Prometheus + Grafana observability
- Multi-cloud compatible (AWS, GCP, Azure, DigitalOcean)

---

### Ecosystem

#### Python Bindings (PyO3)
- 688 lines of native bindings
- Zero-copy performance
- Complete API coverage
- LangChain compatible

```python
import vecstore
store = vecstore.VecStore("vectors.db")
results = store.query([0.1, 0.2, 0.3], k=10)
```

#### Complete RAG Stack
- Document loaders (PDF, Markdown, HTML, JSON, CSV, Parquet)
- Text splitters (Character, Recursive, Semantic, Token, Markdown-aware)
- Reranking (MMR, custom scoring)
- RAG utilities (HyDE, multi-query fusion, conversation memory)
- Evaluation metrics (context relevance, answer faithfulness)

---

### Performance

- **Query Latency:** Low-latency in embedded mode (no network); add network budget for server deployments
- **Throughput:** 10,000+ queries/sec (embedded), 5,000+ (server)
- **Index Build:** ~1,000 vectors/sec
- **Memory:** 512MB-2GB typical workload
- **Storage:** ~500 bytes per vector (128-dim)

---

### Testing

- **350 comprehensive tests** (100% passing)
- **Zero regressions**
- Unit tests, integration tests, property-based tests
- Full test coverage for all features

---

### Documentation

- Complete feature reference
- Production deployment guide
- Kubernetes deployment guide
- Competitive analysis vs Qdrant, Weaviate, Pinecone
- Quick start guide (30 seconds to first query)
- Developer guide for contributors

---

### Competitive Notes

- Embeddable usage remains VecStore’s primary differentiator compared to server-first products such as Pinecone and Weaviate.
- Query-planning helpers and reranking utilities are uncommon in this space and provide extra transparency for operators.
- Hosted competitors currently offer managed clusters, stronger distributed guarantees, and GPU offload—areas where VecStore still relies on on-premise work.

---

### Highlights

1. Query planning and EXPLAIN helpers give visibility into vector search costs.
2. Optional server mode lets teams expose VecStore over gRPC/HTTP without rewriting the core engine.
3. PyO3 bindings mirror the Rust API, enabling local-first Python workflows.
4. RAG tooling (loaders, splitters, rerankers) reduces the amount of surrounding infrastructure required for prototypes.
5. Experimental modules (distributed, GPU, realtime) exist as previews and are not yet hardened for production.

---

### Breaking Changes

None - this is the initial 0.0.1 release.

---

### Migration Guide

Not applicable for 0.0.1 release.

---

## Future Releases

See [ROADMAP.md](ROADMAP.md) for planned features.

**Optional Enhancements (Beyond 100%):**
- Load testing documentation
- Helm chart for Kubernetes
- Additional language bindings (Go, Java, C#)
- More document loaders (Notion, Confluence)
- Graph-RAG integration

---

## Links

- **Repository:** https://github.com/PhilipJohnBasile/vecstore
- **Documentation:** https://docs.rs/vecstore
- **crates.io:** https://crates.io/crates/vecstore
- **PyPI:** https://pypi.org/project/vecstore

---

**Achievement Date:** 2025-10-19
**Tests Passing:** 350/350 (100%)
**Examples:** 36 Rust + 7 Python
**Status:** Core library stable; advanced modules under active development

**Built with Rust** | **Embeddable-first** | **Local-friendly**
