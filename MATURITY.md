# VecStore Feature Maturity Matrix

**Assessment Date:** December 27, 2025
**Repository Version Analyzed:** 0.1.0
**Proposed Target Release:** 0.2.0-alpha

> This is a point-in-time assessment. Verify individual claims against the current code and test results before relying on them.

---

## Purpose

This document provides a clear, honest assessment of the stability and production-readiness of every feature in VecStore. Use this to make informed decisions about what features to use in your project.

---

## Maturity Levels

| Level | Icon | Description | Guarantees |
|-------|------|-------------|------------|
| **Stable** | ✅ | Production-ready. API stable. Well-tested. Battle-tested. | - No breaking API changes without major version bump<br>- Comprehensive test coverage (>90%)<br>- Known edge cases documented<br>- Performance characteristics documented |
| **Beta** | 🟡 | Feature-complete. API may change. Use with caution. | - Feature works for common use cases<br>- API may change in minor versions<br>- Test coverage >70%<br>- Some edge cases may not be handled |
| **Experimental** | 🔴 | Prototype/proof-of-concept. API unstable. No guarantees. | - May not work reliably<br>- API will change<br>- May be removed without notice<br>- Use for research/evaluation only |
| **Deprecated** | ⚫ | Will be removed in future version. | - Do not use in new projects<br>- Migration guide available<br>- Removal date announced |

---

## Core Features

### Vector Storage & Indexing

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **In-Memory Vector Storage** | ✅ Stable | 0.1.0 | `store/mod.rs` | Fast, reliable, well-tested |
| **HNSW Indexing** | ✅ Stable | 0.1.0 | `store/hnsw_backend.rs` | Production-grade approximate NN search |
| **Vector Upsert** | ✅ Stable | 0.1.0 | `store/mod.rs` | Core operation, extensively tested |
| **Vector Query** | ✅ Stable | 0.1.0 | `store/mod.rs` | Core operation, extensively tested |
| **Vector Delete** | ✅ Stable | 0.1.0 | `store/mod.rs` | Soft delete by default, works reliably |
| **Batch Operations** | ✅ Stable | 0.1.0 | `store/mod.rs` | Parallel upsert/query, performance tested |

### Distance Metrics

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Cosine Similarity** | ✅ Stable | 0.1.0 | `vectors.rs` | Most common metric, SIMD optimized |
| **Euclidean Distance** | ✅ Stable | 0.1.0 | `vectors.rs` | SIMD optimized |
| **Dot Product** | ✅ Stable | 0.1.0 | `vectors.rs` | SIMD optimized |
| **Manhattan Distance** | ✅ Stable | 0.1.0 | `vectors.rs` | Basic implementation |
| **Hamming Distance** | 🟡 Beta | 0.1.0 | `vectors.rs` | Works for binary vectors, limited testing |
| **Jaccard Similarity** | 🟡 Beta | 0.1.0 | `vectors.rs` | Works for sets, limited testing |

### Metadata & Filtering

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Metadata Storage** | ✅ Stable | 0.1.0 | `store/types.rs` | JSON metadata, works reliably |
| **Basic Filters** | ✅ Stable | 0.1.0 | `store/filters.rs` | =, !=, >, <, >=, <= operators |
| **Boolean Logic** | ✅ Stable | 0.1.0 | `store/filter_parser.rs` | AND, OR, NOT combinations |
| **CONTAINS Operator** | 🟡 Beta | 0.1.0 | `store/filters.rs` | Works but case-sensitive only |
| **IN / NOT IN Operators** | 🟡 Beta | 0.1.0 | `store/filters.rs` | Works for arrays, needs more testing |
| **Advanced Filters** | 🟡 Beta | 0.2.0 | `advanced_filter.rs` | Regex, wildcards, complex expressions |

### Persistence & Durability

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **File-Based Storage** | ✅ Stable | 0.1.0 | `store/disk.rs` | Reliable persistence via bincode |
| **Snapshots** | 🟡 Beta | 0.1.0 | `store/mod.rs` | Backup works, restore needs more testing |
| **Write-Ahead Log (WAL)** | 🟡 Beta | 0.1.0 | `wal.rs` | Crash recovery works, needs more edge case testing |
| **Memory-Mapped I/O** | 🟡 Beta | 0.2.0 | `mmap.rs` | Works for large datasets, platform-dependent |

---

## Advanced Features

### Hybrid Search

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **BM25 Keyword Search** | 🟡 Beta | 0.1.0 | `store/hybrid.rs` | Works but limited language support |
| **Vector+Keyword Fusion** | 🟡 Beta | 0.1.0 | `store/hybrid.rs` | Multiple fusion strategies, needs tuning |
| **Simple Tokenizer** | 🟡 Beta | 0.1.0 | `store/hybrid.rs` | Basic whitespace splitting |
| **Language Tokenizer** | 🔴 Experimental | 0.1.0 | `store/hybrid.rs` | Limited language support |
| **Phrase Matching** | 🔴 Experimental | 0.1.0 | `store/hybrid.rs` | Position-aware, not fully tested |

### Compression & Optimization

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Product Quantization** | 🔴 Experimental | 0.2.0 | `store/quantization.rs` | Memory compression works, accuracy impact unclear |
| **SIMD Acceleration** | 🟡 Beta | 0.1.0 | `simd.rs` | AVX2/NEON optimizations, platform-dependent |
| **Advanced Quantization** | 🔴 Experimental | 0.2.0 | `advanced_quant.rs` | Ultra-low-bit (1.5-bit, 2-bit), research quality |

### Server Mode

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **gRPC Server** | 🟡 Beta | 0.1.0 | `server/grpc.rs` | Works but lacks auth, rate limiting |
| **HTTP/REST API** | 🟡 Beta | 0.1.0 | `server/http.rs` | Basic endpoints work, incomplete |
| **WebSocket Streaming** | 🔴 Experimental | 0.1.0 | `server/http.rs` | Prototype only |
| **Prometheus Metrics** | 🟡 Beta | 0.1.0 | `server/metrics.rs` | Basic metrics exported |
| **Health Checks** | 🟡 Beta | 0.1.0 | `server/http.rs` | /health endpoint works |

### Multi-Tenancy

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Namespace Isolation** | 🟡 Beta | 0.1.0 | `namespace_manager.rs` | Works, quota enforcement active |
| **Quota Management** | 🟡 Beta | 0.1.0 | `namespace.rs` | Vector, storage, rate limit quotas enforced |
| **Resource Tracking** | 🟡 Beta | 0.1.0 | `namespace.rs` | Usage statistics tracked |
| **Admin API** | 🟡 Beta | 0.1.0 | `server/grpc.rs` | Namespace CRUD operations work |

---

## Language Bindings

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Rust API** | ✅ Stable | 0.1.0 | `lib.rs` | Native, first-class support |
| **Python Bindings** | 🟡 Beta | 0.1.0 | `python.rs` | PyO3 bindings, most features work |
| **WASM/JavaScript** | 🟡 Beta | 0.1.0 | `wasm-pkg/` | Browser support, limited features vs. native |

---

## Innovation Features (Experimental)

> ⚠️ **WARNING:** All features below are experimental prototypes. They may not work reliably, APIs will change, and they may be removed without notice. Use for research/evaluation only.

### Explainability & Debugging

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Explainable Search** | 🔴 Experimental | 0.2.0 | `explainable.rs` | Dimension contributions, prototype quality |
| **Query Explanation** | 🔴 Experimental | 0.1.0 | `query_explain.rs` | EXPLAIN-style analysis, incomplete |
| **Embedding Debugger** | 🔴 Experimental | 0.2.0 | `debugger.rs` | Visualization tools, proof-of-concept |
| **Query Analyzer** | 🔴 Experimental | 0.1.0 | `query_analyzer.rs` | Cost estimation, needs validation |

### Advanced Indexing

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **DiskANN Index** | 🔴 Experimental | 0.2.0 | `diskann.rs` | Billion-scale SSD index, not integrated |
| **Learned Indexes** | 🔴 Experimental | 0.2.0 | `learned_index.rs` | Self-optimizing, research quality |
| **Incremental Indexing** | 🔴 Experimental | 0.2.0 | `incremental_index.rs` | Streaming updates, incomplete |
| **Columnar Storage** | 🔴 Experimental | 0.2.0 | `columnar.rs` | Column-oriented, not fully tested |

### Time & Versioning

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Time-Aware Search** | 🔴 Experimental | 0.2.0 | `temporal.rs` | Temporal decay, drift detection, prototype |
| **Embedding Version Control** | 🔴 Experimental | 0.2.0 | `embedding_vcs.rs` | Model versioning, proof-of-concept |
| **Vector Lineage** | 🔴 Experimental | 0.2.0 | `lineage.rs` | Provenance tracking, incomplete |

### Privacy & Security

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Privacy-Preserving Search** | 🔴 Experimental | 0.2.0 | `privacy.rs` | Differential privacy, research quality |
| **Access Control** | 🔴 Experimental | 0.2.0 | `access_control.rs` | RBAC prototype |
| **Audit Logging** | 🔴 Experimental | 0.2.0 | `audit.rs` | Compliance logs, incomplete |

### Acceleration & Performance

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **GPU Acceleration (CUDA)** | 🔴 Experimental | 0.2.0 | `gpu/cuda_kernels.rs` | **Stubs only**, not functional |
| **GPU Acceleration (Metal)** | 🔴 Experimental | 0.2.0 | N/A | Planned, not implemented |
| **WebGPU Support** | 🔴 Experimental | 0.2.0 | N/A | Planned, not implemented |
| **Auto-Tuning** | 🔴 Experimental | 0.2.0 | `auto_tuning.rs` | Parameter optimization, incomplete |

### AI/ML Integration

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Agentic Search** | 🔴 Experimental | 0.2.0 | `agentic.rs` | Autonomous query refinement, prototype |
| **Neural Rankers** | 🔴 Experimental | 0.2.0 | `neural_ranker.rs` | Learned ranking, incomplete |
| **ColBERT Reranking** | 🔴 Experimental | 0.1.0 | `reranking/colbert.rs` | Late interaction, limited testing |
| **Anomaly Detection** | 🔴 Experimental | 0.2.0 | `anomaly_detection.rs` | Outlier detection, incomplete |
| **Clustering** | 🔴 Experimental | 0.2.0 | `clustering.rs` | K-means, hierarchical, incomplete |

### Advanced Operations

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Graph-Vector Fusion** | 🔴 Experimental | 0.2.0 | `graph_vector.rs` | Hybrid graph traversal, prototype |
| **Semantic Cache** | 🟡 Beta | 0.1.0 | `semantic_cache.rs` | Query caching works, limited testing |
| **Adaptive Cache** | 🔴 Experimental | 0.2.0 | `adaptive_cache.rs` | ML-based caching, incomplete |
| **Federation** | 🔴 Experimental | 0.2.0 | `federation.rs` | Multi-node, not functional |

### Analytics & Observability

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **A/B Testing** | 🔴 Experimental | 0.2.0 | `ab_testing.rs` | Experiment framework, incomplete |
| **Analytics** | 🔴 Experimental | 0.2.0 | `analytics.rs` | Usage analytics, incomplete |
| **Cost Optimization** | 🔴 Experimental | 0.2.0 | `cost_optimizer.rs` | Cloud cost analysis, incomplete |
| **Benchmark Framework** | 🔴 Experimental | 0.2.0 | `benchmark.rs` | Internal benchmarking, incomplete |

### Data Integration

| Feature | Maturity | Since | Module | Notes |
|---------|----------|-------|--------|-------|
| **Change Data Capture** | 🔴 Experimental | 0.2.0 | `cdc.rs` | Database syncing, incomplete |
| **Change Streams** | 🔴 Experimental | 0.2.0 | `change_streams.rs` | Real-time notifications, incomplete |
| **Object Storage** | 🔴 Experimental | 0.2.0 | `object_storage.rs` | S3/GCS tier, incomplete |
| **Backup Management** | 🔴 Experimental | 0.2.0 | `backup.rs` | Advanced backup strategies, incomplete |

---

## Deprecation Policy

When features are deprecated, we will:

1. **Announce** in CHANGELOG and release notes
2. **Mark** as deprecated in documentation (⚫ icon)
3. **Provide** migration guide
4. **Maintain** for at least 2 minor versions before removal
5. **Remove** only in major version updates

---

## How to Use This Matrix

### For Production Use
- ✅ **Only use Stable features** for production workloads
- 🟡 **Use Beta features** only if you can handle API changes
- 🔴 **Avoid Experimental features** unless you're willing to rewrite code

### For Evaluation/Research
- 🔴 **Experimental features** are perfect for research and prototyping
- Provide feedback on experimental features via GitHub issues
- Help us stabilize features by reporting bugs and edge cases

### For Contributors
- See which features need stabilization work
- Focus on moving Beta → Stable features first
- Experimental features welcome but label clearly

---

## Roadmap to Stability

### Path to 1.0 (Stable Release)

**Required for 1.0:**
- ✅ All core features (vector storage, HNSW, basic filters) Stable
- 🟡 Hybrid search → Stable
- 🟡 Server mode → Stable (or removed if not needed)
- 🟡 Python bindings → Stable
- 90%+ test coverage on all Stable features
- Public benchmarks published
- Security audit complete

**Nice to have for 1.0:**
- 🔴 Explainable search → Beta or Stable
- 🔴 One GPU backend (CUDA) → Beta
- 🟡 WASM → Stable

**Deferred to 2.0+:**
- Most experimental features (move to vecstore-labs crate)
- Distributed/federation features
- Advanced ML features (agentic, neural rankers)

---

## Questions?

- **General questions:** GitHub Discussions
- **Bug reports:** GitHub Issues
- **Feature stability questions:** This document
- **Community:** Discord (link in README)

---

**Last Updated:** December 27, 2025
**Maintainer:** VecStore Team
**Review Frequency:** Updated with each release
