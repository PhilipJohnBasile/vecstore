# VecStore

> **High-performance embeddable vector database** with HNSW indexing, hybrid search, and production features
>
> Now available on [crates.io](https://crates.io/crates/vecstore) and [PyPI](https://pypi.org/project/vecstore-rs/)
>
> Rust: `cargo add vecstore` | Python: `pip install vecstore-rs`

[![CI](https://github.com/PhilipJohnBasile/vecstore/workflows/CI/badge.svg)](https://github.com/PhilipJohnBasile/vecstore/actions)
[![Crate](https://img.shields.io/crates/v/vecstore.svg)](https://crates.io/crates/vecstore)
[![Documentation](https://docs.rs/vecstore/badge.svg)](https://docs.rs/vecstore)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-670%20passing-brightgreen)](https://github.com/PhilipJohnBasile/vecstore/actions)
[![Production](https://img.shields.io/badge/production-ready-blue)]()

VecStore is a production-ready vector database with integrated RAG capabilities. Embeddable library with file-based storage—no servers or complex setup required.

**Use cases:** RAG applications, semantic search, recommendation systems, document Q&A, code search

---

## Key Features

- **Query Planning** - Built-in EXPLAIN queries for query cost estimation and optimization
- **OpenTelemetry Instrumentation** - Automatic tracing and metrics for production observability
- **Embeddable** - File-based storage, no server required, sub-millisecond query latency
- **Production-Ready** - WAL recovery, soft deletes, TTL, multi-tenancy, Kubernetes-ready
- **Complete RAG Stack** - Vector DB + text splitters + reranking + evaluation metrics
- **Multi-Language** - Rust (native), Python (PyO3), JavaScript/WASM bindings
- **Advanced Indexing** - HNSW with tunable presets, prefetch queries, MMR diversity, hybrid search
- **Cost-Effective** - No SaaS fees, run on your infrastructure

---

## Quick Start

### Rust

```toml
[dependencies]
vecstore = "1.0"
```

```rust
use vecstore::VecStore;

let mut store = VecStore::open("vectors.db")?;
store.upsert("doc1", &vec![0.1, 0.2, 0.3], metadata)?;
let results = store.query(&vec![0.15, 0.25, 0.85], 10, None)?;
```

### Python

```bash
pip install vecstore-rs
```

```python
import vecstore

store = vecstore.VecStore("vectors.db")
store.upsert("doc1", [0.1, 0.2, 0.3], {"title": "Doc"})
results = store.query([0.15, 0.25, 0.85], k=10)
```

### JavaScript/WASM

```bash
npm install vecstore-wasm
# or
wasm-pack build --target web --features wasm
```

```javascript
import init, { WasmVecStore } from 'vecstore-wasm';

await init();
const store = WasmVecStore.new(384); // 384-dimensional vectors

// Insert vectors
store.upsert('doc1', [0.1, 0.2, ...], { title: 'Document 1' });

// Search with HNSW (sub-millisecond on 100k+ vectors!)
const results = store.query([0.15, 0.25, ...], 10);
```

> **Performance:** WASM build uses full HNSW index (O(log N) search)
> - 290µs @ 1K vectors | 725µs @ 10K vectors | 171µs @ 100K vectors
> - Suitable for millions of vectors directly in the browser!

**See [docs/WASM.md](docs/WASM.md) for TypeScript definitions and complete guide**

---

## Features

### Core Vector Database
- **Query Planning** - EXPLAIN queries for cost estimation and optimization
- **Prefetch Queries** - Multi-stage retrieval (vector → rerank → MMR → final)
- **HNSW Indexing** - Sub-millisecond queries with configurable presets (fast/balanced/high_recall/max_recall)
- **SIMD Acceleration** - 4-8x faster distance calculations (AVX2/NEON)
- **Product Quantization** - 8-32x memory compression
- **Metadata Filtering** - SQL-like queries: `"category = 'tech' AND score > 0.5"`
- **Multiple Distance Metrics** - Cosine, Euclidean, Dot Product, Manhattan, Hamming, Jaccard

### Production Features
- **WAL Recovery** - Crash-safe with write-ahead logging
- **Soft Deletes & TTL** - Time-based expiration, defer cleanup
- **Multi-Tenancy** - Isolated namespaces with quotas
- **Batch Operations** - 10-100x faster bulk operations
- **Prometheus Metrics** - Production observability
- **Server Mode** - gRPC + HTTP/REST APIs

### Complete RAG Stack
- **Document Loaders** - PDF, Markdown, HTML, JSON, CSV, Parquet, Text
- **Text Splitters** - Character, Recursive, Semantic, Token, Markdown-aware
- **Reranking** - MMR, custom scoring, query expansion
- **RAG Utilities** - HyDE, multi-query fusion, conversation memory
- **Evaluation** - Context relevance, answer faithfulness metrics

---

## Documentation

**[Quick Start](QUICKSTART.md)** - Get running in 5 minutes
**[Complete Features](docs/FEATURES.md)** - Comprehensive feature reference
**[Deployment Guide](DEPLOYMENT.md)** - Production deployment (Docker, K8s)
**[Achievements](ACHIEVEMENTS.md)** - Feature completeness and capabilities

**For Contributors:**
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Detailed contributor guide
- [CHANGELOG.md](CHANGELOG.md) - Version history

**Market Position:**
- [docs/COMPETITIVE-ANALYSIS.md](docs/COMPETITIVE-ANALYSIS.md) - vs Qdrant, Weaviate, Pinecone
- [ROADMAP.md](ROADMAP.md) - Future enhancements

---

## Use Cases

- **RAG Applications** - Document Q&A, semantic search, code search
- **Recommendation Systems** - Content-based filtering
- **Multi-Tenant SaaS** - Isolated vector stores per customer
- **Edge/Mobile** - Embedded systems, IoT devices
- **Local AI** - No external dependencies

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for quick start or [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for detailed guide.

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Add tests (`cargo test`)
4. Format code (`cargo fmt`)
5. Submit a PR

**Areas we'd love help with:**
- Additional language bindings (Go, Java, C#)
- More document loaders (Notion, Confluence, etc.)
- Performance benchmarks
- Real-world use case examples

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with Rust** | **Production Ready** | **Zero Cost**

---

## Star History

Star us on GitHub if you find VecStore useful!
