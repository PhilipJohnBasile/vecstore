# VecStore

**The SQLite of vector search.** Embed semantic search directly in your app—no server required.

[![Crate](https://img.shields.io/crates/v/vecstore.svg)](https://crates.io/crates/vecstore)
[![npm](https://img.shields.io/npm/v/vecstore-wasm.svg)](https://www.npmjs.com/package/vecstore-wasm)
[![PyPI](https://img.shields.io/pypi/v/vecstore-rs.svg)](https://pypi.org/project/vecstore-rs/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Why VecStore?

| Feature | VecStore | Pinecone/Weaviate | FAISS |
|---------|----------|-------------------|-------|
| **Runs in browser** | Yes | No | No |
| **No server needed** | Yes | No | Yes |
| **Hybrid search** | Yes | Yes | No |
| **Metadata filtering** | Yes | Yes | No |
| **Python + Rust + JS** | Yes | Partial | Python only |

---

## Quick Start

### Rust

```toml
[dependencies]
vecstore = "0.1.0"
```

```rust
use vecstore::VecStore;

let mut store = VecStore::open("vectors.db")?;

// Insert vectors with metadata
store.upsert("doc1", vec![0.1, 0.2, 0.3], json!({"title": "Hello"}))?;

// Semantic search
let results = store.query(&vec![0.1, 0.2, 0.3], 10)?;

// Filtered search
let results = store.query_with_filter(&query_vec, 10, "category = 'tech'")?;
```

### Python

```bash
pip install vecstore-rs
```

```python
import vecstore

store = vecstore.VecStore("vectors.db")
store.upsert("doc1", [0.1, 0.2, 0.3], {"title": "Hello"})
results = store.query([0.1, 0.2, 0.3], k=10)
```

### JavaScript (Browser)

```bash
npm install vecstore-wasm
```

```javascript
import init, { WasmVecStore } from 'vecstore-wasm';

await init();
const store = new WasmVecStore(384);

store.upsert("doc1", new Float32Array([...]), { title: "Hello" });
const results = store.query(queryVector, 10);
```

---

## Features

### Core

- **HNSW Index** - Sub-millisecond search on 100K+ vectors
- **9 Distance Metrics** - Cosine, Euclidean, Dot Product, Manhattan, Hamming, Jaccard, and more
- **Metadata Filtering** - SQL-like expressions: `category = 'tech' AND score > 0.5`
- **Hybrid Search** - Combine vector similarity with BM25 keyword matching
- **Snapshots** - Point-in-time backups and restore

### Browser-First

- **WASM Support** - Full vector search in the browser, no backend
- **Offline-Capable** - Works without network connection
- **Privacy-First** - Data never leaves the user's device
- **Sub-ms Latency** - 0.2ms search on 100K vectors

### Production

- **Batch Operations** - Parallel ingestion for large datasets
- **Soft Delete + TTL** - Flexible data lifecycle management
- **Write-Ahead Log** - Crash recovery with `wal_enabled: true`
- **gRPC + HTTP Server** - Optional server mode for multi-client access

---

## Use Cases

1. **Local RAG** - Semantic search for LLM context retrieval
2. **Browser Search** - Privacy-first document search (legal, medical, financial)
3. **Offline Apps** - Mobile/desktop apps with embedded search
4. **Prototyping** - Test semantic search ideas without infrastructure
5. **Edge Computing** - IoT devices with local vector search

---

## Performance

| Dataset | Search Latency | Memory |
|---------|----------------|--------|
| 10K vectors (384d) | 0.3ms | ~20MB |
| 100K vectors (384d) | 0.2ms | ~180MB |
| 1M vectors (128d) | 0.2ms | ~200MB |

---

## Documentation

- [WASM Guide](docs/WASM.md) - Browser integration with React, Vue, Next.js
- [Python API](https://pypi.org/project/vecstore-rs/) - Full Python documentation
- [Architecture](docs/ARCHITECTURE.md) - System design overview
- [Security Policy](SECURITY.md) - Vulnerability reporting

---

## Roadmap

### Shipping Now
- [x] HNSW with 9 distance metrics
- [x] Metadata filtering
- [x] Hybrid search (vector + BM25)
- [x] Python bindings
- [x] WASM/browser support
- [x] Snapshots

### Coming Soon
- [ ] LangChain/LlamaIndex integration
- [ ] Product Quantization (8-32x memory reduction)
- [ ] GPU acceleration

---

## Alpha Notice

VecStore is in active development (0.1.x). APIs and file formats may change. Not recommended for production workloads with data you can't regenerate.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

High-impact areas:
1. LangChain/LlamaIndex Python wrappers
2. Browser demo applications
3. Performance benchmarks

---

## License

Apache 2.0 - see [LICENSE](LICENSE).
