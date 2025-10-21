# VecStore (0.0.1 alpha)

> Rust library for embedding a small HNSW index into your application. This is an **alpha** release; APIs, file formats, and packaging will change without notice.

[![Crate](https://img.shields.io/crates/v/vecstore.svg)](https://crates.io/crates/vecstore)
[![Documentation](https://docs.rs/vecstore/badge.svg)](https://docs.rs/vecstore)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-670%20passing-brightgreen)](https://github.com/PhilipJohnBasile/vecstore/actions)

VecStore keeps data local: open a file, upsert vectors, query via HNSW, and take snapshots. Optional bindings (Python) and feature-flagged server code exist for experimentation, but the priority for 0.0.1 is the embedded store.

For an overview of shipped vs. experimental modules, see [`docs/STATUS.md`](docs/STATUS.md).

---

## What Works Today

- **Embedded store API** (`VecStore::open`, `upsert`, `query`, `remove`, `optimize`, `create_snapshot`, `restore_snapshot`).
- **HNSW index** with cosine, Euclidean, and dot-product distance metrics.
- **Metadata filtering** using the built-in expression language.
- **Batch ingestion** via `batch_upsert` (parallel rebuild of the index).
- **Snapshots** on disk for backups/migration.
- **Python bindings** (`pip install vecstore-rs`) that wrap the same embedded API.
- **Optional single-node HTTP/gRPC server** behind the `server` feature flag. Intended for controlled environments only.

These paths are covered by the existing test suite (670+ Rust tests plus Python smoke tests).

---

## Not Ready Yet

You will find ambitious modules in the repo. Treat them as prototypes for now:

- Distributed clustering (`src/distributed/`)
- Realtime indexer (`src/realtime.rs`)
- GPU backends (`src/gpu/`)
- WASM packaging (see [`docs/WASM.md`](docs/WASM.md) for current blockers)
- Packaging directories (Homebrew, MacPorts, AUR, Nix, Scoop, Snap, Winget). Manifests contain placeholder hashes/URLs—see [PACKAGING.md](PACKAGING.md) before using them.

Please open an issue if one of these directions is critical to you; it helps us prioritise the roadmap.

---

## Install

### Rust

```toml
[dependencies]
vecstore = "0.0.1"
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

### JavaScript / WASM

The core library compiles to WASM, but the npm package is not published yet. Follow the instructions in [`docs/WASM.md`](docs/WASM.md) if you want to experiment locally.

---

## Documentation

- [`docs/STATUS.md`](docs/STATUS.md) – Shipped vs. experimental modules
- [`QUICKSTART.md`](QUICKSTART.md) – Embedding VecStore in a Rust project
- [`docs/FEATURES.md`](docs/FEATURES.md) – Feature-by-feature status notes
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) – High-level design
- [`docs/ROADMAP.md`](docs/ROADMAP.md) – Near-term priorities
- [`docs/COMPETITIVE_ANALYSIS.md`](docs/COMPETITIVE_ANALYSIS.md) – Positioning vs. other vector stores

---

## Contributing

We welcome pull requests that improve the core store, expand tests, or add documentation about real-world use. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting patches.

High-impact areas right now:

1. Test coverage for edge cases (large datasets, snapshot workflows)
2. Performance benchmarks that are easy to reproduce
3. Feedback on the Python bindings and server feature flag

---

## License

MIT – see [LICENSE](LICENSE).
