# VecStore

An embeddable vector database for Rust, with optional Python and WebAssembly interfaces. Store vectors alongside metadata, search for similar records, and keep the data inside your application.

[![CI](https://github.com/PhilipJohnBasile/vecstore/actions/workflows/ci.yml/badge.svg)](https://github.com/PhilipJohnBasile/vecstore/actions/workflows/ci.yml) [![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Status: alpha, version 0.1.x.** APIs and file formats can change. Use regenerable data while evaluating the project.

## Run the quick start

With **Rust 1.92+**, start from a fresh checkout:

```bash
git clone https://github.com/PhilipJohnBasile/vecstore.git
cd vecstore
cargo run --locked --example quickstart
```

The [example](examples/quickstart.rs) creates a temporary database, inserts three records, searches with metadata filters, then saves and reopens the store. It uses fixed vectors, so no model, service, or API key is needed. It finishes with `Store reloaded, count: 3`.

## The Rust interface

```rust
use std::collections::HashMap;
use vecstore::{Metadata, Query, VecStore};

fn main() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let mut store = VecStore::open(directory.path().join("vectors"))?;
    let metadata = Metadata {
        fields: HashMap::from([("title".into(), serde_json::json!("Hello"))]),
    };
    store.upsert("doc1".into(), vec![0.1, 0.2, 0.3], metadata)?;
    let neighbors = store.query(Query::new(vec![0.1, 0.2, 0.3]).with_limit(1))?;
    assert_eq!(neighbors[0].id, "doc1");
    Ok(())
}
```

This uses the current checkout's API. A standalone application also needs `anyhow`, `serde_json`, and `tempfile` alongside its `vecstore` dependency.

## What to explore

| Area | Entry point |
| --- | --- |
| Vector indexing and persistence | [Store implementation](src/store), [architecture](docs/ARCHITECTURE.md) |
| Metadata filters | [Quick start](examples/quickstart.rs), [filter examples](examples/filter_parser_demo.rs) |
| Hybrid retrieval | [Hybrid search example](examples/hybrid_search_demo.rs) |
| Python | [Python guide](python/README.md), `python` Cargo feature |
| Browser / WebAssembly | [WASM guide](docs/WASM.md), `wasm` Cargo feature |
| Optional backends | [Cargo features](Cargo.toml), [examples](examples) |

Optional GPU, server, embedding, and distributed components have separate dependencies and validation needs. The default Rust quick start does not establish readiness for those configurations.

## Benchmarks

The repository includes [benchmark harnesses](benches). A useful report needs the dataset, vector dimensions, distance metric, index settings, recall, hardware, and latency distribution together. Use the harness on your intended workload; this README does not make a universal sub-millisecond latency claim.

## Development and releases

```bash
cargo test --locked --lib
cargo fmt --all -- --check
```

[CI](https://github.com/PhilipJohnBasile/vecstore/actions/workflows/ci.yml) runs the project's build, test, and lint checks. See [verification notes](docs/VERIFICATION.md) for the fresh-checkout results and [Releases](https://github.com/PhilipJohnBasile/vecstore/releases) for published GitHub releases. A Cargo version in source is not itself a published release.

[Contributing](CONTRIBUTING.md) · [Security policy](SECURITY.md) · [Issues](https://github.com/PhilipJohnBasile/vecstore/issues)

## License

[MIT](LICENSE), matching the root license and Rust package metadata.
