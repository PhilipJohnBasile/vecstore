# VecStore Status (0.0.1 alpha)

This document tracks which parts of the repository are ready for everyday use and which are still exploratory. The aim is to keep expectations clear while the project is in alpha.

## Shipping (covered by tests)

| Area | Notes |
|------|-------|
| Embedded store (`VecStore`) | Open/create database, upsert/query/remove vectors, batch ingestion, snapshots, compaction. |
| Distance metrics | Cosine (default), Euclidean, dot product. |
| Metadata filtering | Expression parser with `=`, `!=`, `<`, `<=`, `>`, `>=`, `IN`, `NOT IN`, `CONTAINS`, `AND/OR/NOT`. |
| Python bindings (`vecstore-rs`) | Mirrors the embedded API using PyO3. |
| Feature-flagged server | Single-node HTTP/gRPC server behind `--features server`; expect to run it behind your own supervision/observability stack. |
| Text utilities | Splitters, basic hybrid search helpers, reranking scaffolding. |

## Experimental / Incomplete

| Area | Status |
|------|--------|
| Distributed cluster (`src/distributed/`) | Networking, consensus, durability, and rebalancing are unfinished. |
| Realtime indexer (`src/realtime.rs`) | Buffering logic exists; integration with HNSW rebuild is incomplete. |
| GPU acceleration (`src/gpu/`) | CUDA/Metal/WebGPU backends are stubs that fall back to CPU code. |
| WASM packaging (`docs/WASM.md`) | Library compiles locally; npm distribution is blocked on tooling updates. |
| Write-ahead log (`src/wal.rs`) | Implementation is well-tested but not yet integrated into the default write path. |
| Packaging directories | Homebrew, MacPorts, AUR, Nix, Scoop, Snap, Winget, Chocolatey manifests contain placeholder hashes/URLs. Treat them as templates, not releases (see [../PACKAGING.md](../PACKAGING.md)). |

## How to Help

- File issues describing real workloads or blockers in the “Shipping” areas.
- Provide feedback/patches for the experimental modules if you want to see them prioritised.
- Share reproducible benchmarks so we can document performance expectations.

Thanks for testing VecStore while it is still finding its feet!
