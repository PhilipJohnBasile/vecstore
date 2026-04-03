# VecStore Status (0.1.0)

This document tracks which parts of the repository are ready for everyday use and which are still exploratory.

## Shipping (covered by tests)

| Area | Notes |
|------|-------|
| Embedded store (`VecStore`) | Open/create database, upsert/query/remove vectors, batch ingestion, snapshots, compaction. |
| Distance metrics | Cosine (default), Euclidean, dot product, Manhattan, Hamming. |
| Metadata filtering | Expression parser with `=`, `!=`, `<`, `<=`, `>`, `>=`, `IN`, `NOT IN`, `CONTAINS`, `AND/OR/NOT`. |
| Python bindings (`vecstore-rs`) | Mirrors the embedded API using PyO3. |
| Feature-flagged server | Single-node HTTP/gRPC server behind `--features server`. |
| Text utilities | Splitters, hybrid search helpers, reranking scaffolding. |
| GPU acceleration | CUDA, Metal, and WebGPU backends implemented. |
| Distributed cluster | Raft consensus, snapshot installation, shard rebalancing. |
| Write-ahead log | Full implementation with recovery and checkpointing. |

## Experimental / Incomplete

| Area | Status |
|------|--------|
| Realtime indexer (`src/realtime.rs`) | Buffering logic exists; integration with HNSW rebuild is incomplete. |
| WASM packaging (`docs/WASM.md`) | Library compiles locally; npm distribution is blocked on tooling updates. |
| DiskANN (`src/diskann.rs`) | Core implementation complete but accuracy tests need investigation. |
| Packaging directories | Homebrew, MacPorts, AUR, Nix, Scoop, Snap, Winget, Chocolatey manifests contain placeholder hashes/URLs. Treat them as templates, not releases (see [../PACKAGING.md](../PACKAGING.md)). |

## How to Help

- File issues describing real workloads or blockers in the “Shipping” areas.
- Provide feedback/patches for the experimental modules if you want to see them prioritised.
- Share reproducible benchmarks so we can document performance expectations.

Thanks for testing VecStore while it is still finding its feet!
