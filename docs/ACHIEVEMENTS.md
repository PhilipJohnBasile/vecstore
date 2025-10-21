# VecStore Project Status

This document replaces the earlier “achievement” scorecard and aims to provide a grounded view of what currently works, what is in progress, and what remains aspirational.

> VecStore 0.0.1 is an alpha release. Treat the lists below as current status, not long-term guarantees.

## Core Capabilities (Shipping)
- HNSW-based approximate nearest neighbour index (cosine / Euclidean / dot product) with snapshot/restore.
- Metadata-aware search with the expression language.
- Batch ingestion (`batch_upsert`) and manual compaction (`optimize`).
- Python bindings via PyO3 that wrap the embedded API.
- Feature-flagged HTTP/gRPC server for single-node experiments.

## In Progress / Experimental
- Write-ahead log integration (implementation exists; not wired into default path).
- Distributed clustering (`src/distributed`) remains a design prototype.
- GPU acceleration (`src/gpu`) falls back to CPU implementations.
- Realtime indexing (`src/realtime`) does not update the HNSW index in-place yet.
- WASM tooling and npm packaging still need work.
- Packaging directories (Homebrew, MacPorts, AUR, Nix, Scoop, Snap, Winget, Chocolatey) are templates with placeholder values.

## Recent Work
- Expanded query planning utilities and cost estimation helpers.
- Extended metadata filtering and namespace management to cover more production scenarios.
- Added more ingestion tooling (batch operations, loaders) to reduce external glue code.

## Known Gaps and Priorities
- Finish a robust distributed story (network transports, Raft integration, automatic rebalancing).
- Implement real GPU kernels and memory management for the CUDA/Metal backends.
- Hardening for long-running server deployments (observability, backpressure, resource accounting).
- Authoritative benchmarks that document baseline latency/throughput across representative workloads.

If you rely on a feature that is currently marked experimental, please open an issue so priorities can be adjusted accordingly.
