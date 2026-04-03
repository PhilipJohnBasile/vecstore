# VecStore Roadmap

This roadmap captures focus areas for the next releases. It deliberately avoids scorecards and instead lists concrete work items with their current status.

## Guiding Principles
- **Embedded first** – ship a reliable, single-process library before scaling out.
- **Transparency** – call out caveats and unfinished work directly in the docs.
- **Pragmatism** – optimise developer experience for Rust/Python workflows and RAG prototyping.

## Current State (v0.1.0 - December 2025)

> VecStore is in alpha. Priorities below may change rapidly as the core stabilizes.
- HNSW engine with query planning, metadata filtering, batch ingestion, and namespace isolation is stable.
- Python bindings offer parity with the Rust API and are being exercised in tests/examples.
- Optional server mode exists behind a feature flag; it is suitable for controlled environments.
- GPU acceleration (CUDA, Metal, WebGPU) is implemented and feature-gated.
- Distributed mode with Raft consensus is implemented for multi-node deployments.
- WASM support enables browser-based vector search.

## Short-Term Priorities
1. **Distributed durability**
   - Replace the distributed skeleton with real networking, persistence, and leader election.
   - Integrate Raft consensus (log replication, membership changes, snapshotting).
2. **GPU acceleration**
   - Implement CUDA/Metal kernels for distance calculations, knn search, and batching.
   - Introduce feature flags and runtime detection to fall back gracefully when GPU is unavailable.
3. **Realtime indexing**
   - Support incremental index updates without full rebuilds.
   - Close the gap between the realtime buffer and the primary HNSW structure.
4. **Benchmark publishing**
   - Produce reproducible benchmark suites (Rust criterion harness + Python notebooks).
   - Publish datasets, hardware profiles, and baseline numbers.

## Medium-Term Themes
- Disk-backed indexing for larger-than-memory datasets.
- Multi-vector document storage (ColBERT-style late interaction built into the index).
- Connectors for streaming ingestion (Kafka, Redpanda, CDC from relational sources).
- Operational hardening: backpressure, resource quotas, structured logging, alert-friendly metrics.

## How to Influence the Roadmap
Open issues or discussion threads with concrete requirements, sample datasets, or production constraints. Input that comes with reproducible cases or early adopters willing to test pre-releases is especially helpful.
