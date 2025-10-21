# VecStore Architecture

> **Status:** Information reflects the 0.0.1 alpha release. Interfaces and file formats may change.

This note captures how the current codebase is structured so that future doc updates can refer to the real implementation rather than the earlier marketing copy.

## 1. Core Engine (`src/store`)

| Component | What it does | Key Notes |
|-----------|--------------|-----------|
| `VecStore` | Owns vectors, metadata, the on-disk layout, and the HNSW index | Operates in-process and expects exclusive mutable access. |
| `disk.rs` | Persists records, ID mappings, and configuration to JSON + bincode files | Snapshots reuse the same layout under `snapshots/<name>`. |
| `hnsw_backend.rs` | Wraps [`hnsw_rs`](https://github.com/jerry73204/hnsw-rs) for search | Only `Cosine`, `Euclidean`, and `DotProduct` distances are wired up today. |
| `hybrid.rs` | Maintains an inverted index and BM25 scorer for keyword queries | The default tokenizer is “Simple”; pluggable tokenizers live under `src/tokenizer`. |
| `quantization.rs` | Provides a product-quantization helper that callers can opt into | Not automatically engaged by `VecStore`; applications call it explicitly. |
| `filters.rs` | Evaluates SQL-like filter ASTs produced by `filter_parser.rs` | Supports `=`, `!=`, `<`, `<=`, `>`, `>=`, `IN`, `NOT IN`, `CONTAINS`, and boolean operators. |

Insert/query flow:
1. `VecStore::upsert` persists the record (JSON metadata + vector) and feeds the vector into the HNSW index. The first insert establishes the dimension and distance metric.
2. `VecStore::query` invokes the HNSW backend, over-fetches candidates if a filter exists, applies `filters::evaluate_filter`, and returns a `Neighbor` list.
3. `VecStore::hybrid_query` blends BM25 scores from the text index with vector similarity (weighted by `alpha`).

Concurrency: `VecStore` itself is synchronous and not thread-safe. Higher-level wrappers (`AsyncVecStore`, `VecStoreHttpServer`, `VecStoreGrpcServer`) coordinate access through `Arc<RwLock<_>>` and offload blocking work via `tokio::task::spawn_blocking`.

On-disk layout (`DiskLayout`):
```
<root>/
  manifest.json      // schema version, dimension, next_idx, config
  vectors.bin        // JSON array of `Record`
  meta.bin           // bincode-encoded ID/index maps and next index
  hnsw.idx           // persisted HNSW graph (optional)
  text_index.json    // keyword index export (optional)
  snapshots/
    <name>/
      ... same layout as above ...
```

## 2. Supporting Abstractions

| Module | Role | Notes |
|--------|------|-------|
| `namespace.rs` / `namespace_manager.rs` | Define namespace metadata, quotas, and manage a `VecStore` per namespace | Used by `VecDatabase` to offer a Chroma/Qdrant-style “collections” API. |
| `collection.rs` | High-level multi-collection API backed by the namespace manager | Each collection is a separate directory + `VecStore`. |
| `async_api.rs` | Async façade wrapping `VecStore` inside an `Arc<RwLock<_>>` | Every operation delegates to a blocking task. |
| `python.rs` | PyO3 bindings exposing `VecStore`, `VecDatabase`, queries, and text splitters | Keeps metadata as JSON-compatible types; heavy work still happens in Rust. |
| `server/http.rs` | Axum-based HTTP/REST API exposing CRUD, search, snapshots, and metrics | Serves `/metrics`, `/health`, and `/ws/query-stream` routes. |
| `server/grpc.rs` | Tonic-based gRPC service that mirrors the HTTP functionality | Shares the same `Arc<RwLock<VecStore>>` as the HTTP layer when both are enabled. |
| `metrics.rs` | In-memory counters for query latency, throughput, cache stats | No Prometheus endpoint baked in; callers export snapshots themselves. |
| `query_optimizer.rs` | Estimates cost and suggests tuning hints based on store size and query | Uses heuristics; results are informative rather than authoritative. |

## 3. RAG Utilities

Large portions of `src/` are devoted to tooling around the core engine: text splitters, reranking, evaluation helpers, embeddings adapters, etc. They are organised so that production code can depend on just the pieces it needs:

- `text_splitter.rs` and `tokenizer.rs` implement chunking strategies.
- `rag_utils.rs`, `reranking/`, `graph_rag.rs`, and `semantic_cache.rs` provide retrieval helpers.
- `embeddings.rs` (behind feature flags) integrates with ONNX Runtime or other providers.

These modules are broadly “opt-in”: they do not run automatically inside `VecStore`.

## 4. Persistence & Recovery

- Snapshots (`VecStore::create_snapshot`) copy the on-disk layout to `snapshots/<name>`. Restore simply re-loads that layout.
- There is a standalone write-ahead log implementation (`wal.rs`) with comprehensive tests, but the default `VecStore` does not invoke it yet. Integrating the WAL is a future improvement.

## 5. Observability

- The `metrics` module maintains counters in memory and exposes a `snapshot()` for reporting. There is no direct Prometheus or OpenTelemetry exporter baked into the crate.
- `telemetry.rs` wires up `tracing_subscriber` with JSON or human-readable output. To bridge into OpenTelemetry you bring your own `tracing-opentelemetry` configuration, as shown in the examples.

## 6. Experimental/Prototype Modules

Some directories exist to outline future directions; doc comments inside them already carry ⚠️ warnings. They are *not* production-ready today:

- `distributed/` – sketches a coordinator + shard model and an in-progress Raft node. Networking, replication, and durability are stubbed.
- `realtime.rs` – models incremental indexing with buffers and background workers but does not update the HNSW graph yet.
- `gpu/` – contains CUDA/Metal/WebGPU scaffolding; kernels currently return placeholder results.
- `observability/`, `monitoring.rs`, `analytics.rs`, and parts of `rate_limit.rs` describe systems that still need integration.

When updating docs or marketing material, keep these modules in the “roadmap” bucket unless the implementation actually wires them into the default execution path.

## 7. Known Limitations (Documented in Code)

- Only a single writer at a time; no concurrent mutation of `VecStore`.
- HNSW configuration values (`M`, `ef_construction`, `ef_search`) are fixed inside the backend wrapper—callers cannot tune them per-query yet.
- Distance metrics beyond cosine / Euclidean / dot product trigger a warning and fall back to cosine.
- The on-disk layout uses JSON for records; large datasets will benefit from a more compact format.
- No automatic background compaction. Call `optimize()` manually after heavy delete workloads.
- Live server deployments rely on the embedding process for durability (no WAL integration, no replication).

Refer to this file whenever you need the “source of truth” about what currently ships. Update it alongside any major code changes so docs stay honest.
