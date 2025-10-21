# Competitive Notes

This document summarises how VecStore compares to other vector databases as of October 2025. It is intentionally narrative instead of score-based so it can stay honest about strengths and gaps.

> Current release (0.0.1) is alpha quality; treat the comparisons below as directional rather than a guarantee of shipped functionality.
> This replaces the earlier metrics scorecard that lived in `COMPETITIVE-ANALYSIS.md`.

## Positioning

VecStore targets teams that need an embeddable library first and are willing to opt into a server or additional infrastructure later. The sweet spot is local RAG tooling, offline-friendly products, and environments where self-hosting is a hard requirement.

## Strengths

- **Embeddable workflow** – Rust crate with optional Python bindings makes it easy to bundle VecStore into desktop or CLI tooling without running a separate daemon.
- **Hybrid search toolkit** – Metadata filters, reranking strategies, and text splitters reduce the amount of glue code needed for RAG experiments.
- **Query planner and cost hints** – EXPLAIN-style helpers are rare in this space and help operators reason about vector queries.
- **Local observability hooks** – Everything uses the `tracing` crate, so you can plug into OpenTelemetry or logging sinks of your choice when needed.

## Limitations (Compared to Cloud-Focused Databases)

- **No managed cluster** – Qdrant, Pinecone, and Weaviate ship with distributed clustering and hosted offerings. VecStore’s distributed module is still a prototype.
- **GPU compute still on the roadmap** – FAISS and Marqo offer CUDA acceleration today; VecStore’s GPU module is currently stubs that fall back to CPU.
- **Operational polish** – Running VecStore as a long-lived server requires more manual setup for telemetry, capacity planning, and durability than cloud-hosted options provide out of the box.
- **Benchmark transparency** – Public benchmarks are in progress. If you need hard latency numbers, plan to run your own measurements.

## Practical Guidance

- Choose VecStore when you want to keep everything in-process or ship a single binary with embedded search and RAG helpers.
- Reach for hosted or cluster-first products when you need multi-tenant SaaS scale, cross-region replication, or managed GPU search today.
- Treat the experimental modules (distributed, GPU, realtime indexing) as design sketches until their README sections call them production-ready.

Contributions, benchmark results, and case studies are welcome—file an issue or open a discussion if you have data points to share.
