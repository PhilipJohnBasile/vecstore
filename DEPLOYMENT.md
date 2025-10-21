# Running VecStore 0.0.1 (alpha)

VecStore is an embedded library first. The feature-flagged server is provided for experimentation; treat it as an alpha component.

## 1. Embedded (recommended)

Add the crate to your project and open a file-backed store:

```rust
let mut store = VecStore::open("vectors.db")?;
store.upsert("doc1", &vec![0.1, 0.2, 0.3], metadata)?;
let results = store.query(&vec![0.15, 0.25, 0.85], 10, None)?;
```

Backups are ordinary file copies. To migrate between machines, close the process and copy the directory containing `vectors.db` (and any snapshots).

## 2. HTTP/gRPC Server (alpha)

Build with the `server` feature:

```bash
cargo build --release --features server --bin vecstore-server
./target/release/vecstore-server --db-path ./data/vectors.db --http-port 8080 --grpc-port 50051
```

Known limitations:

- Single-node only; no leader election or replication.
- Observability is basic – integrate with your own logging/metrics pipeline.
- Authentication/authorization is not implemented.

Use this mode only behind your own reverse proxy and supervision tooling.

## 3. Containers / Orchestration

Docker and Kubernetes manifests exist in the repository but are examples, not turnkey solutions. Expect to audit and adapt them before use.

## Operational Checklist

- Take regular snapshots (`create_snapshot`) if you rely on on-disk recovery.
- Monitor resource usage (CPU, memory, disk); there is no built-in limiter yet.
- Keep the original data source until you are confident the alpha meets your requirements.

For updates on stability work, follow the roadmap in [`docs/ROADMAP.md`](docs/ROADMAP.md).
