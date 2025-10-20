# 🚀 vecstore Roadmap v3.0 - Path to Production Leadership

**Current Status**: v0.1 Complete — 125 tests passing, production-ready embedded database

**Vision**: Transform vecstore from the best **embedded** vector database into a **production-grade platform** that competes with Qdrant, Milvus, and Pinecone.

---

## 🎯 The Final Three - Production Transformation

These three features will make vecstore **deployable at scale** and accessible to **any programming language**.

---

### **1. HTTP/gRPC Server Mode** 🌐
**Priority**: CRITICAL | **Impact**: MASSIVE | **Complexity**: Medium
**Status**: Not Started | **Target**: v0.2

#### Why This Matters

Currently, vecstore is **Rust-only** (with optional Python/WASM bindings). To compete with Qdrant/Milvus/Pinecone, we need:

- ✅ **Polyglot Access** — Any language can use vecstore (Go, Java, JS, etc.)
- ✅ **No Language Bindings** — Pure network protocol, no PyO3/Neon complexity
- ✅ **Deployment Flexibility** — Run as a service, container, or embedded
- ✅ **Competitive Parity** — Industry standard for vector databases

#### Architecture

```
┌──────────────────────────────────────────────────┐
│           Client Applications                     │
│  (Python, Go, Java, JavaScript, Ruby, etc.)      │
└──────────────────────────────────────────────────┘
         │              │              │
    ┌────┴────┐    ┌────┴────┐   ┌────┴────┐
    │  gRPC   │    │  HTTP   │   │WebSocket│
    │ :50051  │    │  :8080  │   │  :8081  │
    └────┬────┘    └────┬────┘   └────┬────┘
         └──────────────┴─────────────┘
                      │
         ┌────────────────────────────┐
         │   vecstore Server (Rust)   │
         │  • Connection Pool         │
         │  • Rate Limiting           │
         │  • TLS/SSL                 │
         │  • Auth Middleware         │
         └────────────────────────────┘
                      │
         ┌────────────────────────────┐
         │    VecStore Core Engine    │
         │  (All the Rust goodness)   │
         └────────────────────────────┘
```

#### Implementation Plan

**Phase 1: gRPC Server** (Week 1)
```rust
// proto/vecstore.proto
service VecStoreService {
    rpc Upsert(UpsertRequest) returns (UpsertResponse);
    rpc Query(QueryRequest) returns (QueryResponse);
    rpc Delete(DeleteRequest) returns (DeleteResponse);
    rpc SoftDelete(SoftDeleteRequest) returns (SoftDeleteResponse);
    rpc Compact(CompactRequest) returns (CompactResponse);
    rpc CreateSnapshot(SnapshotRequest) returns (SnapshotResponse);
}

// src/server/grpc.rs
use tonic::{transport::Server, Request, Response, Status};

#[tonic::async_trait]
impl VecStoreService for VecStoreServer {
    async fn query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let query_req = request.into_inner();
        let store = self.store.read().await;

        let query = Query {
            vector: query_req.vector,
            k: query_req.limit,
            filter: query_req.filter.map(|f| parse_filter(&f)),
        };

        let results = store.query(query)?;
        Ok(Response::new(QueryResponse { results }))
    }
}
```

**Phase 2: HTTP/REST API** (Week 2)
```rust
// src/server/http.rs
use axum::{
    routing::{get, post},
    Router, Json, extract::State,
};

async fn query_handler(
    State(store): State<Arc<RwLock<VecStore>>>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, StatusCode> {
    let store = store.read().await;
    let results = store.query(payload.into())?;
    Ok(Json(QueryResponse { results }))
}

let app = Router::new()
    .route("/v1/query", post(query_handler))
    .route("/v1/upsert", post(upsert_handler))
    .route("/v1/delete", post(delete_handler))
    .route("/health", get(health_check))
    .route("/metrics", get(metrics_handler))
    .with_state(store_state);
```

**Phase 3: WebSocket Streaming** (Week 3)
```rust
// src/server/websocket.rs
async fn stream_query_handler(
    ws: WebSocketUpgrade,
    State(store): State<Arc<RwLock<VecStore>>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_query_stream(socket, store))
}

async fn handle_query_stream(
    mut socket: WebSocket,
    store: Arc<RwLock<VecStore>>,
) {
    // Stream query results incrementally
    let mut stream = store.query_stream(query)?;

    while let Some(result) = stream.next() {
        socket.send(Message::Text(serde_json::to_string(&result)?)).await?;
    }
}
```

#### Client Libraries

All clients are **thin wrappers** around gRPC/HTTP:

**Python Client**:
```python
# vecstore-client-py (separate repo)
from vecstore import VecStoreClient

client = VecStoreClient("localhost:50051")
results = client.query([0.1, 0.2, 0.3], limit=10)
```

**JavaScript Client**:
```typescript
// vecstore-client-js (separate repo)
import { VecStoreClient } from '@vecstore/client';

const client = new VecStoreClient('localhost:50051');
const results = await client.query([0.1, 0.2, 0.3], { limit: 10 });
```

**Go Client**:
```go
// vecstore-client-go (separate repo)
import "github.com/vecstore/client-go"

client := vecstore.NewClient("localhost:50051")
results, err := client.Query([]float32{0.1, 0.2, 0.3}, 10)
```

#### Deployment

```bash
# Binary mode
vecstore serve \
    --grpc 0.0.0.0:50051 \
    --http 0.0.0.0:8080 \
    --data ./data \
    --tls-cert cert.pem \
    --tls-key key.pem

# Docker mode
docker run -p 50051:50051 -p 8080:8080 \
    -v $(pwd)/data:/data \
    vecstore/vecstore:latest

# Kubernetes (Helm chart)
helm install vecstore vecstore/vecstore \
    --set replicas=3 \
    --set storage.size=100Gi
```

#### Files to Create
- `proto/vecstore.proto` — gRPC service definition
- `src/server/mod.rs` — Server entrypoint
- `src/server/grpc.rs` — gRPC implementation (tonic)
- `src/server/http.rs` — HTTP/REST implementation (axum)
- `src/server/websocket.rs` — WebSocket streaming
- `src/server/auth.rs` — Authentication middleware
- `src/server/rate_limit.rs` — Rate limiting
- `Dockerfile` — Container image
- `helm/vecstore/` — Kubernetes Helm chart

---

### **2. Production Observability Stack** 📊
**Priority**: CRITICAL | **Impact**: HIGH | **Complexity**: Low-Medium
**Status**: Not Started | **Target**: v0.2

#### Why This Matters

You can't run production systems without observability:

- 🔍 **Debugging** — Understand why queries are slow
- 📈 **Monitoring** — Track QPS, latency, cache hits
- 🚨 **Alerting** — Know when things break BEFORE users complain
- 📊 **Capacity Planning** — Predict when to scale

#### Architecture

```
┌─────────────────────────────────────────────┐
│         vecstore Application                 │
│  • Instrumented with tracing spans          │
│  • Emits Prometheus metrics                 │
│  • Structured JSON logs                     │
└─────────────────────────────────────────────┘
         │              │              │
    ┌────┴────┐    ┌────┴────┐   ┌────┴────┐
    │Prometheus│   │  Jaeger  │   │  Loki   │
    │ Metrics  │   │ Traces   │   │  Logs   │
    └────┬────┘   └────┬────┘   └────┬────┘
         └──────────────┴─────────────┘
                      │
         ┌────────────────────────────┐
         │   Grafana Dashboards       │
         │  • Query Performance       │
         │  • System Health           │
         │  • Business Metrics        │
         └────────────────────────────┘
```

#### Implementation Plan

**Phase 1: Prometheus Metrics** (Day 1)
```rust
// src/observability/metrics.rs
use prometheus::{
    Registry, Counter, Histogram, Gauge,
    HistogramOpts, Opts,
};

pub struct VecStoreMetrics {
    // Query metrics
    pub query_total: Counter,
    pub query_duration_seconds: Histogram,
    pub query_results_count: Histogram,

    // Cache metrics
    pub cache_hits_total: Counter,
    pub cache_misses_total: Counter,
    pub cache_size_bytes: Gauge,

    // Storage metrics
    pub vectors_total: Gauge,
    pub deleted_vectors_total: Gauge,
    pub storage_bytes: Gauge,

    // HNSW metrics
    pub hnsw_nodes_visited: Histogram,
    pub hnsw_distance_calculations: Histogram,

    // System metrics
    pub active_connections: Gauge,
    pub memory_usage_bytes: Gauge,
}

impl VecStoreMetrics {
    pub fn new(registry: &Registry) -> Result<Self> {
        let query_total = Counter::with_opts(
            Opts::new("vecstore_queries_total", "Total queries")
        )?;
        registry.register(Box::new(query_total.clone()))?;

        let query_duration = Histogram::with_opts(
            HistogramOpts::new(
                "vecstore_query_duration_seconds",
                "Query latency histogram"
            ).buckets(vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0
            ])
        )?;
        registry.register(Box::new(query_duration.clone()))?;

        // ... register all metrics

        Ok(Self {
            query_total,
            query_duration_seconds: query_duration,
            // ...
        })
    }
}

// Instrument VecStore methods
impl VecStore {
    pub fn query(&self, q: Query) -> Result<Vec<Neighbor>> {
        let timer = self.metrics.query_duration_seconds.start_timer();
        self.metrics.query_total.inc();

        let results = self.query_impl(q)?;

        timer.observe_duration();
        self.metrics.query_results_count.observe(results.len() as f64);

        Ok(results)
    }
}
```

**Phase 2: OpenTelemetry Tracing** (Day 2)
```rust
// src/observability/tracing.rs
use tracing::{instrument, info, warn};
use tracing_subscriber::layer::SubscriberExt;
use opentelemetry::trace::TracerProvider;

#[instrument(skip(self))]
pub fn query(&self, q: Query) -> Result<Vec<Neighbor>> {
    let span = tracing::span!(
        tracing::Level::INFO,
        "vecstore_query",
        query_id = %uuid::Uuid::new_v4(),
        vector_dim = q.vector.len(),
        k = q.k,
    );

    let _enter = span.enter();

    // HNSW search span
    let candidates = {
        let _hnsw_span = tracing::span!(
            tracing::Level::DEBUG,
            "hnsw_search"
        ).entered();

        self.backend.search(&q.vector, q.k)?
    };

    info!(
        candidates_found = candidates.len(),
        "HNSW search complete"
    );

    // Filter span
    let results = {
        let _filter_span = tracing::span!(
            tracing::Level::DEBUG,
            "apply_filters",
            filter_expr = ?q.filter
        ).entered();

        self.apply_filters(candidates, &q)?
    };

    info!(
        results_count = results.len(),
        "Query complete"
    );

    Ok(results)
}
```

**Phase 3: Structured Logging** (Day 3)
```rust
// src/observability/logging.rs
use tracing::{info, warn, error};
use serde_json::json;

// Structured log on query
info!(
    event = "query_complete",
    query_id = %query_id,
    duration_ms = elapsed.as_millis(),
    results_count = results.len(),
    cache_hit = cache_hit,
    filter_selectivity = filter_selectivity,
    user_id = %user_id,
    namespace = %namespace,
);

// Error logging
error!(
    event = "query_error",
    error = %err,
    query_id = %query_id,
    backtrace = ?err.backtrace(),
);
```

**Phase 4: Health Checks** (Day 4)
```rust
// src/server/health.rs
pub struct HealthCheck {
    store: Arc<RwLock<VecStore>>,
}

impl HealthCheck {
    pub async fn liveness(&self) -> HealthStatus {
        // Is the service running?
        HealthStatus {
            status: "healthy",
            timestamp: Utc::now(),
        }
    }

    pub async fn readiness(&self) -> HealthStatus {
        // Can we serve traffic?
        let store = self.store.read().await;

        // Check: Can we perform a simple query?
        let test_query = Query::new(vec![0.0; store.dimension()]);
        match store.query(test_query) {
            Ok(_) => HealthStatus {
                status: "ready",
                timestamp: Utc::now(),
            },
            Err(e) => HealthStatus {
                status: "not_ready",
                error: Some(e.to_string()),
                timestamp: Utc::now(),
            },
        }
    }
}
```

#### Grafana Dashboard

**Included Dashboard** (`grafana/vecstore-dashboard.json`):

**Row 1: Query Performance**
- Query rate (QPS)
- Query latency (p50, p95, p99)
- Query errors rate

**Row 2: Cache Performance**
- Cache hit rate (%)
- Cache size (MB)
- Cache evictions/sec

**Row 3: Storage**
- Total vectors
- Storage size (GB)
- Deleted vectors count

**Row 4: System Health**
- Memory usage
- Active connections
- HNSW nodes visited

#### Files to Create
- `src/observability/mod.rs`
- `src/observability/metrics.rs` — Prometheus metrics
- `src/observability/tracing.rs` — OpenTelemetry
- `src/observability/logging.rs` — Structured logs
- `src/server/health.rs` — Health check endpoints
- `grafana/vecstore-dashboard.json` — Pre-built dashboard
- `docker-compose.yml` — Observability stack (Prometheus, Grafana, Jaeger)

---

### **3. Multi-Tenant Namespaces & Isolation** 🏢
**Priority**: HIGH | **Impact**: VERY HIGH for SaaS | **Complexity**: Medium
**Status**: Not Started | **Target**: v0.2

#### Why This Matters

**SaaS Critical**: Run one vecstore instance for many customers:

- 🔒 **Data Isolation** — Customer A can't see Customer B's vectors
- 💰 **Cost Efficiency** — No separate DB per tenant
- 📊 **Resource Quotas** — Prevent abuse, control costs
- 🎯 **Usage Tracking** — Per-tenant metrics and billing

#### Architecture

```
┌────────────────────────────────────────┐
│      vecstore Multi-Tenant Server      │
└────────────────────────────────────────┘
         │
         ├── Namespace: "acme_corp"
         │   ├── Quota: 1M vectors, 1000 QPS
         │   ├── Storage: /data/ns/acme_corp/
         │   └── Metrics: isolated
         │
         ├── Namespace: "startup_xyz"
         │   ├── Quota: 100K vectors, 100 QPS
         │   ├── Storage: /data/ns/startup_xyz/
         │   └── Metrics: isolated
         │
         └── Namespace: "enterprise_abc"
             ├── Quota: 10M vectors, 10K QPS
             ├── Storage: /data/ns/enterprise_abc/
             └── Metrics: isolated
```

#### Implementation Plan

**Phase 1: Namespace Core** (Day 1)
```rust
// src/store/namespace.rs
pub struct NamespacedVecStore {
    root: PathBuf,
    namespaces: HashMap<String, VecStore>,
    quotas: HashMap<String, Quota>,
    usage: HashMap<String, Usage>,
}

impl NamespacedVecStore {
    pub fn create_namespace(
        &mut self,
        name: &str,
        quota: Quota,
    ) -> Result<()> {
        // Validate namespace name
        if !Self::is_valid_namespace_name(name) {
            return Err(anyhow!("Invalid namespace name"));
        }

        // Create isolated storage
        let ns_path = self.root.join("namespaces").join(name);
        fs::create_dir_all(&ns_path)?;

        // Initialize VecStore for this namespace
        let store = VecStore::open(ns_path)?;

        self.namespaces.insert(name.to_string(), store);
        self.quotas.insert(name.to_string(), quota);
        self.usage.insert(name.to_string(), Usage::default());

        Ok(())
    }

    pub fn upsert(
        &mut self,
        namespace: &str,
        id: String,
        vector: Vec<f32>,
        metadata: Metadata,
    ) -> Result<()> {
        // Check quota BEFORE executing
        self.check_quota(namespace)?;

        // Get namespace store
        let store = self.namespaces.get_mut(namespace)
            .ok_or_else(|| anyhow!("Namespace not found: {}", namespace))?;

        // Execute operation
        store.upsert(id, vector, metadata)?;

        // Update usage tracking
        self.update_usage(namespace);

        Ok(())
    }

    fn check_quota(&self, namespace: &str) -> Result<()> {
        let quota = self.quotas.get(namespace)
            .ok_or_else(|| anyhow!("No quota for namespace"))?;

        let usage = self.usage.get(namespace)
            .ok_or_else(|| anyhow!("No usage data"))?;

        // Check vector count
        if usage.vector_count >= quota.max_vectors {
            return Err(anyhow!("Quota exceeded: max vectors"));
        }

        // Check QPS
        if usage.current_qps >= quota.max_queries_per_second {
            return Err(anyhow!("Quota exceeded: max QPS"));
        }

        // Check storage
        if usage.storage_bytes >= quota.max_storage_bytes {
            return Err(anyhow!("Quota exceeded: max storage"));
        }

        Ok(())
    }
}
```

**Phase 2: Quota System** (Day 2)
```rust
// src/store/quota.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quota {
    pub max_vectors: usize,
    pub max_queries_per_second: usize,
    pub max_storage_bytes: u64,
    pub max_connections: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub vector_count: usize,
    pub current_qps: usize,
    pub storage_bytes: u64,
    pub active_connections: usize,

    // Rolling window for QPS calculation
    query_timestamps: VecDeque<Instant>,
}

impl Usage {
    pub fn record_query(&mut self) {
        let now = Instant::now();

        // Add current query
        self.query_timestamps.push_back(now);

        // Remove queries older than 1 second
        let cutoff = now - Duration::from_secs(1);
        while let Some(&oldest) = self.query_timestamps.front() {
            if oldest < cutoff {
                self.query_timestamps.pop_front();
            } else {
                break;
            }
        }

        // Update QPS
        self.current_qps = self.query_timestamps.len();
    }
}
```

**Phase 3: Admin API** (Day 3)
```rust
// src/server/admin.rs
async fn create_namespace_handler(
    State(store): State<Arc<RwLock<NamespacedVecStore>>>,
    Json(req): Json<CreateNamespaceRequest>,
) -> Result<Json<CreateNamespaceResponse>, StatusCode> {
    let mut store = store.write().await;

    store.create_namespace(&req.name, req.quota)?;

    Ok(Json(CreateNamespaceResponse {
        name: req.name,
        created_at: Utc::now(),
    }))
}

async fn list_namespaces_handler(
    State(store): State<Arc<RwLock<NamespacedVecStore>>>,
) -> Result<Json<Vec<NamespaceInfo>>, StatusCode> {
    let store = store.read().await;
    let namespaces = store.list_namespaces();

    Ok(Json(namespaces))
}

async fn get_namespace_stats_handler(
    Path(namespace): Path<String>,
    State(store): State<Arc<RwLock<NamespacedVecStore>>>,
) -> Result<Json<NamespaceStats>, StatusCode> {
    let store = store.read().await;
    let stats = store.get_stats(&namespace)?;

    Ok(Json(stats))
}
```

**Phase 4: Authentication** (Day 4)
```rust
// src/server/auth.rs
pub struct NamespaceAuth {
    api_keys: HashMap<String, String>, // api_key -> namespace
}

pub async fn auth_middleware(
    req: Request<Body>,
    next: Next<Body>,
) -> Result<Response, StatusCode> {
    // Extract API key from header
    let api_key = req.headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Validate and get namespace
    let namespace = validate_api_key(api_key)
        .ok_or(StatusCode::FORBIDDEN)?;

    // Inject namespace into request extensions
    req.extensions_mut().insert(namespace);

    Ok(next.run(req).await)
}
```

#### API Examples

**Create Namespace**:
```bash
curl -X POST http://localhost:8080/admin/namespaces \
  -H "Content-Type: application/json" \
  -d '{
    "name": "acme_corp",
    "quota": {
      "max_vectors": 1000000,
      "max_qps": 1000,
      "max_storage_bytes": 10737418240
    }
  }'
```

**Query with Namespace**:
```bash
curl -X POST http://localhost:8080/v1/query \
  -H "X-API-Key: acme_corp_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "limit": 10
  }'
```

**Get Namespace Stats**:
```bash
curl http://localhost:8080/admin/namespaces/acme_corp/stats
```

#### Files to Create
- `src/store/namespace.rs` — Namespace management
- `src/store/quota.rs` — Quota system
- `src/server/admin.rs` — Admin API endpoints
- `src/server/auth.rs` — API key authentication

---

## 📅 Implementation Timeline

**Week 1-2**: HTTP/gRPC Server Mode
- gRPC service (tonic)
- HTTP REST API (axum)
- WebSocket streaming
- Client libraries (Python, JS, Go)

**Week 3**: Production Observability
- Prometheus metrics
- OpenTelemetry tracing
- Health checks
- Grafana dashboards

**Week 4**: Multi-Tenant Namespaces
- Namespace core
- Quota system
- Admin API
- Authentication

**Total**: ~4 weeks to production-ready platform

---

## 🎯 Success Metrics

After completing these three features:

| Metric | Current (v0.1) | Target (v0.2) |
|--------|----------------|---------------|
| **Language Support** | Rust only | Any language (gRPC) |
| **Deployment Options** | Embedded only | Embedded + Server |
| **Observability** | Basic logs | Full stack (metrics/traces/logs) |
| **Multi-tenancy** | No | Full isolation + quotas |
| **Production Readiness** | 70% | 95% |

---

## 🚀 Beyond v0.2

**v0.3: Performance & Scale**
- GPU Acceleration (CUDA/Metal)
- Distributed Queries
- Sharding

**v0.4: Enterprise**
- Replication & Clustering
- Advanced RBAC
- Audit Logging

---

## 🤝 Contributing

Ready to help build the future of vector search? Pick a feature and let's go! 🚀

**Built with 🦀 Rust** | **Designed for Production** | **Made for Scale**
