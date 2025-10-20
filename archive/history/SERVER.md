# VecStore Server Deployment Guide

Complete guide for deploying VecStore in server mode with gRPC and HTTP/REST APIs.

## Quick Start

### Local Development

```bash
# Build and run the server
cargo run --bin vecstore-server --features server -- --db-path vectors.db

# Server starts with:
# - gRPC on port 50051
# - HTTP/REST on port 8080
```

### Docker Deployment

```bash
# Build the image
docker build -t vecstore:latest .

# Run with docker-compose
docker-compose up -d

# Check health
curl http://localhost:8080/health
```

## Server Configuration

### Command-line Options

```bash
vecstore-server [OPTIONS]

Options:
  -d, --db-path <PATH>        Path to vector database [default: vecstore.db]
      --grpc-port <PORT>      gRPC server port [default: 50051]
      --http-port <PORT>      HTTP server port [default: 8080]
      --dimension <DIM>       Vector dimension (for new databases)
      --debug                 Enable debug logging
      --no-grpc               Disable gRPC server
      --no-http               Disable HTTP server
  -h, --help                  Print help
```

### Environment Variables

- `RUST_LOG` - Log level (default: `info`)
- `DB_PATH` - Database file path

## API Reference

### gRPC API

Service definition: `proto/vecstore.proto`

#### Example: gRPC Client (Python)

```python
import grpc
from vecstore_pb2 import UpsertRequest, QueryRequest
from vecstore_pb2_grpc import VecStoreServiceStub

channel = grpc.insecure_channel('localhost:50051')
client = VecStoreServiceStub(channel)

# Upsert a vector
client.Upsert(UpsertRequest(
    id="doc1",
    vector=[0.1, 0.2, 0.9],
    metadata={"title": {"string_value": "Hello"}}
))

# Query
results = client.Query(QueryRequest(
    vector=[0.15, 0.25, 0.85],
    limit=10
))

for result in results.results:
    print(f"{result.id}: {result.score}")
```

### HTTP/REST API

Base URL: `http://localhost:8080`

#### Endpoints

##### Upsert Vector
```bash
POST /v1/upsert
Content-Type: application/json

{
  "id": "doc1",
  "vector": [0.1, 0.2, 0.9],
  "metadata": {
    "title": "My Document",
    "category": "tech"
  }
}
```

##### Query Vectors
```bash
POST /v1/query
Content-Type: application/json

{
  "vector": [0.15, 0.25, 0.85],
  "limit": 10,
  "filter": "category = 'tech'"
}
```

##### Batch Upsert
```bash
POST /v1/batch-upsert
Content-Type: application/json

{
  "records": [
    {
      "id": "doc1",
      "vector": [0.1, 0.2, 0.9],
      "metadata": {"title": "Doc 1"}
    },
    {
      "id": "doc2",
      "vector": [0.2, 0.3, 0.8],
      "metadata": {"title": "Doc 2"}
    }
  ]
}
```

##### Delete Vector
```bash
DELETE /v1/delete/doc1
```

##### Soft Delete
```bash
POST /v1/soft-delete/doc1
```

##### Restore
```bash
POST /v1/restore/doc1
```

##### Compact Database
```bash
POST /v1/compact
```

##### Get Statistics
```bash
GET /v1/stats
```

##### Hybrid Search
```bash
POST /v1/hybrid-query
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.9],
  "text_query": "machine learning",
  "limit": 10,
  "alpha": 0.7,
  "filter": "category = 'tech'"
}
```

##### Health Check
```bash
GET /health
GET /ready
```

### WebSocket Streaming

Connect to `ws://localhost:8080/ws/query-stream` for real-time query results.

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/query-stream');

ws.onopen = () => {
  // Send query
  ws.send(JSON.stringify({
    vector: [0.1, 0.2, 0.9],
    limit: 100
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.complete) {
    console.log('Query complete:', data.stats);
  } else {
    console.log('Result:', data.id, data.score);
  }
};
```

## Production Deployment

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vecstore
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vecstore
  template:
    metadata:
      labels:
        app: vecstore
    spec:
      containers:
      - name: vecstore
        image: vecstore:latest
        ports:
        - containerPort: 50051
          name: grpc
        - containerPort: 8080
          name: http
        volumeMounts:
        - name: data
          mountPath: /data
        env:
        - name: RUST_LOG
          value: "info"
        - name: DB_PATH
          value: "/data/vectors.db"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: vecstore-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: vecstore
spec:
  selector:
    app: vecstore
  ports:
  - name: grpc
    port: 50051
    targetPort: 50051
  - name: http
    port: 8080
    targetPort: 8080
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vecstore-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### Performance Tuning

1. **Memory**: Allocate sufficient RAM for HNSW index (roughly 4 bytes × dimensions × vectors)
2. **CPU**: Server uses rayon for parallel operations - allocate multiple cores
3. **Storage**: Use SSD for best I/O performance
4. **Connections**: gRPC supports connection pooling - configure client-side pooling

### Monitoring

#### Prometheus Metrics

VecStore exports Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8080/metrics
```

**Available Metrics:**
- `vecstore_upsert_total` - Total upserts performed
- `vecstore_query_total` - Total queries executed
- `vecstore_delete_total` - Total deletes
- `vecstore_query_duration_seconds` - Query latency histogram
- `vecstore_vectors_total` - Current vector count
- `vecstore_cache_hits_total` / `vecstore_cache_misses_total` - Cache performance
- `vecstore_namespace_vector_count{namespace}` - Per-namespace vector counts
- `vecstore_namespace_quota_utilization{namespace}` - Per-namespace quota usage

**Grafana Dashboard:**

Pre-built dashboards available in `observability/grafana-dashboard.json`:
- Query performance (latency, throughput, error rate)
- Vector operations (upserts, deletes)
- Cache hit rate
- Per-namespace metrics

See [observability/README.md](observability/README.md) for complete monitoring setup.

#### Logging
Set `RUST_LOG=debug` for detailed logs during development.

```bash
# Different log levels
RUST_LOG=info cargo run --bin vecstore-server --features server
RUST_LOG=debug cargo run --bin vecstore-server --features server
RUST_LOG=vecstore=trace cargo run --bin vecstore-server --features server
```

### Security Considerations

⚠️ **Current limitations** (to be addressed in future releases):

- No built-in authentication (use reverse proxy or service mesh)
- No TLS/SSL encryption (terminate SSL at load balancer)
- No rate limiting (implement at API gateway level)

**Recommended production setup:**

```
Internet → Nginx/Traefik (TLS) → VecStore Server
```

Example nginx config:
```nginx
upstream vecstore {
    server vecstore:50051;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        grpc_pass grpc://vecstore;
    }
}
```

## Multi-Tenant Namespaces

VecStore supports **multi-tenant deployments** where multiple isolated vector databases (namespaces) run within a single server instance. Each namespace has its own storage, quotas, and usage tracking.

### Creating Namespaces

#### via HTTP Admin API

```bash
# Create a namespace with free tier quotas
curl -X POST http://localhost:8080/admin/namespaces \
  -H "Content-Type: application/json" \
  -d '{
    "id": "customer-123",
    "name": "Customer 123 Workspace",
    "quotas": {
      "max_vectors": 10000,
      "max_storage_bytes": 104857600,
      "max_requests_per_second": 10.0,
      "max_concurrent_queries": 2,
      "max_dimension": 1536,
      "max_results_per_query": 100,
      "max_batch_size": 100
    }
  }'

# Create with pro tier quotas
curl -X POST http://localhost:8080/admin/namespaces \
  -H "Content-Type: application/json" \
  -d '{
    "id": "premium-customer",
    "name": "Premium Customer",
    "quotas": {
      "max_vectors": 1000000,
      "max_storage_bytes": 10737418240,
      "max_requests_per_second": 100.0,
      "max_concurrent_queries": 20,
      "max_dimension": 4096,
      "max_results_per_query": 1000,
      "max_batch_size": 1000
    }
  }'
```

#### via gRPC Admin API

```python
from vecstore_pb2 import CreateNamespaceRequest, NamespaceQuotas
from vecstore_pb2_grpc import VecStoreAdminServiceStub

admin_client = VecStoreAdminServiceStub(channel)

# Create namespace
response = admin_client.CreateNamespace(CreateNamespaceRequest(
    id="customer-456",
    name="Customer 456",
    quotas=NamespaceQuotas(
        max_vectors=50000,
        max_storage_bytes=524288000,  # 500MB
        max_requests_per_second=50.0
    )
))
```

### Using Namespaces

All vector operations accept an optional `namespace` parameter:

```bash
# Upsert to specific namespace
curl -X POST http://localhost:8080/v1/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "customer-123",
    "id": "doc1",
    "vector": [0.1, 0.2, 0.3],
    "metadata": {"title": "Document 1"}
  }'

# Query within namespace
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "customer-123",
    "vector": [0.15, 0.25, 0.35],
    "limit": 10
  }'

# Delete from namespace
curl -X DELETE http://localhost:8080/v1/vectors/doc1?namespace=customer-123
```

### Managing Namespaces

#### List All Namespaces

```bash
curl http://localhost:8080/admin/namespaces
```

Response:
```json
{
  "namespaces": [
    {
      "id": "customer-123",
      "name": "Customer 123 Workspace",
      "status": "active",
      "created_at": 1705968000,
      "quotas": {...}
    },
    ...
  ]
}
```

#### Get Namespace Details

```bash
curl http://localhost:8080/admin/namespaces/customer-123
```

#### Update Quotas

```bash
curl -X PUT http://localhost:8080/admin/namespaces/customer-123/quotas \
  -H "Content-Type: application/json" \
  -d '{
    "max_vectors": 100000,
    "max_storage_bytes": 1073741824
  }'
```

#### Update Status

Namespace statuses:
- **active** - Fully operational
- **suspended** - All requests rejected (quota violation, payment issue)
- **read_only** - Queries allowed, writes rejected
- **pending_deletion** - Marked for deletion

```bash
# Suspend a namespace
curl -X PUT http://localhost:8080/admin/namespaces/customer-123/status \
  -H "Content-Type: application/json" \
  -d '{"status": "suspended"}'

# Restore to active
curl -X PUT http://localhost:8080/admin/namespaces/customer-123/status \
  -H "Content-Type: application/json" \
  -d '{"status": "active"}'
```

#### Get Namespace Statistics

```bash
curl http://localhost:8080/admin/namespaces/customer-123/stats
```

Response:
```json
{
  "namespace_id": "customer-123",
  "vector_count": 5234,
  "active_count": 5100,
  "deleted_count": 134,
  "dimension": 384,
  "quota_utilization": 0.52,
  "total_requests": 15432,
  "total_queries": 12000,
  "total_upserts": 3200,
  "total_deletes": 232,
  "status": "active"
}
```

#### Get Aggregate Statistics

```bash
curl http://localhost:8080/admin/stats
```

Response:
```json
{
  "total_namespaces": 15,
  "active_namespaces": 12,
  "total_vectors": 125000,
  "total_requests": 500000
}
```

#### Delete Namespace

```bash
# Hard delete (permanent)
curl -X DELETE http://localhost:8080/admin/namespaces/customer-123
```

### Quota Enforcement

Quotas are enforced automatically on every operation:

**Vector Count Limit:**
```bash
# Will return 429 if quota exceeded
curl -X POST http://localhost:8080/v1/upsert \
  -d '{"namespace": "customer-123", "id": "doc1", "vector": [...]}'

# Response:
# HTTP 429 Too Many Requests
# {"error": "Vector quota exceeded: 10000 / 10000 vectors"}
```

**Rate Limit:**
```bash
# Returns 429 if too many requests per second
# Error: "Rate limit exceeded: 10 requests/second"
```

**Storage Limit:**
```bash
# Error: "Storage quota exceeded: 105000000 / 104857600 bytes"
```

**Concurrent Query Limit:**
```bash
# Error: "Concurrent query limit exceeded: 5 / 5"
```

### Quota Tiers

Pre-defined quota templates for common use cases:

#### Free Tier
```json
{
  "max_vectors": 10000,
  "max_storage_bytes": 104857600,        // 100 MB
  "max_requests_per_second": 10.0,
  "max_concurrent_queries": 2,
  "max_dimension": 1536,
  "max_results_per_query": 100,
  "max_batch_size": 100
}
```

#### Pro Tier
```json
{
  "max_vectors": 1000000,
  "max_storage_bytes": 10737418240,      // 10 GB
  "max_requests_per_second": 100.0,
  "max_concurrent_queries": 20,
  "max_dimension": 4096,
  "max_results_per_query": 1000,
  "max_batch_size": 1000
}
```

#### Enterprise Tier
```json
{
  "max_vectors": null,                   // Unlimited
  "max_storage_bytes": null,
  "max_requests_per_second": null,
  "max_concurrent_queries": null,
  "max_dimension": null,
  "max_results_per_query": null,
  "max_batch_size": null
}
```

### Multi-Tenancy Deployment Patterns

#### Pattern 1: Customer Isolation (SaaS)

One namespace per customer for complete data isolation:

```python
# On customer signup
def onboard_customer(customer_id, tier="free"):
    quotas = get_quotas_for_tier(tier)
    admin_client.CreateNamespace(CreateNamespaceRequest(
        id=f"customer-{customer_id}",
        name=f"Customer {customer_id}",
        quotas=quotas
    ))

# On customer upgrade
def upgrade_customer(customer_id, new_tier):
    quotas = get_quotas_for_tier(new_tier)
    admin_client.UpdateNamespaceQuotas(UpdateNamespaceQuotasRequest(
        namespace_id=f"customer-{customer_id}",
        quotas=quotas
    ))

# On customer deletion
def delete_customer(customer_id):
    admin_client.DeleteNamespace(DeleteNamespaceRequest(
        namespace_id=f"customer-{customer_id}"
    ))
```

#### Pattern 2: Environment Separation

Separate namespaces for dev/staging/prod:

```bash
# Create environments
for env in dev staging production; do
  curl -X POST http://localhost:8080/admin/namespaces \
    -H "Content-Type: application/json" \
    -d "{\"id\": \"$env\", \"name\": \"${env^} Environment\"}"
done

# Route requests based on environment
curl -X POST http://localhost:8080/query \
  -d '{"namespace": "'$ENVIRONMENT'", "vector": [...], "limit": 10}'
```

#### Pattern 3: Feature Flags

Use namespaces for A/B testing:

```python
def query_vectors(user, vector):
    namespace = "beta-features" if user.has_beta_access else "stable"

    return client.Query(QueryRequest(
        namespace=namespace,
        vector=vector,
        limit=10
    ))
```

### Persistence & Recovery

Each namespace is stored in its own directory:

```
/data/
├── customer-123/
│   ├── namespace.json      # Metadata (quotas, usage, status)
│   ├── vectors.bin         # Vector data
│   ├── metadata.bin        # Metadata storage
│   └── hnsw_index.bin      # HNSW graph
├── customer-456/
│   └── ...
```

**Backup Strategy:**
```bash
# Backup all namespaces
tar -czf namespaces-backup-$(date +%Y%m%d).tar.gz /data

# Restore specific namespace
tar -xzf backup.tar.gz -C /data customer-123/
```

### Monitoring Namespaces

**Prometheus Metrics:**
```
# Per-namespace vector counts
vecstore_namespace_vector_count{namespace="customer-123"} 5234

# Per-namespace quota utilization (0.0 - 1.0)
vecstore_namespace_quota_utilization{namespace="customer-123"} 0.52

# Total requests per namespace
vecstore_namespace_requests_total{namespace="customer-123"} 15432
```

**Alerting Rules (Prometheus):**
```yaml
groups:
  - name: vecstore_namespaces
    rules:
      - alert: NamespaceNearQuota
        expr: vecstore_namespace_quota_utilization > 0.8
        for: 5m
        annotations:
          summary: "Namespace {{ $labels.namespace }} is near quota ({{ $value | humanizePercentage }})"

      - alert: NamespaceQuotaExceeded
        expr: vecstore_namespace_quota_utilization >= 1.0
        annotations:
          summary: "Namespace {{ $labels.namespace }} has exceeded quota"
```

### Complete Namespace Guide

For detailed documentation including:
- Quota tier details
- Usage examples in multiple languages
- Advanced patterns
- Troubleshooting

See [NAMESPACES.md](NAMESPACES.md)

## Client Libraries

### Python

```bash
# Generate client from proto
python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. proto/vecstore.proto
```

### Node.js

```bash
# Install grpc tools
npm install @grpc/grpc-js @grpc/proto-loader

# Use proto loader
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

const packageDefinition = protoLoader.loadSync('proto/vecstore.proto');
const vecstore = grpc.loadPackageDefinition(packageDefinition).vecstore;

const client = new vecstore.VecStoreService(
    'localhost:50051',
    grpc.credentials.createInsecure()
);
```

### Go

```bash
# Install protoc-gen-go
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Generate client
protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    proto/vecstore.proto
```

## Troubleshooting

### Server won't start

```bash
# Check logs
docker logs vecstore-server

# Check ports
lsof -i :50051
lsof -i :8080

# Check database
ls -lah /data/vectors.db
```

### Connection refused

```bash
# Test gRPC
grpcurl -plaintext localhost:50051 list

# Test HTTP
curl http://localhost:8080/health
```

### Performance issues

1. Check database size: `ls -lh vectors.db`
2. Monitor memory: `docker stats vecstore-server`
3. Enable debug logging: `RUST_LOG=debug`
4. Profile with `cargo flamegraph`

## Roadmap

**✅ Completed:**
- ✅ Prometheus metrics — Exported at `/metrics` with Grafana dashboards
- ✅ Multi-tenant namespaces — Full namespace isolation with quotas and admin API

**Upcoming server features:**

- [ ] TLS/SSL support
- [ ] API key authentication
- [ ] JWT-based authentication
- [ ] OpenTelemetry tracing
- [ ] Horizontal scaling with shared storage
- [ ] Read replicas
- [ ] Connection pooling
- [ ] Request batching

See [ROADMAP_V3.md](./ROADMAP_V3.md) for full production roadmap.
