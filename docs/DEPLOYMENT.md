# VecStore Deployment Guide

Complete guide for deploying VecStore in production environments.

---

## Quick Start

### Local Development

```bash
# Build and run the server
cargo run --bin vecstore-server --features server -- --db-path vectors.db

# Server starts with:
# - gRPC on port 50051
# - HTTP/REST on port 8080
```

### Docker (Recommended for Production)

```bash
# Build the image
docker build -t vecstore:latest .

# Run with docker-compose
docker-compose up -d

# Check health
curl http://localhost:8080/health
```

---

## Deployment Options

### 1. Docker Deployment

**Dockerfile** (included in repository):
```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin vecstore-server --features server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/vecstore-server /usr/local/bin/
EXPOSE 50051 8080
CMD ["vecstore-server", "--db-path", "/data/vectors.db"]
```

**docker-compose.yml**:
```yaml
version: '3.8'
services:
  vecstore:
    image: vecstore:latest
    ports:
      - "50051:50051"  # gRPC
      - "8080:8080"    # HTTP
    volumes:
      - vecstore-data:/data
    environment:
      - RUST_LOG=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  vecstore-data:
    driver: local
```

### 2. Kubernetes Deployment

**vecstore-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vecstore
  labels:
    app: vecstore
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
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
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
  type: LoadBalancer
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
      storage: 50Gi
  storageClassName: standard
```

**Apply the deployment**:
```bash
kubectl apply -f vecstore-deployment.yaml

# Check status
kubectl get pods -l app=vecstore
kubectl logs -l app=vecstore
```

### 3. Standalone Binary

```bash
# Build release binary
cargo build --release --bin vecstore-server --features server

# Run binary
./target/release/vecstore-server \
    --db-path /var/lib/vecstore/vectors.db \
    --grpc-port 50051 \
    --http-port 8080
```

---

## Configuration

### Command-Line Options

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

- `RUST_LOG` - Log level (trace, debug, info, warn, error) [default: info]
- `DB_PATH` - Database file path [default: vecstore.db]

---

## API Reference

### gRPC API

Service definition: `proto/vecstore.proto`

**Example: Python gRPC Client**:
```python
import grpc
from vecstore_pb2 import UpsertRequest, QueryRequest
from vecstore_pb2_grpc import VecStoreServiceStub

channel = grpc.insecure_channel('localhost:50051')
client = VecStoreServiceStub(channel)

# Upsert
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
```

### HTTP/REST API

Base URL: `http://localhost:8080`

**Core Endpoints:**

```bash
# Upsert vector
POST /v1/upsert
{
  "id": "doc1",
  "vector": [0.1, 0.2, 0.9],
  "metadata": {"title": "Document"}
}

# Query vectors
POST /v1/query
{
  "vector": [0.15, 0.25, 0.85],
  "limit": 10,
  "filter": "category = 'tech'"
}

# Batch upsert
POST /v1/batch-upsert
{
  "records": [...]
}

# Delete
DELETE /v1/delete/{id}

# Get statistics
GET /v1/stats

# Health check
GET /health
GET /ready
```

See [SERVER.md](../SERVER.md) for complete API documentation.

---

## Multi-Tenant Namespaces

VecStore supports multi-tenant deployments with isolated namespaces.

### Creating Namespaces

```bash
# Create namespace with quotas
curl -X POST http://localhost:8080/admin/namespaces \
  -H "Content-Type: application/json" \
  -d '{
    "id": "customer-123",
    "name": "Customer 123",
    "quotas": {
      "max_vectors": 10000,
      "max_storage_bytes": 104857600,
      "max_requests_per_second": 10.0,
      "max_concurrent_queries": 2,
      "max_dimension": 1536
    }
  }'
```

### Using Namespaces

```bash
# Upsert to namespace
curl -X POST http://localhost:8080/v1/upsert \
  -d '{
    "namespace": "customer-123",
    "id": "doc1",
    "vector": [0.1, 0.2, 0.3],
    "metadata": {"title": "Doc 1"}
  }'

# Query namespace
curl -X POST http://localhost:8080/v1/query \
  -d '{
    "namespace": "customer-123",
    "vector": [0.15, 0.25, 0.35],
    "limit": 10
  }'
```

See [NAMESPACES.md](../NAMESPACES.md) for complete namespace documentation.

---

## Monitoring & Observability

### Prometheus Metrics

VecStore exports Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8080/metrics
```

**Key Metrics:**
- `vecstore_queries_total` - Total queries
- `vecstore_query_duration_seconds` - Query latency
- `vecstore_vectors_total` - Vector count
- `vecstore_cache_hits_total` / `vecstore_cache_misses_total` - Cache performance
- `vecstore_namespace_vector_count{namespace}` - Per-namespace counts

### Observability Stack

```bash
cd observability
docker-compose -f docker-compose-monitoring.yml up -d
```

**Services:**
- **VecStore**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

**Pre-built Grafana Dashboard** includes:
- Query performance (latency, throughput)
- Vector operations metrics
- Cache hit rates
- Per-namespace statistics

See [observability/README.md](../observability/README.md) for complete monitoring setup.

### Alerting

**Example Prometheus Alerts**:
```yaml
groups:
  - name: vecstore_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(vecstore_errors_total[5m]) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: VecStoreDown
        expr: up{job="vecstore"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "VecStore is down"
```

---

## Security

### Current Limitations

⚠️ **To be addressed in future releases:**
- No built-in TLS/SSL
- No built-in authentication
- No built-in rate limiting

### Recommended Production Setup

Use a reverse proxy (Nginx, Traefik, Envoy) for security:

```
Internet → Nginx/Traefik (TLS + Auth) → VecStore Server
```

**Example Nginx Config**:
```nginx
upstream vecstore {
    server vecstore:50051;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Basic auth
    auth_basic "VecStore API";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        grpc_pass grpc://vecstore;
    }
}
```

---

## Performance Tuning

### Resource Requirements

**Memory:**
- Base overhead: ~100MB
- Per-vector overhead: ~4 bytes × dimension
- HNSW index: ~20 bytes per vector
- Example: 1M vectors × 384 dim = ~1.5GB RAM

**CPU:**
- Server uses Rayon for parallelism
- Allocate multiple cores for best performance
- 4-8 cores recommended for production

**Storage:**
- Use SSD for best I/O performance
- HDD acceptable for read-heavy workloads

### Tuning Parameters

**HNSW Index**:
```rust
// In code (future config file support)
let config = Config {
    hnsw_m: 16,              // Connections per node (higher = faster queries)
    hnsw_ef_construction: 200, // Build quality (higher = better but slower)
    hnsw_ef: 50,             // Query quality (higher = more accurate)
};
```

**Product Quantization**:
```rust
let config = Config {
    use_pq: true,            // Enable compression
    pq_m: 8,                 // Number of subquantizers
};
```

### Connection Pooling

**gRPC** supports connection pooling - configure on client side:

```python
# Python gRPC client
channel = grpc.insecure_channel(
    'localhost:50051',
    options=[
        ('grpc.max_connection_idle_ms', 60000),
        ('grpc.keepalive_time_ms', 30000),
    ]
)
```

---

## Backup & Recovery

### Manual Backup

```bash
# Stop server
docker-compose stop vecstore

# Backup data directory
tar -czf vecstore-backup-$(date +%Y%m%d).tar.gz /path/to/data

# Restart server
docker-compose start vecstore
```

### Automated Backup

**Using cron**:
```bash
# /etc/cron.daily/vecstore-backup
#!/bin/bash
tar -czf /backups/vecstore-$(date +%Y%m%d).tar.gz /var/lib/vecstore/data
find /backups -name "vecstore-*.tar.gz" -mtime +30 -delete
```

### Restore from Backup

```bash
# Stop server
docker-compose stop vecstore

# Restore data
tar -xzf vecstore-backup-20250119.tar.gz -C /

# Restart server
docker-compose start vecstore
```

### Namespace-Specific Backup

```bash
# Backup single namespace
tar -czf customer-123-backup.tar.gz /data/customer-123/

# Restore single namespace
tar -xzf customer-123-backup.tar.gz -C /data/
```

---

## Scaling

### Vertical Scaling

**Increase resources:**
- Add more RAM for larger datasets
- Add more CPU cores for higher throughput
- Use faster storage (NVMe SSD)

### Horizontal Scaling (Future)

**Current limitation:** Single-node only

**Planned features:**
- Read replicas
- Sharding
- Distributed queries
- Consensus (Raft)

**Current workaround:** Use multiple independent instances with client-side routing

---

## Troubleshooting

### Server Won't Start

```bash
# Check logs
docker logs vecstore-server

# Check ports
lsof -i :50051
lsof -i :8080

# Verify database
ls -lah /data/vectors.db
```

### Connection Refused

```bash
# Test gRPC
grpcurl -plaintext localhost:50051 list

# Test HTTP
curl http://localhost:8080/health
```

### Performance Issues

1. **Check database size**: `ls -lh vectors.db`
2. **Monitor memory**: `docker stats vecstore-server`
3. **Enable debug logging**: `RUST_LOG=debug`
4. **Profile with flamegraph**: `cargo flamegraph`

### High Memory Usage

- Enable product quantization
- Reduce HNSW parameters
- Compact database regularly

### Slow Queries

- Check Prometheus metrics for bottlenecks
- Verify HNSW index is built
- Consider caching frequently accessed results

---

## Production Checklist

- [ ] Use Docker or Kubernetes deployment
- [ ] Configure persistent storage
- [ ] Set up reverse proxy with TLS
- [ ] Configure Prometheus monitoring
- [ ] Set up Grafana dashboards
- [ ] Configure alerting rules
- [ ] Set up automated backups
- [ ] Test disaster recovery procedures
- [ ] Configure resource limits
- [ ] Enable health checks
- [ ] Set up logging aggregation
- [ ] Document runbooks for common issues
- [ ] Load test before going live

---

## Next Steps

1. Choose deployment method (Docker recommended)
2. Set up monitoring (Prometheus + Grafana)
3. Configure backups
4. Set up reverse proxy for security
5. Load test your deployment
6. Monitor metrics and adjust resources

---

For more information:
- **Server API**: [SERVER.md](../SERVER.md)
- **Namespaces**: [NAMESPACES.md](../NAMESPACES.md)
- **Monitoring**: [observability/README.md](../observability/README.md)
- **Development**: [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md)
