# 🚀 VecStore Production Deployment Guide

Complete guide for deploying VecStore in production environments.

---

## 📋 Table of Contents

1. [Deployment Modes](#deployment-modes)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Platforms](#cloud-platforms)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Backup & Recovery](#backup--recovery)
8. [Security](#security)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Deployment Modes

VecStore supports three deployment modes:

### 1. Embedded Mode (Library)
**Use when:** Building desktop/mobile apps, CLI tools, or local services

```rust
use vecstore::VecStore;

let mut store = VecStore::open("./vectors.db")?;
// Direct in-process access - <1ms latency
```

**Pros:**
- ✅ <1ms latency (no network)
- ✅ Zero operational overhead
- ✅ Works offline
- ✅ Simple deployment

**Cons:**
- ❌ Single process only
- ❌ No remote access

### 2. Single-Tenant Server
**Use when:** Single application needs remote access

```bash
cargo run --bin vecstore-server --features server
```

**Pros:**
- ✅ gRPC + HTTP APIs
- ✅ Multiple clients
- ✅ Simple configuration

**Cons:**
- ❌ No multi-tenancy
- ❌ Manual scaling

### 3. Multi-Tenant Server
**Use when:** SaaS, hosting multiple customers, or departmental isolation

```bash
cargo run --bin vecstore-server --features server -- --namespaces
```

**Pros:**
- ✅ True tenant isolation
- ✅ Per-tenant quotas
- ✅ Resource management

**Cons:**
- ❌ More complex configuration

---

## 🐳 Docker Deployment

### Quick Start

```bash
# Build image
docker build -t vecstore:latest .

# Run single instance
docker run -d \
  --name vecstore \
  -p 50051:50051 \
  -p 8080:8080 \
  -v vecstore-data:/data \
  vecstore:latest

# Check health
curl http://localhost:8080/health
```

### Docker Compose

```bash
# Single-tenant mode
docker-compose up -d

# Multi-tenant mode
docker-compose --profile multi-tenant up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

### Production Docker Configuration

```yaml
version: '3.8'

services:
  vecstore:
    image: vecstore:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    ports:
      - "50051:50051"  # gRPC
      - "8080:8080"    # HTTP
    volumes:
      - vecstore-data:/data
    environment:
      - RUST_LOG=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 10s

volumes:
  vecstore-data:
    driver: local
```

---

## ☸️ Kubernetes Deployment

### Quick Start

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy VecStore
kubectl apply -f k8s/

# Check status
kubectl get pods -n vecstore
kubectl logs -f deployment/vecstore -n vecstore
```

### Production Deployment

```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Apply configuration
kubectl apply -f k8s/configmap.yaml

# 3. Deploy application
kubectl apply -f k8s/deployment.yaml

# 4. Enable autoscaling
kubectl apply -f k8s/hpa.yaml

# 5. Configure ingress
kubectl apply -f k8s/ingress.yaml

# 6. Verify deployment
kubectl get all -n vecstore
kubectl get hpa -n vecstore
```

### Access Services

```bash
# Get external IPs
kubectl get svc -n vecstore

# HTTP API
export HTTP_IP=$(kubectl get svc vecstore-http -n vecstore -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$HTTP_IP/health

# gRPC API
export GRPC_IP=$(kubectl get svc vecstore-grpc -n vecstore -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
grpcurl -plaintext $GRPC_IP:50051 list
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment vecstore --replicas=5 -n vecstore

# Autoscaling is automatic with HPA
# Scales between 2-10 replicas based on CPU/memory
```

---

## ☁️ Cloud Platforms

### AWS EKS

```bash
# 1. Create EKS cluster
eksctl create cluster \
  --name vecstore-cluster \
  --region us-west-2 \
  --nodes 3 \
  --node-type t3.large

# 2. Deploy VecStore
kubectl apply -f k8s/

# 3. Configure Load Balancer
# AWS automatically provisions ELB for LoadBalancer services

# 4. Get endpoints
kubectl get svc -n vecstore
```

**Estimated Cost:** ~$150-300/month (t3.large × 3 nodes)

### Google GKE

```bash
# 1. Create GKE cluster
gcloud container clusters create vecstore-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2 \
  --region=us-central1

# 2. Get credentials
gcloud container clusters get-credentials vecstore-cluster

# 3. Deploy VecStore
kubectl apply -f k8s/

# 4. Configure Ingress
# GKE automatically provisions Google Cloud Load Balancer
```

**Estimated Cost:** ~$200-400/month (n1-standard-2 × 3 nodes)

### Azure AKS

```bash
# 1. Create resource group
az group create --name vecstore-rg --location eastus

# 2. Create AKS cluster
az aks create \
  --resource-group vecstore-rg \
  --name vecstore-cluster \
  --node-count 3 \
  --node-vm-size Standard_D2s_v3 \
  --generate-ssh-keys

# 3. Get credentials
az aks get-credentials --resource-group vecstore-rg --name vecstore-cluster

# 4. Deploy VecStore
kubectl apply -f k8s/
```

**Estimated Cost:** ~$180-350/month (Standard_D2s_v3 × 3 nodes)

### DigitalOcean Kubernetes

```bash
# 1. Create cluster via web UI or doctl
doctl kubernetes cluster create vecstore-cluster \
  --count 3 \
  --size s-2vcpu-4gb \
  --region nyc1

# 2. Deploy VecStore
kubectl apply -f k8s/

# 3. Configure Load Balancer
# DigitalOcean automatically provisions load balancer
```

**Estimated Cost:** ~$120-250/month (s-2vcpu-4gb × 3 nodes)

---

## ⚙️ Configuration

### Server Options

```bash
vecstore-server --help

Options:
  --db-path <PATH>              Database file path [default: vecstore.db]
  --grpc-port <PORT>           gRPC server port [default: 50051]
  --http-port <PORT>           HTTP server port [default: 8080]
  --dimension <DIM>            Vector dimension (for new DB)
  --debug                      Enable debug logging
  --no-grpc                    Disable gRPC server
  --no-http                    Disable HTTP server
  --namespaces                 Enable multi-tenant mode
  --namespace-root <PATH>      Namespace directory [default: ./namespaces]
```

### Environment Variables

```bash
# Logging level
export RUST_LOG=info  # trace, debug, info, warn, error

# Database path
export DB_PATH=/data/vectors.db

# Server ports
export GRPC_PORT=50051
export HTTP_PORT=8080
```

### Multi-Tenant Configuration

```bash
# Start in multi-tenant mode
vecstore-server \
  --namespaces \
  --namespace-root /data/namespaces \
  --grpc-port 50051 \
  --http-port 8080
```

Each namespace gets:
- Isolated VecStore instance
- Separate data directory
- Configurable quotas
- Independent snapshots

---

## 📊 Monitoring

### Prometheus Metrics

VecStore exposes metrics at `http://localhost:8080/metrics`:

**Key Metrics:**
- `vecstore_query_duration_seconds` - Query latency histogram
- `vecstore_queries_total` - Total queries counter
- `vecstore_vectors_total` - Total vectors gauge
- `vecstore_index_size_bytes` - Index size
- `vecstore_cache_hits_total` - Cache hit rate
- `vecstore_errors_total` - Error counter

### Grafana Dashboard

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:3000  # admin/admin

# Import dashboard
# Dashboard ID: See observability/grafana-dashboard.json
```

**Dashboard Panels:**
- Query latency (p50, p95, p99)
- Throughput (queries/sec)
- Error rate
- Vector count
- Index size
- Cache hit rate
- Resource usage (CPU, memory)

### Health Checks

```bash
# HTTP health endpoint
curl http://localhost:8080/health
# Response: {"status":"SERVING","message":"OK"}

# gRPC health check
grpcurl -plaintext localhost:50051 \
  grpc.health.v1.Health/Check

# Kubernetes liveness probe
kubectl get pods -n vecstore
# Shows READY status
```

---

## 💾 Backup & Recovery

### Snapshot Creation

```rust
// Via API
store.create_snapshot("backup_20251019")?;
```

```bash
# Via gRPC
grpcurl -plaintext \
  -d '{"name":"backup_20251019"}' \
  localhost:50051 \
  vecstore.VecStoreService/CreateSnapshot

# Via HTTP
curl -X POST http://localhost:8080/v1/snapshots \
  -H "Content-Type: application/json" \
  -d '{"name":"backup_20251019"}'
```

### Automated Backups

```bash
# Cron job for daily backups
0 2 * * * /usr/local/bin/vecstore-backup.sh
```

**vecstore-backup.sh:**
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_NAME="backup_$DATE"

# Create snapshot
grpcurl -plaintext \
  -d "{\"name\":\"$SNAPSHOT_NAME\"}" \
  localhost:50051 \
  vecstore.VecStoreService/CreateSnapshot

# Upload to S3 (optional)
aws s3 cp /data/snapshots/$SNAPSHOT_NAME.bin \
  s3://vecstore-backups/snapshots/

# Retention (keep last 7 days)
find /data/snapshots -mtime +7 -delete
```

### Restore from Snapshot

```rust
// Via API
store.restore_snapshot("backup_20251019")?;
```

```bash
# Via gRPC
grpcurl -plaintext \
  -d '{"name":"backup_20251019"}' \
  localhost:50051 \
  vecstore.VecStoreService/RestoreSnapshot
```

### Kubernetes Backup with Velero

```bash
# Install Velero
velero install \
  --provider aws \
  --bucket vecstore-backups \
  --secret-file ./credentials-velero

# Create backup schedule
velero schedule create vecstore-daily \
  --schedule="0 2 * * *" \
  --include-namespaces vecstore

# Restore
velero restore create --from-backup vecstore-daily-20251019
```

---

## 🔒 Security

### TLS Configuration

```bash
# Generate self-signed cert (dev only)
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes

# Start with TLS (when implemented)
vecstore-server \
  --tls-cert cert.pem \
  --tls-key key.pem
```

### Network Security

**Kubernetes Network Policy:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vecstore-netpol
spec:
  podSelector:
    matchLabels:
      app: vecstore
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          access: vecstore
    ports:
    - protocol: TCP
      port: 50051
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
```

### Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: vecstore-quota
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
```

---

## ⚡ Performance Tuning

### Resource Allocation

**CPU Recommendations:**
- Development: 0.25-0.5 cores
- Production (light): 0.5-1 cores
- Production (heavy): 1-2 cores per replica

**Memory Recommendations:**
- Development: 512Mi
- Production (100K vectors): 1-2Gi
- Production (1M vectors): 2-4Gi
- Production (10M+ vectors): 4-8Gi

### HNSW Parameters

```rust
// Configure index for better recall vs speed tradeoff
let config = Config {
    ef_construction: 200,  // Higher = better quality, slower build
    m_max: 16,             // Higher = better recall, more memory
    ..Default::default()
};
```

**Recommendations:**
- **Fast indexing:** ef_construction=100, m_max=8
- **Balanced:** ef_construction=200, m_max=16 (default)
- **High recall:** ef_construction=400, m_max=32

### Database Optimization

```bash
# Enable memory mapping for large datasets
export VECSTORE_MMAP=1

# Increase HNSW search parameter for better recall
export VECSTORE_EF_SEARCH=100
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Check memory usage
kubectl top pods -n vecstore

# Solution: Increase memory limits
kubectl set resources deployment vecstore \
  --limits=memory=4Gi \
  -n vecstore
```

**2. Slow Queries**
```bash
# Check query metrics
curl http://localhost:8080/metrics | grep query_duration

# Solution: Tune HNSW parameters or add Product Quantization
```

**3. Connection Refused**
```bash
# Check if ports are exposed
kubectl get svc -n vecstore

# Check firewall rules
kubectl exec -it deployment/vecstore -- netstat -tlnp
```

**4. Data Corruption**
```bash
# Restore from snapshot
vecstore-server --restore-snapshot backup_latest
```

### Debug Mode

```bash
# Enable debug logging
export RUST_LOG=debug
vecstore-server

# Or via Kubernetes
kubectl set env deployment/vecstore RUST_LOG=debug -n vecstore
```

### Performance Profiling

```bash
# CPU profiling (requires perf)
perf record -F 99 -p $(pgrep vecstore-server)
perf report

# Memory profiling (requires valgrind)
valgrind --tool=massif vecstore-server
```

---

## 📈 Capacity Planning

### Storage Estimation

**Vector Storage:**
- 128-dim float32: ~512 bytes per vector
- 384-dim float32: ~1.5KB per vector
- 768-dim float32: ~3KB per vector

**Example:**
- 100K vectors (384-dim): ~150MB
- 1M vectors (384-dim): ~1.5GB
- 10M vectors (384-dim): ~15GB

**With Product Quantization (16x compression):**
- 10M vectors (384-dim): ~1GB

### Query Throughput

**Single Instance:**
- Embedded mode: 10,000+ queries/sec
- Server mode (local): 5,000-10,000 queries/sec
- Server mode (network): 1,000-5,000 queries/sec

**Kubernetes Cluster (3 replicas):**
- Expected: 3,000-15,000 queries/sec
- With autoscaling (10 replicas): 10,000-50,000 queries/sec

---

## ✅ Production Checklist

- [ ] Configured resource limits
- [ ] Set up health checks
- [ ] Enabled autoscaling
- [ ] Configured backups (daily)
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configured log aggregation
- [ ] Enabled TLS/SSL
- [ ] Set up network policies
- [ ] Configured RBAC
- [ ] Load tested with expected traffic
- [ ] Documented disaster recovery
- [ ] Set up alerting rules
- [ ] Configured retention policies
- [ ] Reviewed security policies

---

**Need help?** Open an issue at https://github.com/yourusername/vecstore/issues
