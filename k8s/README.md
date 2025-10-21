# VecStore Kubernetes Deployment

Production-ready Kubernetes manifests for VecStore server.

## Quick Start

```bash
# Create namespace
kubectl apply -f namespace.yaml

# Apply ConfigMap
kubectl apply -f configmap.yaml

# Deploy VecStore
kubectl apply -f deployment.yaml

# (Optional) Enable autoscaling
kubectl apply -f hpa.yaml

# (Optional) Setup ingress
kubectl apply -f ingress.yaml
```

## Components

### Core Resources
- **namespace.yaml** - Dedicated namespace for VecStore
- **deployment.yaml** - VecStore deployment with 3 replicas
- **configmap.yaml** - Configuration settings
- **hpa.yaml** - Horizontal Pod Autoscaler (2-10 replicas)

### Networking
- **Service (gRPC)** - LoadBalancer on port 50051
- **Service (HTTP)** - LoadBalancer on port 80
- **ingress.yaml** - NGINX ingress with TLS

### Storage
- **PersistentVolumeClaim** - 10Gi storage for vector data

## Architecture

```
┌─────────────────────────────────────┐
│         Load Balancer (HTTP)        │
│         vecstore.example.com        │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         NGINX Ingress               │
│    (TLS termination + routing)      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      VecStore Service (HTTP)        │
│           Port 80 → 8080            │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
┌───────▼─────┐ ┌────▼────────┐
│  Pod 1      │ │  Pod 2      │ ... (up to 10)
│  - gRPC     │ │  - gRPC     │
│  - HTTP     │ │  - HTTP     │
│  - /data    │ │  - /data    │
└─────────────┘ └─────────────┘
```

## Configuration

### Resource Limits
- **Requests:** 250m CPU, 512Mi RAM
- **Limits:** 1000m CPU, 2Gi RAM
- **Storage:** 10Gi per pod

### Autoscaling
- **Min replicas:** 2
- **Max replicas:** 10
- **CPU target:** 70%
- **Memory target:** 80%

### Health Checks
- **Liveness:** HTTP GET /health (10s delay, 30s interval)
- **Readiness:** HTTP GET /health (5s delay, 10s interval)

## Customization

### Change replica count
```bash
kubectl scale deployment vecstore --replicas=5
```

### Update resources
Edit `deployment.yaml`:
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Enable multi-tenant mode
Edit `deployment.yaml` args:
```yaml
args:
- "--namespaces"
- "--namespace-root"
- "/data/namespaces"
```

## Monitoring

### Prometheus scraping
Pods are annotated for automatic Prometheus discovery:
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8080"
  prometheus.io/path: "/metrics"
```

### Check metrics
```bash
kubectl port-forward svc/vecstore-http 8080:80
curl http://localhost:8080/metrics
```

## Access

### Internal (cluster)
```bash
# gRPC
grpc://vecstore-grpc:50051

# HTTP/REST
http://vecstore-http:80
```

### External (via LoadBalancer)
```bash
# Get external IP
kubectl get svc vecstore-http vecstore-grpc

# Access
curl http://<EXTERNAL-IP>/health
grpcurl <EXTERNAL-IP>:50051 list
```

### Via Ingress
```bash
# HTTP/REST
https://vecstore.example.com/v1/query

# gRPC
grpcurl grpc.vecstore.example.com:443 list
```

## Troubleshooting

### Check pod status
```bash
kubectl get pods -n vecstore
kubectl logs -f deployment/vecstore
```

### Check events
```bash
kubectl get events -n vecstore --sort-by='.lastTimestamp'
```

### Exec into pod
```bash
kubectl exec -it deployment/vecstore -- /bin/bash
```

### Check storage
```bash
kubectl get pvc -n vecstore
kubectl describe pvc vecstore-pvc
```

## Production Checklist

- [ ] Configure TLS certificates (cert-manager)
- [ ] Set up backup/restore (Velero)
- [ ] Configure resource quotas
- [ ] Set up network policies
- [ ] Enable Pod Security Policies
- [ ] Configure RBAC
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure log aggregation (ELK/Loki)
- [ ] Set up alerting rules
- [ ] Document disaster recovery procedures
- [ ] Load test with expected traffic

## Security

### Network Policies
```bash
# Restrict egress to only DNS and API server
kubectl apply -f network-policy.yaml
```

### RBAC
```bash
# Create service account with minimal permissions
kubectl apply -f rbac.yaml
```

### Pod Security
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
```

## Updates

### Rolling update
```bash
# Update image
kubectl set image deployment/vecstore vecstore-server=vecstore:v2.0.0

# Monitor rollout
kubectl rollout status deployment/vecstore

# Rollback if needed
kubectl rollout undo deployment/vecstore
```

## Multi-Tenant Deployment

For multi-tenant namespace mode:

```yaml
# deployment-multi.yaml
spec:
  template:
    spec:
      containers:
      - name: vecstore-server
        args:
        - "--namespaces"
        - "--namespace-root"
        - "/data/namespaces"
        - "--grpc-port"
        - "50051"
        - "--http-port"
        - "8080"
```

Deploy:
```bash
kubectl apply -f deployment-multi.yaml
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f .

# Delete namespace
kubectl delete namespace vecstore
```
