# VecStore Observability Stack

Complete monitoring and observability setup for VecStore using Prometheus and Grafana.

## Quick Start

```bash
cd observability
docker-compose -f docker-compose-monitoring.yml up -d
```

Access the services:
- **VecStore Server**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Metrics Endpoint**: http://localhost:8080/metrics

## Architecture

```
┌─────────────┐
│  VecStore   │
│   Server    │──┐
│  :8080      │  │
└─────────────┘  │
                 │ /metrics
┌─────────────┐  │  scrape
│ Prometheus  │◄─┘  every 10s
│   :9090     │
└──────┬──────┘
       │
       │ query
       ▼
┌─────────────┐
│   Grafana   │
│    :3000    │
└─────────────┘
```

## Metrics Exposed

### Database Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `vecstore_vectors_total` | Gauge | Total vectors (active + deleted) |
| `vecstore_vectors_active` | Gauge | Active (non-deleted) vectors |
| `vecstore_vectors_deleted` | Gauge | Soft-deleted vectors |
| `vecstore_dimension` | Gauge | Vector dimension |

### Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vecstore_requests_total` | Counter | endpoint, method | Total requests by endpoint |
| `vecstore_request_duration_seconds` | Histogram | endpoint, method | Request latency distribution |

### Operation Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vecstore_queries_total` | Counter | type | Query operations (vector, hybrid) |
| `vecstore_query_results` | Histogram | type | Results per query |
| `vecstore_upserts_total` | Counter | batch | Upsert operations |
| `vecstore_deletes_total` | Counter | type | Delete operations |

### Error Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vecstore_errors_total` | Counter | error_type | Errors by type |

### Cache Metrics (if enabled)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vecstore_cache_hits_total` | Counter | type | Cache hits |
| `vecstore_cache_misses_total` | Counter | type | Cache misses |

### WebSocket Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `vecstore_websocket_connections` | Gauge | Active WebSocket connections |

## Grafana Dashboard

The pre-configured dashboard includes:

### 1. **Overview Panel**
- Total vectors (active + deleted)
- Active vector count gauge
- Vector dimension gauge

### 2. **Request Metrics**
- Request rate (req/sec) by endpoint
- Request latency (p50, p95, p99)

### 3. **Operation Metrics**
- Query rate by type
- Upsert rate (single vs batch)
- Delete rate

### 4. **Performance Metrics**
- Query result distributions
- Error rates by type

### Accessing the Dashboard

1. Navigate to http://localhost:3000
2. Login with `admin`/`admin`
3. Dashboard loads automatically (VecStore Monitoring Dashboard)

## Example Queries

### PromQL Queries

```promql
# Request rate by endpoint
rate(vecstore_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(vecstore_request_duration_seconds_bucket[5m]))

# Error rate
rate(vecstore_errors_total[5m])

# Active vectors
vecstore_vectors_active

# Query throughput
rate(vecstore_queries_total[5m])

# Average results per query
rate(vecstore_query_results_sum[5m]) / rate(vecstore_query_results_count[5m])

# Cache hit rate
rate(vecstore_cache_hits_total[5m]) /
  (rate(vecstore_cache_hits_total[5m]) + rate(vecstore_cache_misses_total[5m]))
```

## Alerting Rules

Create `alerts.yml` for Prometheus alerting:

```yaml
groups:
  - name: vecstore_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(vecstore_errors_total[5m]) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "VecStore error rate is {{ $value }} errors/sec"

      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(vecstore_request_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency"
          description: "p95 latency is {{ $value }}s"

      # No data
      - alert: VecStoreDown
        expr: up{job="vecstore"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "VecStore is down"
          description: "VecStore server is not responding"

      # Database growth
      - alert: RapidVectorGrowth
        expr: rate(vecstore_vectors_total[1h]) > 1000
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Rapid database growth"
          description: "Vectors growing at {{ $value }}/sec"
```

## Production Setup

### 1. Persistent Storage

Ensure volumes are backed up:
```yaml
volumes:
  vecstore-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/vecstore-data

  prometheus-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/prometheus-data
```

### 2. Retention Policy

Configure Prometheus retention:
```yaml
command:
  - '--storage.tsdb.retention.time=30d'
  - '--storage.tsdb.retention.size=50GB'
```

### 3. Remote Storage (optional)

For long-term storage, configure remote write:
```yaml
remote_write:
  - url: "https://your-remote-storage/api/v1/write"
    basic_auth:
      username: "user"
      password: "pass"
```

### 4. High Availability

Run multiple Prometheus instances with Thanos for HA:
```bash
docker-compose -f docker-compose-monitoring-ha.yml up -d
```

## Kubernetes Deployment

### Prometheus Operator

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vecstore
  labels:
    app: vecstore
spec:
  selector:
    matchLabels:
      app: vecstore
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
```

### Grafana Dashboard ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vecstore-dashboard
  labels:
    grafana_dashboard: "1"
data:
  vecstore.json: |
    <dashboard JSON content>
```

## Troubleshooting

### Metrics not appearing

1. Check VecStore is running:
   ```bash
   curl http://localhost:8080/health
   ```

2. Verify metrics endpoint:
   ```bash
   curl http://localhost:8080/metrics
   ```

3. Check Prometheus targets:
   Navigate to http://localhost:9090/targets

### Grafana not showing data

1. Verify Prometheus datasource:
   - Grafana → Configuration → Data Sources → Prometheus
   - Test connection

2. Check time range in dashboard

3. Verify PromQL queries in panel edit mode

### High memory usage

Reduce Prometheus retention or scrape frequency:
```yaml
--storage.tsdb.retention.time=7d
scrape_interval: 30s
```

## Integration with Other Tools

### Datadog

Export metrics to Datadog:
```yaml
# In prometheus.yml
remote_write:
  - url: "https://api.datadoghq.com/api/v1/series"
    bearer_token: "<DD_API_KEY>"
```

### New Relic

Use New Relic Prometheus integration:
```yaml
remote_write:
  - url: "https://metric-api.newrelic.com/prometheus/v1/write?prometheus_server=vecstore"
    bearer_token: "<NR_LICENSE_KEY>"
```

### AWS CloudWatch

Use CloudWatch exporter for Prometheus.

## Best Practices

1. **Set appropriate scrape intervals**: Balance granularity vs storage
2. **Use recording rules**: Pre-compute expensive queries
3. **Set up alerting**: Don't wait for users to report issues
4. **Monitor the monitors**: Alert if Prometheus/Grafana go down
5. **Regular backups**: Back up Prometheus and Grafana data
6. **Resource limits**: Set memory/CPU limits in production
7. **Security**: Enable authentication in production

## Performance Impact

Metrics collection overhead:
- **CPU**: < 1% additional CPU usage
- **Memory**: ~5MB for Prometheus client
- **Network**: ~10KB/scrape (depends on cardinality)
- **Latency**: < 1ms added to each request

## Next Steps

1. Set up alerting rules
2. Configure alert notifications (Slack, PagerDuty, email)
3. Add custom dashboards for your use case
4. Integrate with APM tools (Jaeger, Zipkin)
5. Set up log aggregation (ELK, Loki)
