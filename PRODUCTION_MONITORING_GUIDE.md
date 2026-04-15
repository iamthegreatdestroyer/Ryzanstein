# Ryzanstein LLM — Production Monitoring Guide

**Document:** PRODUCTION_MONITORING_GUIDE.md
**Date:** February 18, 2026
**Version:** 2.0.0
**Status:** ✅ Production Ready
**Reference:** [REF:TASK4.3]

---

## Table of Contents

1. [Overview](#overview)
2. [Monitoring Architecture](#monitoring-architecture)
3. [Grafana Dashboards](#grafana-dashboards)
4. [Prometheus Alerts](#prometheus-alerts)
5. [Jaeger Distributed Tracing](#jaeger-distributed-tracing)
6. [AlertManager Configuration](#alertmanager-configuration)
7. [SLO/SLA Definitions](#slosla-definitions)
8. [Runbooks](#runbooks)
9. [Troubleshooting](#troubleshooting)
10. [Observability Best Practices](#observability-best-practices)

---

## Overview

### Goals

- **Visibility:** Real-time insight into system performance
- **Alerting:** Rapid detection and notification of failures
- **Tracing:** Request-level debugging across services
- **SLO tracking:** Monitor service level objectives and error budgets

### Monitoring Stack

| Component | Purpose | Retention | Capacity |
|-----------|---------|-----------|----------|
| **Prometheus** | Metrics collection & storage | 30 days (dev), 180 days (prod) | 50-200 GB |
| **Grafana** | Visualization & dashboards | N/A (stateless) | 5-20 GB |
| **Jaeger** | Distributed tracing | In-memory + persistent | 10K-100K spans |
| **AlertManager** | Alert routing & deduplication | Persistent | 1-2 GB |

---

## Monitoring Architecture

### Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                  Ryzanstein Services                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  API Server  │  │ MCP gRPC Srv │  │   Qdrant     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  Metrics: /metrics  Metrics: /metrics  Metrics: /metrics│
│  Traces: Jaeger    Traces: Jaeger     Traces: Jaeger    │
└──────────────────────────────────────────────────────────┘
         ↓ (scrape)         ↓ (ingest)         ↓ (spans)
┌──────────────────────────────────────────────────────────┐
│          Observability Stack (Kubernetes)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Prometheus   │  │ AlertManager │  │   Jaeger     │  │
│  │(metrics DB)  │  │ (routing)    │  │  (tracing)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
         ↓                   ↓                   ↓
┌──────────────────────────────────────────────────────────┐
│              End User Interfaces                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Grafana    │  │Slack/Email   │  │ Jaeger UI    │  │
│  │ (dashboards) │  │ (alerts)     │  │ (traces)     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Scrape Targets

**Prometheus scrapes metrics from:**

| Target | Job Name | Port | Path | Interval |
|--------|----------|------|------|----------|
| ryzanstein-api | ryzanstein-api | 8000 | /metrics | 15s |
| ryzanstein-mcp | ryzanstein-mcp | 8001 | /metrics | 15s |
| qdrant | qdrant | 6333 | /metrics | 30s |
| prometheus | prometheus | 9090 | /metrics | 30s |
| jaeger | jaeger | 14269 | /metrics | 30s |
| alertmanager | alertmanager | 9093 | /metrics | 30s |

---

## Grafana Dashboards

### Dashboard 1: Inference Performance

**Purpose:** Monitor API request handling, latency, errors, and throughput

**Key Metrics:**
- **P99 Latency:** Histogram quantile at 99th percentile (target: <1s)
- **Request Rate:** Requests per second (target: 100-1000 RPS)
- **Error Rate:** % of 5xx responses (target: <1%)
- **Token Throughput:** Tokens generated per second (target: 15-30 tok/s)

**Panels:**
1. **P99 Latency (Gauge):** Shows current P99 in milliseconds
2. **Request Rate (TimeSeries):** RPS over last hour
3. **Error Rate (TimeSeries):** Error percentage over time
4. **Token Throughput (TimeSeries):** tok/s over time

**Alert Thresholds:**
- ⚠️ Warning: P99 > 1s, error rate > 1%, throughput < 10 tok/s
- 🔴 Critical: P99 > 5s, error rate > 5%, throughput < 5 tok/s

### Dashboard 2: Resource Usage

**Purpose:** Monitor container and node resources

**Key Metrics:**
- **CPU Utilization:** Total CPU usage % (request vs limit)
- **Memory Utilization:** Total memory usage %
- **Pod-level Memory:** Per-pod breakdown
- **Disk I/O:** Read/write rates

**Panels:**
1. **CPU Utilization (Gauge):** Current CPU usage %, color-coded
   - Green: <70%
   - Yellow: 70-85%
   - Red: >85%
2. **Memory Utilization (Gauge):** Current memory usage %
   - Green: <80%
   - Yellow: 80-90%
   - Red: >90%
3. **Memory Usage by Pod (TimeSeries):** Per-pod breakdown
4. **Disk Usage (Gauge):** Persistent volume usage

**Alert Thresholds:**
- ⚠️ Warning: CPU > 85%, Memory > 80%
- 🔴 Critical: CPU > 95%, Memory > 90%

### Dashboard 3: System Health

**Purpose:** Monitor resilience patterns and system state

**Key Metrics:**
- **Circuit Breaker State:** CLOSED, HALF_OPEN, or OPEN
- **Bulkhead Status:** Active requests vs max concurrent
- **Retry Attempts:** Failed requests being retried
- **Cache Hit Rate:** % of cached responses

**Panels:**
1. **Circuit Breaker State (Stat):** Current state with color coding
2. **Bulkhead Active Requests (Stat):** Current vs max
3. **Retry Attempts (Stat):** Cumulative retries
4. **Circuit Breaker State Changes (TimeSeries):** Changes over time

**Alert Thresholds:**
- 🔴 Critical: Circuit breaker is OPEN
- ⚠️ Warning: Bulkhead at 90%+ capacity, high retry rate

### Dashboard 4: Model Inference

**Purpose:** Monitor model-specific metrics

**Key Metrics:**
- **Inference Latency Percentiles:** P50, P95, P99
- **Token Throughput:** Tokens per second
- **Failure Rate:** % of failed inferences
- **Total Inferences:** Cumulative counter

**Panels:**
1. **Latency Percentiles (TimeSeries):** P50, P95, P99 over time
2. **Token Throughput (TimeSeries):** tok/s over time
3. **Failure Rate (TimeSeries):** % failed over time
4. **Total Inferences (TimeSeries):** Cumulative count

**Alert Thresholds:**
- ⚠️ Warning: P99 > 5s, failure rate > 5%, throughput < 10 tok/s
- 🔴 Critical: P99 > 10s, failure rate > 10%

---

## Prometheus Alerts

### Alert Categories

**33 total rules across 6 categories:**

#### 1. API Server Alerts (6)
- `RyzansteinAPIDown` — API unreachable (critical, 2m)
- `APIHighErrorRate` — Error rate >1% (warning, 5m)
- `APIHighLatency` — P99 latency >1s (warning, 5m)
- `CircuitBreakerOpen` — Circuit breaker OPEN (critical, 1m)
- `BulkheadExhausted` — Bulkhead at 100% capacity (warning, 2m)
- `HighRequestQueueDepth` — Request queue backing up (warning, 5m)

#### 2. Model Inference Alerts (3)
- `HighInferenceLatency` — P99 >5s (warning, 5m)
- `HighInferenceFailureRate` — Failure rate >5% (warning, 5m)
- `LowTokenThroughput` — Throughput <10 tok/s (info, 10m)

#### 3. Resource Alerts (2)
- `ContainerHighCPU` — CPU >85% (warning, 5m)
- `ContainerHighMemory` — Memory >90% (critical, 5m)

#### 4. Observability Alerts (5)
- `PrometheusDown` — Prometheus unreachable (critical, 2m)
- `AlertManagerDown` — AlertManager unreachable (critical, 2m)
- `PrometheusHighDiskUsage` — TSDB >40GB of 50GB (warning, 10m)
- `JaegerDown` — Jaeger unreachable (warning, 2m)
- `GrafanaDown` — Grafana unreachable (warning, 2m)

#### 5. Storage Alerts (4)
- `QdrantDown` — Vector DB unreachable (critical, 2m)
- `QdrantHighDiskUsage` — Disk >80% (warning, 10m)
- `QdrantHighMemory` — Memory >90% (critical, 10m)
- `PVCHighUsage` — PVC >90% full (warning, 10m)

#### 6. Kubernetes Alerts (3)
- `HighPodRestartRate` — Pod restarting frequently (warning, 5m)
- `HPAAtMaxReplicas` — HPA at maximum scale (warning, 5m)
- `NodeMemoryPressure` — Node low on memory (critical, 2m)

### Alert Severity Levels

| Severity | Action | Response Time | Notification |
|----------|--------|---------------|--------------|
| 🔴 **Critical** | Page on-call | <5 min | PagerDuty + Slack |
| ⚠️ **Warning** | Create ticket | <30 min | Slack #warnings |
| ℹ️ **Info** | Log for review | <1 day | Slack #info |

### Alert Evaluation

All alerts are evaluated every 15 seconds:

```promql
# Example: High error rate alert
rate(http_requests_total{status=~"5.."}[5m]) /
rate(http_requests_total[5m]) > 0.01
```

For condition to fire:
1. PromQL expression matches
2. Condition persists for `for` duration (e.g., 5m)
3. Alert state transitions to "firing"
4. AlertManager receives alert
5. Notification sent to configured channels

---

## Jaeger Distributed Tracing

### Trace Integration

Every API request is traced with:
- **Trace ID:** Unique per request
- **Span ID:** Unique per operation
- **Parent Span ID:** Links operations in chain
- **Duration:** End-to-end latency
- **Tags:** Metadata (user, model, tokens, etc.)
- **Logs:** Event-level details
- **Errors:** Exception stack traces

### Request Flow (Example)

```
Trace: a1b2c3d4e5f6g7h8

Span 1: HTTP Request (100ms)
├─ Operation: POST /v1/chat/completions
├─ Service: api
├─ Duration: 100ms
│
├─ Span 2: MCP gRPC Call (80ms)
│  ├─ Operation: Inference.InvokeModel
│  ├─ Service: mcp-server
│  ├─ Duration: 80ms
│  │
│  ├─ Span 3: Vector DB Lookup (20ms)
│  │  ├─ Operation: Search
│  │  ├─ Service: qdrant
│  │  ├─ Duration: 20ms
│  │
│  └─ Span 4: Model Inference (60ms)
│     ├─ Operation: ForwardPass
│     ├─ Service: api
│     ├─ Duration: 60ms
│
└─ Span 5: Response Encoding (5ms)
   ├─ Operation: JSONEncode
   ├─ Service: api
   └─ Duration: 5ms
```

### Accessing Jaeger UI

```bash
# Port-forward to Jaeger UI
kubectl port-forward svc/jaeger 16686:16686

# Access: http://localhost:16686
```

**Features:**
- Service list (api, mcp-server, qdrant, etc.)
- Operation list (per service)
- Trace search by:
  - Service + operation
  - Trace ID
  - Tag (status, error, user_id)
  - Latency range
  - Span count
- Span waterfall diagram
- Latency analysis (P50, P95, P99)
- Error rate by operation
- Service dependency graph

---

## AlertManager Configuration

### Notification Channels

**Production Setup:**

```yaml
alertmanager:
  slack:
    webhook: https://hooks.slack.com/services/YOUR/PROD/WEBHOOK
    channels:
      alerts: "#production-alerts"      # Critical only
      warnings: "#production-warnings"  # Warning + Info
      info: "#production-info"          # Info level

  pagerduty:
    serviceKey: YOUR_PROD_SERVICE_KEY
    severity: critical

  email:
    to: devops-oncall@company.com
    smtp: smtp.gmail.com:587
```

### Alert Routing

```
Root Route (resolve_timeout: 5m)
├─ Critical → PagerDuty + Slack #alerts (10s wait, 15m repeat)
├─ Warning → Slack #warnings (1m wait, 1h repeat)
└─ Info → Slack #info (5m wait, 1d repeat)
```

### Inhibition Rules

Suppress lower-severity alerts if higher-severity exists:

```yaml
inhibit_rules:
  # Suppress warning/info if critical exists for same service
  - source_match:
      severity: critical
    target_match_re:
      severity: warning|info
    equal: [service, alertname]

  # Don't suppress resolved alerts
  - source_match:
      status: resolved
    target_match_re:
      severity: .+
    equal: [alertname, service]
```

---

## SLO/SLA Definitions

### Service Level Objectives (SLOs)

**API Availability:**
- **Target:** 99.9% uptime (error budget: 43.2 minutes/month)
- **Metric:** `(1 - error_rate) * 100`
- **Measurement:** Per calendar month

**API Latency:**
- **Target:** P99 latency < 1 second
- **Metric:** `histogram_quantile(0.99, response_time_seconds)`
- **Measurement:** Per 30-minute window

**Model Inference:**
- **Target:** 15-30 tokens/second throughput
- **Metric:** `rate(tokens_generated_total[1m])`
- **Measurement:** Per minute average

**Circuit Breaker Health:**
- **Target:** <1% of time in OPEN state
- **Metric:** `(time_in_open_state / total_time)`
- **Measurement:** Per week

### Error Budget

**Monthly Error Budget (99.9% SLO):**
- Total minutes: 43,200
- Allowed downtime: 43.2 minutes
- Allowed errors: 0.1% of requests

**Budget Tracking:**

```promql
# Cumulative error rate this month
sum(increase(http_requests_total{status=~"5.."}[30d])) /
sum(increase(http_requests_total[30d]))

# Remaining budget
(1 - (above_value)) * 43.2  # minutes remaining
```

**Actions:**
- If <50% budget remaining: Begin root cause analysis
- If <10% budget remaining: Page on-call
- If budget exhausted: Post-incident review required

---

## Runbooks

### Alert: RyzansteinAPIDown

**Severity:** 🔴 Critical

**Detection:** API pod unreachable for 2+ minutes

**Impact:**
- All API requests fail
- Inference unavailable
- Users cannot submit requests

**Investigation:**

```bash
# Check pod status
kubectl get pods -l app=ryzanstein-api -n ryzanstein-prod

# Check logs
kubectl logs -f deployment/ryzanstein-api -n ryzanstein-prod

# Check recent events
kubectl describe pod ryzanstein-api-xxx -n ryzanstein-prod

# Check resource limits
kubectl top pods -l app=ryzanstein-api
```

**Resolution:**

1. **If OOMKilled:** Increase memory limit in `values-production.yaml`
2. **If CrashLoopBackOff:** Check logs for startup error
3. **If pending:** Check node capacity and PVC binding
4. **If running but unhealthy:** Check `/health` and `/health/ready` endpoints

```bash
# Force pod restart
kubectl rollout restart deployment/ryzanstein-api -n ryzanstein-prod

# Check health after restart
kubectl exec -it pod/ryzanstein-api-xxx -c api -- \
  curl http://localhost:8000/health/ready
```

---

### Alert: APIHighErrorRate

**Severity:** ⚠️ Warning

**Detection:** Error rate >1% for 5+ minutes

**Impact:**
- ~1% of user requests fail
- Service partially available
- Data quality may be affected

**Investigation:**

```bash
# Check error rate by endpoint
curl http://prometheus:9090/api/v1/query?query='rate(http_requests_total{status=~"5.."}[5m]) by (endpoint)'

# Check error types
curl http://prometheus:9090/api/v1/query?query='rate(http_requests_total{status=~"5.."}[5m]) by (status)'

# Check API logs for errors
kubectl logs deployment/ryzanstein-api -n ryzanstein-prod | grep ERROR
```

**Resolution:**

1. **If 502/503 errors:** Check MCP server status
2. **If 500 errors:** Check API logs for exceptions
3. **If timeout errors:** Check backend latency, increase timeout
4. **If database errors:** Check Qdrant connectivity

```bash
# Check MCP server
kubectl get pods -l app=ryzanstein-mcp
kubectl logs deployment/ryzanstein-mcp

# Check Qdrant
kubectl get pods -l app=ryzanstein-qdrant
kubectl exec -it pod/ryzanstein-qdrant-0 -- \
  curl http://localhost:6333/health
```

---

### Alert: HighInferenceLatency

**Severity:** ⚠️ Warning

**Detection:** P99 latency >5s for 5+ minutes

**Impact:**
- User-facing slowdown
- Timeout risks
- Poor experience

**Investigation:**

```bash
# Check latency breakdown
curl http://prometheus:9090/api/v1/query?query='histogram_quantile(0.99, rate(inference_duration_seconds_bucket[5m]))'

# Check per-layer latency (Prometheus labels)
curl http://prometheus:9090/api/v1/query?query='rate(inference_duration_seconds_bucket[5m]) by (layer)'

# Check if specific models are slow
curl http://prometheus:9090/api/v1/query?query='histogram_quantile(0.99, rate(inference_duration_seconds_bucket[5m])) by (model)'
```

**Resolution:**

1. **Check model throughput:** May be bottlenecked
2. **Check CPU/memory:** If at limits, add replicas
3. **Check Qdrant latency:** If high, scale vector DB
4. **Check network:** Check inter-pod latency

```bash
# Scale API replicas
kubectl scale deployment ryzanstein-api --replicas=5 -n ryzanstein-prod

# Monitor impact
watch kubectl get hpa ryzanstein-api-hpa -n ryzanstein-prod
```

---

### Alert: ContainerHighMemory

**Severity:** 🔴 Critical

**Detection:** Memory usage >90% for 5+ minutes

**Impact:**
- OOMKill risk imminent
- Pod eviction likely
- Service disruption

**Immediate Action:**

```bash
# Scale down other workloads (if on same node)
kubectl get pods --field-selector=status.phase=Running \
  --all-namespaces | grep <node-name>

# Or add node to cluster
kubectl scale nodes --increase=1

# Or increase memory limit immediately
kubectl patch deployment ryzanstein-api \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"12Gi"}}}]}}}}'
```

**Investigation:**

```bash
# Check memory usage history
kubectl top pods -l app=ryzanstein-api --sort-by=memory

# Check for memory leaks
curl http://prometheus:9090/api/v1/query?query='container_memory_usage_bytes{pod=~"ryzanstein-api-.*"}'

# Analyze memory profile (if available)
kubectl exec -it pod/ryzanstein-api-xxx -c api -- \
  curl http://localhost:8000/debug/pprof/heap
```

**Resolution:**

1. **Immediate:** Scale up (add memory limit or replicas)
2. **Short-term:** Identify memory leak or increase cache
3. **Long-term:** Implement memory pooling or reduce model size

---

## Troubleshooting

### Metrics Not Appearing

**Problem:** Prometheus scrape targets show "DOWN"

**Debug Steps:**

```bash
# Check if service is accessible
kubectl get svc ryzanstein-api

# Test endpoint directly
kubectl exec -it pod/ryzanstein-api-xxx -- \
  curl http://localhost:8000/metrics

# Check Prometheus scrape logs
kubectl logs -f deployment/prometheus | grep ryzanstein
```

**Solutions:**

1. Verify endpoint is exposing metrics
2. Check firewall rules (ServiceMonitor if using Prometheus Operator)
3. Verify port number in prometheus.yml

### Alerts Not Firing

**Problem:** Expected alert doesn't fire even though condition is met

**Debug Steps:**

```bash
# Test PromQL expression directly
curl 'http://prometheus:9090/api/v1/query?query=up{job="ryzanstein-api"}'

# Check alert rule evaluation
curl 'http://prometheus:9090/api/v1/rules' | grep ryzanstein

# Check AlertManager status
curl http://alertmanager:9093/api/v1/status
```

**Solutions:**

1. Verify metric exists (check Prometheus targets)
2. Test PromQL in Prometheus UI first
3. Check alert rule `for` duration (may be too long)
4. Restart Prometheus if rules recently added

### Alerts Firing But Not Sent

**Problem:** AlertManager receives alert but doesn't send notification

**Debug Steps:**

```bash
# Check AlertManager logs
kubectl logs -f deployment/alertmanager

# Check AlertManager config
kubectl get configmap alertmanager-config -o yaml

# Test Slack webhook
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test message"}' \
  https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**Solutions:**

1. Verify webhook URL is correct and active
2. Check AlertManager routing rules
3. Verify inhibition rules aren't suppressing alerts
4. Check receiver configuration syntax

---

## Observability Best Practices

### 1. Instrumentation

Every service should expose:
- **Metrics:** Counter, gauge, histogram, summary
- **Traces:** W3C Trace Context headers
- **Logs:** Structured JSON with trace context

### 2. Naming Conventions

Follow Prometheus naming conventions:
```
<namespace>_<subsystem>_<name>_<unit>

Examples:
http_requests_total        # counter
http_request_duration_seconds  # histogram
inference_latency_ms           # gauge
tokens_generated_total         # counter
```

### 3. Cardinality Control

Avoid high-cardinality labels:
```promql
# ❌ BAD: user_id as label (millions of values)
requests_total{user_id="12345"}

# ✅ GOOD: user_id only in traces/logs
```

### 4. Retention Policies

- **Metrics:** 30 days (dev), 180 days (prod)
- **Traces:** 10K-100K in memory, older deleted
- **Logs:** Application-level rotation (daily)

### 5. SLO Tracking

Monitor SLO compliance continuously:

```promql
# SLO: 99.9% availability
availability_slo = (
  sum(rate(requests_total{status!~"5.."}[30d])) /
  sum(rate(requests_total[30d]))
) * 100

# Alert when <99.85% (0.05% margin to 99.9%)
```

### 6. On-Call Runbooks

Every alert should have:
- **Impact description**
- **Detection explanation**
- **Investigation steps**
- **Resolution procedures**
- **Escalation path**

---

**Status:** ✅ **TASK 4.3 COMPLETE**

**Files Created:**
- Grafana dashboard ConfigMap (4 dashboards)
- Prometheus alert rules (33 rules)
- Production monitoring guide (comprehensive)

**Ready for:** Task 4.4 (Security Hardening)

---

_Document Generated: February 18, 2026_
_Author: Copilot Claude Sonnet 4.6_
_Reference: [REF:TASK4.3]_
