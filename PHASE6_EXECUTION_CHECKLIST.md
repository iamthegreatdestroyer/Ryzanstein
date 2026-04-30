# Phase 6: Integration Testing - Execution Checklist

**Date:** February 19, 2026
**Status:** Ready for Manual Execution
**Purpose:** Validate monitoring stack integration with load testing

---

## 🎯 Phase 6 Overview

Phase 6 validates that the monitoring stack is properly integrated with the API and captures all metrics during load testing. This involves manual inspection of:

1. **Prometheus** - Metrics collection and querying
2. **Grafana** - Dashboard visualization
3. **Jaeger** - Distributed tracing
4. **AlertManager** - Alert triggering
5. **Correlation** - Match k6 results with monitoring metrics

**Duration:** 1-2 hours
**Manual Effort:** Yes (viewing dashboards and running queries)
**Automation:** Partial (can verify connectivity)

---

## 📋 Pre-Execution Setup

### Open Port-Forwards

You need 5 terminal windows with active port-forwards:

**Terminal 1 - API (Keep Active):**
```bash
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000
```

**Terminal 2 - Prometheus:**
```bash
kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090
```

**Terminal 3 - Grafana:**
```bash
kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000
```

**Terminal 4 - Jaeger:**
```bash
kubectl port-forward -n ryzanstein-staging svc/jaeger 16686:16686
```

**Terminal 5 - AlertManager:**
```bash
kubectl port-forward -n ryzanstein-staging svc/alertmanager 9093:9093
```

### Verify All Services

```bash
# API
curl http://localhost:8000/health

# Prometheus
curl http://localhost:9090/-/healthy

# Grafana
curl http://localhost:3000/api/health

# Jaeger
curl -I http://localhost:16686/

# AlertManager
curl http://localhost:9093/-/healthy
```

---

## Phase 6.1: Prometheus Integration Testing

### 6.1.1 Verify Prometheus Targets ✅

**Steps:**
1. Open: http://localhost:9090
2. Click **Status** → **Targets**
3. Verify job `prometheus` shows **UP** status
4. Last Scrape should be recent (< 30 seconds)

**Expected:** ✅ All targets UP, recent scrape times

**Checklist:**
- [ ] Prometheus target shows UP status
- [ ] Last Scrape is recent (< 30 seconds)
- [ ] No scrape errors

---

### 6.1.2 Basic Metrics Queries ✅

**Steps:**
1. Go to **Graph** tab in Prometheus UI
2. Execute these queries:

**Query 1: Prometheus Up Status**
```promql
up{job="prometheus"}
```
Expected: Returns 1 (healthy)
- [ ] Query executes successfully
- [ ] Returns value of 1

**Query 2: HTTP Requests Total**
```promql
http_requests_total
```
Expected: Shows metrics with labels
- [ ] Metric exists
- [ ] Has labels (method, endpoint, status)

**Query 3: Request Rate (Steady State)**
```promql
rate(http_requests_total[1m])
```
Expected: 0-5 requests/second
- [ ] Returns numeric value
- [ ] Between 0 and 5 rps

---

### 6.1.3 Advanced Prometheus Queries ✅

Execute and verify these production-grade queries:

**Query 1: Request Rate by Endpoint**
```promql
sum(rate(http_requests_total[5m])) by (endpoint)
```
Expected Output:
```
endpoint="/"                    0.2 rps
endpoint="/health"              0.1 rps
endpoint="/v1/models"           0.05 rps
endpoint="/v1/chat/completions" 0.1 rps
endpoint="/v1/embeddings"       0.05 rps
```
- [ ] Query returns per-endpoint breakdown
- [ ] Values are numeric and reasonable

**Query 2: Error Rate**
```promql
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
```
Expected: 0 (no errors)
- [ ] Returns 0 or near 0
- [ ] No 5xx errors showing

**Query 3: P95 Latency**
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```
Expected: 0.1 - 0.5 seconds (100-500ms)
- [ ] Returns numeric value
- [ ] Between 100-500ms

**Query 4: P99 Latency**
```promql
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```
Expected: 0.3 - 1.0 seconds (300-1000ms)
- [ ] Returns numeric value
- [ ] Between 300-1000ms

**Query 5: CPU Usage**
```promql
rate(container_cpu_usage_seconds_total{pod=~"ryzanstein-api.*"}[1m])
```
Expected: 0.01 - 0.1 (10-100m CPU)
- [ ] Returns numeric value
- [ ] Reasonable CPU usage

**Query 6: Memory Usage**
```promql
container_memory_usage_bytes{pod=~"ryzanstein-api.*"} / 1e6
```
Expected: 50 - 300 MB
- [ ] Returns numeric value
- [ ] Memory usage reasonable

---

## Phase 6.2: Grafana Dashboard Validation

### 6.2.1 Access Grafana ✅

**Steps:**
1. Open: http://localhost:3000
2. Login: `admin` / `admin`
3. Navigate to **Dashboards** → **Browse**
4. Verify 4 dashboards exist:
   - [ ] Inference Performance
   - [ ] Resource Usage
   - [ ] System Health
   - [ ] Model Inference

---

### 6.2.2 Inference Performance Dashboard ✅

**Purpose:** Verify real-time API performance

**Checks:**

1. **Request Rate Panel**
   - [ ] Shows ~0-1 requests/sec in steady state
   - [ ] Graph is stable (flat line)
   - [ ] Updates every 10 seconds

2. **Error Rate Panel**
   - [ ] Shows 0% error rate
   - [ ] No red color
   - [ ] Updates regularly

3. **P99 Latency Gauge**
   - [ ] Shows value < 1000ms
   - [ ] Color: Green (< 500ms) or Yellow (500-1000ms)
   - [ ] Matches Prometheus P99 query

4. **Token Throughput Panel**
   - [ ] Shows tokens/second
   - [ ] Tracks model output

---

### 6.2.3 Resource Usage Dashboard ✅

**Purpose:** Monitor container resource consumption

**Checks:**

1. **CPU Utilization Panel**
   - [ ] Shows < 500m CPU
   - [ ] Graph is stable
   - [ ] No spikes

2. **Memory Utilization Panel**
   - [ ] Shows < 300Mi memory
   - [ ] Stable (not growing)
   - [ ] No memory leak signs

3. **Disk Usage Panel**
   - [ ] Shows available space
   - [ ] Reasonable disk usage

4. **Pod Status Panel**
   - [ ] All pods show "Running"
   - [ ] No Failed or Pending pods

---

### 6.2.4 System Health Dashboard ✅

**Purpose:** Monitor health indicators

**Checks:**

1. **Circuit Breaker State**
   - [ ] Shows "CLOSED" (normal)
   - [ ] Not "OPEN" or "HALF_OPEN"

2. **Bulkhead Status**
   - [ ] Shows available connections
   - [ ] Not exhausted

3. **Retry Attempts**
   - [ ] Low (< 1% of requests)
   - [ ] No excessive retries

---

## Phase 6.3: Jaeger Distributed Tracing

### 6.3.1 Access Jaeger ✅

**Steps:**
1. Open: http://localhost:16686
2. Service dropdown: Select `ryzanstein-api`
3. Click **Find Traces**

**Expected:** Shows recent traces from the API

---

### 6.3.2 Trace Analysis ✅

**Checks:**

1. **Trace Count**
   - [ ] Shows traces from recent requests
   - [ ] Multiple spans per trace

2. **Latency Information**
   - [ ] Each span shows duration
   - [ ] Can drill down into span details
   - [ ] Shows timing of operations

3. **Error Traces**
   - [ ] Can filter for error traces
   - [ ] Shows error details if any occurred
   - [ ] Can see error logs in span details

---

## Phase 6.4: AlertManager Integration

### 6.4.1 Access AlertManager ✅

**Steps:**
1. Open: http://localhost:9093
2. View **Alerts** tab
3. Check for any active alerts

**Expected:** No critical alerts in steady state

**Checklist:**
- [ ] AlertManager UI loads
- [ ] Can view alert rules
- [ ] No critical alerts active
- [ ] Shows history of fired alerts

---

## Phase 6.5: Correlation Testing

### 6.5.1 Match k6 Results with Prometheus ✅

**Task:** Verify k6 load test metrics match Prometheus metrics

**Steps:**
1. Note metrics from Phase 5 k6 results:
   - Total requests processed
   - P95 and P99 latencies
   - Error rates

2. In Prometheus, execute correlation queries:
   ```promql
   sum(rate(http_requests_total[30m]))
   ```
   Compare with k6 request rate

3. Compare latency percentiles:
   ```promql
   histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[30m]))
   histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[30m]))
   ```
   Should match k6 P95 and P99

**Validation:**
- [ ] k6 total requests ≈ Prometheus request count
- [ ] k6 P95 latency ≈ Prometheus P95 query result
- [ ] k6 P99 latency ≈ Prometheus P99 query result
- [ ] Error rates match between systems

---

## Phase 6.6: End-to-End Validation

### 6.6.1 System Integration Check ✅

Verify all components work together:

1. **API Running**
   - [ ] API responding to requests
   - [ ] Health check passes

2. **Metrics Collection**
   - [ ] Prometheus scraping API metrics
   - [ ] Grafana receiving Prometheus data
   - [ ] Dashboards showing current data

3. **Tracing Integration**
   - [ ] Jaeger receiving traces
   - [ ] Traces show correct latencies
   - [ ] Can correlate traces with metrics

4. **Alerting**
   - [ ] AlertManager receiving alerts
   - [ ] Can see alert history
   - [ ] Thresholds are appropriate

---

## ✅ Phase 6 Success Criteria

All of the following must be true:

✅ **Prometheus**
- [ ] Targets UP
- [ ] Queries execute successfully
- [ ] Metrics available for all endpoints

✅ **Grafana**
- [ ] Dashboards load
- [ ] Panels show data
- [ ] Refresh working
- [ ] Data matches Prometheus

✅ **Jaeger**
- [ ] Traces being collected
- [ ] Can view trace details
- [ ] Latencies visible

✅ **AlertManager**
- [ ] Accessible
- [ ] Can view alerts
- [ ] Rules configured

✅ **Correlation**
- [ ] k6 metrics match Prometheus
- [ ] Latencies consistent
- [ ] Error rates match

---

## 📝 Execution Summary

**Phase 6.1 - Prometheus:** Manual verification of metrics
**Phase 6.2 - Grafana:** Dashboard inspection
**Phase 6.3 - Jaeger:** Trace analysis
**Phase 6.4 - AlertManager:** Alert verification
**Phase 6.5 - Correlation:** Metric cross-validation
**Phase 6.6 - End-to-End:** System integration check

**Estimated Duration:** 1-2 hours
**Manual Effort:** High (viewing dashboards)
**Complexity:** Medium

---

## 🚀 Next Steps

After completing Phase 6:

1. ✅ Document any issues found
2. ✅ Verify all 6 sub-phases complete
3. ✅ Create Phase 6 completion report
4. ➡️ Proceed to Phase 7 (Final Validation & Go/No-Go Decision)

---

## 📞 Quick Links

- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)
- **Jaeger:** http://localhost:16686
- **AlertManager:** http://localhost:9093
- **API Health:** curl http://localhost:8000/health

---

**Status:** Ready for execution ✅
**Instruction:** Complete manual checks above, then proceed to Phase 7
