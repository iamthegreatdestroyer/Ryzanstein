# Phase 6: Integration Testing & Monitoring Verification

**Status:** Ready to Execute
**Date:** February 18, 2026
**Purpose:** Correlate load test metrics with monitoring systems and validate observability integration

---

## Overview

Phase 6 validates that the monitoring stack (Prometheus, Grafana, Jaeger, AlertManager) is properly integrated with the API and correctly capturing all relevant metrics and traces during load testing.

### Objectives

1. **Prometheus Integration** — Verify metrics collection and query functionality
2. **Grafana Dashboards** — Validate dashboard data accuracy and visualizations
3. **Jaeger Tracing** — Confirm distributed trace collection and analysis
4. **AlertManager** — Test alert triggering on threshold violations
5. **Correlation** — Match k6 results with monitoring system metrics
6. **End-to-End Validation** — Comprehensive integration verification

---

## Setup

### Prerequisites

Ensure all services are accessible with port-forwarding active:

```bash
# Terminal 1: API (keep port-forward active during tests)
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000 &

# Terminal 2: Prometheus
kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090 &

# Terminal 3: Grafana
kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000 &

# Terminal 4: Jaeger
kubectl port-forward -n ryzanstein-staging svc/jaeger 16686:16686 &

# Terminal 5: AlertManager
kubectl port-forward -n ryzanstein-staging svc/alertmanager 9093:9093 &
```

### Verification

Verify all services are accessible:

```bash
curl http://localhost:8000/health        # API
curl http://localhost:9090/-/healthy     # Prometheus
curl http://localhost:3000/api/health    # Grafana
curl http://localhost:16686/             # Jaeger
curl http://localhost:9093/-/healthy     # AlertManager
```

---

## Phase 6.1: Prometheus Integration Testing

### 6.1.1 Verify Prometheus Targets

**Purpose:** Confirm Prometheus is scraping the API metrics endpoint

**Steps:**

1. Open Prometheus UI: http://localhost:9090

2. Navigate to **Status** → **Targets**

3. Verify target status:
   - Look for job: `prometheus` (self-monitoring)
   - Status should show **UP**
   - Last Scrape: Should be recent (within 15 seconds)

**Expected Output:**
```
Job: prometheus
Instance: localhost:9090
State: UP
Labels: job="prometheus"
Last Scrape: 2s ago
Scrape Duration: 150ms
```

**Checklist:**
- [ ] Prometheus target shows UP status
- [ ] Last Scrape is recent (< 30 seconds)
- [ ] No scrape errors

---

### 6.1.2 Test Basic Metrics Queries

**Purpose:** Verify Prometheus can query API metrics

**Steps:**

1. In Prometheus UI, go to **Graph** tab

2. Execute queries and verify results:

```promql
# Query 1: Verify Prometheus is running
up{job="prometheus"}
```

**Expected Result:** Returns value of 1 (up/healthy)

```promql
# Query 2: Check for HTTP requests total
http_requests_total
```

**Expected Result:** Shows metric with labels (method, endpoint, status)

```promql
# Query 3: Request rate (should be ~0-1 rps in steady state)
rate(http_requests_total[1m])
```

**Expected Result:** Numeric value between 0 and 5 requests/second

**Checklist:**
- [ ] `up{job="prometheus"}` returns 1
- [ ] `http_requests_total` metric exists
- [ ] `rate()` function works correctly

---

### 6.1.3 Advanced Prometheus Queries

**Purpose:** Test production-grade monitoring queries

Execute these queries and verify results:

```promql
# Query 1: Request rate by endpoint
sum(rate(http_requests_total[5m])) by (endpoint)
```

**Expected Output:**
```
endpoint="/"                    0.2 rps
endpoint="/health"              0.1 rps
endpoint="/v1/models"           0.05 rps
endpoint="/v1/chat/completions" 0.1 rps
endpoint="/v1/embeddings"       0.05 rps
```

```promql
# Query 2: Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
```

**Expected Output:** 0 (no errors in steady state)

```promql
# Query 3: Request latency (P95)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

**Expected Output:** Value between 0.1 and 0.5 (100-500ms)

```promql
# Query 4: Request latency (P99)
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```

**Expected Output:** Value between 0.3 and 1.0 (300-1000ms)

```promql
# Query 5: Container metrics (API CPU)
rate(container_cpu_usage_seconds_total{pod=~"ryzanstein-api.*"}[1m])
```

**Expected Output:** Value between 0.01 and 0.1 (10-100m CPU)

```promql
# Query 6: Container metrics (API Memory)
container_memory_usage_bytes{pod=~"ryzanstein-api.*"} / 1e6
```

**Expected Output:** Value between 50 and 300 (50-300 MB)

**Checklist:**
- [ ] Request rate query returns per-endpoint breakdown
- [ ] Error rate is 0% or near 0%
- [ ] P95 latency < 500ms
- [ ] P99 latency < 1000ms
- [ ] CPU usage reasonable (< 500m)
- [ ] Memory usage reasonable (< 512Mi)

---

## Phase 6.2: Grafana Dashboard Validation

### 6.2.1 Access Grafana

**Steps:**

1. Open Grafana: http://localhost:3000
2. Login with credentials: admin / admin123
3. Navigate to **Dashboards**
4. Verify 4 dashboards are provisioned:
   - [ ] Inference Performance
   - [ ] Resource Usage
   - [ ] System Health
   - [ ] Model Inference

---

### 6.2.2 Inference Performance Dashboard

**Purpose:** Verify real-time API performance visualization

**Checks:**

1. **Request Rate Panel**
   - Should show ~0-1 requests per second in steady state
   - Graph line should be stable (flat)
   - Units: RPS (requests per second)

2. **Error Rate Panel**
   - Should show 0% error rate
   - No red color (indicating errors)
   - Updates every 10 seconds

3. **P99 Latency Gauge**
   - Should show value < 1000ms
   - Color coding: Green (< 500ms), Yellow (500-1000ms)
   - Current value should match Prometheus P99 query

4. **Token Throughput Panel**
   - Should show tokens generated per second
   - Tracks model output performance

**Validation Checklist:**
- [ ] Request rate graph updates every 10 seconds
- [ ] Error rate shows 0%
- [ ] P99 latency < 1000ms
- [ ] All panels have data (not "No data")

---

### 6.2.3 Resource Usage Dashboard

**Purpose:** Monitor container resource consumption

**Checks:**

1. **CPU Utilization Panel**
   - Metric: `rate(container_cpu_usage_seconds_total[1m])`
   - Should show < 500m CPU (< 50% of 1000m limit)
   - Graph should be stable

2. **Memory Utilization Panel**
   - Metric: `container_memory_usage_bytes`
   - Should show < 300Mi memory (< 60% of 512Mi limit)
   - Should remain constant (no growth = no leak)

3. **Disk Usage Panel**
   - Metric: `container_fs_usage_bytes`
   - Should show available space

4. **Pod Status Panel**
   - Shows running pods in ryzanstein-staging
   - All should show "Running" status

**Validation Checklist:**
- [ ] CPU usage < 500m
- [ ] Memory usage stable (not growing)
- [ ] Memory < 300Mi
- [ ] All pods showing Running state
- [ ] No warnings or errors

---

### 6.2.4 System Health Dashboard

**Purpose:** Monitor health indicators and resilience

**Checks:**

1. **Circuit Breaker State**
   - Metric: Tracks circuit breaker state
   - Should show "CLOSED" (normal operation)
   - Not "OPEN" (failure) or "HALF_OPEN" (recovering)

2. **Bulkhead Status**
   - Shows concurrent request limiting
   - Should show available threads/connections

3. **Retry Attempts**
   - Tracks automatic retries
   - Should be low (< 1% of requests)

4. **Health Check Status**
   - Shows endpoint health
   - All should show "UP"

**Validation Checklist:**
- [ ] Circuit breaker: CLOSED
- [ ] Bulkhead: Available slots > 90%
- [ ] Retry rate: < 1%
- [ ] All health checks: UP

---

### 6.2.5 Model Inference Dashboard

**Purpose:** Track model-specific metrics

**Checks:**

1. **Latency Percentiles**
   - P50, P95, P99 latency trend
   - Should remain stable over time
   - No sudden spikes

2. **Token Throughput**
   - Tokens generated per second
   - Should be consistent (stable line)

3. **Failure Rate**
   - Should be 0% or very low
   - No red indicators

4. **Total Inferences**
   - Running counter of API invocations
   - Should increase steadily

**Validation Checklist:**
- [ ] P99 latency < 1000ms
- [ ] Token throughput consistent
- [ ] Failure rate = 0%
- [ ] Inference counter increasing

---

## Phase 6.3: Jaeger Distributed Tracing Validation

### 6.3.1 Access Jaeger UI

**Steps:**

1. Open Jaeger: http://localhost:16686
2. Navigate to **Search** tab
3. Verify configuration

---

### 6.3.2 Query Traces

**Purpose:** Verify trace collection is working

**Steps:**

1. In Jaeger Search tab:
   - **Service:** Select dropdown and choose service
   - **Operation:** Leave as "All" or select specific operation
   - **Tags:** Leave empty
   - **Limit Results:** 20

2. Click **Find Traces**

**Expected Result:**
- List of traces appears with:
  - Trace ID
  - Span count
  - Service name
  - Operation name
  - Duration
  - Timestamp

**Checklist:**
- [ ] At least 5 traces are visible
- [ ] Each trace has multiple spans
- [ ] Duration values are reasonable (10-1000ms)
- [ ] Recent timestamps (last few minutes)

---

### 6.3.3 Analyze a Single Trace

**Purpose:** Verify span structure and latency breakdown

**Steps:**

1. Click on any trace to expand it
2. Observe span hierarchy:
   ```
   GET /v1/chat/completions (root span)
   ├── middleware-auth (sub-span)
   ├── model-inference (sub-span)
   │   ├── tokenization (sub-span)
   │   ├── forward-pass (sub-span)
   │   └── detokenization (sub-span)
   └── response-formatting (sub-span)
   ```

3. Verify span details:
   - Span name: Shows operation
   - Service: Shows microservice
   - Duration: Shows execution time
   - Tags: Shows metadata (status, error, etc.)

**Expected Output:**
- Root span duration: ~500-2000ms
- Each sub-span duration proportional to work done
- No error flags (all spans show success)

**Validation Checklist:**
- [ ] Trace shows complete request flow
- [ ] Span hierarchy is logical
- [ ] All spans show successful status (no errors)
- [ ] Durations add up to root span
- [ ] Timestamps are sequential

---

### 6.3.4 Error Trace Analysis (if errors exist)

**Purpose:** Verify error tracing and root cause visibility

**Steps:**

1. In Jaeger Search, add filter:
   - Click **Tags** input
   - Enter: `error=true`
   - Click **Find Traces**

2. Observe error traces (if any):
   - Should show error tags
   - Should show error messages
   - Should identify failing span

**Expected Behavior:**
- In steady state (no load test): No error traces should exist
- During/after load test: May have error traces at breaking point
- Error details should be clear and actionable

**Validation Checklist:**
- [ ] Error traces (if any) have clear error messages
- [ ] Error span is identified
- [ ] Root cause visible in span logs/tags
- [ ] Stack traces included (if applicable)

---

### 6.3.5 Service Dependency Graph

**Purpose:** Visualize service architecture and dependencies

**Steps:**

1. In Jaeger UI, click **Service Topology** tab
2. Observe graph showing:
   - Nodes: Services (ryzanstein-api, prometheus, etc.)
   - Edges: Service-to-service calls
   - Metrics: Request counts, error rates, latencies

**Expected Output:**
- Central node: ryzanstein-api
- Connected to: Monitoring services
- Arrows showing traffic direction
- Labels showing request counts

**Validation Checklist:**
- [ ] Service graph is visible
- [ ] Dependencies are correct
- [ ] No isolated services
- [ ] Traffic metrics visible

---

## Phase 6.4: AlertManager Integration Validation

### 6.4.1 Access AlertManager

**Steps:**

1. Open AlertManager: http://localhost:9093
2. Navigate to **Alerts** tab

---

### 6.4.2 Verify Alert Rules

**Purpose:** Confirm alert rules are loaded and active

**Steps:**

1. In AlertManager, click **Status** dropdown
2. Review loaded alerts:
   - Should show list of alert rules
   - Each rule shows: name, condition, for duration, labels

**Expected Alert Rules (examples):**
```
- APIHighErrorRate: error_rate > 5% for 5 minutes
- APIHighLatency: p99_latency > 1000ms for 5 minutes
- APIDown: up == 0 for 1 minute
- HighCPUUsage: cpu_usage > 80% for 5 minutes
- HighMemoryUsage: memory_usage > 80% for 5 minutes
```

**Validation Checklist:**
- [ ] At least 5 alert rules are loaded
- [ ] Rules have appropriate thresholds
- [ ] Conditions are logical
- [ ] No syntax errors in rules

---

### 6.4.3 Test Alert Triggering (Optional)

**Purpose:** Verify alerts fire when thresholds are exceeded

**Steps:**

During or after load testing, monitor AlertManager:

1. If error rate exceeds threshold, alert fires:
   - Alert name: `APIHighErrorRate`
   - State: FIRING
   - Value: actual error rate percentage

2. If latency exceeds threshold:
   - Alert name: `APIHighLatency`
   - State: FIRING
   - Value: actual P99 latency in milliseconds

**Expected Behavior:**
- Alert appears in AlertManager UI
- Alert shows FIRING state
- Alert includes metric value
- Alert includes labels (service, instance, etc.)

**Validation Checklist:**
- [ ] Alerts trigger when thresholds exceeded
- [ ] Alert messages are clear
- [ ] Severity levels correct (critical, warning, info)
- [ ] Alerts resolve when metrics return to normal

---

## Phase 6.5: Correlation Analysis

### 6.5.1 Match k6 Metrics with Prometheus Metrics

**Purpose:** Verify monitoring system accuracy

**Procedure:**

1. **Before Load Test:**
   - Record baseline metrics in Prometheus
   - Open k6 and Prometheus side-by-side

2. **During Load Test (k6 smoke test):**
   - k6 console shows: Request count, latency, error rate
   - Prometheus should show same values

3. **After Load Test:**
   - Compare final metrics:

**Comparison Template:**

| Metric | k6 Result | Prometheus | Match? |
|--------|-----------|-----------|--------|
| Total Requests | 150 | http_requests_total = 150 | ✓ |
| Request Rate | 5 rps | rate(http_requests_total[1m]) ≈ 5 | ✓ |
| Error Count | 0 | http_requests_total{status=~"5.."} = 0 | ✓ |
| P95 Latency | 450ms | histogram_quantile(0.95,...) ≈ 450 | ✓ |
| P99 Latency | 750ms | histogram_quantile(0.99,...) ≈ 750 | ✓ |

**Acceptable Variance:**
- Request counts: ±1% (network timing differences)
- Latencies: ±10% (measurement timing)
- Error rates: ±0.5% (rounding)

**Validation Checklist:**
- [ ] Request counts match (±1%)
- [ ] Latencies align (±10%)
- [ ] Error rates consistent (±0.5%)
- [ ] All metrics tracked correctly

---

### 6.5.2 Latency Breakdown Analysis

**Purpose:** Understand where time is spent in requests

**Procedure:**

1. In Prometheus, run:
```promql
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```

2. In Jaeger, click on a trace and expand spans
3. Compare breakdown:

**Expected Breakdown (example):**
```
Root Span (total): 850ms
├── Middleware/Auth:  50ms (6%)
├── Model Inference: 700ms (82%)
│   ├── Tokenization: 150ms
│   ├── Forward Pass: 500ms
│   └── Detokenization: 50ms
├── Response Format:  50ms (6%)
└── Network/Other:   50ms (6%)
```

**Validation:**
- [ ] Time breakdown adds up to total
- [ ] Model inference is largest component (expected)
- [ ] No unexpected large sections
- [ ] Spans are logical

---

## Phase 6.6: End-to-End Validation Execution

### 6.6.1 Pre-Test Checklist

Before running load tests for integration validation:

**API Health:**
- [ ] `curl http://localhost:8000/health` returns 200
- [ ] `curl http://localhost:8000/` returns API info

**Prometheus:**
- [ ] Web UI accessible (http://localhost:9090)
- [ ] Targets show UP status
- [ ] Query execution works
- [ ] Metrics are being collected

**Grafana:**
- [ ] Web UI accessible (http://localhost:3000)
- [ ] Login successful (admin/admin123)
- [ ] 4 dashboards visible
- [ ] Dashboards show data

**Jaeger:**
- [ ] Web UI accessible (http://localhost:16686)
- [ ] Can search for services
- [ ] Can view traces
- [ ] Service list is not empty

**AlertManager:**
- [ ] Web UI accessible (http://localhost:9093)
- [ ] Alert rules loaded
- [ ] Status page accessible

**Kubernetes:**
- [ ] All 5 pods running: `kubectl get pods -n ryzanstein-staging`
- [ ] All 5 services created: `kubectl get svc -n ryzanstein-staging`

---

### 6.6.2 Run Integration Test Scenarios

**Scenario 1: Baseline (5 minutes)**

```bash
cd s:\Ryot
k6 run load_test_smoke.js
```

**During Test:**
- Monitor Grafana dashboard
- Verify metrics update in real-time
- Check Jaeger for incoming traces

**After Test:**
- Compare k6 results with Prometheus metrics
- Verify trace collection in Jaeger
- Check AlertManager (should have no firing alerts)

---

**Scenario 2: Load Test (10 minutes)**

```bash
k6 run load_test_load.js
```

**During Test:**
- Watch Grafana Inference Performance dashboard
- Request rate should increase (10 → 25 → 50 VUs)
- Latency should increase slightly
- CPU/Memory should increase in Resource Usage dashboard

**After Test:**
- Create comparison table (k6 vs Prometheus)
- Verify Jaeger traces show load pattern
- Check no alerts fired (shouldn't at 50 VUs)

---

**Scenario 3: Spike Test (10 minutes)**

```bash
k6 run load_test_spike.js
```

**During Test:**
- Watch Grafana as spike occurs
- Request rate should spike to 1000+ VUs
- Latency should spike (acceptable)
- Error rate may spike (acceptable)
- Watch recovery phase - should return to baseline

**After Test:**
- Verify circuit breaker doesn't trip
- Check recovery time in Jaeger
- Verify AlertManager alerts (if configured)

---

### 6.6.3 Create Integration Test Report

**Template:**

```markdown
# Phase 6 Integration Test Report

Date: [Date]
Tester: [Name]
Duration: [Total duration]

## Executive Summary
[Brief summary of findings]

## Test Results

### Test 1: Baseline (Smoke Test)
- Duration: 30 seconds
- k6 Requests: [Count]
- k6 P99 Latency: [Value]
- Prometheus P99: [Value]
- Variance: [Percentage]
- Status: [PASS/FAIL]

### Test 2: Load Test
- Duration: 5 minutes
- Peak VUs: 50
- k6 Error Rate: [Percentage]
- Prometheus Error Rate: [Percentage]
- k6 P99 Latency: [Value]
- Prometheus P99: [Value]
- Status: [PASS/FAIL]

### Test 3: Spike Test
- Duration: 5 minutes
- Peak VUs: 1000
- Recovery Time: [Seconds]
- Alerts Triggered: [List]
- Status: [PASS/FAIL]

## Monitoring System Validation

### Prometheus
- [x] Targets up
- [x] Metrics collected
- [x] Queries working
- [x] Accuracy verified

### Grafana
- [x] Dashboards visible
- [x] Data updating
- [x] Alerts visible
- [x] Accuracy verified

### Jaeger
- [x] Traces collected
- [x] Service topology visible
- [x] Error tracking working
- [x] Latency breakdown accurate

### AlertManager
- [x] Rules loaded
- [x] Alerts triggered appropriately
- [x] Routing configured
- [x] Severity levels correct

## Correlation Results

| Metric | k6 | Prometheus | Variance | Status |
|--------|-----|-----------|----------|--------|
| Total Requests | [n] | [n] | [%] | [✓] |
| Request Rate | [rps] | [rps] | [%] | [✓] |
| Error Rate | [%] | [%] | [%] | [✓] |
| P95 Latency | [ms] | [ms] | [%] | [✓] |
| P99 Latency | [ms] | [ms] | [%] | [✓] |

## Issues & Resolutions

[If any issues encountered]

- Issue: [Description]
  - Root Cause: [Analysis]
  - Resolution: [Action taken]
  - Verification: [How verified]

## Recommendations

[Any recommendations for optimization or improvements]

## Sign-Off

All integration tests completed successfully.
System is ready for Phase 7 (Final Report).

Status: ✅ PASS / ⚠️ CONDITIONAL PASS / ❌ FAIL
```

---

## Phase 6.7: Troubleshooting Guide

### Issue: Prometheus has no data

**Solutions:**
```bash
# Check Prometheus target status
curl http://localhost:9090/api/v1/targets

# Verify metrics endpoint
curl http://localhost:8000/metrics

# Check pod logs
kubectl logs -n ryzanstein-staging -l app=ryzanstein-api | head -50
```

### Issue: Grafana dashboards show "No data"

**Solutions:**
```bash
# Verify Prometheus is configured as datasource
# In Grafana: Configuration → Data Sources → Prometheus

# Verify Prometheus has metrics
# In Prometheus UI: Query → Enter: up

# Check dashboard provisioning
kubectl describe configmap grafana-dashboards -n ryzanstein-staging
```

### Issue: Jaeger shows no traces

**Solutions:**
```bash
# Verify Jaeger collector endpoint in API config
# Check API logs for trace exporting

# Verify Jaeger pod is running
kubectl get pod -n ryzanstein-staging -l app=jaeger

# Check Jaeger logs
kubectl logs -n ryzanstein-staging -l app=jaeger | tail -50
```

### Issue: AlertManager has no alerts

**Solutions:**
```bash
# Verify Prometheus is configured as alertmanager
# In Prometheus: Configuration → Alerting

# Verify alert rules are loaded
curl http://localhost:9090/api/v1/rules

# Check AlertManager configuration
kubectl describe configmap alertmanager-config -n ryzanstein-staging
```

---

## Success Criteria

Phase 6 is PASS when:

✅ **Prometheus:**
- Targets show UP status
- All metrics are queryable
- Queries return expected values

✅ **Grafana:**
- All 4 dashboards visible and functional
- Dashboards display real-time data
- Data matches Prometheus queries

✅ **Jaeger:**
- Traces are collected
- Service topology visible
- Span details accessible
- Error traces tracked

✅ **AlertManager:**
- Alert rules loaded
- Alerts trigger appropriately
- Severity levels correct

✅ **Correlation:**
- k6 metrics match Prometheus (±10%)
- Latency breakdown in Jaeger aligns
- All systems show consistent results

✅ **End-to-End:**
- Load test can be monitored in real-time
- All observability data is correlated
- No significant gaps in monitoring

---

## Next Steps (Phase 7)

After Phase 6 completion:

1. **Compile Results**
   - Gather all metrics and reports
   - Create summary tables
   - Document findings

2. **Generate Final Report**
   - SLO evaluation
   - Go/No-Go decision
   - Recommendations

3. **Archive Artifacts**
   - Save all test results
   - Document configurations
   - Create runbook for future testing

---

## Summary

Phase 6 validates that all monitoring systems are properly integrated with the API and correctly capturing performance data during load testing. The correlation analysis ensures measurement accuracy across all observability tools.

**Expected Duration:** 2-3 hours (depends on load test execution time)

**Status:** 🟢 **READY FOR PHASE 6 EXECUTION**

---

_Generated: February 18, 2026_
_Kubernetes: ryzanstein-staging namespace_
_Tools: Prometheus, Grafana, Jaeger, AlertManager, k6_
