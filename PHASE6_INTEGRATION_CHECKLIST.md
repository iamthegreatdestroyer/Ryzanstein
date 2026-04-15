# Phase 6: Integration Testing - Execution Checklist

**Status:** Ready to Execute
**Date:** February 18, 2026
**Purpose:** Validate monitoring system integration and data correlation

---

## Pre-Execution Checklist

### Environment Setup

- [ ] API is running (`kubectl get pods -n ryzanstein-staging`)
- [ ] All 5 pods in Running state
- [ ] Port-forwarding enabled for:
  - [ ] API (8000)
  - [ ] Prometheus (9090)
  - [ ] Grafana (3000)
  - [ ] Jaeger (16686)
  - [ ] AlertManager (9093)

### Service Accessibility

- [ ] API health check passes: `curl http://localhost:8000/health`
- [ ] Prometheus web UI loads: `curl http://localhost:9090`
- [ ] Grafana web UI loads: `curl http://localhost:3000`
- [ ] Jaeger web UI loads: `curl http://localhost:16686`
- [ ] AlertManager web UI loads: `curl http://localhost:9093`

### Baseline Metrics

Before running load tests, record baseline values:

**Prometheus Baseline:**
- [ ] Request rate (baseline): `rate(http_requests_total[1m])` = _____ rps
- [ ] Error rate (baseline): `rate(http_requests_total{status=~"5.."}[1m])` = _____ %
- [ ] P95 latency: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` = _____ ms
- [ ] P99 latency: `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))` = _____ ms
- [ ] API CPU usage: `rate(container_cpu_usage_seconds_total{pod=~"ryzanstein-api.*"}[1m])` = _____ m
- [ ] API Memory usage: `container_memory_usage_bytes{pod=~"ryzanstein-api.*"} / 1e6` = _____ MB

**Grafana Baseline:**
- [ ] Inference Performance dashboard loads
- [ ] All panels have data
- [ ] Request rate shows baseline value
- [ ] Error rate shows 0%
- [ ] P99 latency shows < 1000ms

**Jaeger Baseline:**
- [ ] Can search for services
- [ ] Service list is populated
- [ ] Can view sample traces
- [ ] Trace details load correctly

**AlertManager Baseline:**
- [ ] Alert rules page accessible
- [ ] No firing alerts in baseline state
- [ ] Alert rules count: _____ rules

---

## Execution Phase 1: Prometheus Integration Testing

### 6.1.1 Verify Targets

- [ ] Navigate to Prometheus: http://localhost:9090
- [ ] Go to Status → Targets
- [ ] Verify `prometheus` target status:
  - [ ] State: UP
  - [ ] Last Scrape: < 30 seconds ago
  - [ ] Scrape Duration: < 500ms
- [ ] Record observations: _______________________

### 6.1.2 Basic Metrics Queries

Execute and verify these queries return data:

- [ ] `up{job="prometheus"}`
  - Result: 1 (should be "up")
  - Status: ✓

- [ ] `http_requests_total`
  - Result: Multiple time series with labels
  - Status: ✓

- [ ] `rate(http_requests_total[1m])`
  - Result: Numeric value (rps)
  - Value: _____ rps
  - Status: ✓

### 6.1.3 Advanced Metrics Queries

- [ ] `sum(rate(http_requests_total[5m])) by (endpoint)`
  - Returns per-endpoint request rates
  - Status: ✓

- [ ] `rate(http_requests_total{status=~"5.."}[5m])`
  - Error rate query
  - Result: 0 or near 0
  - Status: ✓

- [ ] `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
  - P95 latency
  - Result: _____ ms
  - Status: ✓

- [ ] `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))`
  - P99 latency
  - Result: _____ ms
  - Status: ✓

- [ ] `rate(container_cpu_usage_seconds_total{pod=~"ryzanstein-api.*"}[1m])`
  - CPU usage
  - Result: _____ m
  - Status: ✓

- [ ] `container_memory_usage_bytes{pod=~"ryzanstein-api.*"} / 1e6`
  - Memory usage
  - Result: _____ MB
  - Status: ✓

**Prometheus Integration Result:**
- [ ] PASS (all queries work)
- [ ] FAIL (describe issue): _________________

---

## Execution Phase 2: Grafana Dashboard Validation

### Access Grafana

- [ ] Open http://localhost:3000
- [ ] Login: admin / admin123
- [ ] Authentication successful: [ ]

### Dashboard Inventory

- [ ] Verify dashboard count: Should be 4 or more
  - [ ] Inference Performance
  - [ ] Resource Usage
  - [ ] System Health
  - [ ] Model Inference

### Inference Performance Dashboard

Open dashboard and verify panels:

- [ ] **Request Rate Panel**
  - [ ] Visible and displaying data
  - [ ] Current value: _____ rps
  - [ ] Updates every ~10 seconds
  - [ ] Status: ✓

- [ ] **Error Rate Panel**
  - [ ] Visible and displaying data
  - [ ] Current value: _____ %
  - [ ] Shows 0% in baseline
  - [ ] Color: Green (expected)
  - [ ] Status: ✓

- [ ] **P99 Latency Gauge**
  - [ ] Visible and displaying data
  - [ ] Current value: _____ ms
  - [ ] Less than 1000ms
  - [ ] Color: Green/Yellow (expected)
  - [ ] Status: ✓

- [ ] **Token Throughput Panel**
  - [ ] Visible and displaying data
  - [ ] Current value: _____ tok/s
  - [ ] Updates periodically
  - [ ] Status: ✓

**Inference Performance Result:**
- [ ] PASS (all panels functional)
- [ ] PARTIAL (describe missing panels): _______
- [ ] FAIL (describe issue): _________________

### Resource Usage Dashboard

- [ ] **CPU Utilization Panel**
  - [ ] Current value: _____ m (should be < 500m)
  - [ ] Graph stable (no spikes)
  - [ ] Status: ✓

- [ ] **Memory Utilization Panel**
  - [ ] Current value: _____ MB (should be < 300MB)
  - [ ] Graph stable (not growing)
  - [ ] Status: ✓

- [ ] **Pod Status Panel**
  - [ ] All pods showing "Running"
  - [ ] Pod count: _____ (should be 5)
  - [ ] Status: ✓

**Resource Usage Result:**
- [ ] PASS
- [ ] FAIL (describe): _________________

### System Health Dashboard

- [ ] **Circuit Breaker State**
  - [ ] Current state: _____ (should be CLOSED)
  - [ ] Status: ✓

- [ ] **Bulkhead Status**
  - [ ] Available slots: _____ (should be > 90%)
  - [ ] Status: ✓

- [ ] **Retry Attempts**
  - [ ] Retry rate: _____ % (should be < 1%)
  - [ ] Status: ✓

- [ ] **Health Check Status**
  - [ ] All endpoints showing UP
  - [ ] Count: _____ (should be 5)
  - [ ] Status: ✓

**System Health Result:**
- [ ] PASS
- [ ] FAIL (describe): _________________

### Model Inference Dashboard

- [ ] **Latency Percentiles Panel**
  - [ ] P50: _____ ms
  - [ ] P95: _____ ms
  - [ ] P99: _____ ms
  - [ ] No sudden spikes
  - [ ] Status: ✓

- [ ] **Token Throughput Panel**
  - [ ] Current value: _____ tok/s
  - [ ] Consistent (stable line)
  - [ ] Status: ✓

- [ ] **Failure Rate Panel**
  - [ ] Current value: _____ % (should be 0%)
  - [ ] No red indicators
  - [ ] Status: ✓

- [ ] **Total Inferences Counter**
  - [ ] Current value: _____ (should be increasing)
  - [ ] Status: ✓

**Model Inference Result:**
- [ ] PASS
- [ ] FAIL (describe): _________________

**Overall Grafana Result:**
- [ ] PASS (all dashboards functional)
- [ ] PARTIAL (describe): _______
- [ ] FAIL (describe): _________________

---

## Execution Phase 3: Jaeger Distributed Tracing Validation

### 6.3.1 Access Jaeger & Search for Traces

- [ ] Open http://localhost:16686
- [ ] Navigate to Search tab
- [ ] Select service from dropdown
- [ ] Service selected: _____________________
- [ ] Click "Find Traces"
- [ ] Traces appear: [ ]

**Trace Count:**
- [ ] Number of traces: _____ (should be > 5)
- [ ] Trace age: All < 5 minutes old: [ ]

### 6.3.2 Analyze Single Trace

Select and expand one trace:

- [ ] Trace ID: _______________________
- [ ] Span count: _____ (should be > 1)
- [ ] Total duration: _____ ms (should be 10-2000ms)
- [ ] All spans show success (no error flags): [ ]

**Span Hierarchy Verification:**

- [ ] Root span visible
  - [ ] Name: _____________________
  - [ ] Duration: _____ ms

- [ ] Child spans visible
  - [ ] Count: _____ (should be > 1)
  - [ ] All have names: [ ]
  - [ ] All have durations: [ ]

- [ ] Span details accessible
  - [ ] Service name displayed: [ ]
  - [ ] Tags visible: [ ]
  - [ ] Logs visible (if applicable): [ ]

**Jaeger Span Analysis Result:**
- [ ] PASS (traces and spans visible)
- [ ] PARTIAL (describe): _______
- [ ] FAIL (describe): _________________

### 6.3.3 Error Trace Check

- [ ] In Jaeger Search, add filter: `error=true`
- [ ] Click "Find Traces"
- [ ] Expected result in baseline: No error traces
  - [ ] Confirmed: 0 error traces

**Error Tracing Result:**
- [ ] PASS (no unexpected errors)
- [ ] FAIL (unexpected errors found): _______

### 6.3.4 Service Topology

- [ ] Click "Service Topology" tab
- [ ] Service graph visible: [ ]
- [ ] Services shown:
  - [ ] ryzanstein-api
  - [ ] prometheus (optional)
  - [ ] Other: _____________________
- [ ] Dependencies/edges visible: [ ]
- [ ] Metrics on edges (request counts): [ ]

**Jaeger Overall Result:**
- [ ] PASS (traces, spans, topology visible)
- [ ] PARTIAL (describe): _______
- [ ] FAIL (describe): _________________

---

## Execution Phase 4: AlertManager Validation

### 6.4.1 Access AlertManager

- [ ] Open http://localhost:9093
- [ ] Status/Health page loads: [ ]
- [ ] Alerts tab visible: [ ]

### 6.4.2 Verify Alert Rules

- [ ] Click Status dropdown
- [ ] Alert rules listed: [ ]
- [ ] Rule count: _____ (should be > 5)
- [ ] Rules have descriptions: [ ]

**Sample Alert Rules Found:**
- [ ] APIHighErrorRate: [ ]
- [ ] APIHighLatency: [ ]
- [ ] APIDown: [ ]
- [ ] HighCPUUsage: [ ]
- [ ] HighMemoryUsage: [ ]

### 6.4.3 Verify No Firing Alerts (Baseline)

- [ ] Alerts tab shows no FIRING alerts: [ ]
- [ ] Confirmed: 0 firing alerts in baseline

**AlertManager Result:**
- [ ] PASS (rules loaded, no false alarms)
- [ ] FAIL (describe): _________________

---

## Execution Phase 5: Load Test with Monitoring

### Run Smoke Test with Monitoring Active

**Pre-test:**
- [ ] All monitoring services running
- [ ] Grafana dashboards open
- [ ] Jaeger Search tab open
- [ ] AlertManager open
- [ ] Prometheus ready

**Execute:**
```bash
cd s:\Ryot
k6 run load_test_smoke.js
```

- [ ] Test started: _____:_____
- [ ] Test ended: _____:_____
- [ ] Test duration: _____ seconds

**During Test Monitoring:**

Grafana observations:
- [ ] Request rate increased
- [ ] Latency graph updated
- [ ] CPU/Memory increased
- [ ] All panels updated in real-time

Jaeger observations:
- [ ] New traces appeared
- [ ] Trace count increased: from _____ to _____
- [ ] Span details updated

AlertManager observations:
- [ ] No unexpected alerts: [ ]
- [ ] (or describe any alerts): _________________

**k6 Results:**
- [ ] Total requests: _____
- [ ] Request duration (avg): _____ ms
- [ ] Request duration (p95): _____ ms
- [ ] Request duration (p99): _____ ms
- [ ] Error rate: _____ %
- [ ] Success: [ ]

---

## Execution Phase 6: Metrics Correlation Analysis

### Compare k6 Results with Prometheus Metrics

**Request Counts:**
- [ ] k6 reported: _____ total requests
- [ ] Prometheus `http_requests_total` = _____
- [ ] Variance: _____ % (acceptable if < 5%)
- [ ] Status: ✓ / ✗

**Error Rates:**
- [ ] k6 reported: _____ % errors
- [ ] Prometheus error rate = _____ %
- [ ] Variance: _____ % (acceptable if < 1%)
- [ ] Status: ✓ / ✗

**Latency (P95):**
- [ ] k6 reported: _____ ms
- [ ] Prometheus query result: _____ ms
- [ ] Variance: _____ % (acceptable if < 10%)
- [ ] Status: ✓ / ✗

**Latency (P99):**
- [ ] k6 reported: _____ ms
- [ ] Prometheus query result: _____ ms
- [ ] Variance: _____ % (acceptable if < 10%)
- [ ] Status: ✓ / ✗

**Request Rate:**
- [ ] k6 reported: _____ rps
- [ ] Prometheus rate: _____ rps
- [ ] Variance: _____ % (acceptable if < 10%)
- [ ] Status: ✓ / ✗

**Correlation Result:**
- [ ] PASS (all metrics align)
- [ ] PARTIAL (describe mismatches): _______
- [ ] FAIL (describe): _________________

---

## Execution Phase 7: Load Test (Extended)

### Run Load Test with Monitoring

**Configuration:**
- [ ] Test type: Load Test (10→50 VU ramp)
- [ ] Duration: 5 minutes
- [ ] Monitoring: All systems active

**Execute:**
```bash
k6 run load_test_load.js
```

- [ ] Test started: _____:_____
- [ ] Test completed: _____:_____

**Monitoring Observations:**

Grafana during test:
- [ ] Request rate graph shows ramp-up (10→50)
- [ ] Latency increases gradually (expected)
- [ ] CPU/Memory increase with load
- [ ] No sudden drops (no failures)

Jaeger during test:
- [ ] Trace count increases with load
- [ ] No error spikes
- [ ] Latencies track with load
- [ ] All traces show successful completion

**Prometheus Metrics After Test:**
- [ ] Max request rate: _____ rps
- [ ] Peak P99 latency: _____ ms (should be < 1000ms)
- [ ] Error rate: _____ % (should be < 5%)
- [ ] Peak CPU: _____ m (should be < 500m)
- [ ] Peak Memory: _____ MB (should be < 512MB)

**Load Test Result:**
- [ ] PASS (all SLOs met)
- [ ] PARTIAL (describe): _______
- [ ] FAIL (describe): _________________

---

## Final Verification

### All Integration Points Verified

- [ ] Prometheus: Targets UP, metrics collected, queries work
- [ ] Grafana: Dashboards functional, data accurate, updates real-time
- [ ] Jaeger: Traces collected, spans detailed, errors tracked
- [ ] AlertManager: Rules loaded, alerts responsive
- [ ] Correlation: k6 metrics match Prometheus (±10% variance)

### Monitoring System Assessment

**Overall Status:**
- [ ] PASS (all systems integrated and functional)
- [ ] CONDITIONAL PASS (minor issues, documented)
- [ ] FAIL (significant issues, requires remediation)

**Critical Issues Found:**
- [ ] Issue 1: _______________________
  - [ ] Severity: Critical / High / Medium / Low
  - [ ] Resolution: _______________________
  - [ ] Status: Resolved / Pending

- [ ] Issue 2: _______________________
  - [ ] Severity: Critical / High / Medium / Low
  - [ ] Resolution: _______________________
  - [ ] Status: Resolved / Pending

**Sign-Off:**

Performed by: _________________________ Date: _________

Phase 6 Status: [ ] PASS  [ ] CONDITIONAL PASS  [ ] FAIL

Ready for Phase 7: [ ] YES  [ ] NO (describe blockers): ________

---

## Next Steps

Upon completion of Phase 6:

- [ ] Compile all metrics and observations
- [ ] Create summary report
- [ ] Document any issues found
- [ ] Get stakeholder sign-off
- [ ] Proceed to Phase 7 (Final Report)

---

_Checklist Generated: February 18, 2026_
_Phase: 6 / 7 (Integration Testing)_
_Status: Ready for Execution_
