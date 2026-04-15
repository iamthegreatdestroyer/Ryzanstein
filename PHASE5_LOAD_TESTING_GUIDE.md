# Phase 5: Load Testing Execution Guide

**Status:** Ready to Execute
**Date:** February 18, 2026
**Purpose:** Validate SLO thresholds and API performance under various load conditions

---

## Overview

Five load testing scenarios have been prepared to comprehensively validate the Ryzanstein LLM API:

1. **Smoke Test** — Baseline verification (1 VU, 30s)
2. **Load Test** — Sustained load (10→50 VUs, 5 min)
3. **Stress Test** — Breaking point identification (100→2000 VUs, 30 min)
4. **Endurance Test** — Long-running stability (25 VUs, 1 hour)
5. **Spike Test** — Sudden load handling (10→1000 VUs, 5 min)

---

## Prerequisites

### Required
- **k6** installed (performance testing tool)
- **API running** on `http://localhost:8000` (with port-forward)
- **Monitoring** (optional but recommended: Prometheus, Grafana, Jaeger)

### Installation

**Windows (via Chocolatey):**
```bash
choco install k6
```

**Windows (via Direct Download):**
```bash
# Download from: https://github.com/grafana/k6/releases
# Add k6.exe to PATH
```

**macOS:**
```bash
brew install k6
```

**Linux:**
```bash
sudo apt-get install k6
```

### Verify Installation
```bash
k6 version
# Expected: k6 v0.x.x
```

---

## Setup

### Step 1: Ensure API is Accessible

```bash
# In one terminal, keep API port-forward active
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000

# In another terminal, verify API
curl http://localhost:8000/health
# Expected: {"status":"healthy","service":"ryzanstein-api"}
```

### Step 2: (Optional) Enable Monitoring

For real-time monitoring during tests, port-forward all monitoring services:

```bash
# Open new terminal tabs/windows for each port-forward

# Terminal 1: Keep this for test execution
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000

# Terminal 2: Prometheus (metrics)
kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090
# Access: http://localhost:9090

# Terminal 3: Grafana (dashboards)
kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000
# Access: http://localhost:3000 (admin/admin123)

# Terminal 4: Jaeger (tracing)
kubectl port-forward -n ryzanstein-staging svc/jaeger 16686:16686
# Access: http://localhost:16686
```

---

## Test Execution

### Test 1: Smoke Test (30 seconds)

**Purpose:** Quick sanity check that API is responsive
**Load:** 1 VU
**Duration:** 30 seconds
**Expected Time:** < 1 minute

```bash
cd s:\Ryot
k6 run load_test_smoke.js
```

**Expected Output:**
```
✓ GET / status is 200
✓ GET /health status is 200
✓ GET /v1/models status is 200
✓ POST /v1/chat/completions status is 200
✓ POST /v1/embeddings status is 200

http_req_duration..........: avg=400ms, min=100ms, med=350ms, max=800ms, p(95)=750ms, p(99)=800ms
http_req_failed............: 0.00%
http_requests..............: 5 per second
```

**SLO Validation:**
- ✅ All endpoints respond with HTTP 200
- ✅ P99 latency < 1000ms
- ✅ Error rate 0%

---

### Test 2: Load Test (5 minutes)

**Purpose:** Sustained load to measure performance degradation
**Load:** 10→25→50 VUs (ramping)
**Duration:** 5 minutes
**Expected Time:** ~10 minutes (includes k6 startup/shutdown)

```bash
cd s:\Ryot
k6 run load_test_load.js
```

**Expected Output:**
```
Stage 1 (1-2 min): 10 VUs
  http_req_duration: p(99)=600ms
  http_req_failed: 0.00%
  http_requests: 10 rps

Stage 2 (2-4 min): 25 VUs
  http_req_duration: p(99)=800ms
  http_req_failed: 0.00%
  http_requests: 25 rps

Stage 3 (4-5 min): 50 VUs
  http_req_duration: p(99)=950ms
  http_req_failed: 0.00%
  http_requests: 50 rps
```

**SLO Validation:**
- ✅ P99 latency < 1000ms (achieved at 50 VUs)
- ✅ Error rate < 5%
- ✅ Throughput > 10 rps

---

### Test 3: Stress Test (30 minutes)

**Purpose:** Identify system breaking point
**Load:** 100→500→1000→2000 VUs (aggressive ramp)
**Duration:** 30 minutes
**Expected Time:** ~45 minutes (includes setup/teardown)

```bash
cd s:\Ryot
k6 run load_test_stress.js
```

**Expected Behavior:**

| VU Count | Expected P99 | Expected Error Rate | Observations |
|----------|--------------|-------------------|---|
| 100 | 500ms | 0% | Normal performance |
| 500 | 800ms | 0-2% | Slight degradation |
| 1000 | 1500ms | 2-5% | Noticeable slowdown |
| 2000 | 2000ms+ | 5-10% | Breaking point reached |

**Breaking Point Analysis:**
- Identify the VU count where error rate exceeds 5%
- Measure recovery time when scaled back down
- Monitor for cascading failures

**SLO Assessment:**
- ✅ System sustains up to 1000 VUs
- ⚠️ Beyond 1000 VUs: Graceful degradation
- ✅ No cascading failures observed

---

### Test 4: Endurance Test (1 hour)

**Purpose:** Verify system stability and detect resource leaks
**Load:** 25 VUs (constant)
**Duration:** 1 hour
**Expected Time:** ~1.5 hours (includes setup/teardown)

```bash
cd s:\Ryot
k6 run load_test_endurance.js
```

**Expected Behavior:**

Latency and error rates should remain **constant** throughout the test:

```
First 10 minutes:
  P99 latency: 600ms
  Error rate: 0%

Middle 30 minutes:
  P99 latency: 600ms (±50ms)
  Error rate: 0% (±0.1%)

Final 10 minutes:
  P99 latency: 600ms (±50ms)
  Error rate: 0% (±0.1%)
```

**Stability Indicators:**
- ✅ Latency remains constant (no memory leaks)
- ✅ Error rate remains constant (no resource exhaustion)
- ✅ No gradual performance degradation

**Pass Criteria:**
- P99 latency does not increase > 10% over time
- Error rate remains < 2% throughout

---

### Test 5: Spike Test (5 minutes)

**Purpose:** Validate graceful handling of sudden load spikes
**Load:** 10 → 1000 → 10 VUs (spike pattern)
**Duration:** 5 minutes
**Expected Time:** ~10 minutes

```bash
cd s:\Ryot
k6 run load_test_spike.js
```

**Expected Behavior:**

```
Phase 1 (0-30s): 10 VUs
  Response: Normal performance

Phase 2 (30s-1.5m): Spike to 1000 VUs
  Behavior: Brief latency spike, some 503 errors acceptable
  P99 Latency: 1000-2000ms (temporary acceptable)

Phase 3 (1.5m-4.5m): Back to 10 VUs
  Recovery: Latency drops back to baseline < 1 minute
  No cascading failures
```

**Spike Handling Assessment:**
- ✅ System queues requests during spike
- ✅ Recovers quickly when load reduced
- ✅ No permanent performance degradation

---

## Monitoring During Tests

### Real-time Metrics (via Prometheus)

During test execution, query Prometheus for real-time metrics:

```promql
# Request rate (requests per second)
rate(http_requests_total[1m])

# Error rate
rate(http_requests_total{status=~"5.."}[1m])

# Latency (P95)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))

# Latency (P99)
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[1m]))

# API container CPU usage
rate(container_cpu_usage_seconds_total{pod=~"ryzanstein-api.*"}[1m])

# API container memory usage
container_memory_usage_bytes{pod=~"ryzanstein-api.*"}
```

### Grafana Dashboards

Access Grafana at `http://localhost:3000` and view:
- **Inference Performance** dashboard for request metrics
- **Resource Usage** dashboard for CPU/memory during tests
- **System Health** dashboard for circuit breaker state

### Jaeger Traces

Access Jaeger at `http://localhost:16686` to:
- View request flow and span breakdown
- Identify performance bottlenecks
- Analyze latency distribution

---

## SLO Thresholds

### Success Criteria

| Metric | Smoke | Load | Stress | Endurance | Spike |
|--------|-------|------|--------|-----------|-------|
| **P99 Latency** | <1000ms | <1000ms | <2000ms | <700ms | <2000ms |
| **P95 Latency** | <500ms | <800ms | <1500ms | <500ms | <1500ms |
| **Error Rate** | 0% | <5% | <10% | <2% | <10% |
| **Availability** | 100% | 95%+ | 90%+ | 98%+ | 90%+ |
| **Throughput** | 1 rps | 10+ rps | Degradation OK | 12+ rps | Spike recovery |

### Pass/Fail Criteria

**PASS (Production Ready):**
- ✅ All tests complete without critical failures
- ✅ P99 latency < 1000ms under normal load (Load test)
- ✅ Error rate < 5% under normal load
- ✅ System recovers from spikes within 2 minutes
- ✅ No cascading failures observed
- ✅ Resource usage stable (Endurance test)

**FAIL (Requires Optimization):**
- ❌ P99 latency > 1500ms at 50 VUs
- ❌ Error rate > 10% at normal load
- ❌ Cascading failures (exponential error increase)
- ❌ Resource leaks (performance degradation over time)
- ❌ No recovery from spikes (permanent failure)

---

## Test Execution Schedule

### Quick Validation (15 minutes)
```bash
# Run smoke test only
k6 run load_test_smoke.js
```

### Standard Validation (1-2 hours)
```bash
# Run smoke + load + spike tests
k6 run load_test_smoke.js
k6 run load_test_load.js
k6 run load_test_spike.js
```

### Comprehensive Validation (2-3 hours)
```bash
# Run all tests except endurance
k6 run load_test_smoke.js
k6 run load_test_load.js
k6 run load_test_spike.js
k6 run load_test_stress.js
```

### Full Production Validation (5+ hours)
```bash
# Run all tests including endurance
k6 run load_test_smoke.js
k6 run load_test_load.js
k6 run load_test_spike.js
k6 run load_test_stress.js
k6 run load_test_endurance.js
```

---

## Output Analysis

### k6 Console Output

Each test generates:
- Real-time request statistics
- Threshold pass/fail status
- Summary statistics (avg, p95, p99, max latency)
- Error rate and failure counts

### Summary Files

Each test also generates `summary.json`:
```json
{
  "metrics": {
    "http_req_duration": {
      "value": {
        "p": { "50": 400, "95": 750, "99": 950 }
      }
    },
    "http_reqs": { "value": 300 },
    "http_req_failed": { "value": 0 }
  }
}
```

### Comparison Across Tests

Create a results spreadsheet:

| Test | P50 | P95 | P99 | Error % | Status |
|------|-----|-----|-----|---------|--------|
| Smoke | 350ms | 500ms | 800ms | 0% | ✅ PASS |
| Load | 400ms | 750ms | 950ms | 0% | ✅ PASS |
| Stress | 600ms | 1200ms | 1800ms | 5% | ✅ PASS |
| Endurance | 400ms | 600ms | 700ms | 0% | ✅ PASS |
| Spike | 800ms | 1200ms | 1500ms | 5% | ✅ PASS |

---

## Troubleshooting

### Issue: "Connection refused"
```bash
# Verify API is running and accessible
curl http://localhost:8000/health
# If fails, restart port-forward:
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000
```

### Issue: "k6 command not found"
```bash
# Verify k6 installation
k6 version

# If not found, install:
# Windows: choco install k6
# macOS: brew install k6
# Linux: sudo apt-get install k6
```

### Issue: "Extreme latency spike at start"
This is normal - k6 startup and connection pooling causes initial latency spike.
Tests automatically warm up before recording metrics.

### Issue: "High error rate in Stress test"
This is expected behavior at 2000 VUs. The API is intentionally pushed beyond capacity to identify the breaking point. Success is graceful degradation without cascading failures.

---

## Next Steps (Phase 6)

After load testing:

1. **Analyze Results:**
   - Compare metrics against SLO thresholds
   - Identify performance bottlenecks
   - Determine optimization opportunities

2. **Integration Testing:**
   - Verify monitoring data matches k6 results
   - Check Jaeger traces for request flow
   - Review Grafana dashboards for correlations

3. **Optimization (if needed):**
   - Increase API replica count
   - Optimize database queries
   - Implement caching strategies
   - Scale Kubernetes resources

4. **Re-test (if changes made):**
   - Run Load test to verify improvements
   - Run Stress test to check new breaking point

---

## Expected Timeline

| Test | Duration | Total Time |
|------|----------|-----------|
| Smoke | 30s | 1 min |
| Load | 5 min | 10 min |
| Spike | 5 min | 10 min |
| Stress | 30 min | 45 min |
| Endurance | 60 min | 1.5 hours |

**Total for Full Validation:** ~3 hours

---

## Summary

All k6 test scripts are ready to execute. The tests comprehensively validate:

- ✅ API responsiveness (Smoke test)
- ✅ Sustained load handling (Load test)
- ✅ System breaking point (Stress test)
- ✅ Long-term stability (Endurance test)
- ✅ Spike recovery (Spike test)

**Status:** 🟢 READY FOR EXECUTION

**Next Command:**
```bash
cd s:\Ryot
k6 run load_test_smoke.js
```

---

_Generated: February 18, 2026_
_Test Framework: k6 (Grafana k6)_
_API Target: http://localhost:8000_
_Kubernetes: ryzanstein-staging namespace_
