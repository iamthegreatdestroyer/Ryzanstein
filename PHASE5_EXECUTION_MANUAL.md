# Phase 5: Load Testing Execution Manual

**Date:** February 19, 2026
**Status:** Ready for Execution
**Duration:** Approximately 3-4 hours (full suite)
**Prerequisite:** k6 installed + API running with port-forward

---

## Overview

Phase 5 validates the Ryzanstein LLM API performance under various load conditions. This manual guides execution of 5 load test scenarios with SLO validation and results compilation.

**Critical Success Criteria:**
- ✅ All 5 endpoints respond correctly under load
- ✅ P99 latency < 1000ms under normal conditions
- ✅ Error rate < 5% throughout all tests
- ✅ API recovers gracefully from spike events
- ✅ No memory leaks or resource exhaustion over 1 hour

---

## Prerequisites

### 1. k6 Installation Status
k6 is currently being installed via Chocolatey. Verify installation:

```bash
k6 version
```

**Expected Output:**
```
k6 v0.50.0 (go1.20.x, windows/amd64)
```

If not installed yet, install manually:

**Option A: Chocolatey (Recommended)**
```powershell
choco install k6 -y
```

**Option B: Direct Download (Windows)**
1. Visit: https://github.com/grafana/k6/releases
2. Download: `k6-v0.50.0-windows-amd64.zip`
3. Extract to: `C:\Program Files\k6\`
4. Add to PATH if necessary

**Option C: WSL/Linux**
```bash
sudo apt-get update && sudo apt-get install k6
```

### 2. API Verification

Ensure port-forward is active:

```bash
# Terminal 1: Port-forward
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000

# Terminal 2: Verify API
curl http://localhost:8000/health
# Expected: {"status":"healthy","service":"ryzanstein-api"}
```

### 3. Results Directory

Create results directory:

```bash
mkdir -p s:\Ryot\load_test_results
```

---

## Test Suite Overview

| Test | Duration | Load Pattern | Purpose | SLO |
|------|----------|--------------|---------|-----|
| **Smoke** | 30 sec | 1 VU baseline | Sanity check all endpoints | P95 < 500ms |
| **Load** | 5 min | 10→25→50 VU ramp | Normal operation validation | P99 < 2000ms |
| **Spike** | 5 min | 10→100→10 VU spike | Graceful spike handling | P99 < 5000ms |
| **Stress** | 30 min | 100→500→1000 VU | Breaking point identification | P99 < 5000ms |
| **Endurance** | 60 min | 25 VU constant | Memory leak detection | P99 < 2000ms |

---

## Execution Instructions

### Option 1: Run All Tests (Recommended for Full Validation)

**Total Duration:** ~3-4 hours

```bash
cd s:\Ryot
k6 run load_test_smoke.js
k6 run load_test_load.js
k6 run load_test_spike.js
k6 run load_test_stress.js
k6 run load_test_endurance.js
```

**Or using the orchestration script** (if bash working):

```bash
bash run_load_tests.sh full
```

### Option 2: Run Tests Individually

**1. Smoke Test (30 seconds) - ALWAYS RUN FIRST**

```bash
cd s:\Ryot
k6 run load_test_smoke.js \
  --out json=load_test_results/smoke_$(date +%Y%m%d_%H%M%S).json
```

**Expected Output:**
```
✓ health status is 200
✓ models status is 200
✓ chat completion status is 200
✓ embeddings status is 200
✓ root status is 200

checks..................: 100%
http_requests...........: 5 (per VU)
http_req_duration.......: avg=250ms p(95)=400ms p(99)=450ms
```

**2. Load Test (5 minutes)**

```bash
cd s:\Ryot
k6 run load_test_load.js \
  --out json=load_test_results/load_$(date +%Y%m%d_%H%M%S).json
```

**Expected Output:**
- Total requests: ~300-400
- Error rate: < 5%
- P99 latency: < 2000ms
- Throughput: > 10 req/s

**3. Spike Test (5 minutes)**

```bash
cd s:\Ryot
k6 run load_test_spike.js \
  --out json=load_test_results/spike_$(date +%Y%m%d_%H%M%S).json
```

**Expected Behavior:**
- Normal load (first 30s): Latency ~500-800ms
- Spike period (1-2m): Latency increases to 1500-3000ms
- Recovery (2-5m): Latency returns to baseline
- Error rate: < 10%

**4. Stress Test (30 minutes)**

```bash
cd s:\Ryot
k6 run load_test_stress.js \
  --out json=load_test_results/stress_$(date +%Y%m%d_%H%M%S).json
```

**Expected Observations:**
- At 100 VU: ~500ms latency, <1% errors
- At 500 VU: ~1000ms latency, ~2-3% errors
- At 1000 VU: ~2000-3000ms latency, ~5-8% errors
- Breaking point: Identify where error rate exceeds threshold

**5. Endurance Test (60 minutes)**

```bash
cd s:\Ryot
k6 run load_test_endurance.js \
  --out json=load_test_results/endurance_$(date +%Y%m%d_%H%M%S).json
```

**Expected Behavior:**
- Consistent latency throughout: 600-900ms
- Error rate: < 2%
- No performance degradation over time
- Verify memory usage stays constant (check Kubernetes metrics)

### Option 3: Run Subset (Quick Validation)

```bash
# Quick validation: Smoke + Load only (~6 minutes)
cd s:\Ryot
k6 run load_test_smoke.js
k6 run load_test_load.js
```

---

## Monitoring During Tests

### Open in Parallel Terminals

**Terminal 1: Prometheus Metrics**
```bash
# Port-forward and open http://localhost:9090
kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090
```

**Terminal 2: Grafana Dashboard**
```bash
# Port-forward and open http://localhost:3000
kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000
# Login: admin/admin
# Dashboard: "Ryzanstein API Performance"
```

**Terminal 3: Jaeger Traces**
```bash
# Port-forward and open http://localhost:16686
kubectl port-forward -n ryzanstein-staging svc/jaeger 16686:16686
# Search for traces during load test execution
```

### Key Metrics to Monitor

In **Prometheus/Grafana:**
- `http_request_duration_seconds` (should remain < 2000ms)
- `http_requests_total` (track error rate)
- `container_memory_usage_bytes` (should be stable)
- `container_cpu_usage_seconds_total` (should not spike beyond 80%)

---

## Results Analysis

### Interpreting k6 Output

**Example Smoke Test Results:**

```
   13 ↓ 13 requests to /health
   ✓ health status is 200
   ✓ health response time < 100ms
```

**Metrics Definitions:**
- `checks`: Pass rate of assertions (target: 100%)
- `http_requests`: Total requests sent
- `http_req_duration`: Response time percentiles
  - `p(50)`: 50th percentile (median)
  - `p(95)`: 95th percentile (good performance)
  - `p(99)`: 99th percentile (SLO threshold)
- `http_req_failed`: Error count (connection failures, 5xx, timeouts)

### SLO Validation Checklist

After each test, verify:

- [ ] **Smoke Test**: P95 < 500ms (✓ if all endpoints respond)
- [ ] **Load Test**: P99 < 2000ms (✓ if stress is acceptable)
- [ ] **Spike Test**: Error rate < 10% during spike (✓ if recovery is rapid)
- [ ] **Stress Test**: Identify breaking point (✓ if occurs at > 1000 VU)
- [ ] **Endurance Test**: Latency stable over 60 min (✓ if no degradation)

### Example Pass/Fail Criteria

**✅ PASS Condition:**
```
Smoke: P95 < 500ms ✓
Load: P99 < 2000ms, Error < 5% ✓
Spike: Recovery time < 2 min ✓
Stress: Breaking point > 1000 VU ✓
Endurance: Latency stable, Error < 2% ✓
```

**❌ FAIL Condition:**
```
Load: P99 > 5000ms ✗ → Investigate API bottleneck
Stress: Breaking point < 500 VU ✗ → Needs optimization
Endurance: Latency increases > 30% ✗ → Possible memory leak
```

---

## Troubleshooting

### Issue: k6 Command Not Found

```bash
# Option 1: Verify PATH
echo $PATH

# Option 2: Find k6 location
where k6
# Windows
Get-Command k6

# Option 3: Use absolute path
C:\Program Files\k6\k6.exe run load_test_smoke.js
```

### Issue: Connection Refused (http://localhost:8000)

```bash
# Verify port-forward is active
kubectl get pods -n ryzanstein-staging

# Restart port-forward
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000
```

### Issue: API Timeouts (Test Hangs)

**Check API logs:**
```bash
kubectl logs -n ryzanstein-staging deployment/ryzanstein-api -f
```

**Reduce load and retry:**
```bash
# Edit load_test_load.js, change target VU from 50 to 25
# Re-run test
```

### Issue: High Error Rate (>10%)

**Possible Causes:**
1. API container unstable (check CPU/memory)
2. Load exceeds available resources
3. Database connection pool exhausted

**Solutions:**
```bash
# Check resource usage
kubectl top pod -n ryzanstein-staging

# Restart API pod
kubectl rollout restart deployment/ryzanstein-api -n ryzanstein-staging

# Reduce load
# Edit test file, reduce VU target
```

---

## Results Compilation

After completing all tests, compile results:

```bash
# List all results
ls -la s:\Ryot\load_test_results/

# Create summary report
# (See Phase 5 Completion Report template below)
```

### Phase 5 Completion Report Template

**File:** `PHASE5_LOAD_TESTING_RESULTS.md`

```markdown
# Phase 5: Load Testing Results

## Executive Summary
- **Overall Status:** [PASS / FAIL]
- **Date:** [Date]
- **Duration:** [Total time]
- **Tests Executed:** [5/5]

## Test Results

### Smoke Test
- Status: ✓ PASS
- Duration: 30 sec
- Requests: 5
- Error Rate: 0%
- P95 Latency: [X]ms
- P99 Latency: [X]ms

### Load Test
- Status: ✓ PASS / ⚠ WARN / ❌ FAIL
- Duration: 5 min
- Requests: [Count]
- Error Rate: [X]%
- P95 Latency: [X]ms
- P99 Latency: [X]ms

[... repeat for Spike, Stress, Endurance ...]

## SLO Compliance

| SLO | Target | Actual | Status |
|-----|--------|--------|--------|
| P99 Latency (Normal) | < 1000ms | [X]ms | ✓ |
| Error Rate (Normal) | < 5% | [X]% | ✓ |
| P99 Latency (Stress) | < 2000ms | [X]ms | ✓ |
| Error Rate (Spike) | < 10% | [X]% | ✓ |
| Endurance Stability | Latency stable | [Stable/Degrading] | ✓ |

## Recommendations

[Any findings or optimizations needed]

## Sign-off
- Load Testing: ✓ Complete
- Results Analyzed: ✓ Yes
- Ready for Phase 6: ✓ Yes
```

---

## Next Steps

After Phase 5 completion:

1. ✅ **Compile results** into PHASE5_LOAD_TESTING_RESULTS.md
2. ✅ **Verify SLO compliance** (all criteria met?)
3. ✅ **Review Grafana dashboards** for anomalies
4. ✅ **Check Jaeger traces** for bottlenecks
5. ✅ **Proceed to Phase 6** (Integration Testing)

---

## Commands Quick Reference

```bash
# Smoke test (quick check)
cd s:\Ryot && k6 run load_test_smoke.js

# Full suite (3-4 hours)
cd s:\Ryot && for test in smoke load spike stress endurance; do \
  k6 run load_test_$test.js --out json=load_test_results/${test}_$(date +%s).json; \
done

# Monitor API during tests
kubectl logs -n ryzanstein-staging deployment/ryzanstein-api -f

# Check resource usage
kubectl top pods -n ryzanstein-staging

# View Prometheus metrics
kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090
# Open: http://localhost:9090

# View Grafana dashboards
kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000
# Open: http://localhost:3000 (admin/admin)
```

---

**Status:** Ready for Execution
**Next Action:** Install k6 (if needed) → Execute load tests
**Questions?** Review PHASE5_LOAD_TESTING_GUIDE.md for detailed documentation
