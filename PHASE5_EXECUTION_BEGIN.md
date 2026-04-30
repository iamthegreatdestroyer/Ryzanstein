# Phase 5: Load Testing Execution - BEGIN NOW

**Status:** ✅ API Healthy, Ready to Execute Tests
**Date:** February 19, 2026
**API Health:** `{"status":"healthy","service":"ryzanstein-api"}`
**Port-Forward:** Active on localhost:8000

---

## 🚀 Immediate Next Steps

### Step 1: Verify k6 Installation

**Option A: Check if k6 installed (in new PowerShell)**
```powershell
k6 version
# Expected: k6 v0.50.0 (go1.20.x, windows/amd64)
```

**Option B: If k6 not in PATH, install manually**

Download from: https://github.com/grafana/k6/releases/download/v0.50.0/k6-v0.50.0-windows-amd64.zip

Steps:
1. Download the ZIP file
2. Extract to: `C:\Program Files\k6\`
3. Add to PATH:
   - Right-click "This PC" → Properties
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - New (under System variables)
   - Variable name: `PATH`
   - Variable value: `C:\Program Files\k6\` (add to existing, separated by semicolon)
4. Restart PowerShell/CMD
5. Verify: `k6 version`

**Option C: If installation still in progress**
- Wait 5-10 more minutes for Chocolatey to finish
- Then verify: `k6 version`

---

### Step 2: Start Load Tests

Once k6 is verified, open a PowerShell/CMD window and run:

```powershell
cd s:\Ryot
k6 run load_test_smoke.js
```

This will:
1. Run 1 virtual user for 30 seconds
2. Test all 5 API endpoints
3. Display results in console
4. Take ~1 minute total

**Expected Output:**
```
     checks...................: 100%
     data_received.............: 1.2 kB
     data_sent.................: 850 B
     http_req_duration.........: avg=350ms p(95)=420ms p(99)=480ms
     http_reqs.................: 5     (5/sec)
     iterations................: 1
```

---

## 📋 Full Test Suite Execution (Recommended)

After smoke test passes, run all 5 tests in sequence:

```powershell
cd s:\Ryot

# Test 1: Smoke (30 seconds)
Write-Host "========== SMOKE TEST =========="
k6 run load_test_smoke.js

# Test 2: Load (5 minutes)
Write-Host "========== LOAD TEST =========="
k6 run load_test_load.js

# Test 3: Spike (5 minutes)
Write-Host "========== SPIKE TEST =========="
k6 run load_test_spike.js

# Test 4: Stress (30 minutes)
Write-Host "========== STRESS TEST =========="
k6 run load_test_stress.js

# Test 5: Endurance (60 minutes)
Write-Host "========== ENDURANCE TEST =========="
k6 run load_test_endurance.js

Write-Host "========== ALL TESTS COMPLETE =========="
```

**Total Duration:** ~3-4 hours

---

## 🎯 Execution Checklist

Before running tests, verify:

- [ ] k6 installed: `k6 version` returns version number
- [ ] API running: `curl http://localhost:8000/health` returns JSON
- [ ] Port-forward active: API responds in ~50-200ms
- [ ] Results directory exists: `s:\Ryot\load_test_results\`
- [ ] Load test scripts exist: All 5 `.js` files in `s:\Ryot\`

---

## 📊 Quick Test Reference

| Test | Command | Duration | Load | Purpose |
|------|---------|----------|------|---------|
| Smoke | `k6 run load_test_smoke.js` | 30s | 1 VU | Quick check |
| Load | `k6 run load_test_load.js` | 5m | 10→50 VU | Normal ops |
| Spike | `k6 run load_test_spike.js` | 5m | 10→100→10 VU | Recovery |
| Stress | `k6 run load_test_stress.js` | 30m | 100→1000 VU | Breaking point |
| Endurance | `k6 run load_test_endurance.js` | 60m | 25 VU | Stability |

---

## 🖥️ Optional: Monitor During Tests

Open these in separate terminal windows for live monitoring:

**Window 1: Prometheus Metrics**
```bash
kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090
# Open: http://localhost:9090
# Query: rate(http_requests_total[1m])
```

**Window 2: Grafana Dashboards**
```bash
kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000
# Open: http://localhost:3000
# Login: admin/admin
# Dashboard: "Ryzanstein API Performance"
```

**Window 3: Jaeger Traces**
```bash
kubectl port-forward -n ryzanstein-staging svc/jaeger 16686:16686
# Open: http://localhost:16686
# Service: ryzanstein-api
```

**Window 4: API Logs**
```bash
kubectl logs -n ryzanstein-staging deployment/ryzanstein-api -f
```

---

## ✅ When Tests Complete

1. **Review Console Output**
   - Check for PASS/FAIL on each test
   - Note P95 and P99 latency percentiles
   - Verify error rates are within SLO

2. **Verify SLO Compliance**

   | SLO | Target | Pass/Fail |
   |-----|--------|-----------|
   | Smoke P95 latency | < 500ms | ? |
   | Load P99 latency | < 2000ms | ? |
   | Load error rate | < 5% | ? |
   | Spike recovery | < 2 min | ? |
   | Spike error rate | < 10% | ? |
   | Stress breaking point | > 1000 VU | ? |
   | Endurance latency | Stable | ? |
   | Endurance error rate | < 2% | ? |

3. **Save Results** (optional)
   ```bash
   # Re-run with results saved
   k6 run load_test_smoke.js --out json=load_test_results/smoke.json
   ```

4. **Proceed to Phase 6**
   - Review Integration Testing Guide
   - Validate monitoring stack integration

---

## 🔧 Troubleshooting

### k6 Still Not Found
```powershell
# Check installation directory
dir "C:\Program Files\k6\"

# If empty, download manually:
# https://github.com/grafana/k6/releases/download/v0.50.0/k6-v0.50.0-windows-amd64.zip

# Extract ZIP contents to C:\Program Files\k6\
# Restart PowerShell
# Verify: k6 version
```

### API Connection Refused
```powershell
# Restart port-forward in new terminal
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000

# Test connection
curl http://localhost:8000/health
```

### Test Hangs or Timeouts
```powershell
# Check if API has resource issues
kubectl top pods -n ryzanstein-staging

# Check API logs for errors
kubectl logs -n ryzanstein-staging deployment/ryzanstein-api --tail=50

# If CPU/memory maxed, reduce VU count in test script
# Edit load_test_load.js, change target VU from 50 to 25
```

### High Error Rate (>10%)
```powershell
# Reduce load
# Edit test file, decrease VU targets

# Check API pod status
kubectl describe pod -n ryzanstein-staging <pod-name>

# Restart API if needed
kubectl rollout restart deployment/ryzanstein-api -n ryzanstein-staging
```

---

## 📝 Commands Summary

```powershell
# Verify API health
curl http://localhost:8000/health

# Check k6 version
k6 version

# Run smoke test (30 sec)
cd s:\Ryot
k6 run load_test_smoke.js

# Run load test (5 min)
k6 run load_test_load.js

# Run spike test (5 min)
k6 run load_test_spike.js

# Run stress test (30 min)
k6 run load_test_stress.js

# Run endurance test (60 min)
k6 run load_test_endurance.js

# Run all tests with results saved
k6 run load_test_smoke.js --out json=load_test_results/smoke.json
k6 run load_test_load.js --out json=load_test_results/load.json
k6 run load_test_spike.js --out json=load_test_results/spike.json
k6 run load_test_stress.js --out json=load_test_results/stress.json
k6 run load_test_endurance.js --out json=load_test_results/endurance.json

# Check Kubernetes status
kubectl get pods -n ryzanstein-staging
kubectl top pods -n ryzanstein-staging
kubectl logs -n ryzanstein-staging deployment/ryzanstein-api -f
```

---

## 🎯 Success Criteria

Phase 5 is **SUCCESSFUL** when:

✅ All 5 tests execute without critical errors
✅ Smoke test passes (all endpoints respond)
✅ Load test SLOs met (P99 < 2000ms, Error < 5%)
✅ Spike test recovers within 2 minutes
✅ Stress test identifies breaking point at >1000 VU
✅ Endurance test shows stable performance
✅ No cascading failures observed
✅ Results documented

---

## 📚 Documentation Reference

- **Quick Start:** `PHASE5_QUICK_START.txt`
- **Detailed Guide:** `PHASE5_EXECUTION_MANUAL.md`
- **Status Report:** `PHASE5_STATUS_REPORT.md`
- **This File:** `PHASE5_EXECUTION_BEGIN.md`

---

## 🚀 Ready to Begin!

**API Status:** ✅ Healthy
**Port-Forward:** ✅ Active
**Scripts:** ✅ Ready
**Documentation:** ✅ Complete

**Next Action:**
1. Verify k6 installation
2. Run: `k6 run load_test_smoke.js`
3. If PASS, continue with remaining tests

**Estimated Total Duration:** 3-4 hours (full suite)

---

**Generated:** February 19, 2026
**Ready to Execute:** YES ✅
**Phase 5 Status:** EXECUTION IN PROGRESS
