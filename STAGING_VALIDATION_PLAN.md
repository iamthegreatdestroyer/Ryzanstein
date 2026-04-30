# STAGING VALIDATION PLAN — Phase 4 Deliverables

**Date:** February 18, 2026
**Objective:** Validate all Phase 4 deliverables in a Kubernetes test environment before production cutover
**Target Environment:** minikube / local K8s cluster
**Duration:** 2-3 hours
**Reference:** [REF:STAGING-VALIDATION]

---

## VALIDATION SCOPE

### Phase 4 Deliverables to Validate

| Deliverable | Component | Validation Points |
|-------------|-----------|-------------------|
| **Docker Images** | Dockerfile.linux, docker-compose | Image build, startup, health checks |
| **Helm Charts** | Chart structure, values, templates | Linting, template rendering, deployment |
| **Monitoring Stack** | Prometheus, Grafana, Jaeger, AlertManager | Scrape targets, dashboards, alerts, traces |
| **Security Config** | RBAC, secrets, TLS setup | Permission checks, secret injection, mTLS |
| **Load Testing** | k6 scripts, thresholds, runbook | Script execution, threshold validation, SLO checks |

---

## VALIDATION PHASES

### Phase 0: Environment Setup (30 min)
- [ ] Verify minikube/K8s cluster availability
- [ ] Verify kubectl, helm, docker CLI available
- [ ] Check storage classes (need PVC support)
- [ ] Verify network connectivity

### Phase 1: Docker Image Validation (45 min)
- [ ] Build Dockerfile.linux
- [ ] Verify image size (<3 GB)
- [ ] Test container startup
- [ ] Test health check endpoints
- [ ] Verify all dependencies included

### Phase 2: Helm Chart Validation (60 min)
- [ ] Lint Helm chart (`helm lint`)
- [ ] Template rendering validation
- [ ] Development values deployment
- [ ] Production values dry-run
- [ ] Verify all pods reach Running state
- [ ] Verify service discovery

### Phase 3: Monitoring Stack Validation (60 min)
- [ ] Prometheus scrape targets healthy
- [ ] Grafana dashboards auto-load
- [ ] AlertManager receives alerts
- [ ] Jaeger receives traces
- [ ] Alert routing rules test

### Phase 4: Security Validation (45 min)
- [ ] RBAC permissions verified
- [ ] Secrets properly injected
- [ ] API authentication working
- [ ] Network policies applied
- [ ] Pod security standards enforced

### Phase 5: Load Testing Validation (60 min)
- [ ] k6 script syntax validation
- [ ] Execute Smoke test (baseline)
- [ ] Execute Load test (gradual ramp)
- [ ] Execute Stress test (breaking point)
- [ ] Verify SLO thresholds
- [ ] Analyze results and capacity planning

---

## DETAILED VALIDATION STEPS

### Phase 0: Environment Setup

```bash
# Check minikube status
minikube status

# Check cluster version
kubectl version --client
kubectl cluster-info

# Check storage classes
kubectl get storageclass

# Check node resources
kubectl top nodes
kubectl describe nodes

# Set up docker for minikube
eval $(minikube docker-env)
```

### Phase 1: Docker Image Validation

```bash
# Navigate to Ryot repo
cd s:\Ryot

# Build Dockerfile.linux
docker build -f Dockerfile.linux -t ryzanstein:staging .

# Check image size
docker images | grep ryzanstein

# Run container with health check
docker run -d \
  --name ryzanstein-test \
  -p 8000:8000 \
  -e JAEGER_ENABLED=false \
  -e PROMETHEUS_ENABLED=false \
  ryzanstein:staging

# Wait 10 seconds for startup
sleep 10

# Test health check
curl http://localhost:8000/health

# Check logs
docker logs ryzanstein-test

# Cleanup
docker rm -f ryzanstein-test
```

### Phase 2: Helm Chart Validation

```bash
# Validate chart syntax
helm lint ./helm/ryzanstein

# Generate templates (dry-run)
helm template ryzanstein ./helm/ryzanstein -f helm/ryzanstein/values-dev.yaml

# Check template rendering
helm template ryzanstein ./helm/ryzanstein -f helm/ryzanstein/values-production.yaml > /tmp/prod-manifest.yaml

# Validate YAML syntax
kubectl apply -f /tmp/prod-manifest.yaml --dry-run=client

# Deploy to minikube (dev config)
helm install ryzanstein ./helm/ryzanstein \
  -f helm/ryzanstein/values-dev.yaml \
  --namespace ryzanstein-staging \
  --create-namespace

# Wait for rollout
kubectl rollout status deployment/ryzanstein-api -n ryzanstein-staging --timeout=5m

# Check all resources
kubectl get all -n ryzanstein-staging

# Verify ConfigMaps loaded
kubectl get configmap -n ryzanstein-staging

# Test service connectivity
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000 &
curl http://localhost:8000/health
kill %1
```

### Phase 3: Monitoring Stack Validation

```bash
# Port-forward Prometheus
kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090 &

# Check targets (should be "Up")
curl http://localhost:9090/api/v1/targets

# Query metrics
curl 'http://localhost:9090/api/v1/query?query=up'

# Port-forward Grafana
kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000 &

# Check dashboards (via UI at http://localhost:3000)
# Login: admin / admin123 (from values-dev.yaml)
# Verify 4 dashboards visible

# Port-forward Jaeger
kubectl port-forward -n ryzanstein-staging svc/jaeger 16686:16686 &

# Check Jaeger UI at http://localhost:16686
# Verify service dependency graph

# Port-forward AlertManager
kubectl port-forward -n ryzanstein-staging svc/alertmanager 9093:9093 &

# Check AlertManager status
curl http://localhost:9093/api/v1/status

# Check alerts
curl http://localhost:9093/api/v1/alerts

kill %1 %2 %3 %4
```

### Phase 4: Security Validation

```bash
# Check RBAC bindings
kubectl get rolebindings -n ryzanstein-staging
kubectl get clusterrolebindings | grep ryzanstein

# Verify secrets
kubectl get secrets -n ryzanstein-staging

# Check secret data (safely)
kubectl get secret ryzanstein-api-keys -n ryzanstein-staging -o jsonpath='{.data}' | wc -c

# Verify pod security context
kubectl get pod -n ryzanstein-staging -o jsonpath='{.items[*].spec.securityContext}'

# Check NetworkPolicy
kubectl get networkpolicy -n ryzanstein-staging

# Test API authentication
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000 &

# Should fail without auth
curl http://localhost:8000/v1/chat/completions -X POST

# Should work with valid key (from secret)
API_KEY=$(kubectl get secret ryzanstein-api-keys -n ryzanstein-staging -o jsonpath='{.data.API_KEY}' | base64 -d)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/v1/chat/completions -X POST

kill %1
```

### Phase 5: Load Testing Validation

```bash
# Install k6 (if not already installed)
# Download from https://github.com/grafana/k6/releases or use apt/brew

# Create test directory
mkdir -p /tmp/k6-tests

# Copy k6 test scripts from LOAD_TESTING_GUIDE.md to /tmp/k6-tests/

# Port-forward API
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000 &

# Run Smoke test
k6 run /tmp/k6-tests/smoke_test.js

# Run Load test
k6 run /tmp/k6-tests/load_test.js

# Run Stress test
k6 run /tmp/k6-tests/stress_test.js

# Analyze results
# - Check latency (P99 < 1000ms)
# - Check error rate (< 1%)
# - Check throughput (RPS)

kill %1
```

---

## VALIDATION CHECKLIST

### Environment Setup
- [ ] minikube running with sufficient resources (4+ CPU, 8+ GB RAM)
- [ ] kubectl connectivity verified
- [ ] helm version 3.12+
- [ ] docker available and working
- [ ] Storage class available (for PVC)

### Docker Image
- [ ] Build succeeds without errors
- [ ] Final image < 3 GB
- [ ] Container starts and reaches healthy state
- [ ] /health endpoint returns 200 OK
- [ ] /health/ready endpoint returns 200 OK

### Helm Chart
- [ ] `helm lint` passes all checks
- [ ] Template rendering valid for dev AND production values
- [ ] Deployment succeeds (`helm install` completes)
- [ ] All pods reach "Running" state within 5 minutes
- [ ] ConfigMaps contain expected data
- [ ] Services have endpoints assigned

### Prometheus & Monitoring
- [ ] Prometheus targets show "Up" status
- [ ] Metrics queries return data points
- [ ] Grafana dashboards auto-provision
- [ ] All 4 dashboards visible in Grafana UI
- [ ] AlertManager receives alerts properly
- [ ] Jaeger service dependency graph working

### Security
- [ ] RBAC roles/rolebindings present
- [ ] Secrets properly injected into pods
- [ ] API authentication enforced (X-API-Key header required)
- [ ] Pod security context configured correctly
- [ ] NetworkPolicy rules applied (if enabled)

### Load Testing
- [ ] Smoke test passes (1 VU, 30s, no errors)
- [ ] Load test passes (gradual ramp, <10% error)
- [ ] Stress test identifies breaking point
- [ ] P99 latency < 1000ms during Load test
- [ ] Error rate < 1% sustained
- [ ] Capacity planning calculator produces valid results

### SLO Validation
- [ ] Availability: 99.9% uptime during tests
- [ ] Latency: P99 < 1000ms
- [ ] Throughput: 15-30 tok/s minimum
- [ ] Error budget: < 1% error rate

---

## EXPECTED OUTCOMES

### Success Criteria

✅ **Docker Image**
- Builds successfully in < 10 minutes
- Final image size: 2.0-2.8 GB
- Starts in < 5 seconds
- Health checks pass

✅ **Helm Deployment**
- All pods "Running" within 5 minutes
- All services have endpoints
- Resource requests/limits honored

✅ **Monitoring**
- Prometheus scrapes all targets successfully
- Grafana displays all dashboards
- Jaeger receives traces
- AlertManager routes alerts

✅ **Security**
- RBAC enforced
- Secrets protected
- API authentication working
- No security warnings

✅ **Load Testing**
- Smoke test: 0 errors
- Load test: <1% error rate
- Stress test: identifies breaking point at ~100 concurrent requests
- P99 latency: <500ms under Load (if API mock is fast)

### Potential Issues & Mitigation

| Issue | Symptom | Mitigation |
|-------|---------|-----------|
| PVC binding fails | Pod stuck in Pending | Create storage class or use emptyDir |
| Prometheus targets down | Scrape shows "Down" | Check service DNS, port accessibility |
| Grafana dashboards blank | No data in graphs | Wait 2-3 min for metrics, check datasource |
| k6 connection refused | Load test fails immediately | Verify port-forward, check API readiness |
| Memory pressure | Pods OOMKilled | Increase minikube memory or reduce replicas |

---

## VALIDATION TIMELINE

| Phase | Duration | Owner |
|-------|----------|-------|
| Phase 0: Setup | 30 min | Automation |
| Phase 1: Docker | 45 min | Automation |
| Phase 2: Helm | 60 min | Automation + Manual UI checks |
| Phase 3: Monitoring | 60 min | Automation + Manual verification |
| Phase 4: Security | 45 min | Automation + Manual checks |
| Phase 5: Load Testing | 60 min | Automation + Analysis |
| **Total** | **~5 hours** | |

---

## REPORTS & ARTIFACTS

After validation completes, generate:

1. **STAGING_VALIDATION_REPORT.md** — Summary of all validation phases
2. **HELM_DEPLOYMENT_TEST_RESULTS.md** — Pod status, resource usage
3. **MONITORING_VALIDATION_RESULTS.md** — Prometheus targets, Grafana dashboards
4. **LOAD_TEST_RESULTS.md** — k6 test output, latency/RPS/error analysis
5. **SECURITY_VALIDATION_RESULTS.md** — RBAC, secrets, authentication checks

---

## GO/NO-GO DECISION CRITERIA

### GO for Production If:
✅ All 5 phases pass validation
✅ No critical security issues
✅ All pods healthy and communicating
✅ Monitoring stack fully operational
✅ Load test: P99<1s, error rate<1%
✅ All 40+ production checklist items addressed

### NO-GO for Production If:
❌ Any phase fails validation
❌ Security vulnerabilities found
❌ Pods unable to reach Running state
❌ Monitoring stack offline
❌ Load test: P99>5s or error rate>5%
❌ Critical production checklist items incomplete

---

**Status:** Ready to begin staging validation
**Next Step:** Execute Phase 0 (Environment Setup)

_Reference: [REF:STAGING-VALIDATION]_
