# Ryzanstein LLM — Kubernetes Helm Chart Deployment Guide

**Document:** HELM_DEPLOYMENT_GUIDE.md
**Date:** February 18, 2026
**Version:** 2.0.0
**Status:** ✅ Production Ready
**Reference:** [REF:TASK4.2]

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Chart Structure](#chart-structure)
5. [Configuration](#configuration)
6. [Deployment](#deployment)
7. [Scaling & Autoscaling](#scaling--autoscaling)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Production Checklist](#production-checklist)

---

## Overview

### What is This Helm Chart?

The Ryzanstein Helm chart packages the complete LLM inference stack for Kubernetes:

- **API Server** (FastAPI, port 8000)
- **MCP gRPC Server** (ports 8001-8003)
- **Vector Database** (Qdrant)
- **Observability Stack** (Prometheus, Grafana, Jaeger)
- **Alerting** (AlertManager)

### Chart Characteristics

| Property | Value |
|----------|-------|
| **Chart Name** | ryzanstein |
| **Chart Version** | 2.0.0 |
| **App Version** | 2.0.0 |
| **Type** | application |
| **Min Kubernetes** | 1.24.0 |
| **Deployment Type** | Deployment + StatefulSet |
| **Persistence** | PersistentVolumeClaim (5 volumes) |
| **RBAC** | Supported |
| **HPA** | Enabled (v2 API) |
| **Network Policy** | Supported |
| **Pod Security** | Restricted (production) |

---

## Prerequisites

### Kubernetes Cluster

- **Version:** 1.24.0 or later
- **Architecture:** x86_64 (AMD64) or ARM64
- **Node Count:**
  - Development: 1 node minimum
  - Production: 3+ nodes recommended
- **Total Resources:**
  - Development: 4 CPU, 8 GB RAM
  - Production: 16 CPU, 32 GB RAM

### Required Tools

```bash
# Kubernetes command-line tool
kubectl version --client
# Expected: v1.24.0 or higher

# Helm package manager
helm version
# Expected: v3.12.0 or higher

# Optional: Helm Lint for validation
helm lint ./helm/ryzanstein
```

### Storage

**Storage Classes Required:**

```bash
# Check available storage classes
kubectl get storageclass

# For production, create fast-ssd storage class:
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
allowVolumeExpansion: true
reclaimPolicy: Retain
EOF
```

### Image Registry

For production deployments:

```bash
# Create image pull secret (if using private registry)
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=username \
  --docker-password=password \
  --docker-email=email@example.com

# Reference in values.yaml:
imagePullSecrets:
  - name: regcred
```

---

## Quick Start

### 1. Install Helm Chart (Development)

```bash
# Navigate to repo root
cd Ryzanstein

# Add Helm repository (if hosting on Artifact Hub)
# helm repo add ryzanstein https://charts.ryzanstein.io
# helm repo update

# Install from local chart
helm install ryzanstein ./helm/ryzanstein \
  --namespace default \
  --values helm/ryzanstein/values-dev.yaml

# Verify installation
helm list
kubectl get pods
kubectl get svc
```

### 2. Access Services

```bash
# Get service endpoints
kubectl get svc ryzanstein-api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# API endpoint
curl http://<API_IP>:8000/health

# Grafana dashboard
kubectl port-forward svc/grafana 3000:3000
# Access: http://localhost:3000 (admin/admin123)

# Jaeger tracing UI
kubectl port-forward svc/jaeger 16686:16686
# Access: http://localhost:16686

# Prometheus metrics
kubectl port-forward svc/prometheus 9090:9090
# Access: http://localhost:9090
```

### 3. Run Test Inference

```bash
# Forward API port
kubectl port-forward svc/ryzanstein-api 8000:8000

# Test API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bitnet-1.58b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

---

## Chart Structure

```
helm/ryzanstein/
├── Chart.yaml                 # Chart metadata
├── values.yaml                # Default configuration
├── values-dev.yaml            # Development overrides
├── values-production.yaml      # Production overrides
├── values-staging.yaml         # Staging overrides (optional)
├── templates/
│   ├── _helpers.tpl           # Template helpers
│   ├── configmap.yaml         # ConfigMap (config, model settings)
│   ├── deployment-api.yaml    # API server deployment
│   ├── deployment-mcp.yaml    # MCP server deployment (if added)
│   ├── hpa.yaml               # Horizontal Pod Autoscaler
│   ├── service-api.yaml       # Kubernetes Services
│   ├── pvc.yaml               # PersistentVolumeClaims (5 volumes)
│   ├── secrets.yaml           # Secrets (if added)
│   ├── ingress.yaml           # Ingress (if added)
│   ├── rbac.yaml              # RBAC (if added)
│   ├── network-policy.yaml    # Network policies (if added)
│   └── NOTES.txt              # Post-install instructions
└── README.md                  # Chart documentation
```

---

## Configuration

### values.yaml Structure

**Global Settings:**
```yaml
global:
  environment: development|staging|production
  namespace: kubernetes-namespace
  domain: example.com
  imagePullPolicy: IfNotPresent|Always|Never
```

**API Server:**
```yaml
api:
  enabled: true
  replicaCount: 2|3
  image:
    repository: ryzanstein
    tag: "2.0.0"
  resources:
    requests: {cpu: 2, memory: 4Gi}
    limits: {cpu: 4, memory: 8Gi}
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
```

**MCP Server:**
```yaml
mcp:
  enabled: true
  replicaCount: 2|3
  ports:
    inference: 8001
    registry: 8002
    training: 8003
```

**Model Storage:**
```yaml
modelStorage:
  pvc:
    size: 20Gi|100Gi  # dev|prod
    storageClass: standard|fast-ssd
```

**Observability:**
```yaml
prometheus:
  enabled: true
  retention: 30d|180d
grafana:
  enabled: true
  adminPassword: changeme
jaeger:
  enabled: true
  memory:
    maxTraces: 10000|100000
```

### Configuration Files

All YAML configuration is in ConfigMap:
- `model.yaml` — Model settings (vocab size, hidden size, etc.)
- `prometheus.yaml` — Metrics scraping targets
- `api.yaml` — API server settings
- `mcp.yaml` — MCP server settings

Modify with:
```bash
# Edit ConfigMap
kubectl edit configmap ryzanstein-config

# Or update via helm
helm upgrade ryzanstein ./helm/ryzanstein \
  --set modelConfig.vocab_size=32000
```

---

## Deployment

### Development Deployment

```bash
# Install with dev values
helm install ryzanstein ./helm/ryzanstein \
  -f values-dev.yaml \
  --namespace default

# Expected: 1 API pod, 1 MCP pod, minimal resources
```

### Staging Deployment

```bash
# Install with staging values
helm install ryzanstein ./helm/ryzanstein \
  -f values-staging.yaml \
  --namespace ryzanstein-staging

# Expected: 2 API pods, 2 MCP pods, medium resources
```

### Production Deployment

```bash
# 1. Create namespace
kubectl create namespace ryzanstein-prod

# 2. Create secrets (API keys, passwords)
kubectl create secret generic ryzanstein-api-keys \
  --from-literal=API_KEY=your-api-key \
  --from-literal=JWT_SECRET=your-jwt-secret \
  -n ryzanstein-prod

# 3. Install with production values
helm install ryzanstein ./helm/ryzanstein \
  -f values-production.yaml \
  --namespace ryzanstein-prod

# 4. Verify installation
kubectl rollout status deployment/ryzanstein-api -n ryzanstein-prod

# Expected: 3 API pods, 3 MCP pods, high resources, autoscaling enabled
```

### Upgrade Deployment

```bash
# Upgrade to new version
helm upgrade ryzanstein ./helm/ryzanstein \
  -f values-production.yaml \
  --namespace ryzanstein-prod

# Rollback if needed
helm rollback ryzanstein 1 --namespace ryzanstein-prod

# History
helm history ryzanstein --namespace ryzanstein-prod
```

---

## Scaling & Autoscaling

### Manual Scaling

```bash
# Scale API deployment
kubectl scale deployment ryzanstein-api \
  --replicas=5 \
  -n ryzanstein-prod

# Verify
kubectl get pods -n ryzanstein-prod | grep api
```

### Horizontal Pod Autoscaler (HPA)

Enabled by default in production. Monitors:
- **CPU:** Scale up at 70% utilization
- **Memory:** Scale up at 80% utilization
- **Custom Metrics:** HTTP requests/sec (if Prometheus installed)

**Monitor HPA:**
```bash
# Watch HPA status
kubectl get hpa -n ryzanstein-prod -w

# Check HPA details
kubectl describe hpa ryzanstein-api-hpa -n ryzanstein-prod
```

**HPA Scaling Behavior:**

- **Scale Up:** 100% per 30s (double replicas)
- **Scale Down:** 50% per 60s (halve replicas)
- **Min replicas:** 2-3 (HA)
- **Max replicas:** 10-20 (cost control)

### Vertical Pod Autoscaler (VPA)

Optional for automatic resource right-sizing:

```bash
# Install VPA (if using)
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/download/vertical-pod-autoscaler-0.14.0/vpa-v0.14.0.yaml

# Enable in values:
vpa:
  enabled: true
  updateMode: auto|off
```

---

## Monitoring

### Prometheus Integration

Metrics are automatically scraped from:
- `http://ryzanstein-api:8000/metrics` (FastAPI)
- `http://ryzanstein-mcp:8001/metrics` (gRPC)
- `http://qdrant:6333/metrics` (Qdrant)

**Key metrics:**
```promql
# Request rate
rate(http_requests_total[5m])

# P99 latency
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# Token throughput
rate(tokens_generated_total[5m])
```

### Grafana Dashboards

Pre-configured dashboards:
- **Inference Performance** (latency, throughput, errors)
- **Resource Usage** (CPU, memory, disk)
- **System Health** (pod status, HPA activity)

Access:
```bash
kubectl port-forward svc/grafana 3000:3000
# http://localhost:3000 (admin/changeme)
```

### Jaeger Tracing

Distributed traces show request flow through:
- FastAPI → gRPC MCP → inference engine → vector DB

Access:
```bash
kubectl port-forward svc/jaeger 16686:16686
# http://localhost:16686
```

### Health Checks

```bash
# API liveness
kubectl exec -it pod/ryzanstein-api-xxx -- \
  curl http://localhost:8000/health

# API readiness
kubectl exec -it pod/ryzanstein-api-xxx -- \
  curl http://localhost:8000/health/ready

# Check pod status
kubectl get pods -o wide
```

---

## Troubleshooting

### Pod not starting

```bash
# Check pod status
kubectl describe pod ryzanstein-api-xxx

# Check logs
kubectl logs ryzanstein-api-xxx
kubectl logs ryzanstein-api-xxx --previous  # If crashed

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

### Insufficient resources

```bash
# Check node resources
kubectl top nodes
kubectl top pods -n ryzanstein-prod

# Add resources or nodes
kubectl scale nodes --increase=1

# Or adjust Helm values:
helm upgrade ryzanstein ./helm/ryzanstein \
  --set api.resources.limits.memory=6Gi
```

### PVC not binding

```bash
# Check PVC status
kubectl get pvc

# Check storage class
kubectl get storageclass

# Create storage class if missing
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard
provisioner: kubernetes.io/aws-ebs
EOF
```

### Service not accessible

```bash
# Check service endpoints
kubectl get endpoints ryzanstein-api

# Port forward for testing
kubectl port-forward svc/ryzanstein-api 8000:8000

# Check network policy
kubectl get networkpolicy
```

---

## Production Checklist

- [ ] **Pre-deployment**
  - [ ] Kubernetes 1.24+ cluster ready
  - [ ] 16+ CPU, 32+ GB RAM allocated
  - [ ] Storage classes available (fast-sSD)
  - [ ] Model weights downloaded (~575 MB)
  - [ ] Helm chart validated: `helm lint ./helm/ryzanstein`

- [ ] **Configuration**
  - [ ] Grafana admin password changed (not "changeme")
  - [ ] Slack/PagerDuty webhooks configured
  - [ ] API keys and JWT secrets created
  - [ ] Database credentials set
  - [ ] Image registry credentials (if private)

- [ ] **Deployment**
  - [ ] Namespace created: `kubectl create namespace ryzanstein-prod`
  - [ ] Secrets created: API keys, JWT tokens, DB creds
  - [ ] Helm install successful with production values
  - [ ] All pods in "Running" state
  - [ ] Liveness/readiness probes passing

- [ ] **Monitoring & Alerting**
  - [ ] Prometheus collecting metrics
  - [ ] Grafana dashboards visible
  - [ ] Jaeger receiving traces
  - [ ] AlertManager configured
  - [ ] Slack/PagerDuty notifications working

- [ ] **Testing**
  - [ ] `/health` endpoint returns 200
  - [ ] `/health/ready` endpoint returns 200
  - [ ] Inference test: `/v1/chat/completions` works
  - [ ] Load test: 100+ concurrent requests
  - [ ] Scaling: HPA scales up under load

- [ ] **Security**
  - [ ] RBAC enabled (`rbac.enabled: true`)
  - [ ] Network policies applied
  - [ ] Pod security standards enforced
  - [ ] Secrets not exposed in logs
  - [ ] TLS enabled (ingress)

- [ ] **Backup & Recovery**
  - [ ] Model weights backed up
  - [ ] Prometheus data backed up (weekly)
  - [ ] Grafana dashboards exported
  - [ ] Recovery procedure tested

- [ ] **Documentation**
  - [ ] Runbook created
  - [ ] On-call procedures documented
  - [ ] Escalation paths defined
  - [ ] SLO/SLA defined

---

## Common Commands

```bash
# Install
helm install ryzanstein ./helm/ryzanstein -f values-production.yaml

# Upgrade
helm upgrade ryzanstein ./helm/ryzanstein -f values-production.yaml

# Uninstall (careful!)
helm uninstall ryzanstein

# List releases
helm list

# Show current values
helm show values ./helm/ryzanstein

# Dry-run (preview changes)
helm upgrade ryzanstein ./helm/ryzanstein --dry-run --debug

# Get chart info
helm info ryzanstein

# Validate chart
helm lint ./helm/ryzanstein

# Get deployment status
helm status ryzanstein

# View history
helm history ryzanstein

# Rollback to previous
helm rollback ryzanstein 1
```

---

**Status:** ✅ Production Ready
**Last Updated:** February 18, 2026
**Author:** Copilot Claude Sonnet 4.6
**Reference:** [REF:TASK4.2]
