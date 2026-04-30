# Docker Port Audit & Remediation Plan
**Generated:** February 19, 2026
**Purpose:** Isolate Ryzanstein ports from other projects, prevent cross-contamination

---

## Executive Summary

**CRITICAL FINDING:** Port 3000 conflict detected
- **localhost:3000** currently routes to `dashboard-staging` (Faceless YouTube Project)
- **Ryzanstein Grafana** should be on port 3000 but is BLOCKED by Faceless YouTube
- **Action Required:** Reassign Faceless YouTube dashboard to unused port range (40000+)

**Port Availability Status:**
- ✅ Port 8000 (API): Currently unused for external binding (Kubernetes NodePort using 31139 internally)
- ❌ Port 9090 (Prometheus): Claimed by **Hyperbox project**
- ❌ Port 3000 (Grafana): Claimed by **Faceless YouTube project**
- ❌ Port 16686 (Jaeger): Claimed by **Neurectomy ML project**
- ❌ Port 9093 (AlertManager): Currently unused/available ✅

---

## Current Port Mapping by Project

### 🟢 RYZANSTEIN (Kubernetes - Internal Only)
| Service | Int Port | Ext Port | Status | Notes |
|---------|----------|----------|--------|-------|
| ryzanstein-api | 8000 | 31139 (NodePort) | ✅ ISOLATED | Internal K8s cluster |
| prometheus | 9090 | (ClusterIP) | ❌ BLOCKED | Hyperbox using 9090 |
| grafana | 3000 | (ClusterIP) | ❌ BLOCKED | Faceless YouTube using 3000 |
| jaeger | 16686 | (ClusterIP) | ❌ BLOCKED | Neurectomy ML using 16686 |
| alertmanager | 9093 | (ClusterIP) | ✅ AVAILABLE | Currently free |

**Kubernetes Services (Internal ClusterIP - No External Binding):**
```
NAME             TYPE        CLUSTER-IP       PORT(S)
alertmanager     ClusterIP   10.111.8.224     9093/TCP
grafana          ClusterIP   10.104.112.237   3000/TCP
jaeger           ClusterIP   10.111.11.4      16686/TCP,14268/TCP,6831/UDP
prometheus       ClusterIP   10.111.84.110    9090/TCP
ryzanstein-api   NodePort    10.103.11.128    8000:31139/TCP
```

---

### 🟡 HYPERBOX (11 days old)
| Container | Port | Protocol | Status |
|-----------|------|----------|--------|
| hyperbox-prometheus | 9090 | tcp | ⚠️ CONFLICTS WITH RYZANSTEIN |
| hyperbox-postgres | 5432 | tcp | ✅ Isolated |
| hyperboxd | 9999 | tcp | ✅ Isolated |

**Issue:** Prometheus on 9090 conflicts with Ryzanstein
**Recommendation:** Move to 40000 range → 40001 (Prometheus), 40002 (Postgres), 40003 (Hyperbox)

---

### 🟡 AUTOAG (5 weeks old)
| Container | Port | Status |
|-----------|------|--------|
| autoag-nginx | 18530 | ✅ Isolated (40000+ range) |
| autoag-postgres | 18510 | ✅ Isolated (40000+ range) |
| autoag-redis | 18520 | ✅ Isolated (40000+ range) |
| autoag-api | (restarting) | - |

**Status:** Already using isolated 40000+ range - NO CHANGES NEEDED

---

### 🔵 PHANTOM VPN (6 weeks old)
| Container | Port | Protocol | Status |
|-----------|------|----------|--------|
| phantom-agents | 24520 | tcp | ✅ Isolated (40000+ range) |
| phantom-discovery | 24530 | tcp | ✅ Isolated (40000+ range) |
| phantom-primary | 24511 | tcp, 24510 | udp | ✅ Isolated (40000+ range) |
| phantom-node-exporter | 24560 | tcp | ✅ Isolated (40000+ range) |
| phantom-loki | 24550 | tcp | ✅ Isolated (40000+ range) |
| phantom-agent-exporter | 24561 | tcp | ✅ Isolated (40000+ range) |
| phantom-promtail | (no external) | - | ✅ Internal |

**Status:** Already using isolated 24000+ range - NO CHANGES NEEDED

---

### 🔴 NEURECTOMY ML (2 months old)
| Container | Port | Status | Conflict |
|-----------|------|--------|----------|
| neurectomy-jaeger | 16686 | ❌ CONFLICTS WITH RYZANSTEIN | Ryzanstein needs 16686 |
| neurectomy-jaeger | 5775, 5778, 6831-6832, 9411, 14250, 14268 | ✅ Isolated | High port range |
| neurectomy-mlflow | 5000 | ✅ Isolated | Unique low port |
| neurectomy-ml-service | 8002 | ✅ Isolated | Offset from 8000 |
| neurectomy-minio | 9000-9001 | ✅ Isolated | Unique range |
| neurectomy-grafana | 3000 | ⚠️ INTERNAL ONLY | No external binding |
| neurectomy-timescale | 5433 | ✅ Isolated | Offset from 5432 |
| neurectomy-nats | 4222, 8222 | ✅ Isolated | Unique range |
| neurectomy-postgres | 5434 | ✅ Isolated | Offset from 5432 |
| neurectomy-redis | 6379 | ✅ Isolated | Standard Redis |
| neurectomy-neo4j | 7474, 7687 | ✅ Isolated | Unique range |
| neurectomy-optuna-dashboard | (restarting) | - | - |
| neurectomy-alertmanager | (restarting) | - | - |

**Issue:** Jaeger on 16686 conflicts with Ryzanstein
**Recommendation:** Move Neurectomy Jaeger to 40010 (frees 16686 for Ryzanstein)

---

### 🟠 FACELESS YOUTUBE (3 months old)
| Container | Port | Status | Conflict |
|-----------|------|--------|----------|
| dashboard-staging | 3000 | ❌ CONFLICTS WITH RYZANSTEIN | **Critical blocker** |
| api-staging | 8001 | ✅ Isolated | Offset from 8000 |
| mongodb-staging | 27017 | ✅ Isolated | Standard MongoDB |

**Issue:** Dashboard on 3000 is blocking Ryzanstein Grafana
**Recommendation:** Move to 40005 (frees 3000 for Ryzanstein)

---

### 🟠 ARTICLE AUDIO PIPELINE (4 months old)
| Container | Port | Status |
|-----------|------|--------|
| article-audio-web | 8080 | ✅ Isolated |
| article-audio-pipeline | (no external) | ✅ Internal |

**Status:** Isolated - NO CHANGES NEEDED

---

### ⚫ PORTAINER (4 months old - EE License)
| Container | Port | Status |
|-----------|------|--------|
| portainer | 8000 | ⚠️ POTENTIAL FUTURE CONFLICT |
| portainer | 9443 | ✅ Isolated |

**Issue:** Port 8000 used for internal ports; Ryzanstein API exposed via port-forward to 8000
**Current Status:** Not blocking yet (no external binding conflict)
**Recommendation:** Monitor; if needed move to 40020

---

## Port Conflict Resolution Strategy

### Priority 1: CRITICAL (Blocking Phase 6)
✅ **Port 3000 - Grafana**
- **Current User:** Faceless YouTube (`dashboard-staging`)
- **Needed For:** Ryzanstein Grafana
- **Action:** Reassign Faceless YouTube to **40005:3000**
  ```yaml
  Old: 0.0.0.0:3000->3000/tcp
  New: 0.0.0.0:40005->3000/tcp
  ```

### Priority 2: IMPORTANT (Phase 6 Monitoring)
❌ **Port 9090 - Prometheus**
- **Current User:** Hyperbox
- **Needed For:** Ryzanstein Prometheus
- **Action:** Reassign Hyperbox Prometheus to **40001:9090**
  ```yaml
  Old: 0.0.0.0:9090->9090/tcp
  New: 0.0.0.0:40001->9090/tcp
  ```

❌ **Port 16686 - Jaeger**
- **Current User:** Neurectomy ML
- **Needed For:** Ryzanstein Jaeger
- **Action:** Reassign Neurectomy Jaeger to **40010:16686**
  ```yaml
  Old: 0.0.0.0:16686->16686/tcp
  New: 0.0.0.0:40010->16686/tcp
  ```

### Priority 3: AVAILABLE
✅ **Port 9093 - AlertManager**
- **Current User:** None (available)
- **Status:** Ready to use as-is

---

## New Port Assignments

### Ryzanstein - Ports 8000-8004
```
8000 → API (kubectl port-forward)
40001 → Prometheus (external)
40005 → Grafana (external)
40010 → Jaeger (external)
9093 → AlertManager (available)
```

### Reassigned Projects - Ports 40000-40099
```
40001 → Hyperbox Prometheus (moved from 9090)
40002 → Hyperbox Postgres (optional, currently 5432)
40003 → Hyperbox Container (optional, currently 9999)

40005 → Faceless YouTube Dashboard (moved from 3000)
40006 → Faceless YouTube API (optional, currently 8001)

40010 → Neurectomy Jaeger (moved from 16686)
40020 → Portainer (reserved, if needed)

40100+ → Future projects
```

---

## Remediation Instructions

### Step 1: Update Hyperbox docker-compose.yml
**File Location:** Find Hyperbox docker-compose.yml (likely in hyperbox project root)

```yaml
# BEFORE:
services:
  prometheus:
    ports:
      - "9090:9090"

# AFTER:
services:
  prometheus:
    ports:
      - "40001:9090"
```

### Step 2: Update Faceless YouTube docker-compose.yml
**File Location:** Find Faceless YouTube project docker-compose.yml

```yaml
# BEFORE:
services:
  dashboard-staging:
    ports:
      - "3000:3000"

# AFTER:
services:
  dashboard-staging:
    ports:
      - "40005:3000"
```

### Step 3: Update Neurectomy docker-compose.yml
**File Location:** Find Neurectomy ML project docker-compose.yml

```yaml
# BEFORE:
services:
  neurectomy-jaeger:
    ports:
      - "16686:16686"
      - "5775:5775/udp"
      ...

# AFTER:
services:
  neurectomy-jaeger:
    ports:
      - "40010:16686"
      - "5775:5775/udp"
      ...
```

### Step 4: Restart Affected Containers
```bash
# Stop conflicting containers
docker stop hyperbox-prometheus dashboard-staging neurectomy-jaeger

# Re-create with new port mappings (if using docker-compose)
cd /path/to/hyperbox
docker-compose up -d --force-recreate

cd /path/to/faceless-youtube
docker-compose up -d --force-recreate

cd /path/to/neurectomy
docker-compose up -d --force-recreate

# Verify new mappings
docker ps | grep -E "hyperbox-prometheus|dashboard-staging|neurectomy-jaeger"
```

---

## Verification Checklist

After reassignments:

```bash
# Verify Hyperbox on new port
curl -s http://localhost:40001/-/healthy | jq .

# Verify Faceless YouTube dashboard on new port
curl -s http://localhost:40005/api/health | head -20

# Verify Neurectomy Jaeger on new port
curl -s -I http://localhost:40010/ | head -5

# Verify Ryzanstein ports are now available
curl -s http://localhost:3000/api/health        # Should connect to Ryzanstein Grafana
curl -s http://localhost:9090/-/healthy         # Should connect to Ryzanstein Prometheus
curl -s -I http://localhost:16686/              # Should connect to Ryzanstein Jaeger
curl -s http://localhost:9093/-/healthy         # Should connect to Ryzanstein AlertManager
```

---

## Phase 6 Port-Forward Commands (After Remediation)

Once ports are freed, use these commands:

```bash
# Terminal 1: API Port-Forward
kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000

# Terminal 2: Prometheus Port-Forward
kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090

# Terminal 3: Grafana Port-Forward
kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000

# Terminal 4: Jaeger Port-Forward
kubectl port-forward -n ryzanstein-staging svc/jaeger 16686:16686

# Terminal 5: AlertManager Port-Forward
kubectl port-forward -n ryzanstein-staging svc/alertmanager 9093:9093
```

---

## Project Summary (62 containers total)

| Project | Containers | Status | Action |
|---------|-----------|--------|--------|
| **Ryzanstein** | 11 | ✅ Kubernetes (Internal) | Port-forward when needed |
| **Hyperbox** | 3 | ⚠️ Conflicts on 9090 | Move Prometheus to 40001 |
| **AutoAG** | 4 | ✅ Already isolated (18000+ range) | No changes |
| **Phantom VPN** | 8 | ✅ Already isolated (24000+ range) | No changes |
| **Neurectomy ML** | 12 | ❌ Conflicts on 16686 | Move Jaeger to 40010 |
| **Faceless YouTube** | 3 | ❌ Conflicts on 3000 | Move Dashboard to 40005 |
| **Article Audio** | 2 | ✅ Isolated | No changes |
| **Portainer** | 1 | ⚠️ Monitor (8000) | Reserve 40020 if needed |
| **Kubernetes Core** | 15 | ✅ System services | No changes |

---

## Timeline

- **Immediate (NOW):** Share this audit with project teams
- **Within 1 hour:** Teams update their docker-compose.yml files
- **Within 2 hours:** Containers restarted with new port mappings
- **Before Phase 6:** Verify all 5 Ryzanstein ports are accessible
- **Phase 6 Start:** All port-forwards work without conflicts

---

**Generated by:** Claude Code
**Status:** Ready for immediate implementation
**Next Step:** Contact Hyperbox, Faceless YouTube, and Neurectomy ML teams to apply port reassignments
