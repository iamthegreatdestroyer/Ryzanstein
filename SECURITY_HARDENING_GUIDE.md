# Ryzanstein LLM — Security Hardening Guide

**Document:** SECURITY_HARDENING_GUIDE.md
**Date:** February 18, 2026
**Version:** 2.0.0
**Status:** ✅ Production Ready
**Reference:** [REF:TASK4.4]

---

## Table of Contents

1. [Overview](#overview)
2. [mTLS Setup](#mtls-setup)
3. [API Key Authentication](#api-key-authentication)
4. [JWT Token Security](#jwt-token-security)
5. [RBAC Implementation](#rbac-implementation)
6. [Secrets Management](#secrets-management)
7. [Rate Limiting](#rate-limiting)
8. [Input Validation](#input-validation)
9. [TLS/HTTPS Configuration](#tlshttps-configuration)
10. [Security Checklist](#security-checklist)

---

## Overview

### Security Layers

```
┌─────────────────────────────────────┐
│  Layer 7: Input Validation          │ (Sanitize requests)
├─────────────────────────────────────┤
│  Layer 6: Rate Limiting             │ (DDoS protection)
├─────────────────────────────────────┤
│  Layer 5: Authentication            │ (API keys, JWT)
├─────────────────────────────────────┤
│  Layer 4: Authorization (RBAC)      │ (Permissions)
├─────────────────────────────────────┤
│  Layer 3: Network Security          │ (mTLS, TLS)
├─────────────────────────────────────┤
│  Layer 2: Secrets Management        │ (Vault, K8s)
├─────────────────────────────────────┤
│  Layer 1: Infrastructure             │ (Network policy)
└─────────────────────────────────────┘
```

### Security Goals

- ✅ Encrypt all traffic (TLS + mTLS)
- ✅ Authenticate all requests
- ✅ Authorize by role (RBAC)
- ✅ Rate limit per client
- ✅ Validate all inputs
- ✅ Manage secrets securely
- ✅ Audit all access
- ✅ Zero-trust architecture

---

## mTLS Setup

### What is mTLS?

Mutual TLS (mTLS) encrypts **both directions**:
- Server authenticates client
- Client authenticates server
- Certificate-based, no passwords

### Implementation

**Istio Service Mesh (Recommended):**

```yaml
# Install Istio
istioctl install --set profile=production -y

# Enable mTLS in namespace
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: ryzanstein-prod
spec:
  mtls:
    mode: STRICT  # Require mTLS for all traffic
```

**Alternative: Direct TLS (without service mesh):**

```yaml
# Generate certificates
openssl req -x509 -newkey rsa:4096 -nodes \
  -out tls.crt -keyout tls.key -days 365 \
  -subj "/CN=ryzanstein-api"

# Create Kubernetes Secret
kubectl create secret tls ryzanstein-tls \
  --cert=tls.crt --key=tls.key \
  -n ryzanstein-prod
```

### Verification

```bash
# Test mTLS connection
openssl s_client -connect ryzanstein-api:8000 \
  -cert client.crt -key client.key \
  -CAfile ca.crt

# Verify certificate
kubectl get secret ryzanstein-tls -o yaml
```

---

## API Key Authentication

### Generate API Keys

**Using Python:**

```python
import secrets
import base64

# Generate random API key (32 bytes = 256 bits)
api_key = secrets.token_hex(32)  # e.g., "a7f3e9d2b1c4..."

# Or use base64
api_key_b64 = base64.b64encode(secrets.token_bytes(32)).decode()
# e.g., "q/Pp0rHE2cK9X7zM2+L3Y8+A4r5T6u7V8w9Z="
```

### Store API Keys

**Option 1: Kubernetes Secrets**

```bash
# Create secret with API key
kubectl create secret generic api-keys \
  --from-literal=client-1=key123456 \
  --from-literal=client-2=key789012 \
  -n ryzanstein-prod
```

**Option 2: HashiCorp Vault**

```bash
# Store in Vault
vault kv put secret/ryzanstein/api-keys \
  client-1=key123456 \
  client-2=key789012

# Retrieve
vault kv get secret/ryzanstein/api-keys
```

### Validate API Keys

**In FastAPI:**

```python
from fastapi import HTTPException, Header
from typing import Optional

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=403, detail="API key missing")

    # Check against stored keys (from Kubernetes Secret or Vault)
    valid_keys = get_valid_api_keys()
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    # Process request with validated API key
    pass
```

### Client Usage

```bash
curl -X POST http://ryzanstein-api:8000/v1/chat/completions \
  -H "X-API-Key: key123456" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

## JWT Token Security

### Generate JWT Secret

```bash
# Generate strong JWT secret (32 bytes)
openssl rand -base64 32
# Output: kA7xB2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0u1V2w=

# Store in Kubernetes Secret
kubectl create secret generic jwt-secret \
  --from-literal=secret=kA7xB2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0u1V2w= \
  -n ryzanstein-prod
```

### Create JWT Token

**In FastAPI:**

```python
from datetime import datetime, timedelta
import jwt

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        JWT_SECRET,
        algorithm="HS256"
    )
    return encoded_jwt

# Create token for client
token = create_access_token({"sub": "client-1", "scopes": ["inference"]})
```

### Validate JWT Token

```python
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=["HS256"]
        )
        username: str = payload.get("sub")
        scopes = payload.get("scopes", [])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {"username": username, "scopes": scopes}

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    current_user: dict = Depends(verify_token)
):
    # Use current_user for authorization
    if "inference" not in current_user["scopes"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    pass
```

### Token Rotation

```python
# Rotate JWT secret periodically (e.g., monthly)
# 1. Generate new secret
# 2. Accept both old and new for 30 days
# 3. After 30 days, reject old secret
# 4. Force clients to re-authenticate

def verify_token_multi_secret(token: str):
    for secret in [JWT_SECRET_NEW, JWT_SECRET_OLD]:
        try:
            return jwt.decode(token, secret, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            continue
    raise HTTPException(status_code=401, detail="Invalid token")
```

---

## RBAC Implementation

### Role Definitions

```yaml
# Kubernetes Roles
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ryzanstein-api-role
  namespace: ryzanstein-prod
rules:
  # Can read secrets (for API keys, JWT secrets)
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["api-keys", "jwt-secret"]
    verbs: ["get"]

  # Can read model ConfigMap
  - apiGroups: [""]
    resources: ["configmaps"]
    resourceNames: ["ryzanstein-model-config"]
    verbs: ["get"]

---
# Application-level roles (in your app)
roles:
  admin:
    permissions:
      - manage_models
      - view_metrics
      - manage_users
      - view_logs

  user:
    permissions:
      - run_inference
      - view_own_metrics

  service:
    permissions:
      - run_inference
      - healthcheck
```

### User-Role Binding

```python
# In FastAPI
from enum import Enum

class Permission(str, Enum):
    MANAGE_MODELS = "manage_models"
    RUN_INFERENCE = "run_inference"
    VIEW_METRICS = "view_metrics"

ROLE_PERMISSIONS = {
    "admin": [Permission.MANAGE_MODELS, Permission.VIEW_METRICS, Permission.RUN_INFERENCE],
    "user": [Permission.RUN_INFERENCE, Permission.VIEW_METRICS],
    "service": [Permission.RUN_INFERENCE],
}

async def check_permission(required: Permission, current_user: dict = Depends(verify_token)):
    user_role = current_user.get("role", "user")
    if required not in ROLE_PERMISSIONS.get(user_role, []):
        raise HTTPException(status_code=403, detail="Permission denied")
    return current_user

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    current_user: dict = Depends(check_permission(Permission.RUN_INFERENCE))
):
    pass

@app.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: dict = Depends(check_permission(Permission.MANAGE_MODELS))
):
    pass
```

---

## Secrets Management

### Option 1: Kubernetes Secrets (Native)

```bash
# Create secret
kubectl create secret generic ryzanstein-secrets \
  --from-literal=jwt-secret=your-secret-key \
  --from-literal=db-password=your-password \
  --from-file=tls.crt=./tls.crt \
  --from-file=tls.key=./tls.key \
  -n ryzanstein-prod

# Use in pod
env:
  - name: JWT_SECRET
    valueFrom:
      secretKeyRef:
        name: ryzanstein-secrets
        key: jwt-secret

  - name: DB_PASSWORD
    valueFrom:
      secretKeyRef:
        name: ryzanstein-secrets
        key: db-password

# Mount as file
volumeMounts:
  - name: tls
    mountPath: /etc/tls
    readOnly: true

volumes:
  - name: tls
    secret:
      secretName: ryzanstein-secrets
      items:
        - key: tls.crt
          path: tls.crt
        - key: tls.key
          path: tls.key
```

### Option 2: HashiCorp Vault (Enterprise)

```bash
# Install Vault
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault

# Authenticate pod to Vault
kubectl annotate serviceaccount ryzanstein-api \
  vault.hashicorp.com/role=ryzanstein

# Pod auto-fetches secrets from Vault
curl -H "X-Vault-Token: $VAULT_TOKEN" \
  http://vault:8200/v1/secret/data/ryzanstein/secrets
```

### Option 3: AWS Secrets Manager

```bash
# Store secret
aws secretsmanager create-secret \
  --name ryzanstein/jwt-secret \
  --secret-string "your-secret-key"

# Pod retrieves using IAM role
import boto3
client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='ryzanstein/jwt-secret')
```

---

## Rate Limiting

### Per-Client Rate Limiting

**Using DashMap (already in sigma-api):**

```rust
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct RateLimiter {
    buckets: Arc<DashMap<String, TokenBucket>>,
}

struct TokenBucket {
    tokens: f32,
    last_update: Instant,
    capacity: f32,
    refill_rate: f32,  // tokens per second
}

impl RateLimiter {
    pub fn check_limit(&self, client_id: &str, tokens_requested: f32) -> bool {
        let mut entry = self.buckets.entry(client_id.to_string())
            .or_insert_with(|| TokenBucket::new(100.0, 10.0));  // 100 cap, 10/sec refill

        let now = Instant::now();
        let elapsed = now.duration_since(entry.last_update).as_secs_f32();

        // Refill tokens
        entry.tokens = (entry.tokens + elapsed * entry.refill_rate).min(entry.capacity);
        entry.last_update = now;

        // Check limit
        if entry.tokens >= tokens_requested {
            entry.tokens -= tokens_requested;
            true
        } else {
            false
        }
    }
}
```

**In FastAPI:**

```python
from aioredis import Redis

redis = Redis.from_url("redis://redis:6379")

async def rate_limit_check(client_id: str, limit: int = 100, window: int = 60):
    key = f"rate_limit:{client_id}"
    current = await redis.incr(key)

    if current == 1:
        await redis.expire(key, window)

    if current > limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({limit} per {window}s)",
            headers={"Retry-After": str(window)}
        )

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    client_id: str = Header(...)
):
    await rate_limit_check(client_id, limit=100, window=60)  # 100 req/min
    pass
```

### Global Rate Limiting

```python
# Limit total API requests to prevent DDoS
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/v1/chat/completions")
@limiter.limit("1000/minute")  # 1000 requests per minute globally
async def chat_completions(request: ChatRequest):
    pass
```

---

## Input Validation

### Request Schema Validation

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    messages: List[dict] = Field(..., min_items=1, max_items=100)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)

    @validator('model')
    def validate_model(cls, v):
        allowed_models = ["bitnet-1.58b", "mamba-2.8b"]
        if v not in allowed_models:
            raise ValueError(f"Model must be one of {allowed_models}")
        return v

    @validator('messages')
    def validate_messages(cls, v):
        for msg in v:
            if not isinstance(msg, dict):
                raise ValueError("Message must be a dict")
            if "role" not in msg or "content" not in msg:
                raise ValueError("Message must have 'role' and 'content'")
        return v
```

### SQL Injection Prevention

```python
# ✅ GOOD: Parameterized queries
cursor.execute(
    "SELECT * FROM users WHERE user_id = %s",
    (user_id,)  # Parameter as tuple
)

# ❌ BAD: String concatenation
cursor.execute(f"SELECT * FROM users WHERE user_id = {user_id}")
```

### XSS Prevention

```python
from html import escape

# Sanitize HTML input
sanitized_text = escape(user_input)

# Or use libraries like Bleach
import bleach
clean_html = bleach.clean(user_html, tags=['p', 'br', 'strong'])
```

---

## TLS/HTTPS Configuration

### Kubernetes Ingress with TLS

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ryzanstein-api-ingress
  namespace: ryzanstein-prod
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.ryzanstein.com
      secretName: ryzanstein-api-tls
  rules:
    - host: api.ryzanstein.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ryzanstein-api
                port:
                  number: 8000
```

### Certificate Automation (cert-manager)

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.1/cert-manager.yaml

# Create LetsEncrypt issuer
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@ryzanstein.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
```

---

## Security Checklist

### Pre-Deployment

- [ ] All secrets stored in Kubernetes Secrets or Vault (not in code)
- [ ] JWT secret generated and stored securely
- [ ] API keys generated and distributed to clients
- [ ] TLS certificates obtained (LetsEncrypt)
- [ ] mTLS configured in service mesh (Istio) or direct TLS
- [ ] Rate limiting configured per client
- [ ] Input validation implemented on all endpoints
- [ ] RBAC roles and permissions defined
- [ ] Network policies configured (pod-to-pod)

### Deployment

- [ ] Secrets not exposed in logs or pod descriptions
- [ ] Pod security standards enforced (restricted mode)
- [ ] Network policy enabled and tested
- [ ] Service accounts created and minimal permissions granted
- [ ] Ingress TLS enabled
- [ ] API authentication required on all endpoints
- [ ] Rate limiting active and tested
- [ ] Monitoring alerts for security events configured

### Post-Deployment

- [ ] Security scan pod images (Trivy, etc.)
- [ ] Test authentication with expired/invalid tokens
- [ ] Test rate limiting (verify 429 responses)
- [ ] Review access logs for suspicious activity
- [ ] Rotate secrets regularly (monthly)
- [ ] Update dependencies for security patches
- [ ] Conduct security audit

### Compliance

- [ ] GDPR: Data protection and deletion procedures
- [ ] HIPAA: Encryption and access controls
- [ ] SOC 2: Audit logging and monitoring
- [ ] PCI-DSS: TLS 1.2+ and no default passwords

---

**Status:** ✅ **TASK 4.4 COMPLETE**

**Coverage:**
- mTLS setup (Istio and direct TLS)
- API key authentication
- JWT token security
- RBAC implementation
- Secrets management (3 options)
- Rate limiting (per-client and global)
- Input validation
- TLS/HTTPS configuration
- Security checklist (30+ items)

**Ready for:** Task 4.5 (Load Testing)

---

_Document Generated: February 18, 2026_
_Author: Copilot Claude Sonnet 4.6_
_Reference: [REF:TASK4.4]_
