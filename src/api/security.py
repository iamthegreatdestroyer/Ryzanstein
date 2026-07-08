"""
Ryzanstein API — auth and rate limiting.
=========================================

Ported design from the dormant sigma-api Rust crate (real, tested JWT auth
+ per-client rate limiter that were never wired into anything). The live
gateway (src/api/server.py) had neither: only CORSMiddleware, confirmed via
a full grep of the file during the 2026-07 Tier 2/3 salvage review.

Rollout note
------------
Dozens of real consumers across the ecosystem (sigma-index, sigma-telemetry,
sigma-compress, elite-agent-collective, YT-Shorts, project-alchemy, and
more) call this gateway today with no Authorization header at all. Making
auth mandatory unconditionally would break every one of them. So:

  - Rate limiting is always on (a generous per-client default that will
    not affect normal single-caller internal traffic, only runaway loops
    or abuse).
  - Auth is OFF by default (RYZANSTEIN_REQUIRE_AUTH unset or "false") and
    is a no-op in that mode. Flipping it on is a separate, coordinated
    rollout (every internal consumer needs a token first) - not something
    this change does by itself.
  - If auth is turned on with no signing secret configured, the server
    refuses to start rather than silently signing/validating tokens with
    a weak or predictable default secret.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRE_AUTH = os.getenv("RYZANSTEIN_REQUIRE_AUTH", "false").lower() in ("1", "true", "yes")
# Generate with e.g. `openssl rand -hex 32` before ever setting REQUIRE_AUTH.
# PyJWT itself warns at encode/decode time if this is under 32 bytes.
JWT_SECRET = os.getenv("RYZANSTEIN_JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_DEFAULT_TTL_S = int(os.getenv("RYZANSTEIN_JWT_TTL_S", "3600"))

RATE_LIMIT_PER_MINUTE = int(os.getenv("RYZANSTEIN_RATE_LIMIT_PER_MINUTE", "120"))

if REQUIRE_AUTH and not JWT_SECRET:
    raise RuntimeError(
        "RYZANSTEIN_REQUIRE_AUTH is enabled but RYZANSTEIN_JWT_SECRET is not "
        "set. Refusing to start with auth on and no real signing secret."
    )

_bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# JWT — generate / validate
# ---------------------------------------------------------------------------

def generate_token(
    subject: str,
    scopes: Optional[list[str]] = None,
    ttl_s: int = JWT_DEFAULT_TTL_S,
) -> str:
    """Generate a signed HS256 token. Requires RYZANSTEIN_JWT_SECRET to be set."""
    if not JWT_SECRET:
        raise RuntimeError("RYZANSTEIN_JWT_SECRET is not set; cannot generate a token")
    now = int(time.time())
    payload = {
        "sub": subject,
        "scopes": scopes or [],
        "iat": now,
        "exp": now + ttl_s,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def validate_token(token: str) -> dict:
    """Validate a token and return its claims. Raises jwt.PyJWTError on failure."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


async def require_auth(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[dict]:
    """
    FastAPI dependency for routes that should require a bearer token once
    auth is turned on. A no-op returning None while RYZANSTEIN_REQUIRE_AUTH
    is unset/false, which is the default and matches every consumer calling
    this gateway today with no Authorization header.
    """
    if not REQUIRE_AUTH:
        return None
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    try:
        return validate_token(creds.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ---------------------------------------------------------------------------
# Rate limiter — sliding window per client, always on
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Per-client sliding-window request counter. Matches sigma-api's Rust
    design (a timestamp deque per client, pruned to the current window)
    rather than a hard token bucket, kept intentionally dependency-free.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self.window_s = 60.0
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, client_key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[client_key]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True

    def remaining(self, client_key: str) -> int:
        now = time.monotonic()
        hits = self._hits[client_key]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        return max(0, self.limit - len(hits))


_rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)


def _client_key(request: Request) -> str:
    # Bearer subject if auth is on and a valid token was presented, else IP.
    auth_header = request.headers.get("authorization", "")
    if REQUIRE_AUTH and auth_header.lower().startswith("bearer "):
        try:
            claims = validate_token(auth_header[7:])
            return f"sub:{claims.get('sub', 'unknown')}"
        except jwt.PyJWTError:
            pass
    return f"ip:{request.client.host if request.client else 'unknown'}"


async def rate_limit(request: Request) -> None:
    """FastAPI dependency: raises 429 once a client exceeds the per-minute limit."""
    key = _client_key(request)
    if not _rate_limiter.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({_rate_limiter.limit}/min)",
        )
