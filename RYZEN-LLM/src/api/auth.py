"""
API key authentication for Ryzanstein LLM.

Keys are loaded from the API_KEYS environment variable (comma-separated),
or from the file path set in API_KEYS_FILE.

Usage:
    @app.post("/v1/chat/completions")
    async def chat(request: ..., api_key: str = Depends(verify_api_key)):
        ...

Disable auth for local dev with AUTH_DISABLED=1.
"""

import os
import secrets
from typing import Optional, Set

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def _load_api_keys() -> Set[str]:
    """Load valid API keys from environment or file at call time (hot-reloadable)."""
    keys: Set[str] = set()

    raw = os.environ.get("API_KEYS", "")
    for k in raw.split(","):
        k = k.strip()
        if k:
            keys.add(k)

    keys_file = os.environ.get("API_KEYS_FILE", "")
    if keys_file:
        try:
            with open(keys_file) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        keys.add(line)
        except OSError:
            pass

    return keys


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> str:
    """
    FastAPI dependency — validates Bearer token.

    Returns the validated API key string so downstream handlers can log it.
    Raises HTTP 401 on missing or invalid credentials.
    AUTH_DISABLED=1 bypasses all checks (dev/test only).
    """
    if os.environ.get("AUTH_DISABLED", "").lower() in ("1", "true", "yes"):
        return "dev"

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Expected: Bearer <api-key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    valid_keys = _load_api_keys()

    if not valid_keys:
        # No keys configured — reject everything (fail-closed)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server has no API keys configured. Set API_KEYS env var.",
        )

    matched = any(secrets.compare_digest(token.encode(), k.encode()) for k in valid_keys)
    if not matched:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token
