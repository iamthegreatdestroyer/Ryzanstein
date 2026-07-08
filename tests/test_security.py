"""
Tests for src/api/security.py — JWT auth (opt-in) and the always-on
per-client rate limiter.
"""

import importlib
import sys
from pathlib import Path

import jwt
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from fastapi.testclient import TestClient

from src.api import security as sec
from src.api.server import app


# ---------------------------------------------------------------------------
# JWT generate / validate
# ---------------------------------------------------------------------------

def test_generate_and_validate_round_trip(monkeypatch):
    monkeypatch.setattr(sec, "JWT_SECRET", "test-secret-value")
    token = sec.generate_token("client-a", scopes=["chat", "embed"])
    claims = sec.validate_token(token)
    assert claims["sub"] == "client-a"
    assert claims["scopes"] == ["chat", "embed"]
    assert "exp" in claims and "iat" in claims


def test_validate_rejects_wrong_secret(monkeypatch):
    monkeypatch.setattr(sec, "JWT_SECRET", "real-secret")
    token = sec.generate_token("client-a")
    monkeypatch.setattr(sec, "JWT_SECRET", "different-secret")
    with pytest.raises(jwt.PyJWTError):
        sec.validate_token(token)


def test_validate_rejects_expired_token(monkeypatch):
    monkeypatch.setattr(sec, "JWT_SECRET", "test-secret-value")
    token = sec.generate_token("client-a", ttl_s=-10)
    with pytest.raises(jwt.ExpiredSignatureError):
        sec.validate_token(token)


def test_generate_token_requires_a_secret(monkeypatch):
    monkeypatch.setattr(sec, "JWT_SECRET", "")
    with pytest.raises(RuntimeError):
        sec.generate_token("client-a")


# ---------------------------------------------------------------------------
# require_auth dependency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_require_auth_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(sec, "REQUIRE_AUTH", False)
    result = await sec.require_auth(creds=None)
    assert result is None


@pytest.mark.asyncio
async def test_require_auth_rejects_missing_token_when_enabled(monkeypatch):
    monkeypatch.setattr(sec, "REQUIRE_AUTH", True)
    monkeypatch.setattr(sec, "JWT_SECRET", "test-secret-value")
    with pytest.raises(HTTPException) as exc_info:
        await sec.require_auth(creds=None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_auth_accepts_a_valid_token_when_enabled(monkeypatch):
    monkeypatch.setattr(sec, "REQUIRE_AUTH", True)
    monkeypatch.setattr(sec, "JWT_SECRET", "test-secret-value")
    token = sec.generate_token("client-a")

    class _Creds:
        credentials = token

    claims = await sec.require_auth(creds=_Creds())
    assert claims["sub"] == "client-a"


@pytest.mark.asyncio
async def test_require_auth_rejects_an_invalid_token_when_enabled(monkeypatch):
    monkeypatch.setattr(sec, "REQUIRE_AUTH", True)
    monkeypatch.setattr(sec, "JWT_SECRET", "test-secret-value")

    class _Creds:
        credentials = "not-a-real-token"

    with pytest.raises(HTTPException) as exc_info:
        await sec.require_auth(creds=_Creds())
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Import-time safety check: refuse to start with auth on, no secret
# ---------------------------------------------------------------------------

def test_module_refuses_to_import_with_auth_on_and_no_secret(monkeypatch):
    monkeypatch.setenv("RYZANSTEIN_REQUIRE_AUTH", "true")
    monkeypatch.delenv("RYZANSTEIN_JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="no real signing secret"):
        importlib.reload(sec)
    # Restore a working module state for any tests that run after this one.
    monkeypatch.setenv("RYZANSTEIN_REQUIRE_AUTH", "false")
    importlib.reload(sec)


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

def test_rate_limiter_allows_up_to_the_limit_then_blocks():
    limiter = sec.RateLimiter(limit_per_minute=3)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False


def test_rate_limiter_tracks_clients_independently():
    limiter = sec.RateLimiter(limit_per_minute=1)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True
    assert limiter.allow("client-a") is False
    assert limiter.allow("client-b") is False


def test_rate_limiter_remaining_reflects_usage():
    limiter = sec.RateLimiter(limit_per_minute=5)
    assert limiter.remaining("client-a") == 5
    limiter.allow("client-a")
    limiter.allow("client-a")
    assert limiter.remaining("client-a") == 3


@pytest.mark.asyncio
async def test_rate_limit_dependency_raises_429_once_exceeded(monkeypatch):
    monkeypatch.setattr(sec, "_rate_limiter", sec.RateLimiter(limit_per_minute=1))

    class _Client:
        host = "203.0.113.5"

    class _Req:
        client = _Client()
        headers: dict = {}

    await sec.rate_limit(_Req())  # first call: allowed
    with pytest.raises(HTTPException) as exc_info:
        await sec.rate_limit(_Req())  # second call: blocked
    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# End-to-end: the dependency actually fires through a real route, not just
# when called directly as a plain async function.
# ---------------------------------------------------------------------------

def test_rate_limit_fires_through_a_real_http_call(monkeypatch):
    monkeypatch.setattr(sec, "_rate_limiter", sec.RateLimiter(limit_per_minute=2))
    client = TestClient(app, raise_server_exceptions=True)

    r1 = client.post("/v1/embeddings", json={"input": "hello"})
    r2 = client.post("/v1/embeddings", json={"input": "hello"})
    r3 = client.post("/v1/embeddings", json={"input": "hello"})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


def test_no_auth_header_still_works_by_default(monkeypatch):
    """The default (REQUIRE_AUTH=false) must not break any existing caller."""
    monkeypatch.setattr(sec, "_rate_limiter", sec.RateLimiter(limit_per_minute=1000))
    client = TestClient(app, raise_server_exceptions=True)
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
