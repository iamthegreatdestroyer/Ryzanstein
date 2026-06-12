"""Sprint 5: Auth + rate limiting unit tests."""

import asyncio
import os
import pytest


# ---------------------------------------------------------------------------
# auth.py tests
# ---------------------------------------------------------------------------

def test_verify_api_key_disabled(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "1")
    # Import fresh after env change
    import importlib
    import sys
    sys.modules.pop("auth", None)
    sys.modules.pop("RYZEN_LLM.src.api.auth", None)
    try:
        import sys as _sys
        _sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "RYZEN-LLM" / "src" / "api"))
        import auth as _auth
        result = _auth.verify_api_key(None)
        assert result == "dev"
    finally:
        monkeypatch.delenv("AUTH_DISABLED", raising=False)


def test_verify_api_key_no_credentials(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "0")
    monkeypatch.setenv("API_KEYS", "secret123")
    import sys
    from pathlib import Path
    api_path = str(Path(__file__).parent.parent / "RYZEN-LLM" / "src" / "api")
    if api_path not in sys.path:
        sys.path.insert(0, api_path)
    import importlib
    import auth as _auth
    importlib.reload(_auth)

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        _auth.verify_api_key(None)
    assert exc_info.value.status_code == 401


def test_verify_api_key_valid(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "0")
    monkeypatch.setenv("API_KEYS", "mykey,otherkey")
    import sys
    from pathlib import Path
    api_path = str(Path(__file__).parent.parent / "RYZEN-LLM" / "src" / "api")
    if api_path not in sys.path:
        sys.path.insert(0, api_path)
    import importlib
    import auth as _auth
    importlib.reload(_auth)

    from fastapi.security import HTTPAuthorizationCredentials
    creds = HTTPAuthorizationCredentials(scheme="bearer", credentials="mykey")
    result = _auth.verify_api_key(creds)
    assert result == "mykey"


def test_verify_api_key_invalid(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "0")
    monkeypatch.setenv("API_KEYS", "real-key")
    import sys
    from pathlib import Path
    api_path = str(Path(__file__).parent.parent / "RYZEN-LLM" / "src" / "api")
    if api_path not in sys.path:
        sys.path.insert(0, api_path)
    import importlib
    import auth as _auth
    importlib.reload(_auth)

    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials
    creds = HTTPAuthorizationCredentials(scheme="bearer", credentials="wrong-key")
    with pytest.raises(HTTPException) as exc_info:
        _auth.verify_api_key(creds)
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# rate_limiter.py tests
# ---------------------------------------------------------------------------

def test_rate_limiter_allows_within_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "5")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
    import sys
    from pathlib import Path
    api_path = str(Path(__file__).parent.parent / "RYZEN-LLM" / "src" / "api")
    if api_path not in sys.path:
        sys.path.insert(0, api_path)
    import importlib
    import rate_limiter as _rl
    importlib.reload(_rl)

    limiter = _rl.SlidingWindowRateLimiter()

    async def run():
        for _ in range(5):
            await limiter.check("test-key-allow")

    asyncio.run(run())


def test_rate_limiter_blocks_over_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
    import sys
    from pathlib import Path
    api_path = str(Path(__file__).parent.parent / "RYZEN-LLM" / "src" / "api")
    if api_path not in sys.path:
        sys.path.insert(0, api_path)
    import importlib
    import rate_limiter as _rl
    importlib.reload(_rl)
    from fastapi import HTTPException

    limiter = _rl.SlidingWindowRateLimiter()

    async def run():
        for _ in range(3):
            await limiter.check("test-key-block")
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check("test-key-block")
        assert exc_info.value.status_code == 429

    asyncio.run(run())


def test_rate_limiter_keys_independent(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
    import sys
    from pathlib import Path
    api_path = str(Path(__file__).parent.parent / "RYZEN-LLM" / "src" / "api")
    if api_path not in sys.path:
        sys.path.insert(0, api_path)
    import importlib
    import rate_limiter as _rl
    importlib.reload(_rl)
    from fastapi import HTTPException

    limiter = _rl.SlidingWindowRateLimiter()

    async def run():
        await limiter.check("key-a")
        await limiter.check("key-a")
        # key-b should still work even though key-a is maxed
        await limiter.check("key-b")
        await limiter.check("key-b")
        # key-a now blocked
        with pytest.raises(HTTPException):
            await limiter.check("key-a")

    asyncio.run(run())
