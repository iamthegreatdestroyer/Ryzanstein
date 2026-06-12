"""
Sliding-window rate limiter for the Ryzanstein LLM API.

Default limits are read from environment variables:
    RATE_LIMIT_REQUESTS  — max requests per window  (default: 60)
    RATE_LIMIT_WINDOW_S  — window size in seconds    (default: 60)

Each API key gets its own independent window.
Uses asyncio.Lock for per-key concurrency safety.
"""

import asyncio
import os
import time
from collections import deque
from typing import Deque, Dict

from fastapi import HTTPException, status


def _cfg() -> tuple[int, float]:
    limit = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
    window = float(os.environ.get("RATE_LIMIT_WINDOW_S", "60"))
    return max(1, limit), max(1.0, window)


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter keyed by API key."""

    def __init__(self) -> None:
        # key → deque of request timestamps (monotonic)
        self._windows: Dict[str, Deque[float]] = {}
        # key → asyncio.Lock (created lazily)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def check(self, api_key: str) -> None:
        """
        Record a request for `api_key`.

        Raises HTTP 429 if the rate limit is exceeded.
        """
        limit, window = _cfg()
        lock = await self._get_lock(api_key)

        async with lock:
            now = time.monotonic()
            cutoff = now - window

            if api_key not in self._windows:
                self._windows[api_key] = deque()

            dq = self._windows[api_key]

            # Evict timestamps outside the current window
            while dq and dq[0] < cutoff:
                dq.popleft()

            if len(dq) >= limit:
                oldest = dq[0]
                retry_after = int(window - (now - oldest)) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {limit} requests per {int(window)}s window.",
                    headers={"Retry-After": str(retry_after)},
                )

            dq.append(now)

    def stats(self, api_key: str) -> dict:
        """Return current window usage for an API key (for health/debug endpoints)."""
        limit, window = _cfg()
        dq = self._windows.get(api_key, deque())
        now = time.monotonic()
        cutoff = now - window
        active = sum(1 for t in dq if t >= cutoff)
        return {"key_prefix": api_key[:8] + "...", "requests_in_window": active, "limit": limit}


# Module-level singleton
_limiter: SlidingWindowRateLimiter | None = None


def get_limiter() -> SlidingWindowRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = SlidingWindowRateLimiter()
    return _limiter


async def check_rate_limit(api_key: str) -> None:
    """Convenience function — check rate limit for api_key."""
    await get_limiter().check(api_key)
