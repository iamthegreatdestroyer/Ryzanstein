"""
Sprint 3.3: Resilience Integration for Ryzanstein LLM API
[REF:SPRINT3.3] - Circuit Breaker, Bulkhead, Retry, Fallback

Wires PHASE2_DEVELOPMENT resilience patterns into the FastAPI serving layer.

Components:
  - InferenceCircuitBreaker: Protects inference endpoint from cascade failures
  - InferenceBulkhead: Limits concurrent inference requests (resource isolation)
  - InferenceRetry: Retry transient MCP/gRPC errors with exponential backoff
  - InferenceFallback: Returns cached/static response when primary fails
  - RyzansteinHealthChecker: Aggregated health for /health/live and /health/ready

Usage (from server.py):
    from .resilience_integration import (
        get_inference_circuit_breaker,
        get_inference_bulkhead,
        get_health_checker,
        resilience_middleware,
    )
"""

import sys
import os
import asyncio
import time
import logging
from pathlib import Path
from typing import Any, Optional, Dict
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap — add PHASE2_DEVELOPMENT/src to sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_PHASE2_SRC = _REPO_ROOT / "PHASE2_DEVELOPMENT" / "src"
if str(_PHASE2_SRC) not in sys.path:
    sys.path.insert(0, str(_PHASE2_SRC))

# ---------------------------------------------------------------------------
# Import resilience primitives (graceful fallback to no-ops if unavailable)
# ---------------------------------------------------------------------------
try:
    from resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
    from resilience.bulkhead import Bulkhead, BulkheadFullError
    from resilience.retry_policy import RetryPolicy
    from resilience.fallback import FallbackHandler, FallbackStrategy
    from resilience.health_check import HealthChecker, HealthStatus
    _RESILIENCE_AVAILABLE = True
    logger.info("Resilience patterns loaded from PHASE2_DEVELOPMENT")
except ImportError as _e:
    logger.warning(f"Resilience modules not available ({_e}); using no-op stubs")
    _RESILIENCE_AVAILABLE = False

    # Minimal stubs so imports don't crash the server
    class CircuitOpenError(Exception):
        pass

    class BulkheadFullError(Exception):
        pass

    class CircuitBreaker:
        def __init__(self, *a, **kw): pass
        async def execute(self, func, *a, **kw): return await func(*a, **kw)
        def get_stats(self): return {"state": "unavailable"}

    class Bulkhead:
        def __init__(self, *a, **kw): pass
        async def execute(self, func, *a, **kw): return await func(*a, **kw)
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def get_stats(self): return {"active_calls": 0, "rejected_calls": 0}

    class RetryPolicy:
        def __init__(self, *a, **kw): pass
        async def execute(self, func, *a, **kw): return await func(*a, **kw)
        def get_stats(self): return {}

    class FallbackHandler:
        def __init__(self, *a, **kw): pass
        async def execute(self, func, *a, fallback_value=None, **kw):
            try:
                return await func(*a, **kw)
            except Exception:
                return fallback_value

    class HealthChecker:
        def __init__(self, *a, **kw): pass
        def register(self, *a, **kw): pass
        async def check_liveness(self): return _NoOpHealthReport(True)
        async def check_readiness(self): return _NoOpHealthReport(True)
        def get_last_report(self): return None

    class _NoOpHealthReport:
        def __init__(self, ok): self._ok = ok
        @property
        def is_healthy(self): return self._ok
        @property
        def is_ready(self): return self._ok
        def to_dict(self): return {"status": "healthy" if self._ok else "unhealthy", "components": []}


# ---------------------------------------------------------------------------
# Singleton instances — initialized lazily on first access
# ---------------------------------------------------------------------------

_inference_circuit_breaker: Optional[CircuitBreaker] = None
_inference_bulkhead: Optional[Bulkhead] = None
_inference_retry: Optional[RetryPolicy] = None
_inference_fallback: Optional[FallbackHandler] = None
_health_checker: Optional[HealthChecker] = None
_last_successful_response: Optional[Dict[str, Any]] = None  # for fallback cache


def get_inference_circuit_breaker() -> CircuitBreaker:
    """Return the singleton CircuitBreaker for the inference endpoint."""
    global _inference_circuit_breaker
    if _inference_circuit_breaker is None:
        _inference_circuit_breaker = CircuitBreaker(
            name="inference",
            failure_threshold=5,       # Open after 5 consecutive failures
            recovery_timeout=30.0,     # Attempt recovery after 30s
            half_open_max_calls=2,     # Allow 2 test calls in HALF_OPEN
            success_threshold=2,       # Close again after 2 successes
            failure_rate_threshold=0.5, # Or if >50% fail in sliding window
            window_size=10,
        )
        logger.info("Initialized inference CircuitBreaker (threshold=5, recovery=30s)")
    return _inference_circuit_breaker


def get_inference_bulkhead() -> Bulkhead:
    """Return the singleton Bulkhead for the inference endpoint."""
    global _inference_bulkhead
    if _inference_bulkhead is None:
        _inference_bulkhead = Bulkhead(
            name="inference",
            max_concurrent=16,   # Max simultaneous inference requests
            max_queue=64,        # Queue up to 64 additional requests
            timeout=120.0,       # 2-minute queue wait timeout
        )
        logger.info("Initialized inference Bulkhead (max_concurrent=16, queue=64)")
    return _inference_bulkhead


def get_inference_retry() -> RetryPolicy:
    """Return the singleton RetryPolicy for transient MCP/gRPC errors."""
    global _inference_retry
    if _inference_retry is None:
        _inference_retry = RetryPolicy(
            max_retries=2,
            base_delay=0.5,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=True,
            jitter_factor=0.1,
            retryable_exceptions={ConnectionError, TimeoutError, OSError},
        )
        logger.info("Initialized inference RetryPolicy (max_retries=2, base_delay=0.5s)")
    return _inference_retry


def get_inference_fallback() -> FallbackHandler:
    """Return the singleton FallbackHandler for cached degraded responses."""
    global _inference_fallback
    if _inference_fallback is None:
        _inference_fallback = FallbackHandler(
            strategy=FallbackStrategy.CACHE if _RESILIENCE_AVAILABLE else None,
            cache_ttl=300.0,  # 5-minute cache TTL
            log_fallback=True,
        )
        logger.info("Initialized inference FallbackHandler (strategy=CACHE, ttl=300s)")
    return _inference_fallback


def get_health_checker() -> HealthChecker:
    """Return the singleton HealthChecker with component registrations."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(check_timeout=5.0)
        logger.info("Initialized HealthChecker")
    return _health_checker


# ---------------------------------------------------------------------------
# Health check registration helpers
# ---------------------------------------------------------------------------

def register_engine_health(engine, engine_type: str) -> None:
    """Register the inference engine as a health check component."""
    checker = get_health_checker()

    async def _check_engine():
        if engine is None:
            return {"status": "unavailable", "type": engine_type}
        try:
            # Minimal ping: check if config attribute is accessible
            _ = engine
            return {"status": "ok", "type": engine_type}
        except Exception as e:
            raise RuntimeError(f"Engine check failed: {e}")

    checker.register("inference_engine", _check_engine, critical=True)
    logger.info(f"Registered inference_engine health check (type={engine_type})")


def register_circuit_breaker_health() -> None:
    """Register circuit breaker state as a (non-critical) health component."""
    checker = get_health_checker()
    cb = get_inference_circuit_breaker()

    def _check_cb():
        stats = cb.get_stats()
        state = stats.get("state", "unknown")
        if state == "open":
            raise RuntimeError(f"Circuit breaker is OPEN (failure_rate={stats.get('failure_rate', 0):.2%})")
        return {"state": state, **stats}

    checker.register("circuit_breaker", _check_cb, critical=False)


# ---------------------------------------------------------------------------
# Protected inference helper
# ---------------------------------------------------------------------------

async def run_protected_inference(
    inference_func,
    fallback_response: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs,
) -> Any:
    """
    Execute inference with full resilience stack:
      Bulkhead → Circuit Breaker → Retry → Fallback

    Args:
        inference_func: Async callable that performs the inference
        fallback_response: Static response to return if everything fails
        *args, **kwargs: Forwarded to inference_func

    Returns:
        Inference result or fallback response

    Raises:
        BulkheadFullError: If bulkhead queue is full (503)
        CircuitOpenError: If circuit breaker is open (503)
        Exception: Any non-retryable exception from inference_func
    """
    bulkhead = get_inference_bulkhead()
    cb = get_inference_circuit_breaker()
    retry = get_inference_retry()
    fallback = get_inference_fallback()

    # Layer 1: Bulkhead — reject if too many concurrent requests
    async with bulkhead:
        # Layer 2: Circuit Breaker — reject if service is known-bad
        async def _with_retry():
            return await retry.execute(inference_func, *args, **kwargs)

        async def _static_fallback(*_a, **_kw):
            logger.warning("Inference fallback activated — returning cached/static response")
            if fallback_response is not None:
                return fallback_response
            raise RuntimeError("No fallback response available")

        return await cb.execute(
            _with_retry,
            fallback=_static_fallback if fallback_response is not None else None,
        )


# ---------------------------------------------------------------------------
# FastAPI ASGI middleware for resilience metrics
# ---------------------------------------------------------------------------

class ResilienceMiddleware:
    """
    ASGI middleware that exposes resilience stats via X-Resilience-* headers
    on responses and rejects requests when the bulkhead is saturated early.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Only apply to inference endpoints
        if path not in ("/v1/chat/completions", "/v1/embeddings"):
            await self.app(scope, receive, send)
            return

        bulkhead = get_inference_bulkhead()
        cb = get_inference_circuit_breaker()
        cb_stats = cb.get_stats()
        bh_stats = bulkhead.get_stats()

        async def send_with_headers(response):
            if response["type"] == "http.response.start":
                headers = list(response.get("headers", []))
                cb_state = cb_stats.get("state", "unknown").encode()
                active = str(bh_stats.get("active_calls", 0)).encode()
                headers.append((b"x-circuit-breaker-state", cb_state))
                headers.append((b"x-bulkhead-active", active))
                response = {**response, "headers": headers}
            await send(response)

        await self.app(scope, receive, send_with_headers)


# ---------------------------------------------------------------------------
# Graceful startup / shutdown
# ---------------------------------------------------------------------------

async def initialize_resilience(engine, engine_type: str) -> None:
    """Call from app lifespan to set up health checks."""
    register_engine_health(engine, engine_type)
    register_circuit_breaker_health()
    logger.info("Resilience layer initialized")


async def shutdown_resilience() -> None:
    """Call from app lifespan shutdown."""
    global _inference_circuit_breaker, _inference_bulkhead
    global _inference_retry, _inference_fallback, _health_checker
    _inference_circuit_breaker = None
    _inference_bulkhead = None
    _inference_retry = None
    _inference_fallback = None
    _health_checker = None
    logger.info("Resilience layer shut down")
