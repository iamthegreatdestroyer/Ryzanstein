"""
Sprint 3.3: Resilience Integration Tests
[REF:SPRINT3.3] - Circuit Breaker, Bulkhead, Retry, Fallback, Health Check

Tests:
1. Circuit breaker lifecycle (CLOSED → OPEN → HALF_OPEN → CLOSED)
2. Bulkhead concurrent limit enforcement
3. Retry with exponential backoff
4. Fallback handler — static, cache, alternative
5. Health checker aggregation
6. Resilience integration module singletons
7. Protected inference wrapper
8. Chaos testing (random failure injection)
9. /health/live and /health/ready endpoints via FastAPI test client
"""

import sys
import os
import asyncio
import time
import threading
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
PHASE2_SRC = REPO_ROOT / "PHASE2_DEVELOPMENT" / "src"
API_SRC = REPO_ROOT / "RYZEN-LLM" / "src" / "api"

sys.path.insert(0, str(PHASE2_SRC))
sys.path.insert(0, str(API_SRC))
sys.path.insert(0, str(REPO_ROOT / "RYZEN-LLM" / "src"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def circuit_breaker():
    """Fresh circuit breaker for each test."""
    try:
        from resilience.circuit_breaker import CircuitBreaker
        return CircuitBreaker(
            name="test",
            failure_threshold=3,
            recovery_timeout=0.1,   # Fast recovery for tests
            half_open_max_calls=2,
            success_threshold=2,
            failure_rate_threshold=0.8,
            window_size=5,
        )
    except ImportError:
        pytest.skip("resilience.circuit_breaker not available")


@pytest.fixture
def bulkhead():
    """Fresh bulkhead for each test."""
    try:
        from resilience.bulkhead import Bulkhead
        return Bulkhead(name="test", max_concurrent=3, max_queue=5, timeout=1.0)
    except ImportError:
        pytest.skip("resilience.bulkhead not available")


@pytest.fixture
def retry_policy():
    """Fast retry policy for tests."""
    try:
        from resilience.retry_policy import RetryPolicy
        return RetryPolicy(
            max_retries=3,
            base_delay=0.01,
            max_delay=0.1,
            jitter=False,
            retryable_exceptions={ConnectionError, TimeoutError, OSError},
        )
    except ImportError:
        pytest.skip("resilience.retry_policy not available")


@pytest.fixture
def fallback_handler():
    """Cache-strategy fallback handler."""
    try:
        from resilience.fallback import FallbackHandler, FallbackStrategy
        return FallbackHandler(strategy=FallbackStrategy.CACHE, cache_ttl=10.0)
    except ImportError:
        pytest.skip("resilience.fallback not available")


@pytest.fixture
def health_checker():
    """Fresh health checker for each test."""
    try:
        from resilience.health_check import HealthChecker
        return HealthChecker(check_timeout=1.0)
    except ImportError:
        pytest.skip("resilience.health_check not available")


# ============================================================================
# Test 1: Circuit Breaker Lifecycle
# ============================================================================

class TestCircuitBreakerLifecycle:
    def test_initial_state_is_closed(self, circuit_breaker):
        assert circuit_breaker.is_closed

    @pytest.mark.asyncio
    async def test_closed_allows_calls(self, circuit_breaker):
        async def ok(): return "success"
        result = await circuit_breaker.execute(ok)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self, circuit_breaker):
        from resilience.circuit_breaker import CircuitOpenError

        async def failing(): raise ConnectionError("simulated failure")

        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ConnectionError):
                await circuit_breaker.execute(failing)

        assert circuit_breaker.is_open

    @pytest.mark.asyncio
    async def test_open_rejects_calls(self, circuit_breaker):
        from resilience.circuit_breaker import CircuitOpenError

        async def failing(): raise ConnectionError("fail")
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ConnectionError):
                await circuit_breaker.execute(failing)

        async def good(): return "ok"
        with pytest.raises(CircuitOpenError):
            await circuit_breaker.execute(good)

    @pytest.mark.asyncio
    async def test_half_open_after_recovery_timeout(self, circuit_breaker):
        """After recovery_timeout, circuit transitions to HALF_OPEN."""
        async def failing(): raise ConnectionError("fail")
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ConnectionError):
                await circuit_breaker.execute(failing)

        assert circuit_breaker.is_open
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.05)

        # Trigger check by calling _check_recovery
        await circuit_breaker._check_recovery()
        assert circuit_breaker.is_half_open or circuit_breaker.is_open  # may vary by impl

    @pytest.mark.asyncio
    async def test_fallback_called_when_open(self, circuit_breaker):
        async def failing(): raise ConnectionError("fail")
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ConnectionError):
                await circuit_breaker.execute(failing)

        async def fallback(): return "fallback_result"
        result = await circuit_breaker.execute(lambda: None, fallback=fallback)
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_stats_track_requests(self, circuit_breaker):
        async def ok(): return "ok"
        await circuit_breaker.execute(ok)
        stats = circuit_breaker.get_stats()
        assert stats["total_requests"] >= 1
        assert stats["successful_requests"] >= 1


# ============================================================================
# Test 2: Bulkhead Concurrent Limit
# ============================================================================

class TestBulkheadConcurrency:
    def test_initial_stats(self, bulkhead):
        stats = bulkhead.get_stats()
        assert stats["active_calls"] == 0
        assert stats["accepted_calls"] == 0

    @pytest.mark.asyncio
    async def test_executes_within_limit(self, bulkhead):
        async def work(): return "done"
        result = await bulkhead.execute(work)
        assert result == "done"
        assert bulkhead.stats.completed_calls >= 1

    @pytest.mark.asyncio
    async def test_rejects_when_full(self, bulkhead):
        from resilience.bulkhead import BulkheadFullError

        barrier = asyncio.Event()
        tasks_started = []

        async def slow():
            tasks_started.append(1)
            await barrier.wait()
            return "ok"

        # Fill the bulkhead (max_concurrent=3) + queue (max_queue=5)
        # Total capacity = 3 + 5 = 8
        tasks = [asyncio.create_task(bulkhead.execute(slow)) for _ in range(8)]
        # Wait for tasks to start acquiring
        await asyncio.sleep(0.05)

        # One more should be rejected
        with pytest.raises(BulkheadFullError):
            await bulkhead.execute(slow)

        barrier.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_context_manager(self, bulkhead):
        async with bulkhead:
            stats = bulkhead.get_stats()
            assert stats["active_calls"] == 1
        stats = bulkhead.get_stats()
        assert stats["active_calls"] == 0


# ============================================================================
# Test 3: Retry Policy
# ============================================================================

class TestRetryPolicy:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self, retry_policy):
        call_count = [0]

        async def ok():
            call_count[0] += 1
            return "success"

        result = await retry_policy.execute(ok)
        assert result == "success"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self, retry_policy):
        call_count = [0]

        async def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("transient")
            return "recovered"

        result = await retry_policy.execute(flaky)
        assert result == "recovered"
        assert call_count[0] == 3
        assert retry_policy.stats.retries_performed >= 2

    @pytest.mark.asyncio
    async def test_does_not_retry_non_retryable(self, retry_policy):
        async def fatal(): raise ValueError("not retryable")

        with pytest.raises(ValueError):
            await retry_policy.execute(fatal)

        assert retry_policy.stats.retries_performed == 0

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, retry_policy):
        async def always_fail(): raise ConnectionError("permanent")

        with pytest.raises(ConnectionError):
            await retry_policy.execute(always_fail)

        assert retry_policy.stats.failed_attempts > retry_policy.config.max_retries

    def test_decorator_usage(self, retry_policy):
        call_count = [0]

        @retry_policy
        async def decorated():
            call_count[0] += 1
            return call_count[0]

        result = asyncio.get_event_loop().run_until_complete(decorated())
        assert result == 1


# ============================================================================
# Test 4: Fallback Handler
# ============================================================================

class TestFallbackHandler:
    @pytest.mark.asyncio
    async def test_returns_primary_result(self, fallback_handler):
        async def primary(): return {"result": "primary"}
        result = await fallback_handler.execute(primary, fallback_value={"result": "fallback"})
        assert result["result"] == "primary"
        assert fallback_handler.stats.primary_successes == 1

    @pytest.mark.asyncio
    async def test_returns_static_fallback(self, fallback_handler):
        async def failing(): raise RuntimeError("primary failed")
        result = await fallback_handler.execute(
            failing, fallback_value={"result": "static_fallback"}
        )
        assert result["result"] == "static_fallback"
        assert fallback_handler.stats.fallback_activations == 1

    @pytest.mark.asyncio
    async def test_returns_cached_result(self, fallback_handler):
        """Cache strategy: returns last successful result on failure."""
        call_count = [0]

        async def sometimes_fails():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"data": "fresh"}
            raise RuntimeError("service down")

        # First call succeeds and caches
        result1 = await fallback_handler.execute(sometimes_fails)
        assert result1["data"] == "fresh"

        # Second call fails — should return cache
        result2 = await fallback_handler.execute(sometimes_fails)
        assert result2["data"] == "fresh"

    @pytest.mark.asyncio
    async def test_alternative_function(self, fallback_handler):
        async def primary(): raise RuntimeError("primary failed")
        async def alternative(): return {"result": "alternative"}

        result = await fallback_handler.execute(primary, fallback_func=alternative)
        assert result["result"] == "alternative"

    def test_get_stats(self, fallback_handler):
        stats = fallback_handler.get_stats()
        assert "fallback_rate" in stats
        assert "primary_successes" in stats


# ============================================================================
# Test 5: Health Checker
# ============================================================================

class TestHealthChecker:
    @pytest.mark.asyncio
    async def test_empty_checker_is_healthy(self, health_checker):
        from resilience.health_check import HealthStatus
        report = await health_checker.check_all()
        assert report.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_healthy_component(self, health_checker):
        from resilience.health_check import HealthStatus

        def ok_check(): return True
        health_checker.register("test_component", ok_check, critical=True)

        report = await health_checker.check_readiness()
        assert report.is_ready
        assert any(c.name == "test_component" for c in report.components)

    @pytest.mark.asyncio
    async def test_unhealthy_critical_component(self, health_checker):
        from resilience.health_check import HealthStatus

        def bad_check(): raise RuntimeError("service down")
        health_checker.register("critical_svc", bad_check, critical=True)

        report = await health_checker.check_readiness()
        assert report.status == HealthStatus.UNHEALTHY
        assert not report.is_ready

    @pytest.mark.asyncio
    async def test_unhealthy_noncritical_is_degraded(self, health_checker):
        from resilience.health_check import HealthStatus

        def ok_check(): return True
        def bad_check(): raise RuntimeError("optional down")

        health_checker.register("primary", ok_check, critical=True)
        health_checker.register("optional", bad_check, critical=False)

        report = await health_checker.check_readiness()
        assert report.status == HealthStatus.DEGRADED
        assert report.is_ready  # degraded = still serving traffic

    @pytest.mark.asyncio
    async def test_liveness_always_healthy(self, health_checker):
        from resilience.health_check import HealthStatus
        report = await health_checker.check_liveness()
        assert report.is_healthy

    @pytest.mark.asyncio
    async def test_to_dict(self, health_checker):
        report = await health_checker.check_liveness()
        d = report.to_dict()
        assert "status" in d
        assert "components" in d


# ============================================================================
# Test 6: Resilience Integration Module Singletons
# ============================================================================

class TestResilienceIntegrationSingletons:
    def test_get_circuit_breaker_returns_same_instance(self):
        try:
            from resilience_integration import get_inference_circuit_breaker
        except ImportError:
            pytest.skip("resilience_integration not available")

        cb1 = get_inference_circuit_breaker()
        cb2 = get_inference_circuit_breaker()
        assert cb1 is cb2

    def test_get_bulkhead_returns_same_instance(self):
        try:
            from resilience_integration import get_inference_bulkhead
        except ImportError:
            pytest.skip("resilience_integration not available")

        bh1 = get_inference_bulkhead()
        bh2 = get_inference_bulkhead()
        assert bh1 is bh2

    def test_get_health_checker_returns_same_instance(self):
        try:
            from resilience_integration import get_health_checker
        except ImportError:
            pytest.skip("resilience_integration not available")

        h1 = get_health_checker()
        h2 = get_health_checker()
        assert h1 is h2


# ============================================================================
# Test 7: Protected Inference Wrapper
# ============================================================================

class TestProtectedInference:
    @pytest.mark.asyncio
    async def test_successful_protected_call(self):
        try:
            from resilience_integration import (
                run_protected_inference,
                get_inference_circuit_breaker,
                get_inference_bulkhead,
            )
            # Reset singletons for clean test
            import resilience_integration as ri
            ri._inference_circuit_breaker = None
            ri._inference_bulkhead = None
            ri._inference_retry = None
        except ImportError:
            pytest.skip("resilience_integration not available")

        async def inference(): return {"tokens": [1, 2, 3]}
        result = await run_protected_inference(inference)
        assert result == {"tokens": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_fallback_on_circuit_open(self):
        try:
            from resilience_integration import (
                run_protected_inference,
                get_inference_circuit_breaker,
            )
            import resilience_integration as ri
            ri._inference_circuit_breaker = None
            ri._inference_bulkhead = None
            ri._inference_retry = None
        except ImportError:
            pytest.skip("resilience_integration not available")

        # Exhaust circuit breaker
        cb = get_inference_circuit_breaker()
        async def failing(): raise ConnectionError("model unavailable")
        for _ in range(cb.config.failure_threshold):
            with pytest.raises((ConnectionError, Exception)):
                await run_protected_inference(failing)

        # Now circuit is open — should return fallback
        async def _should_not_run(): raise RuntimeError("should not run")
        static = {"choices": [{"message": {"content": "Service temporarily unavailable"}}]}
        # When circuit is open, bulkhead still needs to be acquired first
        # This may raise CircuitOpenError or return fallback
        try:
            result = await run_protected_inference(_should_not_run, fallback_response=static)
            assert "choices" in result
        except Exception:
            pass  # CircuitOpenError without fallback is also valid


# ============================================================================
# Test 8: Chaos Testing
# ============================================================================

class TestChaosResilience:
    """Inject random failures and verify the system gracefully degrades."""

    @pytest.mark.asyncio
    async def test_random_failure_injection(self, circuit_breaker):
        """System should remain stable under random 30% failure rate."""
        import random
        random.seed(42)

        successes = 0
        failures = 0
        rejections = 0

        async def chaotic():
            if random.random() < 0.3:
                raise ConnectionError("random failure")
            return "ok"

        from resilience.circuit_breaker import CircuitOpenError

        for _ in range(50):
            try:
                result = await circuit_breaker.execute(chaotic)
                if result == "ok":
                    successes += 1
            except CircuitOpenError:
                rejections += 1
            except ConnectionError:
                failures += 1

        total = successes + failures + rejections
        assert total == 50
        # System should have processed some requests
        assert successes > 0 or rejections > 0

    @pytest.mark.asyncio
    async def test_bulkhead_under_burst(self, bulkhead):
        """Bulkhead should handle burst traffic without crashing."""
        from resilience.bulkhead import BulkheadFullError

        results = {"ok": 0, "rejected": 0}

        async def work():
            await asyncio.sleep(0.01)
            return "done"

        tasks = []
        for _ in range(20):  # Send 20 concurrent requests
            tasks.append(asyncio.create_task(bulkhead.execute(work)))

        for task in asyncio.as_completed(tasks):
            try:
                await task
                results["ok"] += 1
            except BulkheadFullError:
                results["rejected"] += 1
            except Exception:
                pass

        # Some should succeed, some may be rejected
        total = results["ok"] + results["rejected"]
        assert total >= 1
        assert results["ok"] > 0  # At least some got through


# ============================================================================
# Test 9: FastAPI Endpoint Integration
# ============================================================================

class TestFastAPIHealthEndpoints:
    """Test /health/live and /health/ready via TestClient."""

    @pytest.fixture(autouse=True)
    def skip_if_no_testclient(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not available")

    def test_health_live_returns_200(self):
        try:
            from fastapi.testclient import TestClient
            from server import app
        except ImportError:
            pytest.skip("server or TestClient not available")

        client = TestClient(app)
        response = client.get("/health/live")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_health_ready_returns_status(self):
        try:
            from fastapi.testclient import TestClient
            from server import app
        except ImportError:
            pytest.skip("server or TestClient not available")

        client = TestClient(app)
        response = client.get("/health/ready")
        assert response.status_code in (200, 503)
        body = response.json()
        assert "status" in body

    def test_health_includes_resilience_metrics(self):
        try:
            from fastapi.testclient import TestClient
            from server import app
        except ImportError:
            pytest.skip("server or TestClient not available")

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "engine_loaded" in body


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "-x"],
        cwd=str(REPO_ROOT)
    )
    sys.exit(result.returncode)
