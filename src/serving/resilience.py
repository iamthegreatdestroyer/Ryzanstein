"""
src/serving/resilience.py — Resilience Layer for Ryzanstein LLM

Sprint 7.2: Resilience Layer
Autonomy Level: 80%

Implements:
- CircuitBreaker (per-endpoint, half-open probing)
- GracefulDegrader (mock engine fallback when C++ engine fails)
- WorkerWatchdog (restarts crashed async workers)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation — requests pass through
    OPEN = "open"           # Tripped — requests fast-fail immediately
    HALF_OPEN = "half_open" # Probing — one request allowed to test recovery


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # failures before opening
    success_threshold: int = 2          # successes in half-open before closing
    timeout_seconds: float = 60.0       # time in OPEN before probing
    half_open_max_calls: int = 1        # concurrent calls allowed in half-open


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""


class CircuitBreaker:
    """
    Per-endpoint circuit breaker following the standard three-state pattern.

    Usage::

        cb = CircuitBreaker("bitnet-engine", CircuitBreakerConfig())

        async def call():
            async with cb:
                return await engine.generate(prompt)

        result = await call()
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def __aenter__(self):
        async with self._lock:
            await self._check_state()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            if exc_type is None:
                await self._on_success()
            elif exc_type is not CircuitOpenError:
                await self._on_failure()
        return False  # never suppress the exception

    async def _check_state(self) -> None:
        if self._state == CircuitState.CLOSED:
            return

        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0)
            if elapsed >= self.config.timeout_seconds:
                self._transition(CircuitState.HALF_OPEN)
            else:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is OPEN "
                    f"(retry in {self.config.timeout_seconds - elapsed:.1f}s)"
                )

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is HALF_OPEN — probe slot taken"
                )
            self._half_open_calls += 1

    async def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0  # reset on any success

    async def _on_failure(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            # Probe failed — re-open immediately
            self._transition(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        old = self._state
        self._state = new_state
        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
            self._half_open_calls = 0
            self._success_count = 0
            logger.warning("CircuitBreaker '%s': %s → OPEN", self.name, old.value)
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
            logger.info("CircuitBreaker '%s': OPEN → HALF_OPEN (probing)", self.name)
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            logger.info("CircuitBreaker '%s': HALF_OPEN → CLOSED", self.name)

    def metrics(self) -> dict:
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Graceful Degradation (mock engine fallback)
# ─────────────────────────────────────────────────────────────────────────────

class MockInferenceEngine:
    """
    Fallback inference engine returned when the real C++ engine fails.

    Returns a clearly-labelled degraded response so the API layer never
    returns an empty body, preserving HTTP 200 contracts for load balancers.
    """

    async def generate(self, prompt: str, max_tokens: int = 64) -> str:
        await asyncio.sleep(0.05)  # simulate minimal latency
        return (
            "[DEGRADED MODE — C++ engine unavailable. "
            f"Echo: {prompt[:80]}{'...' if len(prompt) > 80 else ''}]"
        )

    async def embed(self, text: str) -> list[float]:
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
        import random
        rng = random.Random(seed)
        return [rng.uniform(-1, 1) for _ in range(768)]


class GracefulDegrader:
    """
    Wraps the real inference engine with fallback logic.

    If the real engine raises within ``max_failures`` consecutive calls or
    the circuit breaker opens, all subsequent calls automatically route to
    the MockInferenceEngine until the primary recovers.

    Usage::

        real_engine = BitNetEngine(config)
        degrader = GracefulDegrader(real_engine)

        # Transparent — callers don't know which engine is used
        result = await degrader.generate(prompt)
    """

    def __init__(
        self,
        primary: Any,
        fallback: Optional[Any] = None,
        max_failures: int = 3,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or MockInferenceEngine()
        self._circuit = CircuitBreaker(
            "primary-engine",
            CircuitBreakerConfig(failure_threshold=max_failures),
        )
        self._using_fallback = False

    async def generate(self, prompt: str, **kwargs) -> str:
        try:
            async with self._circuit:
                result = await self.primary.generate(prompt, **kwargs)
                if self._using_fallback:
                    logger.info("Primary engine recovered — switching back from fallback")
                    self._using_fallback = False
                return result
        except CircuitOpenError:
            if not self._using_fallback:
                logger.warning("Primary engine circuit open — routing to MockInferenceEngine")
                self._using_fallback = True
            return await self.fallback.generate(prompt, **kwargs)
        except Exception as exc:
            logger.error("Primary engine error: %s — routing to fallback", exc)
            return await self.fallback.generate(prompt, **kwargs)

    async def embed(self, text: str) -> list[float]:
        try:
            async with self._circuit:
                return await self.primary.embed(text)
        except (CircuitOpenError, Exception):
            return await self.fallback.embed(text)

    def metrics(self) -> dict:
        return {
            "circuit": self._circuit.metrics(),
            "using_fallback": self._using_fallback,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Worker Watchdog
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WatchdogConfig:
    restart_delay_seconds: float = 1.0   # wait before restarting
    max_restarts: int = 10               # give up after N restarts
    restart_window_seconds: float = 60.0 # restarts within this window count


class WorkerWatchdog:
    """
    Supervises an async worker coroutine and restarts it if it crashes.

    Implements exponential back-off and a restart-rate limiter so a
    persistently failing worker doesn't spin-loop consuming CPU.

    Usage::

        async def my_worker():
            while True:
                await process_queue()

        watchdog = WorkerWatchdog("batch-flusher", my_worker)
        await watchdog.start()
        # watchdog runs until cancelled or max_restarts exceeded
    """

    def __init__(
        self,
        name: str,
        factory: Callable[[], Coroutine],
        config: Optional[WatchdogConfig] = None,
    ) -> None:
        self.name = name
        self.factory = factory
        self.config = config or WatchdogConfig()
        self._task: Optional[asyncio.Task] = None
        self._restart_times: list[float] = []
        self._total_restarts = 0
        self._running = False

    async def start(self) -> None:
        """Start the watchdog supervision loop."""
        self._running = True
        logger.info("Watchdog '%s': starting", self.name)
        await self._supervise()

    async def stop(self) -> None:
        """Gracefully stop the watchdog and the worker."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Watchdog '%s': stopped", self.name)

    async def _supervise(self) -> None:
        while self._running:
            self._task = asyncio.create_task(
                self.factory(), name=f"watchdog-{self.name}"
            )
            try:
                await self._task
                # Worker returned cleanly — no restart needed
                logger.info("Watchdog '%s': worker exited cleanly", self.name)
                break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._running:
                    break

                self._total_restarts += 1
                now = time.monotonic()

                # Prune old restart timestamps
                self._restart_times = [
                    t for t in self._restart_times
                    if now - t < self.config.restart_window_seconds
                ]
                self._restart_times.append(now)

                recent = len(self._restart_times)
                logger.error(
                    "Watchdog '%s': worker crashed (#%d in window, %d total) — %s",
                    self.name, recent, self._total_restarts, exc,
                )

                if self._total_restarts >= self.config.max_restarts:
                    logger.critical(
                        "Watchdog '%s': giving up after %d restarts",
                        self.name, self.config.max_restarts,
                    )
                    self._running = False
                    break

                # Exponential back-off capped at 30s
                delay = min(
                    self.config.restart_delay_seconds * (2 ** min(recent - 1, 5)),
                    30.0,
                )
                logger.info(
                    "Watchdog '%s': restarting in %.1fs…", self.name, delay
                )
                await asyncio.sleep(delay)

    def metrics(self) -> dict:
        return {
            "name": self.name,
            "running": self._running,
            "total_restarts": self._total_restarts,
            "recent_restarts": len(self._restart_times),
            "task_done": self._task.done() if self._task else True,
        }
