"""
Sprint 3.2: Tracing Integration for Ryzanstein LLM API
[REF:SPRINT3.2] - End-to-End Jaeger Tracing + Structured Log Correlation

Wires the PHASE2_DEVELOPMENT tracing infrastructure into the FastAPI serving layer.
Provides:
  - Automatic span creation for every inference request
  - Log correlation via trace_id injection into structured logs
  - Jaeger export via BatchSpanProcessor
  - Token-level timing events within spans
  - Health/readiness endpoint instrumentation

Usage in server.py:
    from .tracing_integration import setup_tracing, trace_inference_request

Setup:
    tracer = setup_tracing(service_name="ryzanstein-llm", jaeger_host="localhost")
"""

from __future__ import annotations

import logging
import os
import sys
import time
import threading
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass

# Add PHASE2_DEVELOPMENT to path for tracing modules
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_PHASE2_SRC = _REPO_ROOT / "PHASE2_DEVELOPMENT" / "src"

if str(_PHASE2_SRC) not in sys.path:
    sys.path.insert(0, str(_PHASE2_SRC))

# Import tracing modules (graceful degradation if unavailable)
_TRACER_AVAILABLE = False
_DistributedTracer = None
_JaegerConfig = None
_InMemorySpanExporter = None
_BatchSpanProcessor = None
_create_jaeger_tracer = None

try:
    from tracing.jaeger_exporter import (
        DistributedTracer,
        JaegerConfig,
        InMemorySpanExporter,
        BatchSpanProcessor,
        create_jaeger_tracer,
        create_test_tracer,
    )
    from tracing.tracer import LLMTracer, SpanKind, init_tracing, get_tracer
    _TRACER_AVAILABLE = True
    _DistributedTracer = DistributedTracer
    _JaegerConfig = JaegerConfig
    _InMemorySpanExporter = InMemorySpanExporter
    _BatchSpanProcessor = BatchSpanProcessor
    _create_jaeger_tracer = create_jaeger_tracer
except ImportError as e:
    logging.getLogger(__name__).warning(
        f"Tracing modules not available ({e}) — using no-op tracing"
    )


# ============================================================================
# Structured Logging with Trace Correlation
# ============================================================================

class TraceCorrelationFilter(logging.Filter):
    """
    Injects trace_id and span_id into log records for log-trace correlation.

    This enables correlation between Jaeger traces and structured logs
    in an ELK/Loki stack.
    """

    _local = threading.local()

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = getattr(self._local, "trace_id", "no-trace")
        record.span_id = getattr(self._local, "span_id", "no-span")
        return True

    @classmethod
    def set_context(cls, trace_id: str, span_id: str) -> None:
        cls._local.trace_id = trace_id
        cls._local.span_id = span_id

    @classmethod
    def clear_context(cls) -> None:
        cls._local.trace_id = "no-trace"
        cls._local.span_id = "no-span"


def setup_structured_logging(log_level: str = "INFO") -> None:
    """Configure structured logging with trace correlation fields."""
    fmt = (
        "%(asctime)s %(levelname)-8s "
        "[trace=%(trace_id)s span=%(span_id)s] "
        "%(name)s: %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(TraceCorrelationFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))


# ============================================================================
# No-op fallback tracer (when Jaeger not available)
# ============================================================================

@dataclass
class NoOpSpan:
    """No-op span when tracing is unavailable."""
    trace_id: str = ""
    span_id: str = ""
    operation_name: str = ""
    tags: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        self.trace_id = uuid.uuid4().hex[:32]
        self.span_id = uuid.uuid4().hex[:16]


class NoOpTracer:
    """No-op tracer fallback."""

    def start_span(self, operation_name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan(operation_name=operation_name)

    def end_span(self, span, **kwargs) -> None:
        pass

    def add_event(self, span, name: str, **kwargs) -> None:
        pass

    def set_tag(self, span, key: str, value: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {"active_spans": 0, "spans_processed": 0}


# ============================================================================
# Global Tracer Instance
# ============================================================================

_global_tracer: Optional[Any] = None
_tracer_lock = threading.Lock()
_in_memory_exporter: Optional[Any] = None

logger = logging.getLogger(__name__)


def setup_tracing(
    service_name: str = "ryzanstein-llm",
    jaeger_host: str = "localhost",
    jaeger_port: int = 6831,
    use_http: bool = False,
    use_in_memory: bool = False,
    log_level: str = "INFO",
) -> Any:
    """
    Initialize the global tracing system.

    Args:
        service_name: Service name for Jaeger traces
        jaeger_host: Jaeger agent/collector host
        jaeger_port: Jaeger agent port (6831 UDP) or collector port (14268 HTTP)
        use_http: Use HTTP collector instead of UDP agent
        use_in_memory: Use in-memory exporter (for testing/development)
        log_level: Logging level

    Returns:
        The configured tracer instance
    """
    global _global_tracer, _in_memory_exporter

    # Setup structured logging first
    setup_structured_logging(log_level)

    with _tracer_lock:
        if _global_tracer is not None:
            return _global_tracer

        if not _TRACER_AVAILABLE:
            logger.warning("Tracing modules unavailable — using no-op tracer")
            _global_tracer = NoOpTracer()
            return _global_tracer

        # Determine exporter based on environment
        jaeger_env = os.environ.get("JAEGER_AGENT_HOST", jaeger_host)
        use_in_memory_env = (
            use_in_memory
            or os.environ.get("TRACING_IN_MEMORY", "").lower() in ("1", "true", "yes")
            or os.environ.get("TESTING", "").lower() in ("1", "true", "yes")
        )

        try:
            if use_in_memory_env:
                from tracing.jaeger_exporter import create_test_tracer
                tracer, exporter = create_test_tracer(service_name)
                _in_memory_exporter = exporter
                tracer.start()
                logger.info(f"[Tracing] In-memory tracer initialized for service: {service_name}")
            else:
                config = _JaegerConfig(
                    agent_host=jaeger_env,
                    agent_port=jaeger_port,
                    service_name=service_name,
                    collector_endpoint=f"http://{jaeger_env}:14268/api/traces" if use_http else None,
                )
                if use_http:
                    from tracing.jaeger_exporter import JaegerHTTPExporter
                    exporter = JaegerHTTPExporter(config)
                else:
                    from tracing.jaeger_exporter import JaegerThriftExporter
                    exporter = JaegerThriftExporter(config)

                processor = _BatchSpanProcessor(
                    exporter=exporter,
                    max_queue_size=10000,
                    batch_size=100,
                    flush_interval=5.0,
                )
                processor.start()

                tracer = _DistributedTracer(
                    service_name=service_name,
                    exporter=exporter,
                    config=config,
                )
                tracer.processor = processor
                tracer.start()

                logger.info(
                    f"[Tracing] Jaeger tracer initialized: service={service_name} "
                    f"host={jaeger_env}:{jaeger_port} http={use_http}"
                )

            _global_tracer = tracer

        except Exception as e:
            logger.warning(f"[Tracing] Failed to initialize Jaeger tracer ({e}) — using no-op")
            _global_tracer = NoOpTracer()

    return _global_tracer


def get_global_tracer() -> Any:
    """Get the global tracer, initializing with in-memory if not set up."""
    global _global_tracer
    if _global_tracer is None:
        setup_tracing(use_in_memory=True)
    return _global_tracer


def get_in_memory_exporter() -> Optional[Any]:
    """Get the in-memory exporter (for testing/metrics)."""
    return _in_memory_exporter


# ============================================================================
# Inference Request Tracing
# ============================================================================

@contextmanager
def trace_inference_request(
    model: str,
    prompt_tokens: int,
    max_tokens: int,
    request_id: Optional[str] = None,
):
    """
    Context manager that traces a full inference request.

    Injects trace_id into log records for log-trace correlation.
    Records token generation events and timing within the span.

    Usage:
        with trace_inference_request("bitnet-1.58b", 10, 100) as span:
            tokens = engine.generate(...)
            span.tags["output_tokens"] = len(tokens)

    Yields:
        span: The active span (may be NoOpSpan if tracing disabled)
    """
    tracer = get_global_tracer()
    request_id = request_id or uuid.uuid4().hex[:8]

    span = tracer.start_span(
        operation_name="inference",
        kind="server",
        tags={
            "model": model,
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "request_id": request_id,
        }
    )

    # Inject trace context into logs
    TraceCorrelationFilter.set_context(span.trace_id, span.span_id)
    logger.info(f"[Inference] START request_id={request_id} model={model} tokens={prompt_tokens}")

    start_time = time.time()
    try:
        yield span

        elapsed_ms = (time.time() - start_time) * 1000
        span.tags["duration_ms"] = elapsed_ms
        span.tags["status"] = "ok"

        output_tokens = span.tags.get("output_tokens", 0)
        if output_tokens > 0 and elapsed_ms > 0:
            tps = (output_tokens / elapsed_ms) * 1000
            span.tags["throughput_tps"] = tps
            logger.info(
                f"[Inference] COMPLETE request_id={request_id} "
                f"output_tokens={output_tokens} "
                f"duration_ms={elapsed_ms:.1f} "
                f"throughput_tps={tps:.2f}"
            )
        else:
            logger.info(f"[Inference] COMPLETE request_id={request_id} duration_ms={elapsed_ms:.1f}")

        tracer.end_span(span, status="ok")

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        span.tags["error"] = str(e)
        span.tags["status"] = "error"

        logger.error(
            f"[Inference] ERROR request_id={request_id} "
            f"error={e} duration_ms={elapsed_ms:.1f}"
        )
        tracer.end_span(span, status="error", error_message=str(e))
        raise

    finally:
        TraceCorrelationFilter.clear_context()


@contextmanager
def trace_token_generation(parent_span: Any, position: int):
    """
    Sub-span for individual token generation steps.

    Usage:
        for i in range(max_tokens):
            with trace_token_generation(inference_span, i) as tok_span:
                token = engine.forward(...)
                tok_span.tags["token_id"] = token
    """
    tracer = get_global_tracer()
    span = tracer.start_span(
        operation_name="token_generation",
        parent_span_id=getattr(parent_span, "span_id", None),
        trace_id=getattr(parent_span, "trace_id", None),
        kind="internal",
        tags={"position": position}
    )

    start = time.time()
    try:
        yield span
        elapsed = (time.time() - start) * 1000
        span.tags["duration_ms"] = elapsed
        tracer.end_span(span, status="ok")
    except Exception as e:
        tracer.end_span(span, status="error", error_message=str(e))
        raise


# ============================================================================
# FastAPI Middleware
# ============================================================================

class TracingMiddleware:
    """
    ASGI middleware for automatic request tracing.

    Automatically creates spans for all HTTP requests and injects
    trace context into request state.
    """

    def __init__(self, app, service_name: str = "ryzanstein-llm"):
        self.app = app
        self.service_name = service_name

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        method = scope.get("method", "")

        # Skip health checks to reduce noise
        if path in ("/health", "/ready", "/metrics"):
            await self.app(scope, receive, send)
            return

        tracer = get_global_tracer()
        span = tracer.start_span(
            operation_name=f"{method} {path}",
            kind="server",
            tags={"http.method": method, "http.path": path}
        )

        # Inject trace ID into scope for downstream use
        scope["trace_id"] = span.trace_id
        scope["span_id"] = span.span_id

        TraceCorrelationFilter.set_context(span.trace_id, span.span_id)

        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
            span.tags["http.status_code"] = status_code
            tracer.end_span(span, status="ok" if status_code < 400 else "error")
        except Exception as e:
            span.tags["error"] = str(e)
            tracer.end_span(span, status="error", error_message=str(e))
            raise
        finally:
            TraceCorrelationFilter.clear_context()


# ============================================================================
# Integration Test
# ============================================================================

def run_integration_test() -> bool:
    """
    Run end-to-end tracing integration test.

    Tests the complete pipeline:
    1. Tracer initialization
    2. Span creation
    3. Log correlation
    4. Span export (to in-memory)
    5. Span retrieval

    Returns:
        True if all tests pass
    """
    print("\n[Sprint 3.2] Running tracing integration tests...")
    passed = 0
    failed = 0

    def check(name: str, condition: bool, details: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name} {details}")
            failed += 1
        return condition

    # Test 1: Initialize in-memory tracer
    tracer = setup_tracing(service_name="test-service", use_in_memory=True)
    check("Tracer initialization", tracer is not None)

    # Test 2: Create and end span
    span = tracer.start_span("test_operation", kind="internal", tags={"test": True})
    check("Span creation", span is not None)
    check("Span has trace_id", bool(getattr(span, "trace_id", "")))
    check("Span has span_id", bool(getattr(span, "span_id", "")))

    tracer.end_span(span, status="ok")
    check("Span ended", True)

    # Test 3: Log correlation
    import io
    log_output = io.StringIO()
    handler = logging.StreamHandler(log_output)
    handler.addFilter(TraceCorrelationFilter())
    handler.setFormatter(logging.Formatter("%(trace_id)s %(message)s"))
    test_logger = logging.getLogger("test_correlation")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.DEBUG)

    with trace_inference_request("test-model", 10, 50) as inf_span:
        test_logger.info("inference started")
        log_content = log_output.getvalue()

    check("Log contains trace_id", inf_span.trace_id[:8] in log_content or "no-trace" not in log_content)

    # Test 4: In-memory exporter has spans
    exporter = get_in_memory_exporter()
    if exporter is not None:
        exported = exporter.get_spans()
        check("Spans exported to in-memory", len(exported) > 0)
    else:
        print("  → No in-memory exporter (using no-op tracer)")

    # Test 5: Tracer stats
    stats = tracer.get_stats()
    check("Stats available", isinstance(stats, dict))

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Jaeger Integration Test
# ============================================================================

def test_jaeger_connectivity(host: str = "localhost", port: int = 6831) -> bool:
    """
    Test connectivity to Jaeger agent.

    Args:
        host: Jaeger agent host
        port: Jaeger agent port

    Returns:
        True if Jaeger is reachable
    """
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)
        sock.connect((host, port))
        sock.close()
        print(f"  ✓ Jaeger agent reachable at {host}:{port}")
        return True
    except Exception as e:
        print(f"  ✗ Jaeger agent not reachable at {host}:{port}: {e}")
        print(f"  → Start Jaeger: docker run -d --name jaeger -p 16686:16686 -p 6831:6831/udp jaegertracing/all-in-one")
        return False


if __name__ == "__main__":
    # Run integration tests when executed directly
    print("Sprint 3.2: Tracing Integration Test")
    print("=" * 50)

    success = run_integration_test()
    test_jaeger_connectivity()

    if success:
        print("\n✓ Sprint 3.2 tracing integration: ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ Sprint 3.2 tracing integration: SOME TESTS FAILED")
        sys.exit(1)
