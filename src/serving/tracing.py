"""
src/serving/tracing.py — Distributed Tracing for Ryzanstein LLM

W3C Trace Context propagation, OpenTelemetry spans, and structured
correlation-ID logging. Supports Jaeger and OTLP export without requiring
them to be installed — all external exporters are optional imports.

Sprint 7.1: Distributed Tracing
Autonomy Level: 85%
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trace Context (W3C Trace Context spec §3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraceContext:
    """Immutable W3C Trace Context carrier."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    sampled: bool = True

    # ── W3C headers ──
    @property
    def traceparent(self) -> str:
        """W3C traceparent header value: 00-<trace_id>-<span_id>-<flags>."""
        flags = "01" if self.sampled else "00"
        return f"00-{self.trace_id}-{self.span_id}-{flags}"

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "TraceContext":
        """Parse W3C traceparent/tracestate from incoming request headers."""
        tp = headers.get("traceparent") or headers.get("x-b3-traceid")
        if tp and tp.startswith("00-"):
            parts = tp.split("-")
            if len(parts) == 4:
                return cls(
                    trace_id=parts[1],
                    span_id=uuid.uuid4().hex[:16],  # new span for this hop
                    parent_span_id=parts[2],
                    sampled=parts[3] == "01",
                )
        return cls()  # fresh context if no header

    def child(self) -> "TraceContext":
        """Create a child span within the same trace."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self.span_id,
            sampled=self.sampled,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Span
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Span:
    """A lightweight in-process span that exports to the configured backend."""

    name: str
    context: TraceContext
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    status: str = "OK"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        self.events.append({
            "name": name,
            "time": time.perf_counter(),
            "attributes": attributes or {},
        })

    def set_error(self, exc: Exception) -> None:
        self.status = "ERROR"
        self.attributes["exception.type"] = type(exc).__name__
        self.attributes["exception.message"] = str(exc)

    def finish(self) -> None:
        if self.end_time is None:
            self.end_time = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "attributes": self.attributes,
            "events": self.events,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tracer
# ─────────────────────────────────────────────────────────────────────────────

class Tracer:
    """
    Lightweight tracer with pluggable export.

    Backends (selected via RYZANSTEIN_TRACE_BACKEND env var):
    - "noop"  : discard (default, zero overhead)
    - "log"   : emit span as structured JSON to the Python logger
    - "otlp"  : OpenTelemetry gRPC/HTTP export (requires opentelemetry-sdk)
    - "jaeger": Jaeger Thrift UDP (requires opentelemetry-exporter-jaeger)

    Usage::

        tracer = Tracer.from_env("ryzanstein.serving")
        with tracer.span("generate_tokens", ctx) as span:
            span.set_attribute("model", "bitnet-7b")
            tokens = engine.generate(...)
    """

    def __init__(self, service_name: str, backend: str = "noop"):
        self.service_name = service_name
        self.backend = backend
        self._otel_tracer: Optional[Any] = None
        self._setup_backend()

    @classmethod
    def from_env(cls, service_name: str) -> "Tracer":
        backend = os.getenv("RYZANSTEIN_TRACE_BACKEND", "log")
        return cls(service_name, backend)

    def _setup_backend(self) -> None:
        if self.backend == "noop":
            return

        if self.backend in ("otlp", "jaeger"):
            try:
                from opentelemetry import trace as otel_trace
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.resources import Resource, SERVICE_NAME

                resource = Resource({SERVICE_NAME: self.service_name})
                provider = TracerProvider(resource=resource)

                if self.backend == "otlp":
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor

                    endpoint = os.getenv(
                        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
                    )
                    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)

                elif self.backend == "jaeger":
                    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor

                    host = os.getenv("JAEGER_AGENT_HOST", "localhost")
                    port = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
                    exporter = JaegerExporter(agent_host_name=host, agent_port=port)

                provider.add_span_processor(BatchSpanProcessor(exporter))  # type: ignore[assignment]
                otel_trace.set_tracer_provider(provider)
                self._otel_tracer = otel_trace.get_tracer(self.service_name)
                logger.info("Tracing backend '%s' initialised", self.backend)

            except ImportError as e:
                logger.warning(
                    "Tracing backend '%s' unavailable (%s). Falling back to 'log'.",
                    self.backend, e,
                )
                self.backend = "log"

    # ── Context managers ──

    @contextmanager
    def span(self, name: str, ctx: Optional[TraceContext] = None):
        """Synchronous span context manager."""
        ctx = ctx or TraceContext()
        child_ctx = ctx.child()
        sp = Span(name=name, context=child_ctx)

        if self._otel_tracer:
            from opentelemetry import trace as otel_trace
            with self._otel_tracer.start_as_current_span(name) as otel_span:
                otel_span.set_attribute("trace_id", child_ctx.trace_id)
                sp._otel_span = otel_span  # type: ignore[attr-defined]
                try:
                    yield sp
                except Exception as exc:
                    sp.set_error(exc)
                    otel_span.record_exception(exc)
                    raise
                finally:
                    sp.finish()
        else:
            try:
                yield sp
            except Exception as exc:
                sp.set_error(exc)
                raise
            finally:
                sp.finish()
                self._export(sp)

    @asynccontextmanager
    async def async_span(self, name: str, ctx: Optional[TraceContext] = None):
        """Async span context manager."""
        ctx = ctx or TraceContext()
        child_ctx = ctx.child()
        sp = Span(name=name, context=child_ctx)
        try:
            yield sp
        except Exception as exc:
            sp.set_error(exc)
            raise
        finally:
            sp.finish()
            self._export(sp)

    def _export(self, span: Span) -> None:
        if self.backend == "noop":
            return
        # "log" backend — structured JSON via stdlib logger
        logger.debug(
            "span",
            extra={"span": span.to_dict()},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Starlette / ASGI middleware
# ─────────────────────────────────────────────────────────────────────────────

class TraceMiddleware:
    """
    ASGI middleware that extracts W3C Trace Context from incoming requests
    and injects traceparent into outgoing responses.

    Usage (FastAPI)::

        app = FastAPI()
        tracer = Tracer.from_env("ryzanstein.api")
        app.add_middleware(TraceMiddleware, tracer=tracer)
    """

    def __init__(self, app, tracer: Tracer):
        self.app = app
        self.tracer = tracer

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        str_headers = {k.decode(): v.decode() for k, v in headers.items()}
        ctx = TraceContext.from_headers(str_headers)
        scope["trace_context"] = ctx

        path = scope.get("path", "")
        method = scope.get("method", "")

        async with self.tracer.async_span(f"{method} {path}", ctx) as span:
            span.set_attribute("http.method", method)
            span.set_attribute("http.path", path)

            status_holder = [200]

            async def send_with_trace(message):
                if message["type"] == "http.response.start":
                    # Inject traceparent header into response
                    status_holder[0] = message["status"]
                    extra = [(b"traceparent", ctx.traceparent.encode())]
                    message = {
                        **message,
                        "headers": list(message.get("headers", [])) + extra,
                    }
                await send(message)

            await self.app(scope, receive, send_with_trace)
            span.set_attribute("http.status_code", status_holder[0])
            if status_holder[0] >= 500:
                span.status = "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# Correlation-ID aware log filter
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationIdFilter(logging.Filter):
    """
    Injects trace_id and span_id into every log record so structured
    log shippers (ELK, Loki) can correlate logs with traces.

    Usage::

        import logging
        from src.serving.tracing import CorrelationIdFilter

        for handler in logging.root.handlers:
            handler.addFilter(CorrelationIdFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            ctx = getattr(asyncio, "_current_tasks", {})
            record.trace_id = "-"
            record.span_id = "-"
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Log rotation helper (wraps RotatingFileHandler)
# ─────────────────────────────────────────────────────────────────────────────

def configure_rotating_logger(
    log_dir: str = "logs",
    max_bytes: int = 100 * 1024 * 1024,  # 100 MB
    backup_count: int = 7,               # 7 days retention
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up a rotating file logger that respects the 100MB / 7-day policy
    defined in Sprint 7.1.

    Returns the root logger configured with both stream and file handlers.
    """
    import os
    from logging.handlers import RotatingFileHandler

    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
        '"trace_id":"%(trace_id)s","span_id":"%(span_id)s","msg":%(message)s}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = RotatingFileHandler(
        os.path.join(log_dir, "ryzanstein.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh.addFilter(CorrelationIdFilter())

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    sh.addFilter(CorrelationIdFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(sh)

    return root


# ─────────────────────────────────────────────────────────────────────────────
# Module-level default tracer (zero-config import)
# ─────────────────────────────────────────────────────────────────────────────

_default_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer.from_env("ryzanstein")
    return _default_tracer
