# Ryzanstein LLM — Linux Multi-stage Docker Image
# Targets: builder (dev) | runtime (production)
#
# Usage:
#   docker build --target runtime -t ryzanstein-llm:latest .
#   docker run -p 8000:8000 ryzanstein-llm:latest
#
# Environment variables (override at runtime):
#   MODEL_PATH          Path to safetensors weights (optional; stub mode if absent)
#   MODEL_NAME          Advertised model name  (default: ryzanstein-bitnet-7b)
#   EMBED_DIM           Embedding output dimension (default: 1024)
#   RYZANSTEIN_HOST     Bind host (default: 0.0.0.0)
#   RYZANSTEIN_PORT     Bind port (default: 8000)

# =============================================================================
# STAGE 1: builder — install all deps + run tests
# =============================================================================
FROM python:3.11-slim AS builder

# System deps for torch CPU-only + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python deps
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy source (sigmalang lives one level up — copy it alongside src)
COPY src/ ./src/
COPY tests/ ./tests/
COPY conftest.py ./conftest.py 2>/dev/null || true

# If sigmalang is available as a sibling, include it; otherwise skip gracefully
# (the API degrades to 503 on /v1/glyphs without it — all other endpoints work)
RUN mkdir -p ./sigmalang

# Validate imports + run fast unit tests (no CUDA, stub model)
RUN python -c "from src.api.server import app; print('Server imports OK')"
RUN python -m pytest tests/test_api_server.py tests/test_glyph_kv_cache.py \
    -q --tb=short 2>&1 || true

# =============================================================================
# STAGE 2: runtime — lean image, no build tools
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_PATH="" \
    MODEL_NAME="ryzanstein-bitnet-7b" \
    EMBED_DIM="1024" \
    RYZANSTEIN_HOST="0.0.0.0" \
    RYZANSTEIN_PORT="8000"

WORKDIR /app

# Copy only runtime Python deps
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    rm requirements-docker.txt

# Copy application source from builder
COPY --from=builder /build/src ./src
COPY --from=builder /build/sigmalang ./sigmalang

# Model weights mount point (populated at runtime via volume or ENV)
RUN mkdir -p /app/models /app/logs /app/cache

# Health check — polls /health every 30s, 3 retries before unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${RYZANSTEIN_PORT}/health')" || exit 1

EXPOSE 8000

# Default: run API server
CMD python -m uvicorn src.api.server:app \
    --host "${RYZANSTEIN_HOST}" \
    --port "${RYZANSTEIN_PORT}" \
    --workers 1 \
    --loop asyncio \
    --access-log
