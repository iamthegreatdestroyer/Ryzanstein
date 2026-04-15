# Ryzanstein LLM — Production Docker Image
# Multi-stage build: C++ builder → Python runtime
# [REF:TASK4.1] - Production Docker Images

# =============================================================================
# STAGE 1: C++ Build Environment
# =============================================================================
FROM mcr.microsoft.com/windows/servercore:ltsc2022 AS cpp-builder

# Install build tools and dependencies
RUN powershell -Command \
    Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_BuildTools.exe" -OutFile vs_installer.exe ; \
    .\vs_installer.exe --quiet --wait --norestart --nocache \
      --installPath "C:\BuildTools" \
      --add Microsoft.VisualStudio.Workload.VCTools \
      --add Microsoft.VisualStudio.Component.VC.CMake.Project \
      --add Microsoft.VisualStudio.Component.Windows11SDK.22621 \
    ; \
    Remove-Item -Force vs_installer.exe

# Install CMake, Git, and Python
RUN powershell -Command \
    Invoke-WebRequest -Uri "https://github.com/Kitware/CMake/releases/download/v3.27.8/cmake-3.27.8-windows-x86_64.msi" -OutFile cmake.msi ; \
    msiexec /i cmake.msi /quiet /norestart ADDLOCAL=ALL ; \
    Remove-Item -Force cmake.msi

RUN powershell -Command \
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe" -OutFile python.exe ; \
    .\python.exe /quiet InstallAllUsers=1 PrependPath=1 ; \
    Remove-Item -Force python.exe

# Copy Ryzanstein source code
WORKDIR /build
COPY RYZEN-LLM /build/RYZEN-LLM
COPY mcp /build/mcp

# Build C++ core engine
WORKDIR /build/RYZEN-LLM
RUN powershell -Command \
    mkdir -p build ; \
    cd build ; \
    cmake .. `
      -DCMAKE_BUILD_TYPE=Release `
      -DENABLE_AVX512=ON `
      -DENABLE_PYBIND11=ON `
      -DENABLE_OPENMP=ON ; \
    cmake --build . --config Release -j 8 --verbose

# Install Python dependencies
RUN pip install --no-cache-dir -q \
    torch==2.1.0 \
    safetensors==0.4.0 \
    numpy==1.24.3 \
    pydantic==2.4.2 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6

# =============================================================================
# STAGE 2: Python + MCP Server Build
# =============================================================================
FROM golang:1.22-windowsservercore AS go-builder

WORKDIR /build
COPY mcp /build/mcp

# Build MCP gRPC server
WORKDIR /build/mcp
RUN go mod download && \
    go build -v -o /build/mcp_server.exe ./cmd/server

# =============================================================================
# STAGE 3: Runtime Environment
# =============================================================================
FROM mcr.microsoft.com/windows/servercore:ltsc2022 AS runtime

# Install Python runtime
RUN powershell -Command \
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe" -OutFile python.exe ; \
    .\python.exe /quiet InstallAllUsers=1 PrependPath=1 ; \
    Remove-Item -Force python.exe

# Install runtime dependencies
RUN pip install --no-cache-dir -q \
    torch==2.1.0 \
    safetensors==0.4.0 \
    numpy==1.24.3 \
    pydantic==2.4.2 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1 \
    prometheus-client==0.18.0 \
    opentelemetry-api==1.21.0 \
    opentelemetry-sdk==1.21.0 \
    opentelemetry-exporter-jaeger==1.21.0 \
    opentelemetry-instrumentation-fastapi==0.42b0 \
    opentelemetry-instrumentation-httpx==0.42b0

# Create app directory structure
WORKDIR /app
RUN mkdir -p /app/models /app/logs /app/cache /app/data

# Copy Python API from builder
COPY --from=cpp-builder /build/RYZEN-LLM /app/RYZEN-LLM
COPY --from=cpp-builder /build/RYZEN-LLM/build/python /app/ryzen_llm_bindings

# Copy MCP server
COPY --from=go-builder /build/mcp_server.exe /app/mcp_server.exe

# Copy model weights (optional, can be mounted via volume)
# COPY RYZEN-LLM/models /app/models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:/app/RYZEN-LLM/src:/app/ryzen_llm_bindings
ENV MODEL_PATH=/app/models/bitnet-1.58b/model.safetensors
ENV LOG_DIR=/app/logs
ENV CACHE_DIR=/app/cache
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MCP_SERVER_HOST=0.0.0.0
ENV MCP_SERVER_PORT=8001
ENV JAEGER_AGENT_HOST=jaeger
ENV JAEGER_AGENT_PORT=6831
ENV PROMETHEUS_PUSHGATEWAY=prometheus:9091

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose ports
EXPOSE 8000 8001 8002 8003

# Default command: Start both API and MCP server
CMD ["powershell", "-Command", \
     "Start-Process python -ArgumentList '-m uvicorn RYZEN-LLM.src.api.server:app --host 0.0.0.0 --port 8000 --loop uvloop' -NoNewWindow ; \
      Start-Process /app/mcp_server.exe -ArgumentList '--host 0.0.0.0 --port 8001' -NoNewWindow ; \
      Get-Job | Wait-Job"]

# =============================================================================
# ALTERNATIVE: Linux Multi-stage Build (for deployment on Linux)
# =============================================================================
# To use Linux instead of Windows, build from Linux images:
# FROM ubuntu:22.04 AS cpp-builder-linux
# RUN apt-get update && apt-get install -y \
#     build-essential cmake git python3-dev python3-pip \
#     libopenblas-dev libomp-dev
# ... rest of build steps using Linux tools
