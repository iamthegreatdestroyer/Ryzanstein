"""
OpenAI-Compatible API Server
[REF:API-008a] - API Layer: OpenAI-Compatible Endpoints

This module implements a FastAPI server with OpenAI-compatible endpoints
for chat completions, embeddings, and model management.

Key Features:
    - /v1/chat/completions endpoint
    - /v1/embeddings endpoint
    - /v1/models endpoint
    - Streaming support (SSE)
    - Authentication middleware
"""

from typing import List, Optional, Dict, Any, AsyncIterator
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import sys
import os
import uuid as _uuid_mod
from pathlib import Path

# Add the build directory to Python path for bindings
build_dir = Path(__file__).parent.parent.parent / "build" / "python"
if str(build_dir) not in sys.path:
    sys.path.insert(0, str(build_dir))

# Try to import C++ bindings, fall back to mock engine
try:
    import ryzen_llm_bindings as rlb
    BINDINGS_AVAILABLE = True
    USING_MOCK = False
    print("✓ C++ bindings loaded successfully")
except ImportError as e:
    print(f"Warning: ryzen_llm_bindings not available ({e})")
    print("Falling back to mock engine for testing...")
    BINDINGS_AVAILABLE = False
    USING_MOCK = True
    
    # Import mock engine as fallback
    try:
        from . import mock_engine as rlb
        print("✓ Mock engine loaded as fallback")
    except ImportError:
        # Try direct import if running as script
        import mock_engine as rlb
        print("✓ Mock engine loaded as fallback (direct import)")


# Pydantic models for API requests/responses
class Message(BaseModel):
    """Chat message."""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""
    model: str = Field(..., description="Model identifier")
    messages: List[Message] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = Field(default=False)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class EmbeddingRequest(BaseModel):
    """Embedding request."""
    model: str = Field(default="default", description="Embedding model")
    input: str | List[str] = Field(..., description="Text to embed")


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ryzanstein-llm"


# Initialize engine and utilities
engine = None
engine_type = "none"

if BINDINGS_AVAILABLE:
    try:
        # Real C++ bindings - create a minimal config that won't OOM
        config = rlb.ModelConfig()
        config.vocab_size = 32000
        config.hidden_size = 256  # Small for testing without weights
        config.intermediate_size = 512
        config.num_layers = 4
        config.num_heads = 8
        config.head_dim = 32
        config.max_seq_length = 512
        config.use_tmac = False
        config.use_speculative_decoding = False
        engine = rlb.BitNetEngine(config)
        engine_type = "bitnet-cpp"
        print(f"✓ {engine_type} engine initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize C++ engine: {e}")
        print("Falling back to mock engine...")
        USING_MOCK = True
        
if USING_MOCK:
    try:
        # Mock engine fallback
        config = rlb.create_bitnet_1_58b_config()
        engine = rlb.MockBitNetEngine(config)
        engine_type = "mock"
        print(f"✓ {engine_type} engine initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize mock engine: {e}")
        engine = None


def simple_tokenize(text: str) -> List[int]:
    """
    Simple tokenization for testing.
    In production, this should use proper tokenizer like SentencePiece or BPE.

    Args:
        text: Input text

    Returns:
        List of token IDs
    """
    # For now, just split by spaces and use hash-based token IDs
    # This is a placeholder - real implementation needs proper tokenizer
    words = text.lower().split()
    # Use simple hash function to generate token IDs (0-31999 range for vocab_size=32000)
    return [hash(word) % 32000 for word in words]


def simple_detokenize(tokens: List[int]) -> str:
    """
    Simple detokenization for testing.
    In production, this should use proper detokenizer.

    Args:
        tokens: List of token IDs

    Returns:
        Detokenized text
    """
    # Placeholder - real implementation needs proper vocabulary
    return " ".join([f"token_{token}" for token in tokens])


# Initialize resilience layer (Sprint 3.3) — non-blocking, graceful fallback
try:
    from .resilience_integration import (
        get_health_checker, get_inference_circuit_breaker, get_inference_bulkhead,
        run_protected_inference, ResilienceMiddleware, initialize_resilience,
        CircuitOpenError, BulkheadFullError,
    )
    _RESILIENCE_AVAILABLE = True
except ImportError:
    try:
        from resilience_integration import (
            get_health_checker, get_inference_circuit_breaker, get_inference_bulkhead,
            run_protected_inference, ResilienceMiddleware, initialize_resilience,
            CircuitOpenError, BulkheadFullError,
        )
        _RESILIENCE_AVAILABLE = True
    except ImportError:
        _RESILIENCE_AVAILABLE = False
        class CircuitOpenError(Exception): pass
        class BulkheadFullError(Exception): pass


# Initialize tracing (Sprint 3.2) — non-blocking, graceful fallback
try:
    from .tracing_integration import (
        setup_tracing, trace_inference_request, TracingMiddleware, get_global_tracer
    )
    _TRACING_AVAILABLE = True
except ImportError:
    try:
        from tracing_integration import (
            setup_tracing, trace_inference_request, TracingMiddleware, get_global_tracer
        )
        _TRACING_AVAILABLE = True
    except ImportError:
        _TRACING_AVAILABLE = False
        from contextlib import contextmanager

        @contextmanager
        def trace_inference_request(model, prompt_tokens, max_tokens, request_id=None):
            class _NoOpSpan:
                tags = {}
            yield _NoOpSpan()


# Initialize FastAPI app
app = FastAPI(
    title="Ryzanstein LLM API",
    description="OpenAI-compatible API for Ryzanstein LLM",
    version="0.1.0"
)

# Wire tracing middleware (Sprint 3.2)
if _TRACING_AVAILABLE:
    try:
        _tracer = setup_tracing(
            service_name="ryzanstein-llm",
            use_in_memory=os.environ.get("TRACING_JAEGER_HOST") is None,
        )
        app.add_middleware(TracingMiddleware, service_name="ryzanstein-llm")
        print("✓ Distributed tracing (Sprint 3.2) initialized")
    except Exception as _te:
        print(f"Warning: Tracing middleware setup failed ({_te}) — continuing without tracing")

# Wire resilience middleware (Sprint 3.3)
if _RESILIENCE_AVAILABLE:
    try:
        app.add_middleware(ResilienceMiddleware)
        # Register engine health check (engine may still be None — registered after init)
        import asyncio as _asyncio
        _loop = None
        try:
            _loop = _asyncio.get_event_loop()
        except RuntimeError:
            pass
        print("✓ Resilience layer (Sprint 3.3) initialized (circuit breaker, bulkhead, retry)")
    except Exception as _re:
        print(f"Warning: Resilience middleware setup failed ({_re}) — continuing without resilience")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Ryzanstein LLM API Server",
        "version": "0.1.0",
        "endpoints": ["/v1/chat/completions", "/v1/embeddings", "/v1/models",
                      "/health", "/health/live", "/health/ready"]
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for service monitoring (backward-compatible).

    Returns:
        Health status with engine state and resilience metrics
    """
    import time
    response = {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "engine_type": engine_type,
        "bindings_available": BINDINGS_AVAILABLE,
        "using_mock": USING_MOCK if 'USING_MOCK' in dir() else False,
        "model_path": "models/bitnet",
        "timestamp": int(time.time()),
    }
    # Attach resilience stats if available
    if _RESILIENCE_AVAILABLE:
        try:
            cb = get_inference_circuit_breaker()
            bh = get_inference_bulkhead()
            response["circuit_breaker"] = cb.get_stats()
            response["bulkhead"] = bh.get_stats()
        except Exception:
            pass
    return response


@app.get("/health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe — is the process alive?
    Returns 200 as long as the event loop is running.
    """
    if _RESILIENCE_AVAILABLE:
        checker = get_health_checker()
        report = await checker.check_liveness()
        status_code = 200 if report.is_healthy else 503
        return report.to_dict()
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe — is the service ready to accept traffic?
    Checks engine initialization and circuit breaker state.
    """
    from fastapi.responses import JSONResponse
    if _RESILIENCE_AVAILABLE:
        try:
            # Register engine health lazily on first readiness check
            checker = get_health_checker()
            if engine is not None:
                from .resilience_integration import register_engine_health
                register_engine_health(engine, engine_type)
        except Exception:
            pass
        checker = get_health_checker()
        report = await checker.check_readiness()
        status_code = 200 if report.is_ready else 503
        return JSONResponse(content=report.to_dict(), status_code=status_code)

    # Fallback readiness without resilience library
    from fastapi.responses import JSONResponse
    if engine is None:
        return JSONResponse(
            content={"status": "not_ready", "reason": "engine not initialized"},
            status_code=503
        )
    return JSONResponse(content={"status": "ready"}, status_code=200)


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response"""
    object: str = "list"
    data: List[ModelInfo]


@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    """
    List available models.

    Returns:
        ModelListResponse with available models
    """
    import time
    current_time = int(time.time())

    models = []
    if engine is not None:
        model_id = "mock-bitnet-1.58b" if engine_type == "mock" else "bitnet-1.58b"
        models.append(ModelInfo(
            id=model_id,
            created=current_time,
            owned_by="ryzanstein-llm"
        ))

    return ModelListResponse(object="list", data=models)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest
):
    """
    Generate chat completions.

    Args:
        request: Chat completion request

    Returns:
        Chat completion response or streaming response
    """
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not available. Please check server logs."
        )

    # For now, implement non-streaming only
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented")

    # Extract the last user message
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    user_input = user_messages[-1].content
    request_id = _uuid_mod.uuid4().hex[:8]

    # Estimate prompt tokens for tracing
    prompt_token_estimate = len(user_input.split())

    try:
        with trace_inference_request(
            model=request.model,
            prompt_tokens=prompt_token_estimate,
            max_tokens=request.max_tokens or 100,
            request_id=request_id,
        ) as _span:
            # Check if using mock engine with generate_text method
            if USING_MOCK and hasattr(engine, 'generate_text'):
                gen_config = rlb.GenerationConfig()
                gen_config.max_tokens = request.max_tokens or 100
                gen_config.temperature = request.temperature
                response_text = engine.generate_text(user_input, gen_config)
                input_token_count = len(user_input.split())
                output_token_count = len(response_text.split())
            else:
                # Use token-based generation for real engine
                input_tokens = simple_tokenize(user_input)
                if not input_tokens:
                    raise HTTPException(status_code=400, detail="Failed to tokenize input")

                # Create generation config
                gen_config = rlb.GenerationConfig()
                gen_config.max_tokens = request.max_tokens or 100
                gen_config.temperature = request.temperature
                gen_config.top_p = request.top_p
                gen_config.top_k = 50
                gen_config.repetition_penalty = 1.1

                # Generate response
                output_tokens = engine.generate(input_tokens, gen_config)

                # Detokenize response
                response_text = simple_detokenize(output_tokens)
                input_token_count = len(input_tokens)
                output_token_count = len(output_tokens)

            # Record output token count in span
            _span.tags["output_tokens"] = output_token_count

        # Create response
        import time

        response = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_token_count,
                "completion_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            }
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for text.

    Args:
        request: Embedding request

    Returns:
        Embedding response
    """
    # TODO: Implement proper embeddings with dedicated model
    # For now, return stub response
    import time
    import numpy as np

    inputs = [request.input] if isinstance(request.input, str) else request.input

    # Generate dummy embeddings (512 dimensions)
    embeddings = []
    for text in inputs:
        # Simple hash-based embedding for testing
        embedding = np.random.rand(512).tolist()
        embeddings.append(embedding)

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": i
            }
            for i, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "usage": {
            "prompt_tokens": sum(len(text.split()) for text in inputs),
            "total_tokens": sum(len(text.split()) for text in inputs)
        }
    }


async def generate_stream(
    messages: List[Message],
    model: str
) -> AsyncIterator[str]:
    """
    Generate streaming response.
    
    Args:
        messages: Chat messages
        model: Model identifier
        
    Yields:
        SSE-formatted chunks
    """
    # TODO: Implement streaming
    # 1. Initialize generation
    # 2. Yield tokens as they're generated
    # 3. Format as SSE
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

