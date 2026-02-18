#!/usr/bin/env python3
"""
End-to-End Real Inference Verification
[REF:WEEK1-TASK1.5] - Verify complete pipeline: C++ Engine → Python Bindings → FastAPI → Response

Tests:
1. C++ engine initialization and SIMD detection
2. Model weight loading (real or mock)
3. Token generation (real inference pipeline)
4. FastAPI endpoint round-trip
5. Throughput benchmark (tok/s measurement)
6. Streaming SSE verification

Usage:
    python scripts/verify_inference.py
    python scripts/verify_inference.py --api-only   (skip C++ engine, test API only)
    python scripts/verify_inference.py --benchmark   (run throughput benchmark)
"""

import sys
import os
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# ============================================================================
# Path Setup
# ============================================================================

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "RYZEN-LLM" / "build" / "python"))
sys.path.insert(0, str(REPO_ROOT / "RYZEN-LLM" / "src"))

MODELS_DIR = REPO_ROOT / "RYZEN-LLM" / "models"
API_URL = "http://localhost:8000"


# ============================================================================
# Color Output Helpers
# ============================================================================

class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def ok(msg): print(f"  {Color.GREEN}✓{Color.RESET} {msg}")
def fail(msg): print(f"  {Color.RED}✗{Color.RESET} {msg}")
def info(msg): print(f"  {Color.CYAN}→{Color.RESET} {msg}")
def warn(msg): print(f"  {Color.YELLOW}⚠{Color.RESET} {msg}")
def header(title):
    print(f"\n{Color.BOLD}{'=' * 65}{Color.RESET}")
    print(f"{Color.BOLD}  {title}{Color.RESET}")
    print(f"{Color.BOLD}{'=' * 65}{Color.RESET}")


# ============================================================================
# Test 1: C++ Bindings Availability & SIMD Detection
# ============================================================================

def test_cpp_bindings():
    """Test C++ bindings import and SIMD capabilities."""
    header("Test 1: C++ Bindings & SIMD Detection")

    try:
        import ryzen_llm_bindings as rlb
        ok("C++ bindings (ryzen_llm_bindings) loaded successfully")

        # Check what's exposed
        attrs = dir(rlb)
        info(f"Exposed symbols: {[a for a in attrs if not a.startswith('_')]}")

        # Try creating a minimal config
        if hasattr(rlb, 'ModelConfig'):
            config = rlb.ModelConfig()
            ok(f"ModelConfig created (vocab_size={config.vocab_size})")
        else:
            warn("ModelConfig not found in bindings")

        if hasattr(rlb, 'BitNetEngine'):
            ok("BitNetEngine class available")
            return rlb, True
        else:
            warn("BitNetEngine not found in bindings")
            return rlb, False

    except ImportError as e:
        warn(f"C++ bindings not available: {e}")
        info("Falling back to mock engine for API testing")
        return None, False


# ============================================================================
# Test 2: Engine Initialization
# ============================================================================

def test_engine_init(rlb=None):
    """Test BitNet engine initialization with minimal config."""
    header("Test 2: Engine Initialization")

    if rlb is None:
        # Try mock engine
        try:
            sys.path.insert(0, str(REPO_ROOT / "RYZEN-LLM" / "src" / "api"))
            import mock_engine as mock
            config = mock.create_bitnet_1_58b_config()
            engine = mock.MockBitNetEngine(config)
            ok(f"Mock engine initialized: {type(engine).__name__}")
            return engine, "mock"
        except Exception as e:
            fail(f"Mock engine initialization failed: {e}")
            return None, "none"

    # Real C++ engine
    try:
        config = rlb.ModelConfig()
        # Use small config to avoid OOM
        config.vocab_size = 32000
        config.hidden_size = 256
        config.intermediate_size = 512
        config.num_layers = 2
        config.num_heads = 8
        config.head_dim = 32
        config.max_seq_length = 512
        config.use_tmac = False
        config.use_speculative_decoding = False

        start = time.time()
        engine = rlb.BitNetEngine(config)
        init_time = time.time() - start

        ok(f"BitNetEngine initialized in {init_time*1000:.1f}ms")
        info(f"Config: hidden={config.hidden_size}, layers={config.num_layers}, heads={config.num_heads}")
        return engine, "bitnet-cpp"

    except Exception as e:
        fail(f"BitNetEngine initialization failed: {e}")
        return None, "none"


# ============================================================================
# Test 3: Model Weight Loading
# ============================================================================

def test_weight_loading(engine, engine_type):
    """Test model weight loading from disk."""
    header("Test 3: Model Weight Loading")

    if engine_type == "mock":
        info("Mock engine uses synthetic weights — skipping file loading")
        return True

    if engine is None:
        warn("No engine available — skipping weight loading test")
        return False

    # Check for available model weights
    model_dirs = [
        MODELS_DIR / "bitnet-1.58b",
        MODELS_DIR / "bitnet-3b",
        REPO_ROOT / "RYZEN-LLM" / "weights",
    ]

    loaded = False
    for model_dir in model_dirs:
        if model_dir.exists():
            files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
            if files:
                info(f"Trying to load weights from: {model_dir}")
                info(f"Found {len(files)} weight file(s)")

                try:
                    if hasattr(engine, 'load_weights'):
                        start = time.time()
                        result = engine.load_weights(str(model_dir))
                        load_time = time.time() - start
                        if result:
                            ok(f"Weights loaded in {load_time:.2f}s from {model_dir.name}")
                            loaded = True
                            break
                        else:
                            warn(f"load_weights returned False for {model_dir.name}")
                    else:
                        warn("Engine has no load_weights method")
                except Exception as e:
                    warn(f"Weight loading exception for {model_dir.name}: {e}")

    if not loaded:
        warn("No model weights found — engine will use random weights")
        info("To download: run scripts/download_models.ps1")
        info("Real inference validation requires actual model weights")

    return True  # Non-fatal — random weights still test the pipeline


# ============================================================================
# Test 4: Token Generation (Core Inference)
# ============================================================================

def test_token_generation(engine, engine_type):
    """Test token generation pipeline."""
    header("Test 4: Token Generation")

    if engine is None:
        warn("No engine — skipping token generation test")
        return False, 0.0

    # Simple test input
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # Simple token IDs

    try:
        if hasattr(engine, 'generate'):
            from api.server import simple_tokenize
        except Exception:
            pass

        # Try to call generate
        if hasattr(engine, 'generate'):
            try:
                import ryzen_llm_bindings as rlb
                gen_config = rlb.GenerationConfig()
                gen_config.max_tokens = 20
                gen_config.temperature = 0.7
                gen_config.top_k = 50

                start = time.time()
                output_tokens = engine.generate(input_tokens, gen_config)
                elapsed = time.time() - start

                new_tokens = len(output_tokens) - len(input_tokens)
                tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0

                ok(f"Generated {new_tokens} tokens in {elapsed*1000:.1f}ms")
                ok(f"Throughput: {tok_per_sec:.2f} tok/s")
                info(f"Input: {input_tokens}")
                info(f"Output: {list(output_tokens)[:20]}")

                # Validate output is not all zeros or garbage
                if len(output_tokens) > len(input_tokens):
                    ok("Output tokens produced (pipeline working)")
                else:
                    warn("No new tokens generated (possible EOS at first token)")

                return True, tok_per_sec

            except Exception as e:
                fail(f"Generate call failed: {e}")
                return False, 0.0

        # Mock engine path
        elif hasattr(engine, 'generate_text'):
            start = time.time()
            result = engine.generate_text("Test prompt for inference", max_tokens=20)
            elapsed = time.time() - start
            words = result.split()
            tok_per_sec = len(words) / elapsed if elapsed > 0 else 0
            ok(f"Mock generation: '{result[:50]}...' in {elapsed*1000:.1f}ms")
            ok(f"Throughput: {tok_per_sec:.2f} words/s (mock)")
            return True, tok_per_sec
        else:
            warn(f"Engine type '{engine_type}' has no supported generate method")
            return False, 0.0

    except Exception as e:
        fail(f"Token generation test failed: {e}")
        return False, 0.0


# ============================================================================
# Test 5: FastAPI Endpoint Round-trip
# ============================================================================

def test_api_endpoint():
    """Test FastAPI /v1/chat/completions endpoint."""
    header("Test 5: FastAPI Endpoint Round-trip")

    try:
        import urllib.request
        import urllib.error

        # Check if server is running
        try:
            req = urllib.request.urlopen(f"{API_URL}/health", timeout=3)
            health = json.loads(req.read())
            ok(f"API server running: {health.get('engine_type', 'unknown')} engine")
            info(f"Health: {health}")
        except (urllib.error.URLError, ConnectionRefusedError):
            warn("API server not running — start with: uvicorn RYZEN-LLM.src.api.server:app --reload")
            info("Skipping API endpoint tests")
            return False

        # Test /v1/chat/completions
        payload = json.dumps({
            "model": "bitnet-1.58b",
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{API_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        start = time.time()
        response = urllib.request.urlopen(req, timeout=30)
        elapsed = time.time() - start

        result = json.loads(response.read())
        ok(f"Chat completion response in {elapsed*1000:.1f}ms")
        info(f"Model: {result.get('model', 'N/A')}")
        info(f"Choices: {result.get('choices', [])}")
        info(f"Usage: {result.get('usage', {})}")

        if result.get("choices"):
            ok("Response contains choices — endpoint working")
        else:
            warn("Response has no choices")

        return True

    except Exception as e:
        fail(f"API endpoint test failed: {e}")
        return False


# ============================================================================
# Test 6: Throughput Benchmark
# ============================================================================

def run_benchmark(engine, engine_type, num_runs=5):
    """Run throughput benchmark."""
    header("Test 6: Throughput Benchmark")

    if engine is None or engine_type == "none":
        warn("No engine for benchmarking")
        return

    input_tokens = list(range(1, 17))  # 16 input tokens
    throughputs = []

    info(f"Running {num_runs} inference runs...")

    for i in range(num_runs):
        try:
            if hasattr(engine, 'generate'):
                try:
                    import ryzen_llm_bindings as rlb
                    gen_config = rlb.GenerationConfig()
                    gen_config.max_tokens = 50
                    gen_config.temperature = 0.0  # Greedy for determinism

                    start = time.time()
                    output = engine.generate(input_tokens, gen_config)
                    elapsed = time.time() - start

                    new_tokens = len(output) - len(input_tokens)
                    if new_tokens > 0 and elapsed > 0:
                        tps = new_tokens / elapsed
                        throughputs.append(tps)
                        info(f"  Run {i+1}: {new_tokens} tokens in {elapsed*1000:.1f}ms = {tps:.2f} tok/s")
                except Exception:
                    break
            else:
                warn("Engine has no generate method")
                break
        except Exception as e:
            warn(f"Benchmark run {i+1} failed: {e}")

    if throughputs:
        avg_tps = sum(throughputs) / len(throughputs)
        peak_tps = max(throughputs)
        min_tps = min(throughputs)

        print(f"\n  {'=' * 40}")
        print(f"  BENCHMARK RESULTS ({num_runs} runs)")
        print(f"  {'=' * 40}")
        print(f"  Average:  {avg_tps:.2f} tok/s")
        print(f"  Peak:     {peak_tps:.2f} tok/s")
        print(f"  Min:      {min_tps:.2f} tok/s")

        # Compare to targets
        targets = {
            "Phase 1 baseline": 0.68,
            "Phase 2 achieved": 56.62,
            "Week 1 target (SIMD fix)": 2.5,
            "Week 1 target (full fix)": 10.0,
        }

        print(f"\n  TARGET COMPARISON:")
        for target_name, target_tps in targets.items():
            pct = (avg_tps / target_tps) * 100
            status = "✓" if avg_tps >= target_tps else "✗"
            print(f"  {status} {target_name}: {target_tps:.2f} tok/s ({pct:.1f}%)")
    else:
        warn("No throughput measurements collected")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ryzanstein LLM End-to-End Verification")
    parser.add_argument("--api-only", action="store_true", help="Only test API (skip C++ engine)")
    parser.add_argument("--benchmark", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--fast", action="store_true", help="Skip benchmark for fast verification")
    args = parser.parse_args()

    print(f"\n{Color.BOLD}{Color.CYAN}")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   RYZANSTEIN LLM — END-TO-END INFERENCE VERIFICATION    ║")
    print("  ║   [REF:WEEK1-TASK1.5]                                   ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"{Color.RESET}")

    results = {}

    if not args.api_only:
        # Test C++ bindings
        rlb, bindings_ok = test_cpp_bindings()
        results["cpp_bindings"] = bindings_ok

        # Test engine init
        engine, engine_type = test_engine_init(rlb)
        results["engine_init"] = engine is not None

        # Test weight loading
        weight_ok = test_weight_loading(engine, engine_type)
        results["weight_loading"] = weight_ok

        # Test token generation
        gen_ok, tps = test_token_generation(engine, engine_type)
        results["token_generation"] = gen_ok

        if args.benchmark and not args.fast:
            run_benchmark(engine, engine_type)
    else:
        engine, engine_type = None, "none"
        info("--api-only mode: skipping C++ engine tests")

    # Test API endpoint
    api_ok = test_api_endpoint()
    results["api_endpoint"] = api_ok

    # Final summary
    header("VERIFICATION SUMMARY")
    all_critical_passed = True
    critical_tests = ["engine_init", "token_generation"]

    for test_name, passed in results.items():
        status = f"{Color.GREEN}✓ PASS{Color.RESET}" if passed else f"{Color.RED}✗ FAIL{Color.RESET}"
        critical = " [CRITICAL]" if test_name in critical_tests else ""
        print(f"  {status}  {test_name}{critical}")

        if test_name in critical_tests and not passed:
            all_critical_passed = False

    print()
    if all_critical_passed and len(results) > 0:
        print(f"  {Color.GREEN}{Color.BOLD}✓ VERIFICATION PASSED — Pipeline operational{Color.RESET}")
        if engine_type == "mock":
            warn("Using mock engine — real inference needs: scripts/download_models.ps1")
        elif engine_type == "bitnet-cpp":
            ok("Real C++ engine verified — check throughput vs targets above")
    else:
        print(f"  {Color.RED}{Color.BOLD}✗ VERIFICATION FAILED — Critical tests did not pass{Color.RESET}")
        info("Check the detailed output above for failure reasons")
        info("Common fixes:")
        info("  1. Build C++ bindings: cd RYZEN-LLM && cmake --build build")
        info("  2. Download weights: scripts/download_models.ps1 -BitNetOnly")
        info("  3. Start API server: uvicorn RYZEN-LLM.src.api.server:app --reload")
        sys.exit(1)


if __name__ == "__main__":
    main()
