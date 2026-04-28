#!/usr/bin/env python3
"""
Performance regression gate for Ryzanstein CI.

Runs a synthetic benchmark against the FastAPI inference server and checks
that throughput, latency, and error-rate meet the specified thresholds.

Usage (CI):
    python scripts/check_regression.py --min-rps 1500 --output reports/perf_<sha>.json

Usage (local):
    python scripts/check_regression.py --host http://localhost:8000 --requests 200 --concurrency 8

Exit codes:
    0  — all gates passed
    1  — one or more gates failed (regression detected)
    2  — server unreachable (soft-fail in CI unless --strict)
"""

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

SAMPLE_PROMPT = "What is the capital of France? Answer in one word."
SAMPLE_MODEL = "ryzanstein-7b"


def _send_chat(host: str, timeout: int) -> tuple[float, bool]:
    """POST /v1/chat/completions, return (latency_ms, success)."""
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{host}/v1/chat/completions",
            json={
                "model": SAMPLE_MODEL,
                "messages": [{"role": "user", "content": SAMPLE_PROMPT}],
                "max_tokens": 32,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        ok = r.status_code == 200
    except Exception:
        ok = False
    latency_ms = (time.perf_counter() - t0) * 1000
    return latency_ms, ok


def run_benchmark(
    host: str,
    n_requests: int,
    concurrency: int,
    per_request_timeout: int,
) -> dict:
    """Run benchmark, return raw results dict."""
    latencies: list[float] = []
    errors = 0

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_send_chat, host, per_request_timeout) for _ in range(n_requests)]
        for f in as_completed(futures):
            lat, ok = f.result()
            latencies.append(lat)
            if not ok:
                errors += 1
    wall_elapsed = time.perf_counter() - wall_start

    latencies.sort()
    n = len(latencies)

    def pct(p: float) -> float:
        return latencies[int((n - 1) * p)] if n else 0.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "n_requests": n_requests,
        "concurrency": concurrency,
        "wall_time_s": round(wall_elapsed, 3),
        "rps": round(n_requests / wall_elapsed, 2) if wall_elapsed > 0 else 0,
        "errors": errors,
        "error_rate": round(errors / n_requests, 4) if n_requests else 0,
        "latency_ms": {
            "min": round(latencies[0], 2) if n else 0,
            "p50": round(pct(0.50), 2),
            "p95": round(pct(0.95), 2),
            "p99": round(pct(0.99), 2),
            "max": round(latencies[-1], 2) if n else 0,
            "mean": round(statistics.mean(latencies), 2) if n else 0,
        },
    }


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def evaluate_gates(results: dict, args: argparse.Namespace) -> list[str]:
    """Return list of failure messages (empty = all passed)."""
    failures = []

    if args.min_rps and results["rps"] < args.min_rps:
        failures.append(
            f"RPS gate FAILED: {results['rps']:.1f} < {args.min_rps} (min)"
        )

    if args.max_p95_ms and results["latency_ms"]["p95"] > args.max_p95_ms:
        failures.append(
            f"P95 latency gate FAILED: {results['latency_ms']['p95']:.1f}ms > {args.max_p95_ms}ms (max)"
        )

    if args.max_p99_ms and results["latency_ms"]["p99"] > args.max_p99_ms:
        failures.append(
            f"P99 latency gate FAILED: {results['latency_ms']['p99']:.1f}ms > {args.max_p99_ms}ms (max)"
        )

    if args.max_error_rate is not None and results["error_rate"] > args.max_error_rate:
        failures.append(
            f"Error-rate gate FAILED: {results['error_rate']:.2%} > {args.max_error_rate:.2%} (max)"
        )

    return failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Ryzanstein performance regression gate")
    parser.add_argument("--host", default=os.getenv("RYZANSTEIN_API_URL", "http://localhost:8000"))
    parser.add_argument("--requests", type=int, default=100, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent workers")
    parser.add_argument("--timeout", type=int, default=30, help="Per-request timeout (s)")
    parser.add_argument("--output", help="Write JSON report to this path")
    # Gate thresholds
    parser.add_argument("--min-rps", type=float, help="Minimum acceptable RPS")
    parser.add_argument("--max-p95-ms", type=float, dest="max_p95_ms", help="Max P95 latency (ms)")
    parser.add_argument("--max-p99-ms", type=float, dest="max_p99_ms", help="Max P99 latency (ms)")
    parser.add_argument("--max-error-rate", type=float, dest="max_error_rate",
                        default=0.05, help="Max error rate (0.0–1.0, default 0.05)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 (not 2) when server is unreachable")
    args = parser.parse_args()

    # Connectivity probe
    try:
        r = requests.get(f"{args.host}/health", timeout=5)
        if r.status_code != 200:
            print(f"WARNING: /health returned {r.status_code}", file=sys.stderr)
    except Exception as e:
        print(f"Server unreachable at {args.host}: {e}", file=sys.stderr)
        return 1 if args.strict else 2

    print(f"Running {args.requests} requests @ concurrency={args.concurrency} against {args.host}")
    results = run_benchmark(args.host, args.requests, args.concurrency, args.timeout)

    # Print summary
    lat = results["latency_ms"]
    print(f"\n{'='*55}")
    print(f"  Throughput : {results['rps']:.1f} req/s  ({results['n_requests']} reqs in {results['wall_time_s']:.1f}s)")
    print(f"  Errors     : {results['errors']} ({results['error_rate']:.1%})")
    print(f"  Latency P50: {lat['p50']:.1f}ms")
    print(f"  Latency P95: {lat['p95']:.1f}ms")
    print(f"  Latency P99: {lat['p99']:.1f}ms")
    print(f"  Latency Max: {lat['max']:.1f}ms")
    print(f"{'='*55}\n")

    # Evaluate gates
    failures = evaluate_gates(results, args)
    results["gates_passed"] = len(failures) == 0
    results["gate_failures"] = failures

    # Write report
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Report written to {out_path}")

    if failures:
        print("REGRESSION DETECTED:")
        for msg in failures:
            print(f"  ✗ {msg}")
        return 1

    print("All performance gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
