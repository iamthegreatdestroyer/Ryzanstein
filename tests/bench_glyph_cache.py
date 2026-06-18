"""
Sprint 1 Benchmark: Glyph-Native vs Token-Exact KV Cache
=========================================================

Measures cache hit rate, effective compression, and simulated time savings
across three request workloads:

  - EXACT:    100% repeated prompts (upper bound for token cache)
  - SEMANTIC: Prompts that share meaning but differ in token spelling
              (only the glyph cache can hit these)
  - MIXED:    Realistic 70/15/15 split of unique/exact/semantic repeats

Run with:
    cd S:\\repos\\Layer-4-Storage\\Ryot
    python -m pytest tests/bench_glyph_cache.py -v -s
  or standalone:
    python tests/bench_glyph_cache.py
"""

import sys
import time
import random
from pathlib import Path
from typing import List, Tuple

# Ensure sigmalang is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.recycler.semantic_kv_cache import SemanticKVCache
from src.recycler.glyph_kv_cache import GlyphKVCache, HybridKVCache, TokenGlyphMapper

CHUNK = 16          # small chunk for bench (hit rate visible in short seqs)
ENTRIES = 1024
N_REQUESTS = 500
SEQ_LEN = 32        # tokens per request
VOCAB_SIZE = 512    # token ID range


# ---------------------------------------------------------------------------
# Synthetic workload generators
# ---------------------------------------------------------------------------

def _rand_seq(length: int = SEQ_LEN) -> List[int]:
    return [random.randint(0, VOCAB_SIZE - 1) for _ in range(length)]


def build_exact_workload(n: int) -> List[List[int]]:
    """Same 10 prompts repeated across all n requests."""
    templates = [_rand_seq() for _ in range(10)]
    return [random.choice(templates) for _ in range(n)]


def build_semantic_workload(n: int) -> List[List[int]]:
    """
    Prompts that are semantically similar but token-different.

    Two tokens that map to the same Σ-glyph (because they wrap to the
    same primitive_id in TokenGlyphMapper) are used interchangeably.
    Tokens 0 and 16 are distinct glyphs; tokens 256 and 384 both wrap
    to primitive_id 0x80 — swapping them changes the token sequence but
    not the glyph sequence.
    """
    mapper = TokenGlyphMapper()
    templates = [_rand_seq() for _ in range(10)]
    result = []
    for _ in range(n):
        base = list(random.choice(templates))
        # Replace up to 4 tokens with semantic equivalents (same glyph)
        for i in random.sample(range(len(base)), min(4, len(base))):
            t = base[i]
            if t >= 128:
                # Any token ≥ 128 + 256 wraps to the same primitive
                base[i] = t + 256 if t + 256 < VOCAB_SIZE * 4 else t
        result.append(base)
    return result


def build_mixed_workload(n: int) -> List[List[int]]:
    """70% unique, 15% exact repeats, 15% semantic repeats."""
    n_unique   = int(n * 0.70)
    n_exact    = int(n * 0.15)
    n_semantic = n - n_unique - n_exact

    unique   = [_rand_seq() for _ in range(n_unique)]
    exact    = build_exact_workload(n_exact)
    semantic = build_semantic_workload(n_semantic)

    combined = unique + exact + semantic
    random.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def _make_kv():
    return [([0.0] * 64, [0.0] * 64)]   # dummy KV — no torch dependency


def run_workload(cache, requests: List[List[int]], label: str) -> dict:
    cache.invalidate()
    hits = 0
    total = 0
    t0 = time.perf_counter()

    for tokens in requests:
        kv, length = cache.lookup(tokens)
        if length > 0:
            hits += 1
        else:
            cache.store(tokens, _make_kv())
        total += 1

    elapsed = time.perf_counter() - t0
    hit_rate = hits / total if total else 0

    return {
        "label": label,
        "requests": total,
        "hits": hits,
        "hit_rate": round(hit_rate * 100, 1),
        "elapsed_ms": round(elapsed * 1000, 2),
        "throughput_rps": round(total / elapsed, 0),
    }


def run_benchmark():
    random.seed(42)

    token_cache  = SemanticKVCache(max_entries=ENTRIES, chunk_size=CHUNK)
    glyph_cache  = GlyphKVCache(max_entries=ENTRIES,   chunk_size=CHUNK)
    hybrid_cache = HybridKVCache(max_entries=ENTRIES,  chunk_size=CHUNK)

    workloads = [
        ("EXACT",    build_exact_workload(N_REQUESTS)),
        ("SEMANTIC", build_semantic_workload(N_REQUESTS)),
        ("MIXED",    build_mixed_workload(N_REQUESTS)),
    ]

    results = []
    for wl_name, requests in workloads:
        for cache, cache_name in [
            (token_cache,  "TokenCache "),
            (glyph_cache,  "GlyphCache "),
            (hybrid_cache, "HybridCache"),
        ]:
            r = run_workload(cache, requests, f"{cache_name} / {wl_name}")
            results.append(r)

    # Print table
    print()
    print("=" * 72)
    print(f"  Sprint 1 Benchmark — {N_REQUESTS} requests, seq_len={SEQ_LEN}, chunk={CHUNK}")
    print("=" * 72)
    print(f"  {'Cache / Workload':<32} {'Hits':>6} {'Hit%':>7} {'ms':>8} {'rps':>8}")
    print("  " + "-" * 68)
    prev_wl = None
    for r in results:
        wl = r["label"].split("/")[1].strip()
        if prev_wl and wl != prev_wl:
            print()
        prev_wl = wl
        print(f"  {r['label']:<32} {r['hits']:>6} {r['hit_rate']:>6.1f}% "
              f"{r['elapsed_ms']:>7.1f}ms {r['throughput_rps']:>7.0f}")
    print("=" * 72)

    # Assert glyph cache ≥ token cache on semantic workload
    sem_token = next(r for r in results if "TokenCache" in r["label"] and "SEMANTIC" in r["label"])
    sem_glyph = next(r for r in results if "GlyphCache" in r["label"] and "SEMANTIC" in r["label"])
    print(f"\n  Semantic dedup: token={sem_token['hit_rate']}%  glyph={sem_glyph['hit_rate']}%")
    assert sem_glyph["hit_rate"] >= sem_token["hit_rate"], (
        "GlyphCache should match or beat TokenCache on semantic workload"
    )
    print("  [OK] Glyph cache >= token cache on semantic workload\n")

    # Compression stats from glyph cache
    g_stats = glyph_cache.stats()
    if g_stats["glyph_bytes_saved"] > 0:
        print(f"  Hash input compression: {g_stats['glyph_bytes_saved']:,} bytes saved "
              f"vs raw int32 encoding")
    return results


# pytest entry point
def test_glyph_cache_benchmark():
    results = run_benchmark()
    sem_glyph = next(r for r in results if "GlyphCache" in r["label"] and "SEMANTIC" in r["label"])
    assert sem_glyph["hit_rate"] >= 0  # always true; real assertion inside run_benchmark


if __name__ == "__main__":
    run_benchmark()
