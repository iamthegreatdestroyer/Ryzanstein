"""
Innovation #5: Multi-Node Negative-Space Consensus — Benchmark
==============================================================

Measures:
  1. Bandwidth compression vs standard distributed inference
     (negative-space 32B wire format vs full logit vectors)
  2. Consensus strength vs number of nodes (2, 4, 8)
  3. Latency of consensus round (fan-out + aggregation)
  4. Agreed-absent region size vs seq_len (how many primitives are safely skipped)

Run:
    python tests/bench_negative_space.py

Expected results:
  - Bandwidth ratio: >1,000,000x vs full logit sharing at large seq/vocab
  - Consensus latency: < 1ms per round (pure Python, no I/O)
  - Agreed-absent grows with node agreement (synthetic nodes show ~60% absent)
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consensus.negative_space import (
    ConsensusNode,
    NegativeSpaceConsensusEngine,
    NegativeSpaceDescriptor,
    NegativeSpaceExtractor,
)
from src.models.glyph_mamba import GlyphMambaModel
from src.consensus.negative_space import GlyphMambaWithNegativeSpace

# ─── Config ─────────────────────────────────────────────────────────────────

VOCAB_SIZES = [1_000, 8_000, 32_000]
SEQ_LENGTHS = [64, 128, 256, 512, 1024]
NODE_COUNTS = [2, 4, 8]
N_ROUNDS    = 20
N_WARMUP    = 5


# ─── 1. Bandwidth compression table ─────────────────────────────────────────

def bench_bandwidth() -> None:
    print("\n" + "=" * 70)
    print("  Bandwidth: Negative-Space (32B) vs Full Logit Sharing")
    print("=" * 70)
    print(f"  {'Nodes':>5} {'SeqLen':>7} {'VocabSize':>10} "
          f"{'NegSpace B':>12} {'Logits MB':>10} {'Ratio':>12}")
    print("-" * 70)

    for n_nodes in [2, 4, 8]:
        for seq_len in [64, 512]:
            for vocab in [8_000, 32_000]:
                neg_bytes    = n_nodes * 32              # always 32 bytes per node
                logit_bytes  = n_nodes * seq_len * vocab * 4  # float32
                ratio        = logit_bytes / neg_bytes
                print(
                    f"  {n_nodes:>5} {seq_len:>7} {vocab:>10,} "
                    f"{neg_bytes:>12,} {logit_bytes/1e6:>10.1f} {ratio:>12,.0f}x"
                )
    print()


# ─── 2. Consensus round latency ─────────────────────────────────────────────

async def bench_latency() -> None:
    print("=" * 70)
    print("  Consensus Round Latency (N nodes, synthetic descriptors)")
    print("=" * 70)
    print(f"  {'Nodes':>5} {'Median ms':>10} {'P90 ms':>10}")
    print("-" * 70)

    token_ids = torch.zeros(1, 64, dtype=torch.long)

    for n_nodes in NODE_COUNTS:
        engine = NegativeSpaceConsensusEngine(
            [ConsensusNode(node_id=f"n{i}") for i in range(n_nodes)]
        )

        # Warmup
        for _ in range(N_WARMUP):
            await engine.run_consensus(token_ids)

        latencies: List[float] = []
        for _ in range(N_ROUNDS):
            t0 = time.perf_counter()
            await engine.run_consensus(token_ids)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        latencies.sort()
        median = latencies[len(latencies) // 2]
        p90    = latencies[int(len(latencies) * 0.9)]
        print(f"  {n_nodes:>5} {median:>10.3f} {p90:>10.3f}")
    print()


# ─── 3. Consensus strength vs node count ────────────────────────────────────

async def bench_consensus_strength() -> None:
    print("=" * 70)
    print("  Consensus Strength vs Node Count (synthetic, absent_fraction=0.6)")
    print("  Strength = mean pairwise Jaccard of absence descriptors")
    print("=" * 70)
    print(f"  {'Nodes':>5} {'Strength':>10} {'Agreed abs':>12} {'Disputed':>10}")
    print("-" * 70)

    ext = NegativeSpaceExtractor()
    token_ids = torch.zeros(1, 8, dtype=torch.long)

    for n_nodes in [2, 3, 4, 6, 8, 12]:
        nodes = [ConsensusNode(node_id=f"node-{i}") for i in range(n_nodes)]
        engine = NegativeSpaceConsensusEngine(nodes)

        # Collect descriptors once (seed-based, deterministic)
        descriptors = [
            ext.extract_random_for_test(seed=i, absent_fraction=0.6)
            for i in range(n_nodes)
        ]
        result = engine.aggregate_descriptors(
            descriptors, seq_len=64, vocab_size=1_000
        )
        print(
            f"  {n_nodes:>5} {result.strength:>10.4f} "
            f"{len(result.agreed_absent):>12} {len(result.disputed):>10}"
        )
    print()


# ─── 4. Absent region growth with glyph diversity ───────────────────────────

def bench_absent_vs_seq_len() -> None:
    print("=" * 70)
    print("  Absent Primitive Count vs Sequence Length")
    print("  (GlyphMamba nodes, coord_threshold=0.30)")
    print("=" * 70)
    print(f"  {'SeqLen':>7} {'Node A absent':>14} {'Node B absent':>14} {'Agreed abs':>12}")
    print("-" * 70)

    torch.manual_seed(42)
    base_a = GlyphMambaModel(vocab_size=256, d_model=32, d_glyph=16, n_layers=1, use_sigmalang=False)
    base_b = GlyphMambaModel(vocab_size=256, d_model=32, d_glyph=16, n_layers=1, use_sigmalang=False)
    model_a = GlyphMambaWithNegativeSpace(base_a)
    model_b = GlyphMambaWithNegativeSpace(base_b)
    ext = NegativeSpaceExtractor(coord_threshold=0.30)

    for seq_len in SEQ_LENGTHS:
        token_ids = torch.randint(0, 256, (1, seq_len))
        with torch.no_grad():
            model_a(token_ids)
            model_b(token_ids)
        desc_a = ext.extract_from_glyph_coords(token_ids, model_a.last_glyph_coords)
        desc_b = ext.extract_from_glyph_coords(token_ids, model_b.last_glyph_coords)
        agreed = desc_a.merge(desc_b)
        print(
            f"  {seq_len:>7} {desc_a.absent_count:>14} {desc_b.absent_count:>14} "
            f"{agreed.absent_count:>12}"
        )
    print()


# ─── 5. Sanity check: wire format integrity ──────────────────────────────────

def bench_wire_format() -> None:
    print("=" * 70)
    print("  Wire Format Sanity: to_bytes() -> from_bytes() round-trip")
    print("=" * 70)
    ext = NegativeSpaceExtractor()
    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
        d = ext.extract_random_for_test(seed=0, absent_fraction=fraction)
        raw = d.to_bytes()
        d2  = NegativeSpaceDescriptor.from_bytes(raw)
        ok  = d == d2
        print(
            f"  absent_fraction={fraction:.2f}  "
            f"absent={d.absent_count:>3}/256  "
            f"wire={len(raw):>2}B  "
            f"round_trip={'[OK]' if ok else '[FAIL]'}"
        )
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

async def main_async() -> None:
    bench_bandwidth()
    await bench_latency()
    await bench_consensus_strength()
    bench_absent_vs_seq_len()
    bench_wire_format()
    print("[bench_negative_space] DONE\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
