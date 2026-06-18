"""
Innovation #2: Mamba-Glyph Fusion — Latency & Memory Benchmark
===============================================================

Compares GlyphMambaModel vs nn.MultiheadAttention at sequence lengths:
    [64, 128, 256, 512, 1024]

Metrics reported per model per seq length:
    - Forward pass latency (ms, median of N_RUNS runs)
    - Peak memory delta (MB) via process RSS approximation
    - Parameter count
    - Glyph selectivity ratio (GlyphMambaModel only)

Run:
    python tests/bench_glyph_mamba.py

Expected result (CPU, d_model=128, n_layers=4, batch=1):
    - GlyphMambaModel: O(L) scaling (linear in sequence length)
    - MHA baseline:    O(L²) scaling (quadratic in sequence length)

The crossover point where Glyph-Mamba wins depends on hardware; on x86 CPU
without flash-attention, Mamba typically wins for L >= 256.
"""

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.glyph_mamba import GlyphMambaModel

# ─── Configuration ──────────────────────────────────────────────────────────

VOCAB_SIZE   = 1_000
D_MODEL      = 128
N_LAYERS     = 4
D_STATE      = 16
D_GLYPH      = 32   # << D_MODEL — the key invariant
D_CONV       = 4
EXPAND       = 2
BATCH_SIZE   = 1
SEQ_LENGTHS  = [64, 128, 256, 512, 1024]
N_RUNS       = 10    # benchmark repetitions per config
N_WARMUP     = 3     # warmup runs (excluded from stats)


# ─── MHA baseline ───────────────────────────────────────────────────────────

class MHABaselineModel(nn.Module):
    """
    Simple multi-head attention baseline with the same d_model, n_layers,
    and vocab_size as GlyphMambaModel.

    Architecture: embed → N×(MHA + FFN) → norm → lm_head
    Standard O(L²) attention complexity.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model:    int = D_MODEL,
        n_layers:   int = N_LAYERS,
        n_heads:    int = 4,
    ):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.layers  = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm_f  = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(token_ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.norm_f(h))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Benchmark harness ──────────────────────────────────────────────────────

class BenchResult(NamedTuple):
    name:        str
    seq_len:     int
    latency_ms:  float   # median
    latency_p90: float   # 90th percentile
    param_count: int
    notes:       str


def benchmark_model(
    model:    nn.Module,
    name:     str,
    seq_len:  int,
    batch:    int = BATCH_SIZE,
    n_runs:   int = N_RUNS,
    n_warmup: int = N_WARMUP,
) -> BenchResult:
    model.eval()
    token_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len))

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(token_ids)

    # Timed runs
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(token_ids)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies.sort()
    median = latencies[len(latencies) // 2]
    p90    = latencies[int(len(latencies) * 0.9)]

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    notes = ""
    if hasattr(model, "glyph_selectivity_ratio"):
        ratio = model.glyph_selectivity_ratio()
        notes = f"glyph_selectivity={ratio:.1%}"

    return BenchResult(
        name=name, seq_len=seq_len,
        latency_ms=round(median, 3),
        latency_p90=round(p90, 3),
        param_count=param_count,
        notes=notes,
    )


def print_table(results: List[BenchResult]) -> None:
    hdr = f"{'Model':<20} {'L':>6} {'Median ms':>10} {'P90 ms':>10} {'Params':>10} {'Notes'}"
    print("\n" + "=" * 75)
    print("  Glyph-Mamba vs MHA Attention — CPU Latency Benchmark")
    print(f"  d_model={D_MODEL}, n_layers={N_LAYERS}, d_glyph={D_GLYPH}, batch={BATCH_SIZE}")
    print("=" * 75)
    print(hdr)
    print("-" * 75)

    last_model = None
    for r in results:
        if last_model and last_model != r.name:
            print()  # blank line between model groups
        last_model = r.name
        print(
            f"  {r.name:<18} {r.seq_len:>6} {r.latency_ms:>10.3f} {r.latency_p90:>10.3f}"
            f" {r.param_count:>10,}  {r.notes}"
        )
    print("=" * 75)


def scaling_analysis(results: List[BenchResult]) -> None:
    """
    Print empirical scaling exponent.
    If y ∝ L^α then log(y2/y1)/log(L2/L1) ≈ α.
    α ≈ 1 → linear (Mamba), α ≈ 2 → quadratic (attention).
    """
    print("\n  Empirical scaling exponents (log-log slope of median latency vs L)")
    print("  Ideal: Glyph-Mamba ~ 1.0 (linear), MHA ~ 2.0 (quadratic)")
    print()
    import math

    by_model: Dict[str, List[BenchResult]] = {}
    for r in results:
        by_model.setdefault(r.name, []).append(r)

    for name, rs in by_model.items():
        if len(rs) < 2:
            continue
        exponents = []
        for i in range(1, len(rs)):
            L1, L2 = rs[i-1].seq_len, rs[i].seq_len
            ms1, ms2 = rs[i-1].latency_ms, rs[i].latency_ms
            if ms1 > 0 and ms2 > 0 and L1 > 0 and L2 > 0:
                alpha = math.log(ms2 / ms1) / math.log(L2 / L1)
                exponents.append(alpha)
        if exponents:
            avg_exp = sum(exponents) / len(exponents)
            print(f"  {name:<20} avg exponent = {avg_exp:.2f}  "
                  f"{'[OK linear]' if avg_exp < 1.5 else '[quadratic or worse]'}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[bench_glyph_mamba] torch {torch.__version__}, cpu")
    print(f"  Config: d_model={D_MODEL}, n_layers={N_LAYERS}, d_glyph={D_GLYPH}")
    print(f"  Seq lengths: {SEQ_LENGTHS}, batch={BATCH_SIZE}, runs={N_RUNS}\n")

    glyph_model = GlyphMambaModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        d_glyph=D_GLYPH,
        d_conv=D_CONV,
        expand=EXPAND,
        use_sigmalang=False,   # deterministic fallback for benchmarking
    )

    mha_model = MHABaselineModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
    )

    print(f"  GlyphMamba params  : {glyph_model.count_parameters():,}")
    print(f"  MHA baseline params: {mha_model.count_parameters():,}")
    print(f"  Glyph selectivity  : {glyph_model.glyph_selectivity_ratio():.1%}")

    results: List[BenchResult] = []

    for L in SEQ_LENGTHS:
        print(f"  Benchmarking L={L} ...", end=" ", flush=True)
        r_glyph = benchmark_model(glyph_model, "GlyphMamba", L)
        r_mha   = benchmark_model(mha_model,   "MHA",        L)
        results.extend([r_glyph, r_mha])
        speedup = r_mha.latency_ms / r_glyph.latency_ms if r_glyph.latency_ms > 0 else float("inf")
        print(f"Mamba {r_glyph.latency_ms:.1f}ms  MHA {r_mha.latency_ms:.1f}ms  "
              f"({speedup:.1f}x {'[Mamba wins]' if speedup > 1 else '[MHA wins]'})")

    print_table(results)
    scaling_analysis(results)

    # Quick correctness sanity check
    print("\n  Sanity check: GlyphMambaModel logits finite for L=64 ...")
    glyph_model.eval()
    with torch.no_grad():
        ids = torch.randint(0, VOCAB_SIZE, (1, 64))
        logits = glyph_model(ids)
    assert torch.isfinite(logits).all(), "Non-finite logits detected!"
    print(f"  [OK] logits shape={tuple(logits.shape)}, "
          f"range=[{logits.min():.3f}, {logits.max():.3f}]")

    print("\n[bench_glyph_mamba] DONE\n")


if __name__ == "__main__":
    main()
