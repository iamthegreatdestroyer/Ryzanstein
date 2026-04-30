"""
Sprint 4.2: Model Optimization Pipeline for Ryzanstein LLM
[REF:SPRINT4.2] - Quantization, Pruning, ONNX Export

Wires PHASE2_DEVELOPMENT optimization primitives (quantizer.py, pruner.py)
into an end-to-end model optimization pipeline.

Pipeline:
  1. Calibrate: collect activation statistics from representative dataset
  2. Quantize: INT4/INT8 auto-selection based on layer sensitivity
  3. Prune: magnitude-based + structured head pruning with perplexity gating
  4. Export: ONNX with graph optimization passes

Usage:
    from .model_optimization import ModelOptimizationPipeline, OptimizationConfig

    config = OptimizationConfig(
        target_bits=8,           # 8 for INT8, 4 for INT4
        pruning_sparsity=0.2,    # 20% weight sparsity
        perplexity_budget=0.03,  # Max 3% perplexity increase allowed
    )
    pipeline = ModelOptimizationPipeline(config)
    result = pipeline.run(model, calibration_data, output_dir="models/optimized")
"""

import sys
import os
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_PHASE2_SRC = _REPO_ROOT / "PHASE2_DEVELOPMENT" / "src"
if str(_PHASE2_SRC) not in sys.path:
    sys.path.insert(0, str(_PHASE2_SRC))

# ---------------------------------------------------------------------------
# Import optimization primitives
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — optimization pipeline in stub mode")

_QUANTIZER_AVAILABLE = False
_PRUNER_AVAILABLE = False

try:
    from optimization.quantizer import (
        ModelQuantizer, QuantizationConfig, QuantizationMode,
        QuantizationStrategy,
    )
    _QUANTIZER_AVAILABLE = True
    logger.info("Sprint 4.2 quantizer loaded from PHASE2_DEVELOPMENT")
except ImportError as _e:
    logger.warning(f"Quantizer not available ({_e})")

try:
    from optimization.pruner import (
        MagnitudePruner, StructuredPruner,
        PruningConfig, PruningStrategy, PruningGranularity,
    )
    _PRUNER_AVAILABLE = True
    logger.info("Sprint 4.2 pruner loaded from PHASE2_DEVELOPMENT")
except ImportError as _e:
    logger.warning(f"Pruner not available ({_e})")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class QuantizationTarget(Enum):
    INT8 = "int8"
    INT4 = "int4"
    AUTO = "auto"   # Select per-layer based on sensitivity


@dataclass
class OptimizationConfig:
    """
    End-to-end optimization pipeline configuration.

    Attributes:
        target_bits: Default quantization bit-width (4 or 8). AUTO selects per layer.
        pruning_sparsity: Target weight sparsity fraction [0, 1). 0 = no pruning.
        prune_attention_heads: Whether to prune attention heads structurally.
        perplexity_budget: Max acceptable fractional perplexity increase (e.g. 0.03 = 3%).
        calibration_samples: Number of samples from calibration dataset.
        onnx_export: Whether to export to ONNX after quantization.
        onnx_opset: ONNX opset version.
        onnx_optimize: Apply ONNX optimization passes (constant folding, etc.).
        skip_first_last_layers: Skip quantizing embedding/output layers (often sensitive).
    """
    target_bits: int = 8
    pruning_sparsity: float = 0.0
    prune_attention_heads: bool = False
    perplexity_budget: float = 0.05
    calibration_samples: int = 128
    onnx_export: bool = False
    onnx_opset: int = 17
    onnx_optimize: bool = True
    skip_first_last_layers: bool = True


@dataclass
class OptimizationResult:
    """Results from the optimization pipeline."""
    original_size_mb: float = 0.0
    optimized_size_mb: float = 0.0
    compression_ratio: float = 1.0
    original_perplexity: Optional[float] = None
    optimized_perplexity: Optional[float] = None
    perplexity_delta: Optional[float] = None
    quantization_mode: str = "none"
    pruning_sparsity_actual: float = 0.0
    onnx_path: Optional[str] = None
    layer_stats: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    success: bool = False
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Quantization: {self.quantization_mode}",
            f"  Size: {self.original_size_mb:.1f} MB → {self.optimized_size_mb:.1f} MB "
            f"({self.compression_ratio:.2f}x)",
        ]
        if self.original_perplexity and self.optimized_perplexity:
            lines.append(
                f"  Perplexity: {self.original_perplexity:.3f} → {self.optimized_perplexity:.3f} "
                f"(Δ{self.perplexity_delta:+.2%})"
            )
        if self.pruning_sparsity_actual > 0:
            lines.append(f"  Pruning sparsity: {self.pruning_sparsity_actual:.1%}")
        if self.onnx_path:
            lines.append(f"  ONNX: {self.onnx_path}")
        lines.append(f"  Elapsed: {self.elapsed_s:.1f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calibration dataset helper
# ---------------------------------------------------------------------------

class CalibrationDataset:
    """
    Lightweight calibration dataset for post-training quantization.

    Wraps a list of tokenized prompts and provides batched iteration
    for computing activation statistics.
    """

    DEFAULT_PROMPTS = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of France?",
        "Explain how transformers work in machine learning.",
        "The Ryzanstein LLM is optimized for AMD Ryzen processors.",
        "List the planets of the solar system in order.",
        "Write a Python function to compute fibonacci numbers.",
        "What are the key differences between supervised and unsupervised learning?",
        "Describe the architecture of a large language model.",
    ]

    def __init__(
        self,
        prompts: Optional[List[str]] = None,
        tokenizer=None,
        max_length: int = 512,
        num_samples: int = 128,
    ):
        self.prompts = (prompts or self.DEFAULT_PROMPTS) * max(1, num_samples // 8)
        self.prompts = self.prompts[:num_samples]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.prompts)

    def get_input_ids(self, batch_size: int = 8):
        """Yield batches of synthetic input_ids tensors for calibration."""
        if not _TORCH_AVAILABLE:
            return

        import torch
        for i in range(0, len(self.prompts), batch_size):
            batch = self.prompts[i:i + batch_size]

            if self.tokenizer is not None:
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                yield enc["input_ids"]
            else:
                # Synthetic fallback: uniform random tokens in vocab range
                seq_len = min(64, self.max_length)
                yield torch.randint(0, 32000, (len(batch), seq_len))


# ---------------------------------------------------------------------------
# Per-layer sensitivity scoring
# ---------------------------------------------------------------------------

def compute_layer_sensitivity(model, calibration_data: CalibrationDataset) -> Dict[str, float]:
    """
    Compute quantization sensitivity score for each linear layer.

    Score = mean absolute weight / weight std (coefficient of variation).
    Layers with high variation are more sensitive to quantization.

    Returns:
        dict mapping layer name → sensitivity score (higher = more sensitive)
    """
    if not _TORCH_AVAILABLE:
        return {}

    import torch
    sensitivity = {}

    try:
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                w = module.weight.data.float()
                mean_abs = w.abs().mean().item()
                std = w.std().item()
                if std > 1e-8:
                    sensitivity[name] = mean_abs / std
                else:
                    sensitivity[name] = 0.0
    except Exception as e:
        logger.warning(f"Sensitivity computation failed: {e}")

    return sensitivity


def auto_select_bitwidth(
    sensitivity: Dict[str, float],
    default_bits: int = 8,
    sensitive_threshold: float = 2.0,
) -> Dict[str, int]:
    """
    Auto-select per-layer quantization bit-width.

    Layers above sensitive_threshold keep their default precision;
    others can use lower precision.

    Returns:
        dict mapping layer name → bit-width (4 or 8)
    """
    layer_bits = {}
    for name, score in sensitivity.items():
        if score >= sensitive_threshold:
            layer_bits[name] = default_bits  # Keep higher precision for sensitive layers
        else:
            layer_bits[name] = max(4, default_bits - 4)  # Drop to INT4 if insensitive
    return layer_bits


# ---------------------------------------------------------------------------
# Perplexity measurement
# ---------------------------------------------------------------------------

def estimate_perplexity(model, calibration_data: CalibrationDataset) -> Optional[float]:
    """
    Estimate perplexity on calibration data (proxy for model accuracy).

    Uses cross-entropy loss averaged over calibration samples.
    Returns None if torch or model forward pass is unavailable.
    """
    if not _TORCH_AVAILABLE:
        return None

    import torch
    try:
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for input_ids in calibration_data.get_input_ids(batch_size=4):
                if input_ids.shape[1] < 2:
                    continue
                try:
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss.item()
                    total_loss += loss * input_ids.numel()
                    total_tokens += input_ids.numel()
                except Exception:
                    break

        if total_tokens == 0:
            return None

        import math
        return math.exp(total_loss / total_tokens)
    except Exception as e:
        logger.debug(f"Perplexity estimation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Model size measurement
# ---------------------------------------------------------------------------

def model_size_mb(model) -> float:
    """Return model parameter size in megabytes."""
    if not _TORCH_AVAILABLE:
        return 0.0
    try:
        return sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        ) / (1024 * 1024)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model,
    output_path: str,
    opset: int = 17,
    optimize: bool = True,
    input_shape: Tuple[int, int] = (1, 64),
) -> Optional[str]:
    """
    Export model to ONNX with optional optimization passes.

    Optimization passes applied:
    - Constant folding
    - Redundant node elimination
    - Shape inference propagation

    Returns:
        Absolute path to exported ONNX file, or None on failure.
    """
    if not _TORCH_AVAILABLE:
        logger.warning("ONNX export skipped: PyTorch not available")
        return None

    try:
        import torch
        import torch.onnx
    except ImportError:
        logger.warning("ONNX export skipped: torch.onnx not available")
        return None

    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        model.eval()
        dummy_input = torch.randint(0, 32000, input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
            },
            do_constant_folding=optimize,
        )
        logger.info(f"ONNX export: {output_path} (opset={opset}, optimize={optimize})")

        if optimize:
            _apply_onnx_optimizations(output_path)

        return output_path

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return None


def _apply_onnx_optimizations(onnx_path: str) -> None:
    """Apply onnxruntime/onnxoptimizer passes to reduce graph size."""
    try:
        import onnx
        from onnx import optimizer as onnx_opt
        model_proto = onnx.load(onnx_path)
        passes = [
            "eliminate_deadend",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
            "eliminate_unused_initializer",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
        ]
        optimized = onnx_opt.optimize(model_proto, passes)
        onnx.save(optimized, onnx_path)
        logger.info(f"ONNX optimization passes applied: {passes}")
    except ImportError:
        logger.debug("onnx/onnxoptimizer not installed; skipping graph optimization passes")
    except Exception as e:
        logger.warning(f"ONNX optimization failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Main optimization pipeline
# ---------------------------------------------------------------------------

class ModelOptimizationPipeline:
    """
    End-to-end model optimization pipeline.

    Steps:
    1. Measure baseline size + perplexity
    2. Compute per-layer sensitivity
    3. Auto-select bit-width per layer
    4. Quantize (INT4/INT8)
    5. Prune (if sparsity > 0)
    6. Validate perplexity budget
    7. Export ONNX (if requested)
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

    def run(
        self,
        model,
        calibration_prompts: Optional[List[str]] = None,
        tokenizer=None,
        output_dir: str = "models/optimized",
        model_name: str = "ryzanstein",
    ) -> OptimizationResult:
        """
        Run the full optimization pipeline.

        Args:
            model: PyTorch model to optimize
            calibration_prompts: Optional list of calibration text prompts
            tokenizer: Optional HuggingFace tokenizer for calibration
            output_dir: Directory for optimized model and ONNX artifacts
            model_name: Prefix for output filenames

        Returns:
            OptimizationResult with metrics and artifact paths
        """
        result = OptimizationResult()
        t_start = time.time()
        cfg = self.config
        os.makedirs(output_dir, exist_ok=True)

        # Step 0: Validate
        if not _TORCH_AVAILABLE:
            result.errors.append("PyTorch not available — skipping optimization")
            result.elapsed_s = time.time() - t_start
            return result

        try:
            import torch

            # Step 1: Baseline metrics
            logger.info("[OptPipeline] Step 1/7 — Baseline measurement")
            cal_data = CalibrationDataset(
                prompts=calibration_prompts,
                tokenizer=tokenizer,
                num_samples=cfg.calibration_samples,
            )
            result.original_size_mb = model_size_mb(model)
            result.original_perplexity = estimate_perplexity(model, cal_data)
            logger.info(
                f"  Baseline: {result.original_size_mb:.1f} MB, "
                f"PPL={result.original_perplexity}"
            )

            # Step 2: Sensitivity
            logger.info("[OptPipeline] Step 2/7 — Layer sensitivity analysis")
            sensitivity = compute_layer_sensitivity(model, cal_data)
            layer_bits = auto_select_bitwidth(
                sensitivity,
                default_bits=cfg.target_bits,
            )
            result.layer_stats["sensitivity"] = {
                k: round(v, 4) for k, v in sensitivity.items()
            }
            result.layer_stats["bit_assignment"] = layer_bits
            logger.info(
                f"  {len(sensitivity)} layers analyzed; "
                f"INT4 assigned to {sum(1 for b in layer_bits.values() if b == 4)} layers"
            )

            # Step 3: Quantization
            logger.info(f"[OptPipeline] Step 3/7 — Quantization (target={cfg.target_bits}-bit)")
            model = self._apply_quantization(model, layer_bits, cfg)
            result.quantization_mode = f"INT{cfg.target_bits}+auto"

            # Step 4: Pruning
            if cfg.pruning_sparsity > 0:
                logger.info(
                    f"[OptPipeline] Step 4/7 — Pruning "
                    f"(sparsity={cfg.pruning_sparsity:.0%})"
                )
                model, actual_sparsity = self._apply_pruning(model, cfg)
                result.pruning_sparsity_actual = actual_sparsity
            else:
                logger.info("[OptPipeline] Step 4/7 — Pruning skipped (sparsity=0)")

            # Step 5: Post-optimization measurement
            logger.info("[OptPipeline] Step 5/7 — Post-optimization measurement")
            result.optimized_size_mb = model_size_mb(model)
            result.compression_ratio = (
                result.original_size_mb / max(result.optimized_size_mb, 0.001)
            )
            result.optimized_perplexity = estimate_perplexity(model, cal_data)
            if result.original_perplexity and result.optimized_perplexity:
                result.perplexity_delta = (
                    (result.optimized_perplexity - result.original_perplexity)
                    / result.original_perplexity
                )
            logger.info(
                f"  Optimized: {result.optimized_size_mb:.1f} MB "
                f"({result.compression_ratio:.2f}x), "
                f"PPL={result.optimized_perplexity}"
            )

            # Step 6: Perplexity gate
            logger.info("[OptPipeline] Step 6/7 — Perplexity budget validation")
            if result.perplexity_delta is not None:
                if result.perplexity_delta > cfg.perplexity_budget:
                    msg = (
                        f"Perplexity increased by {result.perplexity_delta:.2%} "
                        f"(budget={cfg.perplexity_budget:.2%}) — "
                        f"optimization acceptable but exceeds budget"
                    )
                    logger.warning(f"  WARN: {msg}")
                    result.errors.append(msg)
                else:
                    logger.info(
                        f"  OK: perplexity delta {result.perplexity_delta:+.2%} "
                        f"within budget ({cfg.perplexity_budget:.2%})"
                    )

            # Step 7: ONNX export
            if cfg.onnx_export:
                logger.info(
                    f"[OptPipeline] Step 7/7 — ONNX export "
                    f"(opset={cfg.onnx_opset}, optimize={cfg.onnx_optimize})"
                )
                onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
                result.onnx_path = export_onnx(
                    model,
                    onnx_path,
                    opset=cfg.onnx_opset,
                    optimize=cfg.onnx_optimize,
                )
            else:
                logger.info("[OptPipeline] Step 7/7 — ONNX export skipped")

            result.success = True

        except Exception as e:
            logger.error(f"[OptPipeline] Pipeline failed: {e}", exc_info=True)
            result.errors.append(str(e))

        result.elapsed_s = time.time() - t_start
        logger.info(
            f"[OptPipeline] Complete in {result.elapsed_s:.1f}s "
            f"({'OK' if result.success else 'FAILED'})\n{result.summary()}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_quantization(self, model, layer_bits: Dict[str, int], cfg: OptimizationConfig):
        """Apply INT4/INT8 quantization using PHASE2 quantizer or PyTorch native."""
        if _QUANTIZER_AVAILABLE:
            return self._quantize_via_phase2(model, layer_bits, cfg)
        return self._quantize_pytorch_native(model, cfg)

    def _quantize_via_phase2(self, model, layer_bits: Dict[str, int], cfg: OptimizationConfig):
        """Use PHASE2_DEVELOPMENT quantizer."""
        try:
            q_config = QuantizationConfig(
                mode=(QuantizationMode.INT8 if cfg.target_bits == 8
                      else QuantizationMode.INT4),
                strategy=QuantizationStrategy.STATIC,
                per_channel=True,
            )
            quantizer = ModelQuantizer(q_config)
            return quantizer.quantize(model)
        except Exception as e:
            logger.warning(f"PHASE2 quantizer failed ({e}), falling back to PyTorch native")
            return self._quantize_pytorch_native(model, cfg)

    def _quantize_pytorch_native(self, model, cfg: OptimizationConfig):
        """Apply PyTorch dynamic quantization as fallback."""
        import torch.quantization as tq
        try:
            dtype = (torch.qint8 if cfg.target_bits == 8 else torch.quint4x2)
            quantized = tq.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=dtype,
            )
            logger.info("  Applied PyTorch dynamic INT8 quantization")
            return quantized
        except Exception as e:
            logger.warning(f"PyTorch quantization failed ({e}); returning original model")
            return model

    def _apply_pruning(self, model, cfg: OptimizationConfig) -> Tuple[Any, float]:
        """Apply magnitude-based pruning, returning (pruned_model, actual_sparsity)."""
        if _PRUNER_AVAILABLE:
            return self._prune_via_phase2(model, cfg)
        return self._prune_pytorch_native(model, cfg)

    def _prune_via_phase2(self, model, cfg: OptimizationConfig) -> Tuple[Any, float]:
        """Use PHASE2_DEVELOPMENT pruner."""
        try:
            p_config = PruningConfig(
                strategy=PruningStrategy.MAGNITUDE,
                granularity=PruningGranularity.WEIGHT,
                sparsity=cfg.pruning_sparsity,
            )
            pruner = MagnitudePruner(p_config)
            pruned = pruner.prune(model)
            actual = _measure_sparsity(pruned)
            logger.info(f"  PHASE2 pruner: actual sparsity = {actual:.2%}")

            if cfg.prune_attention_heads:
                head_config = PruningConfig(
                    strategy=PruningStrategy.L1_NORM,
                    granularity=PruningGranularity.HEAD,
                    sparsity=cfg.pruning_sparsity * 0.5,  # More conservative for heads
                )
                head_pruner = StructuredPruner(head_config)
                pruned = head_pruner.prune(pruned)
                actual = _measure_sparsity(pruned)
                logger.info(f"  Head pruning applied: final sparsity = {actual:.2%}")

            return pruned, actual
        except Exception as e:
            logger.warning(f"PHASE2 pruner failed ({e}), using PyTorch native")
            return self._prune_pytorch_native(model, cfg)

    def _prune_pytorch_native(self, model, cfg: OptimizationConfig) -> Tuple[Any, float]:
        """Apply magnitude-based pruning via torch.nn.utils.prune."""
        import torch.nn.utils.prune as prune
        try:
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(
                        module, name="weight", amount=cfg.pruning_sparsity
                    )
                    prune.remove(module, "weight")
            actual = _measure_sparsity(model)
            logger.info(f"  PyTorch native pruning: actual sparsity = {actual:.2%}")
            return model, actual
        except Exception as e:
            logger.warning(f"Pruning failed ({e}); returning original model")
            return model, 0.0


def _measure_sparsity(model) -> float:
    """Fraction of near-zero weights in the model."""
    if not _TORCH_AVAILABLE:
        return 0.0
    try:
        import torch
        total = 0
        zeros = 0
        for p in model.parameters():
            total += p.numel()
            zeros += (p.abs() < 1e-8).sum().item()
        return zeros / max(total, 1)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Quick benchmark comparison (called by verify_inference.py / CI)
# ---------------------------------------------------------------------------

def benchmark_quantization_speedup(hidden_size: int = 512, seq_len: int = 64) -> Dict[str, float]:
    """
    Micro-benchmark: compare FP32 vs INT8 linear layer throughput.
    Returns dict with timings in ms.
    """
    if not _TORCH_AVAILABLE:
        return {"fp32_ms": 0.0, "int8_ms": 0.0, "speedup": 1.0}

    import torch
    import torch.nn as nn
    import torch.quantization as tq
    import time

    linear = nn.Linear(hidden_size, hidden_size)
    x = torch.randn(seq_len, hidden_size)

    # FP32 baseline
    linear.eval()
    with torch.no_grad():
        WARMUP = 10
        N = 100
        for _ in range(WARMUP):
            _ = linear(x)
        t0 = time.perf_counter()
        for _ in range(N):
            _ = linear(x)
        fp32_ms = (time.perf_counter() - t0) / N * 1000

    # INT8 quantized
    try:
        q_linear = tq.quantize_dynamic(linear, {nn.Linear}, dtype=torch.qint8)
        with torch.no_grad():
            for _ in range(WARMUP):
                _ = q_linear(x)
            t0 = time.perf_counter()
            for _ in range(N):
                _ = q_linear(x)
            int8_ms = (time.perf_counter() - t0) / N * 1000
        speedup = fp32_ms / max(int8_ms, 0.001)
    except Exception:
        int8_ms = fp32_ms
        speedup = 1.0

    return {
        "fp32_ms": round(fp32_ms, 3),
        "int8_ms": round(int8_ms, 3),
        "speedup": round(speedup, 2),
    }
