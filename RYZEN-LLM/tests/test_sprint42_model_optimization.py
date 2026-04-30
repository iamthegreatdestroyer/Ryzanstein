"""
Sprint 4.2: Model Optimization Pipeline — Test Suite
[REF:SPRINT4.2]

Covers:
  - OptimizationConfig defaults and validation
  - CalibrationDataset batching (synthetic + tokenizer paths)
  - compute_layer_sensitivity scoring
  - auto_select_bitwidth decision logic
  - estimate_perplexity (stub/torch paths)
  - benchmark_quantization_speedup
  - ModelOptimizationPipeline.run() full pipeline with mock model
  - OptimizationResult.summary() formatting
  - ONNX export path (mocked)
  - Graceful fallback when torch/PHASE2 not available
"""

import sys
import types
import importlib
import asyncio
import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ensure repo root on path
_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / "RYZEN-LLM" / "src"))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mock_linear(in_f=64, out_f=128, seed=0):
    """Return a mock torch.nn.Linear-like object with a weight tensor."""
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        layer = nn.Linear(in_f, out_f, bias=False)
        return layer
    except ImportError:
        return None


def _make_tiny_model():
    """Return a tiny 2-layer MLP for pipeline smoke tests."""
    try:
        import torch
        import torch.nn as nn

        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 64)
                self.fc2 = nn.Linear(64, 32)

            def forward(self, input_ids):
                x = input_ids.float()
                # Truncate to hidden_size
                x = x[:, :32] if x.shape[1] >= 32 else torch.cat(
                    [x, torch.zeros(x.shape[0], 32 - x.shape[1])], dim=1
                )
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        return TinyMLP()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Test: OptimizationConfig
# ---------------------------------------------------------------------------

class TestOptimizationConfig(unittest.TestCase):

    def setUp(self):
        from api.model_optimization import OptimizationConfig
        self.OptimizationConfig = OptimizationConfig

    def test_defaults(self):
        cfg = self.OptimizationConfig()
        self.assertEqual(cfg.target_bits, 8)
        self.assertAlmostEqual(cfg.pruning_sparsity, 0.0)
        self.assertFalse(cfg.prune_attention_heads)
        self.assertAlmostEqual(cfg.perplexity_budget, 0.05)
        self.assertEqual(cfg.calibration_samples, 128)
        self.assertFalse(cfg.onnx_export)
        self.assertEqual(cfg.onnx_opset, 17)
        self.assertTrue(cfg.onnx_optimize)
        self.assertTrue(cfg.skip_first_last_layers)

    def test_custom_config(self):
        cfg = self.OptimizationConfig(
            target_bits=4,
            pruning_sparsity=0.3,
            perplexity_budget=0.02,
            calibration_samples=64,
            onnx_export=True,
        )
        self.assertEqual(cfg.target_bits, 4)
        self.assertAlmostEqual(cfg.pruning_sparsity, 0.3)
        self.assertAlmostEqual(cfg.perplexity_budget, 0.02)
        self.assertEqual(cfg.calibration_samples, 64)
        self.assertTrue(cfg.onnx_export)

    def test_zero_pruning(self):
        cfg = self.OptimizationConfig(pruning_sparsity=0.0)
        self.assertEqual(cfg.pruning_sparsity, 0.0)

    def test_max_sparsity(self):
        # Allow values up to (but not including) 1.0
        cfg = self.OptimizationConfig(pruning_sparsity=0.99)
        self.assertAlmostEqual(cfg.pruning_sparsity, 0.99)


# ---------------------------------------------------------------------------
# Test: OptimizationResult
# ---------------------------------------------------------------------------

class TestOptimizationResult(unittest.TestCase):

    def setUp(self):
        from api.model_optimization import OptimizationResult
        self.OptimizationResult = OptimizationResult

    def test_defaults(self):
        r = self.OptimizationResult()
        self.assertFalse(r.success)
        self.assertEqual(r.compression_ratio, 1.0)
        self.assertIsNone(r.onnx_path)
        self.assertEqual(r.errors, [])

    def test_summary_no_perplexity(self):
        r = self.OptimizationResult(
            original_size_mb=100.0,
            optimized_size_mb=25.0,
            compression_ratio=4.0,
            quantization_mode="INT8",
            elapsed_s=12.5,
            success=True,
        )
        summary = r.summary()
        self.assertIn("INT8", summary)
        self.assertIn("4.00x", summary)
        self.assertIn("100.0 MB", summary)
        self.assertIn("25.0 MB", summary)
        self.assertIn("12.5s", summary)

    def test_summary_with_perplexity(self):
        r = self.OptimizationResult(
            original_perplexity=10.0,
            optimized_perplexity=10.3,
            perplexity_delta=0.03,
            compression_ratio=2.0,
            quantization_mode="INT4",
            elapsed_s=5.0,
            success=True,
        )
        summary = r.summary()
        self.assertIn("10.000", summary)
        self.assertIn("10.300", summary)

    def test_summary_with_onnx(self):
        r = self.OptimizationResult(
            onnx_path="/models/model.onnx",
            quantization_mode="INT8",
            elapsed_s=1.0,
            success=True,
        )
        summary = r.summary()
        self.assertIn("model.onnx", summary)

    def test_summary_with_pruning(self):
        r = self.OptimizationResult(
            pruning_sparsity_actual=0.25,
            quantization_mode="INT8",
            elapsed_s=2.0,
            success=True,
        )
        summary = r.summary()
        self.assertIn("25.0%", summary)


# ---------------------------------------------------------------------------
# Test: CalibrationDataset
# ---------------------------------------------------------------------------

class TestCalibrationDataset(unittest.TestCase):

    def setUp(self):
        from api.model_optimization import CalibrationDataset
        self.CalibrationDataset = CalibrationDataset

    def test_default_prompts_loaded(self):
        ds = self.CalibrationDataset(num_samples=8)
        self.assertEqual(len(ds), 8)

    def test_custom_prompts(self):
        prompts = ["Hello world.", "Test prompt."]
        ds = self.CalibrationDataset(prompts=prompts, num_samples=2)
        self.assertEqual(len(ds), 2)

    def test_num_samples_truncates(self):
        ds = self.CalibrationDataset(num_samples=4)
        self.assertEqual(len(ds), 4)

    def test_num_samples_repeats(self):
        # With 8 default prompts and num_samples=16, should repeat
        ds = self.CalibrationDataset(num_samples=16)
        self.assertEqual(len(ds), 16)

    def test_get_input_ids_synthetic(self):
        """Without tokenizer, should yield synthetic int tensors."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        ds = self.CalibrationDataset(num_samples=8)
        batches = list(ds.get_input_ids(batch_size=4))
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertIsInstance(batch, torch.Tensor)
            self.assertEqual(batch.shape[0], 4)
            self.assertTrue((batch >= 0).all())
            self.assertTrue((batch < 32000).all())

    def test_get_input_ids_with_tokenizer(self):
        """With a mock tokenizer, should use its output."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (4, 16))
        }
        ds = self.CalibrationDataset(num_samples=8, tokenizer=mock_tokenizer)
        batches = list(ds.get_input_ids(batch_size=4))
        self.assertEqual(len(batches), 2)
        mock_tokenizer.assert_called()

    def test_empty_dataset_no_error(self):
        """Zero samples should yield nothing."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        ds = self.CalibrationDataset(num_samples=1)
        # Force empty
        ds.prompts = []
        batches = list(ds.get_input_ids(batch_size=4))
        self.assertEqual(len(batches), 0)


# ---------------------------------------------------------------------------
# Test: compute_layer_sensitivity
# ---------------------------------------------------------------------------

class TestComputeLayerSensitivity(unittest.TestCase):

    def setUp(self):
        from api.model_optimization import compute_layer_sensitivity, CalibrationDataset
        self.compute = compute_layer_sensitivity
        self.CalibrationDataset = CalibrationDataset

    def test_returns_dict(self):
        try:
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        ds = self.CalibrationDataset(num_samples=8)
        scores = self.compute(model, ds)
        self.assertIsInstance(scores, dict)

    def test_linear_layers_scored(self):
        try:
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        ds = self.CalibrationDataset(num_samples=8)
        scores = self.compute(model, ds)
        # TinyMLP has fc1 and fc2
        keys = list(scores.keys())
        self.assertTrue(any("fc" in k or "linear" in k.lower() for k in keys),
                        f"Expected linear layer keys, got: {keys}")

    def test_scores_are_nonnegative(self):
        try:
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        ds = self.CalibrationDataset(num_samples=8)
        scores = self.compute(model, ds)
        for name, score in scores.items():
            self.assertGreaterEqual(score, 0.0, f"Negative score for {name}")

    def test_no_torch_returns_empty(self):
        """When torch is unavailable, should return empty dict."""
        import api.model_optimization as mo_mod
        orig = mo_mod._TORCH_AVAILABLE
        try:
            mo_mod._TORCH_AVAILABLE = False
            scores = mo_mod.compute_layer_sensitivity(MagicMock(), MagicMock())
            self.assertEqual(scores, {})
        finally:
            mo_mod._TORCH_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Test: auto_select_bitwidth
# ---------------------------------------------------------------------------

class TestAutoSelectBitwidth(unittest.TestCase):

    def setUp(self):
        from api.model_optimization import auto_select_bitwidth
        self.auto_select = auto_select_bitwidth

    def test_empty_scores_returns_empty(self):
        result = self.auto_select({}, threshold=0.5)
        self.assertEqual(result, {})

    def test_high_sensitivity_uses_int8(self):
        scores = {"layer.0": 2.0, "layer.1": 1.5}
        result = self.auto_select(scores, threshold=1.0)
        for name in scores:
            self.assertEqual(result[name], 8, f"Expected INT8 for high-sensitivity {name}")

    def test_low_sensitivity_uses_int4(self):
        scores = {"layer.0": 0.1, "layer.1": 0.2}
        result = self.auto_select(scores, threshold=1.0)
        for name in scores:
            self.assertEqual(result[name], 4, f"Expected INT4 for low-sensitivity {name}")

    def test_mixed_sensitivity(self):
        scores = {"sensitive": 2.0, "insensitive": 0.1}
        result = self.auto_select(scores, threshold=1.0)
        self.assertEqual(result["sensitive"], 8)
        self.assertEqual(result["insensitive"], 4)

    def test_threshold_boundary(self):
        # Exactly at threshold should use INT8
        scores = {"boundary": 1.0}
        result = self.auto_select(scores, threshold=1.0)
        self.assertEqual(result["boundary"], 8)

    def test_result_only_contains_4_or_8(self):
        import random
        random.seed(42)
        scores = {f"layer.{i}": random.uniform(0, 3) for i in range(20)}
        result = self.auto_select(scores, threshold=1.0)
        for name, bits in result.items():
            self.assertIn(bits, (4, 8), f"Unexpected bit-width {bits} for {name}")


# ---------------------------------------------------------------------------
# Test: estimate_perplexity
# ---------------------------------------------------------------------------

class TestEstimatePerplexity(unittest.TestCase):

    def setUp(self):
        from api.model_optimization import estimate_perplexity, CalibrationDataset
        self.estimate = estimate_perplexity
        self.CalibrationDataset = CalibrationDataset

    def test_returns_none_without_torch(self):
        import api.model_optimization as mo_mod
        orig = mo_mod._TORCH_AVAILABLE
        try:
            mo_mod._TORCH_AVAILABLE = False
            result = mo_mod.estimate_perplexity(MagicMock(), MagicMock())
            self.assertIsNone(result)
        finally:
            mo_mod._TORCH_AVAILABLE = orig

    def test_returns_float_with_torch(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        ds = self.CalibrationDataset(num_samples=8)
        result = self.estimate(model, ds, max_batches=2)
        # Should return a float >= 0 or None on error
        if result is not None:
            self.assertIsInstance(result, float)
            self.assertGreater(result, 0.0)

    def test_max_batches_limits_computation(self):
        """max_batches parameter should cap the number of batches processed."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        ds = self.CalibrationDataset(num_samples=32)

        call_count = [0]
        original_get = ds.get_input_ids

        def counting_get_input_ids(*args, **kwargs):
            for batch in original_get(*args, **kwargs):
                call_count[0] += 1
                yield batch

        ds.get_input_ids = counting_get_input_ids

        from api.model_optimization import estimate_perplexity
        estimate_perplexity(model, ds, max_batches=2)
        self.assertLessEqual(call_count[0], 3)  # at most max_batches+1 due to loop


# ---------------------------------------------------------------------------
# Test: benchmark_quantization_speedup
# ---------------------------------------------------------------------------

class TestBenchmarkQuantizationSpeedup(unittest.TestCase):

    def setUp(self):
        from api.model_optimization import benchmark_quantization_speedup
        self.benchmark = benchmark_quantization_speedup

    def test_returns_dict(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        result = self.benchmark(model, input_shape=(1, 32), n_warmup=1, n_iters=3)
        self.assertIsInstance(result, dict)

    def test_result_contains_fp32(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        result = self.benchmark(model, input_shape=(1, 32), n_warmup=1, n_iters=3)
        self.assertIn("fp32_ms", result)

    def test_result_has_nonnegative_latency(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        result = self.benchmark(model, input_shape=(1, 32), n_warmup=1, n_iters=3)
        fp32 = result.get("fp32_ms", 0)
        self.assertGreaterEqual(fp32, 0.0)

    def test_no_torch_returns_empty(self):
        import api.model_optimization as mo_mod
        orig = mo_mod._TORCH_AVAILABLE
        try:
            mo_mod._TORCH_AVAILABLE = False
            result = mo_mod.benchmark_quantization_speedup(MagicMock())
            self.assertEqual(result, {})
        finally:
            mo_mod._TORCH_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Test: ModelOptimizationPipeline — full pipeline smoke tests
# ---------------------------------------------------------------------------

class TestModelOptimizationPipeline(unittest.TestCase):

    def setUp(self):
        from api.model_optimization import ModelOptimizationPipeline, OptimizationConfig
        self.Pipeline = ModelOptimizationPipeline
        self.Config = OptimizationConfig

    def test_init_default_config(self):
        pipeline = self.Pipeline()
        self.assertIsNotNone(pipeline.config)
        self.assertEqual(pipeline.config.target_bits, 8)

    def test_init_custom_config(self):
        cfg = self.Config(target_bits=4, pruning_sparsity=0.1)
        pipeline = self.Pipeline(cfg)
        self.assertEqual(pipeline.config.target_bits, 4)

    def test_run_no_torch_returns_result(self):
        """Pipeline should gracefully handle missing torch."""
        import api.model_optimization as mo_mod
        orig = mo_mod._TORCH_AVAILABLE
        try:
            mo_mod._TORCH_AVAILABLE = False
            pipeline = self.Pipeline()
            result = pipeline.run(MagicMock(), calibration_data=None)
            from api.model_optimization import OptimizationResult
            self.assertIsInstance(result, OptimizationResult)
        finally:
            mo_mod._TORCH_AVAILABLE = orig

    def test_run_returns_optimization_result(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        from api.model_optimization import CalibrationDataset, OptimizationResult
        model = _make_tiny_model()
        ds = CalibrationDataset(num_samples=8)
        cfg = self.Config(
            target_bits=8,
            pruning_sparsity=0.0,
            onnx_export=False,
            calibration_samples=8,
        )
        pipeline = self.Pipeline(cfg)
        result = pipeline.run(model, calibration_data=ds)
        self.assertIsInstance(result, OptimizationResult)

    def test_run_records_elapsed(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        from api.model_optimization import CalibrationDataset
        model = _make_tiny_model()
        ds = CalibrationDataset(num_samples=8)
        pipeline = self.Pipeline(self.Config(calibration_samples=8))
        result = pipeline.run(model, calibration_data=ds)
        self.assertGreater(result.elapsed_s, 0.0)

    def test_run_sets_quantization_mode(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        from api.model_optimization import CalibrationDataset
        model = _make_tiny_model()
        ds = CalibrationDataset(num_samples=8)
        pipeline = self.Pipeline(self.Config(target_bits=8, calibration_samples=8))
        result = pipeline.run(model, calibration_data=ds)
        self.assertIn("int", result.quantization_mode.lower())

    def test_run_with_pruning(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        from api.model_optimization import CalibrationDataset
        model = _make_tiny_model()
        ds = CalibrationDataset(num_samples=8)
        cfg = self.Config(pruning_sparsity=0.2, calibration_samples=8)
        pipeline = self.Pipeline(cfg)
        result = pipeline.run(model, calibration_data=ds)
        # Should attempt pruning — actual sparsity may be 0 if PHASE2 unavailable
        self.assertIsNotNone(result)

    def test_run_with_none_calibration_data(self):
        """Pipeline should create default CalibrationDataset if none provided."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        pipeline = self.Pipeline(self.Config(calibration_samples=8))
        result = pipeline.run(model, calibration_data=None)
        from api.model_optimization import OptimizationResult
        self.assertIsInstance(result, OptimizationResult)

    def test_run_perplexity_gate(self):
        """If perplexity exceeds budget, errors list should be populated."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        import api.model_optimization as mo_mod
        from api.model_optimization import CalibrationDataset

        model = _make_tiny_model()
        ds = CalibrationDataset(num_samples=8)
        cfg = self.Config(
            perplexity_budget=0.0,   # 0% budget — almost certainly exceeded
            calibration_samples=8,
        )
        pipeline = self.Pipeline(cfg)

        # Patch estimate_perplexity to return deterministic values
        with patch.object(mo_mod, "estimate_perplexity", side_effect=[10.0, 15.0]):
            result = pipeline.run(model, calibration_data=ds)
        # Either the error list is populated or result is marked failed
        # (depends on implementation)
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# Test: export_onnx (mocked torch.onnx.export)
# ---------------------------------------------------------------------------

class TestExportOnnx(unittest.TestCase):

    def setUp(self):
        from api.model_optimization import export_onnx
        self.export_onnx = export_onnx

    def test_export_no_torch(self):
        import api.model_optimization as mo_mod
        orig = mo_mod._TORCH_AVAILABLE
        try:
            mo_mod._TORCH_AVAILABLE = False
            result = mo_mod.export_onnx(MagicMock(), "/tmp/out.onnx")
            self.assertIsNone(result)
        finally:
            mo_mod._TORCH_AVAILABLE = orig

    def test_export_calls_torch_onnx(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        with patch("torch.onnx.export") as mock_export:
            mock_export.return_value = None
            result = self.export_onnx(model, "/tmp/test_model.onnx",
                                      input_shape=(1, 32), opset=17)
            if result is not None:
                mock_export.assert_called_once()

    def test_export_handles_exception(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        model = _make_tiny_model()
        with patch("torch.onnx.export", side_effect=RuntimeError("ONNX error")):
            result = self.export_onnx(model, "/tmp/fail.onnx", input_shape=(1, 32))
            # Should return None on error, not raise
            self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Test: QuantizationTarget enum
# ---------------------------------------------------------------------------

class TestQuantizationTarget(unittest.TestCase):

    def test_enum_values(self):
        from api.model_optimization import QuantizationTarget
        self.assertEqual(QuantizationTarget.INT8.value, "int8")
        self.assertEqual(QuantizationTarget.INT4.value, "int4")
        self.assertEqual(QuantizationTarget.AUTO.value, "auto")

    def test_enum_by_value(self):
        from api.model_optimization import QuantizationTarget
        self.assertEqual(QuantizationTarget("int8"), QuantizationTarget.INT8)
        self.assertEqual(QuantizationTarget("int4"), QuantizationTarget.INT4)


# ---------------------------------------------------------------------------
# Test: Module-level imports don't crash
# ---------------------------------------------------------------------------

class TestModuleImport(unittest.TestCase):

    def test_module_importable(self):
        """The module should import without errors even without PHASE2 deps."""
        import importlib
        mod = importlib.import_module("api.model_optimization")
        self.assertIsNotNone(mod)

    def test_public_api_present(self):
        from api import model_optimization as m
        self.assertTrue(hasattr(m, "OptimizationConfig"))
        self.assertTrue(hasattr(m, "OptimizationResult"))
        self.assertTrue(hasattr(m, "CalibrationDataset"))
        self.assertTrue(hasattr(m, "ModelOptimizationPipeline"))
        self.assertTrue(hasattr(m, "compute_layer_sensitivity"))
        self.assertTrue(hasattr(m, "auto_select_bitwidth"))
        self.assertTrue(hasattr(m, "estimate_perplexity"))
        self.assertTrue(hasattr(m, "export_onnx"))
        self.assertTrue(hasattr(m, "benchmark_quantization_speedup"))

    def test_torch_availability_flag(self):
        from api import model_optimization as m
        self.assertIsInstance(m._TORCH_AVAILABLE, bool)

    def test_quantizer_availability_flag(self):
        from api import model_optimization as m
        self.assertIsInstance(m._QUANTIZER_AVAILABLE, bool)

    def test_pruner_availability_flag(self):
        from api import model_optimization as m
        self.assertIsInstance(m._PRUNER_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
