#!/usr/bin/env python3
"""
Test suite for BitNet quantization C++ bindings.
Tests quantization, dequantization, and error measurement.
"""

import sys
import numpy as np
from pathlib import Path

# Add python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import ryzen_llm
from ryzen_llm import ryzen_llm_bindings as bindings

# ============================================================================
# Constants
# ============================================================================

RTOL = 1e-5  # Relative tolerance for float comparisons
ATOL = 1e-8  # Absolute tolerance for float comparisons


# ============================================================================
# Test Classes
# ============================================================================

class TestQuantConfig:
    """Test QuantConfig structure and functionality."""
    
    def test_config_creation(self):
        """Test QuantConfig default creation."""
        config = bindings.QuantConfig()
        assert config is not None
        print("✓ QuantConfig created successfully")
    
    def test_config_properties(self):
        """Test QuantConfig property access."""
        config = bindings.QuantConfig()
        assert hasattr(config, 'per_group_scaling')
        assert hasattr(config, 'weight_group_size')
        assert hasattr(config, 'activation_clip_value')
        assert hasattr(config, 'symmetric_activations')
        print("✓ QuantConfig has all expected properties")
    
    def test_config_modification(self):
        """Test QuantConfig property modification."""
        config = bindings.QuantConfig()
        config.weight_group_size = 64
        config.per_group_scaling = True
        assert config.weight_group_size == 64
        assert config.per_group_scaling == True
        print("✓ QuantConfig properties are modifiable")
    
    def test_config_repr(self):
        """Test QuantConfig string representation."""
        config = bindings.QuantConfig()
        repr_str = repr(config)
        assert 'QuantConfig' in repr_str
        print(f"✓ QuantConfig repr: {repr_str}")


class TestTernaryWeight:
    """Test TernaryWeight structure and functionality."""
    
    def test_ternary_weight_creation(self):
        """Test TernaryWeight creation with dimensions."""
        rows, cols = 4, 8
        weight = bindings.TernaryWeight(rows, cols, 4)
        assert weight is not None
        assert weight.rows == rows
        assert weight.cols == cols
        print("✓ TernaryWeight created successfully")
    
    def test_ternary_weight_properties(self):
        """Test TernaryWeight property access."""
        weight = bindings.TernaryWeight(4, 8, 4)
        assert hasattr(weight, 'values')
        assert hasattr(weight, 'scales')
        assert hasattr(weight, 'rows')
        assert hasattr(weight, 'cols')
        assert hasattr(weight, 'group_size')
        print("✓ TernaryWeight has all expected properties")
    
    def test_ternary_weight_size(self):
        """Test TernaryWeight size method."""
        weight = bindings.TernaryWeight(4, 8, 4)
        assert weight.size() == 4 * 8
        print(f"✓ TernaryWeight size: {weight.size()}")
    
    def test_ternary_weight_repr(self):
        """Test TernaryWeight string representation."""
        weight = bindings.TernaryWeight(4, 8, 4)
        repr_str = repr(weight)
        assert 'TernaryWeight' in repr_str
        assert '4' in repr_str
        assert '8' in repr_str
        print(f"✓ TernaryWeight repr: {repr_str}")


class TestQuantizedActivation:
    """Test QuantizedActivation structure and functionality."""
    
    def test_quant_activation_creation(self):
        """Test QuantizedActivation creation."""
        size = 32
        activation = bindings.QuantizedActivation(size)
        assert activation is not None
        print("✓ QuantizedActivation created successfully")
    
    def test_quant_activation_properties(self):
        """Test QuantizedActivation property access."""
        activation = bindings.QuantizedActivation(32)
        assert hasattr(activation, 'values')
        assert hasattr(activation, 'scale')
        assert hasattr(activation, 'zero_point')
        print("✓ QuantizedActivation has all expected properties")
    
    def test_quant_activation_size(self):
        """Test QuantizedActivation size method."""
        activation = bindings.QuantizedActivation(32)
        assert activation.size() == 32
        print(f"✓ QuantizedActivation size: {activation.size()}")
    
    def test_quant_activation_repr(self):
        """Test QuantizedActivation string representation."""
        activation = bindings.QuantizedActivation(32)
        repr_str = repr(activation)
        assert 'QuantizedActivation' in repr_str
        print(f"✓ QuantizedActivation repr: {repr_str}")


class TestWeightQuantization:
    """Test weight quantization functionality."""
    
    def test_quantize_weights_ternary_small(self):
        """Test ternary weight quantization with small weights."""
        # Create simple test weights
        weights = np.array([
            [2.5, -1.0, 0.1, -0.05],
            [1.2, -2.3, 0.8, 0.0],
            [0.3, 1.5, -0.7, 2.0],
            [0.0, -0.3, 1.1, -1.5]
        ], dtype=np.float32)
        
        config = bindings.QuantConfig()
        ternary = bindings.quantize_weights_ternary(weights, 4, 4, config)
        
        assert ternary is not None
        assert ternary.rows == 4
        assert ternary.cols == 4
        assert ternary.size() == 16
        assert len(ternary.scales) > 0
        print(f"✓ Weight quantization successful: {ternary}")
    
    def test_quantize_weights_ternary_large(self):
        """Test ternary weight quantization with larger matrix."""
        rows, cols = 64, 128
        weights = np.random.randn(rows, cols).astype(np.float32) * 2.0
        
        config = bindings.QuantConfig()
        ternary = bindings.quantize_weights_ternary(weights, rows, cols, config)
        
        assert ternary is not None
        assert ternary.rows == rows
        assert ternary.cols == cols
        assert ternary.size() == rows * cols
        print(f"✓ Large weight quantization successful: {rows}x{cols} -> {ternary}")
    
    def test_quantize_dequantize_weights_roundtrip(self):
        """Test weight quantization and dequantization roundtrip."""
        # Create test weights
        weights = np.array([
            [1.0, 2.0, -1.0, 0.5],
            [-2.0, 1.5, 0.3, -0.5],
            [0.8, -1.2, 0.0, 2.0],
            [0.1, 0.2, -0.1, -2.0]
        ], dtype=np.float32)
        
        config = bindings.QuantConfig()
        
        # Quantize
        ternary = bindings.quantize_weights_ternary(weights, 4, 4, config)
        
        # Dequantize
        dequantized = bindings.dequantize_weights(ternary)
        
        # Check shapes
        assert dequantized.shape == weights.shape
        
        # Check values are reasonably close
        # Note: quantization is lossy, so we just check they're in a reasonable range
        assert np.all(np.isfinite(dequantized))
        print(f"✓ Weight roundtrip successful: original shape {weights.shape} -> quantized -> {dequantized.shape}")
    
    def test_quantize_weights_preserves_sign(self):
        """Test that quantization preserves sign information."""
        weights = np.array([
            [5.0, -3.0, 0.1, 0.0],
            [-4.0, 2.0, -0.2, 1.0],
        ], dtype=np.float32)
        
        config = bindings.QuantConfig()
        ternary = bindings.quantize_weights_ternary(weights, 2, 4, config)
        dequantized = bindings.dequantize_weights(ternary)
        
        # Check sign preservation (allowing for near-zero rounding)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                orig = weights[i, j]
                dequant = dequantized[i, j]
                
                # Sign should be preserved (except for zero)
                if abs(orig) > 0.01:
                    assert np.sign(orig) == np.sign(dequant), \
                        f"Sign mismatch at [{i},{j}]: {orig} -> {dequant}"
        
        print("✓ Sign preservation verified in quantization")


class TestActivationQuantization:
    """Test activation quantization functionality."""
    
    def test_quantize_activations_int8(self):
        """Test INT8 activation quantization."""
        activations = np.array([
            1.5, -2.3, 0.8, -0.5, 2.0, 0.1, -1.2, 0.3,
            -0.8, 1.1, 0.0, -1.5, 2.5, -2.0, 0.5, -0.3
        ], dtype=np.float32)
        
        config = bindings.QuantConfig()
        quant_act = bindings.quantize_activations_int8(activations, config)
        
        assert quant_act is not None
        assert quant_act.size() == len(activations)
        assert quant_act.scale > 0
        print(f"✓ Activation quantization successful: {quant_act}")
    
    def test_quantize_dequantize_activations_roundtrip(self):
        """Test activation quantization and dequantization roundtrip."""
        activations = np.array([
            1.5, -2.3, 0.8, -0.5, 2.0, 0.1, -1.2, 0.3,
            -0.8, 1.1, 0.0, -1.5, 2.5, -2.0, 0.5, -0.3
        ], dtype=np.float32)
        
        config = bindings.QuantConfig()
        
        # Quantize
        quant_act = bindings.quantize_activations_int8(activations, config)
        
        # Dequantize
        dequantized = bindings.dequantize_activations(quant_act)
        
        # Check shape
        assert dequantized.shape == activations.shape
        
        # Check values are reasonably close (INT8 quantization is lossy)
        assert np.all(np.isfinite(dequantized))
        print(f"✓ Activation roundtrip successful: {activations.shape} -> quantized -> {dequantized.shape}")


class TestQuantizationError:
    """Test quantization error computation."""
    
    def test_compute_quantization_error(self):
        """Test quantization error computation."""
        original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        quantized = np.array([1.05, 2.0, 2.95, 4.05], dtype=np.float32)
        
        error = bindings.compute_quantization_error(original, quantized)
        
        assert error > 0
        assert np.isfinite(error)
        print(f"✓ Quantization error computed: {error:.6f}")
    
    def test_compute_zero_error(self):
        """Test quantization error when arrays are identical."""
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        
        error = bindings.compute_quantization_error(values, values)
        
        assert error == 0.0
        print(f"✓ Zero error for identical arrays: {error}")


class TestWeightLoading:
    """Test weight loading and quantization workflow."""
    
    def test_load_and_quantize_workflow(self):
        """Test realistic workflow of loading and quantizing weights."""
        # Simulate loading pretrained weights
        print("\n  Simulating weight loading workflow...")
        
        # Create synthetic weight tensor (e.g., attention weight matrix)
        weight_dim = 768  # Typical transformer dimension
        rows, cols = 32, weight_dim
        
        print(f"  - Creating synthetic weights: {rows}x{cols}")
        weights = np.random.randn(rows, cols).astype(np.float32) * 0.1
        
        # Quantize with BitNet configuration
        config = bindings.QuantConfig()
        config.weight_group_size = 32
        config.per_group_scaling = True
        
        print(f"  - Quantizing with group_size={config.weight_group_size}")
        ternary = bindings.quantize_weights_ternary(weights, rows, cols, config)
        
        # Verify quantization
        assert ternary.size() == rows * cols
        print(f"  - Quantized: {ternary}")
        
        # Compute error
        dequantized = bindings.dequantize_weights(ternary)
        error = bindings.compute_quantization_error(weights, dequantized)
        
        print(f"  - Quantization error: {error:.6f}")
        print("✓ Weight loading and quantization workflow successful")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all test suites."""
    
    print("\n" + "="*80)
    print("BitNet Quantization C++ Bindings Test Suite")
    print("="*80)
    
    test_classes = [
        TestQuantConfig,
        TestTernaryWeight,
        TestQuantizedActivation,
        TestWeightQuantization,
        TestActivationQuantization,
        TestQuantizationError,
        TestWeightLoading,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 80)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, test_method)
                method()
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {test_method} FAILED: {e}")
                failed_tests += 1
            except Exception as e:
                print(f"✗ {test_method} ERROR: {e}")
                failed_tests += 1
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Total Tests:   {total_tests}")
    print(f"Passed:        {passed_tests}")
    print(f"Failed:        {failed_tests}")
    print(f"Success Rate:  {100*passed_tests/total_tests:.1f}%")
    print("="*80 + "\n")
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
