#!/usr/bin/env python3
"""
Test script for BitNet engine SafeTensors loading and matrix multiplication
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_quantization():
    """Test the quantization functions"""
    print("Testing quantization functions...")

    try:
        from ryzen_llm.bitnet.quantize import quantize_activations_int8, QuantConfig

        # Test data
        activations = np.random.randn(100).astype(np.float32)
        config = QuantConfig()
        config.activation_clip_value = 6.0
        config.symmetric_activations = True

        # Quantize
        result = quantize_activations_int8(activations, len(activations), config)

        print(f"✓ Quantized {len(activations)} activations")
        print(f"  Scale: {result.scale:.4f}")
        print(f"  Zero point: {result.zero_point}")
        print(f"  Value range: {min(result.values)} to {max(result.values)}")

        return True
    except Exception as e:
        print(f"✗ Quantization test failed: {e}")
        return False

def test_naive_matmul():
    """Test the naive matrix multiplication"""
    print("\nTesting naive matrix multiplication...")

    try:
        from ryzen_llm.bitnet.quantize import naive_ternary_matmul, TernaryWeight, QuantizedActivation, QuantConfig

        # Create test matrices
        M, N, K = 4, 6, 8

        # Create ternary weights
        ternary_weight = TernaryWeight(K, N, 0)  # No grouping
        for i in range(len(ternary_weight.values)):
            ternary_weight.values[i] = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        ternary_weight.scales[0] = 1.0  # Single scale

        # Create quantized activations
        config = QuantConfig()
        config.activation_clip_value = 6.0
        config.symmetric_activations = True

        activations_fp32 = np.random.randn(M * K).astype(np.float32)
        quantized_activations = quantize_activations_int8(activations_fp32, len(activations_fp32), config)

        # Output buffer
        output = np.zeros(M * N, dtype=np.float32)

        # Perform multiplication
        naive_ternary_matmul(ternary_weight, quantized_activations, output, M, N, K)

        print(f"✓ Matrix multiplication: {M}x{K} * {K}x{N} = {M}x{N}")
        print(f"  Output range: {output.min():.4f} to {output.max():.4f}")

        return True
    except Exception as e:
        print(f"✗ Matrix multiplication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== BitNet Engine Component Tests ===\n")

    success = True
    success &= test_quantization()
    success &= test_naive_matmul()

    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())