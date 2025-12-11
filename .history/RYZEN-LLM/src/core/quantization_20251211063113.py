"""
High-level Python quantization API for BitNet.

This module provides a Pythonic interface to the C++ quantization engine,
handling common workflows and providing sensible defaults.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings

from ryzen_llm.ryzen_llm_bindings import (
    QuantConfig as CppQuantConfig,
    TernaryWeight,
    QuantizedActivation,
    quantize_weights_ternary as cpp_quantize_weights,
    quantize_activations_int8 as cpp_quantize_activations,
    dequantize_weights as cpp_dequantize_weights,
    dequantize_activations as cpp_dequantize_activations,
    compute_quantization_error as cpp_compute_error,
)


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class QuantizationConfig:
    """
    Configuration for BitNet quantization.
    
    Attributes:
        weight_group_size: Size of groups for per-group scaling (default: 128)
        per_group_scaling: Enable per-group scaling (default: True)
        activation_clip_value: Clipping threshold for activations (default: 6.0)
        symmetric_activations: Use symmetric INT8 quantization (default: True)
        dtype_weights: Data type for input weights (default: np.float32)
        dtype_activations: Data type for input activations (default: np.float32)
    """
    weight_group_size: int = 128
    per_group_scaling: bool = True
    activation_clip_value: float = 6.0
    symmetric_activations: bool = True
    dtype_weights: np.dtype = np.float32
    dtype_activations: np.dtype = np.float32
    
    def to_cpp_config(self) -> CppQuantConfig:
        """Convert to C++ QuantConfig."""
        config = CppQuantConfig()
        config.weight_group_size = self.weight_group_size
        config.per_group_scaling = self.per_group_scaling
        config.activation_clip_value = self.activation_clip_value
        config.symmetric_activations = self.symmetric_activations
        return config
    
    @classmethod
    def from_cpp_config(cls, cpp_config: CppQuantConfig) -> "QuantizationConfig":
        """Create from C++ QuantConfig."""
        return cls(
            weight_group_size=cpp_config.weight_group_size,
            per_group_scaling=cpp_config.per_group_scaling,
            activation_clip_value=cpp_config.activation_clip_value,
            symmetric_activations=cpp_config.symmetric_activations,
        )
    
    def __repr__(self) -> str:
        return (f"QuantizationConfig("
                f"group_size={self.weight_group_size}, "
                f"per_group={self.per_group_scaling}, "
                f"clip={self.activation_clip_value})")


# ============================================================================
# Quantization Engine
# ============================================================================

class QuantizationEngine:
    """
    High-level interface to BitNet quantization.
    
    Handles weight quantization, activation quantization, and provides
    utilities for measuring quantization accuracy.
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize quantization engine.
        
        Args:
            config: QuantizationConfig instance (uses defaults if None)
        """
        self.config = config or QuantizationConfig()
        self._cpp_config = self.config.to_cpp_config()
        self._weight_cache: Dict[str, TernaryWeight] = {}
        self._activation_cache: Dict[str, QuantizedActivation] = {}
    
    def quantize_weights(
        self,
        weights: np.ndarray,
        name: Optional[str] = None,
        cache: bool = False,
    ) -> TernaryWeight:
        """
        Quantize weight matrix to ternary.
        
        Args:
            weights: FP32 weight array (can be 1D or 2D)
            name: Optional name for caching
            cache: Whether to cache the quantized result
            
        Returns:
            TernaryWeight object containing quantized values and scales
            
        Raises:
            ValueError: If weights are not FP32
            ValueError: If weights are empty
            
        Example:
            >>> engine = QuantizationEngine()
            >>> weights = np.random.randn(768, 3072).astype(np.float32)
            >>> ternary = engine.quantize_weights(weights, name="attn_weights")
            >>> print(f"Quantized: {ternary.rows}×{ternary.cols}")
        """
        if weights.dtype != np.float32:
            raise ValueError(
                f"Expected FP32 weights, got {weights.dtype}. "
                f"Convert with: weights.astype(np.float32)"
            )
        
        if weights.size == 0:
            raise ValueError("Cannot quantize empty weight array")
        
        # Check cache
        if cache and name and name in self._weight_cache:
            return self._weight_cache[name]
        
        # Handle 1D weights by reshaping
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        elif weights.ndim != 2:
            raise ValueError(
                f"Weights must be 1D or 2D, got shape {weights.ndim}D"
            )
        
        rows, cols = weights.shape
        
        # Quantize
        ternary = cpp_quantize_weights(weights, rows, cols, self._cpp_config)
        
        # Cache if requested
        if cache and name:
            self._weight_cache[name] = ternary
        
        return ternary
    
    def quantize_activations(
        self,
        activations: np.ndarray,
        name: Optional[str] = None,
        cache: bool = False,
    ) -> QuantizedActivation:
        """
        Quantize activation tensor to INT8.
        
        Args:
            activations: FP32 activation array
            name: Optional name for caching
            cache: Whether to cache the quantized result
            
        Returns:
            QuantizedActivation object containing quantized values and scale
            
        Raises:
            ValueError: If activations are not FP32
            ValueError: If activations are empty
            
        Example:
            >>> engine = QuantizationEngine()
            >>> acts = np.random.randn(1024).astype(np.float32)
            >>> quant_acts = engine.quantize_activations(acts)
            >>> print(f"Scale factor: {quant_acts.scale}")
        """
        if activations.dtype != np.float32:
            raise ValueError(
                f"Expected FP32 activations, got {activations.dtype}. "
                f"Convert with: activations.astype(np.float32)"
            )
        
        if activations.size == 0:
            raise ValueError("Cannot quantize empty activation array")
        
        # Check cache
        if cache and name and name in self._activation_cache:
            return self._activation_cache[name]
        
        # Flatten to 1D for quantization
        activations_1d = activations.ravel()
        
        # Quantize
        quant_act = cpp_quantize_activations(activations_1d, self._cpp_config)
        
        # Cache if requested
        if cache and name:
            self._activation_cache[name] = quant_act
        
        return quant_act
    
    def dequantize_weights(self, ternary: TernaryWeight) -> np.ndarray:
        """
        Recover FP32 weights from ternary representation.
        
        Args:
            ternary: TernaryWeight object
            
        Returns:
            FP32 numpy array of shape (rows, cols)
            
        Example:
            >>> ternary = engine.quantize_weights(weights)
            >>> recovered = engine.dequantize_weights(ternary)
            >>> error = np.mean((weights - recovered)**2)
        """
        return cpp_dequantize_weights(ternary)
    
    def dequantize_activations(self, quant_act: QuantizedActivation) -> np.ndarray:
        """
        Recover FP32 activations from INT8 representation.
        
        Args:
            quant_act: QuantizedActivation object
            
        Returns:
            FP32 numpy array
            
        Example:
            >>> quant_acts = engine.quantize_activations(acts)
            >>> recovered = engine.dequantize_activations(quant_acts)
        """
        return cpp_dequantize_activations(quant_act)
    
    def compute_error(
        self,
        original: np.ndarray,
        quantized: np.ndarray,
    ) -> float:
        """
        Compute mean squared error between original and quantized values.
        
        Args:
            original: Original FP32 array
            quantized: Quantized FP32 array (same shape as original)
            
        Returns:
            MSE error (float)
            
        Raises:
            ValueError: If array shapes don't match
            
        Example:
            >>> error = engine.compute_error(weights, recovered)
            >>> print(f"Quantization MSE: {error:.6f}")
        """
        if original.shape != quantized.shape:
            raise ValueError(
                f"Shape mismatch: original {original.shape} vs "
                f"quantized {quantized.shape}"
            )
        
        return float(cpp_compute_error(original, quantized))
    
    def quantize_and_measure(
        self,
        weights: np.ndarray,
        recover: bool = True,
    ) -> Dict[str, Union[TernaryWeight, np.ndarray, float]]:
        """
        Quantize weights and measure quantization error.
        
        Args:
            weights: FP32 weight array
            recover: Whether to compute error (requires dequantization)
            
        Returns:
            Dictionary with keys:
            - 'ternary': TernaryWeight object
            - 'recovered': Recovered FP32 array (if recover=True)
            - 'error': MSE error (if recover=True)
            - 'compression': Compression ratio
            
        Example:
            >>> result = engine.quantize_and_measure(weights)
            >>> print(f"Error: {result['error']:.6f}")
            >>> print(f"Compression: {result['compression']:.1f}×")
        """
        ternary = self.quantize_weights(weights)
        
        # Calculate compression ratio
        # Original: FP32 = 4 bytes/element
        # Ternary: 2 bits/weight + scales (FP32 per group)
        original_bytes = weights.nbytes
        ternary_bytes = ternary.size() // 4 + len(ternary.scales()) * 4  # Rough estimate
        
        result = {
            'ternary': ternary,
            'compression': original_bytes / max(ternary_bytes, 1),  # Avoid division by zero
        }
        
        if recover:
            recovered = self.dequantize_weights(ternary)
            error = self.compute_error(weights, recovered)
            result['recovered'] = recovered
            result['error'] = error
        
        return result
    
    def clear_cache(self, weights: bool = True, activations: bool = True):
        """
        Clear quantization caches.
        
        Args:
            weights: Clear weight cache
            activations: Clear activation cache
        """
        if weights:
            self._weight_cache.clear()
        if activations:
            self._activation_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_weights': len(self._weight_cache),
            'cached_activations': len(self._activation_cache),
        }


# ============================================================================
# Batch Quantization
# ============================================================================

class BatchQuantizer:
    """
    Quantize multiple weight matrices with consistent configuration.
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize batch quantizer.
        
        Args:
            config: QuantizationConfig instance
        """
        self.engine = QuantizationEngine(config)
    
    def quantize_dict(
        self,
        weights_dict: Dict[str, np.ndarray],
        measure_error: bool = False,
    ) -> Dict[str, Union[TernaryWeight, float]]:
        """
        Quantize a dictionary of weight matrices.
        
        Args:
            weights_dict: Dict mapping names to FP32 weight arrays
            measure_error: Whether to compute quantization errors
            
        Returns:
            Dictionary mapping names to quantized weights (and errors if requested)
            
        Example:
            >>> quantizer = BatchQuantizer()
            >>> weights = {
            ...     'layer1_weight': np.random.randn(768, 3072).astype(np.float32),
            ...     'layer2_weight': np.random.randn(3072, 768).astype(np.float32),
            ... }
            >>> ternary_weights = quantizer.quantize_dict(weights)
        """
        results = {}
        
        for name, weights in weights_dict.items():
            try:
                ternary = self.engine.quantize_weights(weights, name=name, cache=True)
                results[name] = ternary
                
                if measure_error:
                    recovered = self.engine.dequantize_weights(ternary)
                    error = self.engine.compute_error(weights, recovered)
                    results[f"{name}_error"] = error
                
            except Exception as e:
                warnings.warn(f"Failed to quantize {name}: {e}")
                continue
        
        return results
    
    def quantize_layer_weights(
        self,
        layer_dict: Dict[str, np.ndarray],
    ) -> Dict[str, TernaryWeight]:
        """
        Quantize all weight matrices in a transformer layer.
        
        Typical layer structure:
        - self_attn.q_proj: (hidden_size, hidden_size)
        - self_attn.k_proj: (hidden_size, hidden_size)
        - self_attn.v_proj: (hidden_size, hidden_size)
        - self_attn.o_proj: (hidden_size, hidden_size)
        - mlp.fc1: (hidden_size, intermediate_size)
        - mlp.fc2: (intermediate_size, hidden_size)
        
        Args:
            layer_dict: Dict of layer weights
            
        Returns:
            Dictionary of quantized weights with same keys
        """
        return self.quantize_dict(layer_dict, measure_error=False)


# ============================================================================
# Utilities
# ============================================================================

def create_default_config() -> QuantizationConfig:
    """Create default quantization configuration optimized for BitNet."""
    return QuantizationConfig(
        weight_group_size=128,
        per_group_scaling=True,
        activation_clip_value=6.0,
        symmetric_activations=True,
    )


def create_aggressive_config() -> QuantizationConfig:
    """Create aggressive quantization config for maximum compression."""
    return QuantizationConfig(
        weight_group_size=256,  # Larger groups = more compression
        per_group_scaling=True,
        activation_clip_value=4.0,  # Tighter clipping
        symmetric_activations=True,
    )


def estimate_model_size(
    weights_dict: Dict[str, Tuple[int, int]],
    original_dtype_bits: int = 32,
) -> Dict[str, float]:
    """
    Estimate model size before and after quantization.
    
    Args:
        weights_dict: Dict mapping names to (rows, cols) tuples
        original_dtype_bits: Bits per element in original format
        
    Returns:
        Dictionary with size estimates in MB
        
    Example:
        >>> weights_shapes = {
        ...     'attn_weights': (768, 768),
        ...     'mlp_weights': (768, 3072),
        ... }
        >>> sizes = estimate_model_size(weights_shapes)
        >>> print(f"Original: {sizes['original_mb']:.1f}MB")
        >>> print(f"Ternary: {sizes['ternary_mb']:.1f}MB")
    """
    total_elements = sum(rows * cols for rows, cols in weights_dict.values())
    original_bytes = total_elements * (original_dtype_bits // 8)
    
    # Ternary: 1.3 bits per element (1 bit value + 1.3 bits scaling)
    # Estimate: ~21% of original size
    ternary_bytes = original_bytes * 0.21
    
    return {
        'original_mb': original_bytes / (1024 ** 2),
        'ternary_mb': ternary_bytes / (1024 ** 2),
        'compression_ratio': original_bytes / ternary_bytes,
    }
