"""
Ryzen LLM Python Package

This package provides Python bindings for the Ryzen LLM C++ library,
enabling efficient inference on AMD Ryzen processors with AVX-512 support.
"""

try:
    from .ryzen_llm_bindings import *
except ImportError:
    import ryzen_llm_bindings
    from ryzen_llm_bindings import *