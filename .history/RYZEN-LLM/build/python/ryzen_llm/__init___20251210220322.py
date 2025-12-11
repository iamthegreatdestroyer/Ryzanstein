# Ryzen LLM Python Package

# Import the C++ extension
try:
    from .ryzen_llm_bindings import *
except ImportError:
    # Fallback for direct import
    import ryzen_llm_bindings
    from ryzen_llm_bindings import *