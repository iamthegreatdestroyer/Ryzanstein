#!/usr/bin/env python3
"""
Quick test to validate C++ extension loads and functions
"""
import sys
import os
from pathlib import Path

# Add build directory to Python path
build_dir = Path(__file__).parent / "build" / "python"
sys.path.insert(0, str(build_dir))

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}")
print(f"Looking for extension in: {build_dir}")

try:
    # Try to import the compiled extension
    import ryzen_llm.ryzen_llm_bindings as bindings
    print("✅ Successfully imported ryzen_llm_bindings C++ extension")
    
    # Test basic function
    result = bindings.test_function()
    print(f"✅ test_function() returned: {result}")
    assert result == 42, f"Expected 42, got {result}"
    
    # Test quantization functions
    print("\n--- Testing C++ quantization functions ---")
    
    # Test simple scalar quantization
    scalar_result = bindings.test_quantize_scalar()
    print(f"✅ test_quantize_scalar() returned: {scalar_result}")
    
    # Test simple loop
    loop_result = bindings.test_simple_loop()
    print(f"✅ test_simple_loop() returned: {loop_result} (expected: 136)")
    assert loop_result == 136, f"Expected 136, got {loop_result}"
    
    # Test nested loops
    nested_result = bindings.test_nested_loops()
    print(f"✅ test_nested_loops() returned: {nested_result} (expected: 16)")
    assert nested_result == 16, f"Expected 16, got {nested_result}"
    
    print("\n✅ All C++ extension tests passed!")
    print("\nC++ extension is ready for use.")
    
except ImportError as e:
    print(f"❌ Failed to import C++ extension: {e}")
    print(f"\nAvailable files in {build_dir}:")
    if build_dir.exists():
        for f in build_dir.rglob("*"):
            if f.is_file():
                print(f"  {f.relative_to(build_dir)}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error testing C++ extension: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
