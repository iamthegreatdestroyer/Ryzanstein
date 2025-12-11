import sys
import os
import ctypes

print("Current directory:", os.getcwd())
print("Python path before:", sys.path)

sys.path.insert(0, 'build/python')

print("Python path after:", sys.path)

# Try loading the DLL directly with ctypes
dll_path = 'build/python/ryzen_llm/test_module.pyd'
print(f"Trying to load DLL directly: {dll_path}")
try:
    dll = ctypes.CDLL(dll_path)
    print("DLL loaded successfully with ctypes")
except Exception as e:
    print(f"DLL load failed: {e}")

try:
    print("Attempting to import test_module directly...")
    import test_module
    print('Direct import successful')
    print('Available functions:', [attr for attr in dir(test_module) if not attr.startswith('_')])
    print(f"test_function result: {test_module.test_function()}")
except ImportError as e:
    print(f'Direct import failed: {e}')
    
    # Try importing as a package module
    try:
        print("Attempting to import as ryzen_llm.test_module...")
        import ryzen_llm.test_module
        print('Package import successful')
        print('Available functions:', [attr for attr in dir(ryzen_llm.test_module) if not attr.startswith('_')])
        print(f"test_function result: {ryzen_llm.test_module.test_function()}")
    except ImportError as e2:
        print(f'Package import failed: {e2}')
        import traceback
        traceback.print_exc()

try:
    print("Attempting to import ryzen_llm package...")
    import ryzen_llm
    print('Package import successful')
    print('Available functions:', [attr for attr in dir(ryzen_llm) if not attr.startswith('_')])
except ImportError as e:
    print(f'Package import failed: {e}')
    import traceback
    traceback.print_exc()