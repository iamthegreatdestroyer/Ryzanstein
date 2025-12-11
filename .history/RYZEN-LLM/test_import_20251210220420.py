import sys
import os

print("Current directory:", os.getcwd())
print("Python path before:", sys.path)

sys.path.insert(0, 'build/python')

print("Python path after:", sys.path)

try:
    print("Attempting to import ryzen_llm_bindings directly...")
    import ryzen_llm_bindings
    print('Direct import successful')
    print('Available functions:', [attr for attr in dir(ryzen_llm_bindings) if not attr.startswith('_')])
except ImportError as e:
    print(f'Direct import failed: {e}')
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