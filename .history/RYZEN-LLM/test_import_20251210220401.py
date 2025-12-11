import sys
import os

print("Current directory:", os.getcwd())
print("Python path before:", sys.path)

sys.path.insert(0, 'build/python')

print("Python path after:", sys.path)

try:
    print("Attempting to import ryzen_llm...")
    import ryzen_llm
    print('Import successful')
    print('Available functions:', [attr for attr in dir(ryzen_llm) if not attr.startswith('_')])
except ImportError as e:
    print(f'Import failed: {e}')
    import traceback
    traceback.print_exc()