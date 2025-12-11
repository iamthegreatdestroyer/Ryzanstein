import sys
sys.path.insert(0, 'build/python')

try:
    import ryzen_llm
    print('Import successful')
    print('Available functions:', [attr for attr in dir(ryzen_llm) if not attr.startswith('_')])
except ImportError as e:
    print(f'Import failed: {e}')