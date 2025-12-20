#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to run benchmark with proper encoding
"""
import subprocess
import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    # Change code page to UTF-8 on Windows
    os.system('chcp 65001 > nul 2>&1')
    
# Run the benchmark script
result = subprocess.run(
    [sys.executable, 'tests/benchmark_phase2_comprehensive.py'],
    cwd='.',
    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
)

sys.exit(result.returncode)
