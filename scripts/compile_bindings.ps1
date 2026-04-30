#Requires -Version 7.0
<#
.SYNOPSIS
    Automated C++ binding compilation for Ryzanstein LLM.

.DESCRIPTION
    Compiles the pybind11 C++ engine bindings (ryzen_llm_bindings.pyd / .so)
    enabling real BitNet inference instead of the mock engine.

    Resolves: Blocker #1 (C++ Bindings) and Blocker #2 (Real Inference).

.PARAMETER Clean
    Remove the build directory before building (default: false).

.PARAMETER EnableAVX512
    Enable AVX-512 + VNNI SIMD optimizations (default: true).

.EXAMPLE
    .\scripts\compile_bindings.ps1
    .\scripts\compile_bindings.ps1 -Clean -EnableAVX512:$false
#>
param(
    [switch] $Clean,
    [bool]   $EnableAVX512 = $true
)

$ErrorActionPreference = "Stop"
$Root     = "s:\Ryot\RYZEN-LLM"
$BuildDir = "$Root\build"

function Write-Section([string]$title) {
    Write-Host "`n$('='*60)" -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host "$('='*60)" -ForegroundColor Cyan
}

function Test-Prerequisite([string]$Name, [scriptblock]$Check, [string]$InstallHint) {
    try {
        & $Check | Out-Null
        Write-Host "  [OK] $Name" -ForegroundColor Green
    } catch {
        Write-Host "  [MISSING] $Name — $InstallHint" -ForegroundColor Red
        throw "Missing prerequisite: $Name"
    }
}

# ─────────────────────────────────────────────────
# STEP 1: Prerequisite Check
# ─────────────────────────────────────────────────
Write-Section "Step 1: Prerequisites"

Test-Prerequisite "CMake 3.20+" {
    $v = (cmake --version 2>&1 | Select-Object -First 1) -replace 'cmake version ', ''
    $parts = $v.Split('.')
    if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 20)) {
        throw "CMake too old: $v"
    }
    Write-Host "    cmake $v" -ForegroundColor DarkGray
} "Install CMake >= 3.20 from https://cmake.org/"

Test-Prerequisite "Python 3.11+" {
    $v = python --version 2>&1
    Write-Host "    $v" -ForegroundColor DarkGray
} "Install Python 3.11+ from https://python.org/"

Test-Prerequisite "pybind11" {
    python -c "import pybind11; print(pybind11.get_cmake_dir())"
} "pip install pybind11"

Test-Prerequisite "C++ Compiler (cl.exe or g++)" {
    $cl  = Get-Command cl.exe  -ErrorAction SilentlyContinue
    $gxx = Get-Command g++     -ErrorAction SilentlyContinue
    if (-not $cl -and -not $gxx) { throw "No C++ compiler" }
} "Install Visual Studio 2022 or MinGW-w64"

if (-not (Test-Path "$Root\CMakeLists.txt")) {
    Write-Host "  [WARN] CMakeLists.txt not found at $Root — using workspace root" -ForegroundColor Yellow
    $Root = "s:\Ryot"
}

# ─────────────────────────────────────────────────
# STEP 2: Clean (optional)
# ─────────────────────────────────────────────────
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Section "Step 2: Clean Build"
    Remove-Item $BuildDir -Recurse -Force
    Write-Host "  Removed: $BuildDir" -ForegroundColor Gray
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# ─────────────────────────────────────────────────
# STEP 3: CMake Configure
# ─────────────────────────────────────────────────
Write-Section "Step 3: CMake Configure"

$pybind11Dir = python -c "import pybind11; print(pybind11.get_cmake_dir())"
$pythonExe   = (python -c "import sys; print(sys.executable)")
$avx512Flag  = if ($EnableAVX512) { "ON" } else { "OFF" }

$cmakeArgs = @(
    $Root,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DENABLE_AVX512=$avx512Flag",
    "-DENABLE_VNNI=$avx512Flag",
    "-DENABLE_PYBIND11=ON",
    "-Dpybind11_DIR=$pybind11Dir",
    "-DPYTHON_EXECUTABLE=$pythonExe"
)

Write-Host "  cmake $($cmakeArgs -join ' ')" -ForegroundColor DarkGray
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)" }

# ─────────────────────────────────────────────────
# STEP 4: Build
# ─────────────────────────────────────────────────
Write-Section "Step 4: Build"

$cores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "  Building with $cores cores..." -ForegroundColor DarkGray

cmake --build $BuildDir --config Release -j $cores
if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)" }

# ─────────────────────────────────────────────────
# STEP 5: Locate Output
# ─────────────────────────────────────────────────
Write-Section "Step 5: Locate Binding"

# Search for the compiled module (platform-specific name)
$bindingFiles = Get-ChildItem -Recurse $BuildDir -Include "ryzen_llm_bindings.pyd","ryzen_llm_bindings*.so" -ErrorAction SilentlyContinue
if (-not $bindingFiles) {
    # Fallback: any pyd in build dir
    $bindingFiles = Get-ChildItem -Recurse $BuildDir -Filter "*.pyd" -ErrorAction SilentlyContinue
}

if (-not $bindingFiles) {
    throw "Compiled binding not found in $BuildDir — check build output above"
}

$bindingPath = $bindingFiles[0].FullName
Write-Host "  Found: $bindingPath" -ForegroundColor Green

# Copy to expected location
$destDir = "$Root\build\python"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null
Copy-Item $bindingPath $destDir -Force
Write-Host "  Copied to: $destDir" -ForegroundColor Green

# ─────────────────────────────────────────────────
# STEP 6: Smoke Test
# ─────────────────────────────────────────────────
Write-Section "Step 6: Smoke Test"

$smokeScript = @"
import sys
sys.path.insert(0, r'$destDir')
try:
    from ryzen_llm_bindings import BitNetEngine, ModelConfig
    cfg = ModelConfig(vocab_size=32000, hidden_size=2048, num_layers=24, num_heads=16)
    engine = BitNetEngine(cfg)
    simd = engine.simd_capabilities()
    print(f'  Engine: OK')
    print(f'  SIMD:   {simd}')
    print('SMOKE_TEST=PASS')
except ImportError as e:
    print(f'  ImportError: {e}')
    print('SMOKE_TEST=FAIL')
except Exception as e:
    print(f'  Error: {e}')
    print('SMOKE_TEST=FAIL')
"@

$result = python -c $smokeScript 2>&1
Write-Host $result

if ($result -notmatch "SMOKE_TEST=PASS") {
    Write-Host "`n  [WARN] Smoke test failed — bindings may have init issues." -ForegroundColor Yellow
    Write-Host "  Run: python scripts\verify_inference.py --smoke-only" -ForegroundColor Yellow
} else {
    Write-Host "`n  Smoke test PASSED" -ForegroundColor Green
}

# ─────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────
Write-Section "Done"
Write-Host @"
  Binding: $bindingPath
  Copied:  $destDir

  Next steps:
    1. python scripts\verify_inference.py --benchmark
    2. python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"@ -ForegroundColor Green
