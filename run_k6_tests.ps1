# Ryzanstein Phase 5 Load Testing - k6 Execution Script (PowerShell)
# Usage: .\run_k6_tests.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Phase 5: Load Testing Execution" -ForegroundColor Cyan
Write-Host "  Ryzanstein LLM API" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Set k6 location
$k6Path = "C:\Users\sgbil\Downloads\k6-v0.50.0-windows-amd64\k6-v0.50.0-windows-amd64"
$k6Exe = "$k6Path\k6.exe"

# Verify k6 exists
if (!(Test-Path $k6Exe)) {
    Write-Host "ERROR: k6.exe not found at:" -ForegroundColor Red
    Write-Host $k6Exe -ForegroundColor Red
    Write-Host ""
    Write-Host "Please verify the k6 download location." -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] k6 found at: $k6Exe" -ForegroundColor Green
Write-Host ""

# Add k6 to PATH for this session
$env:PATH = "$k6Path;$env:PATH"

# Verify k6 works
Write-Host "[CHECK] Verifying k6 version..." -ForegroundColor Yellow
& $k6Exe version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: k6 command failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Navigate to project directory
Set-Location "s:\Ryot"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Cannot navigate to s:\Ryot" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Working directory: $(Get-Location)" -ForegroundColor Green
Write-Host ""

# Verify API is running
Write-Host "[CHECK] Verifying API health..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -ErrorAction SilentlyContinue
    Write-Host "[OK] API is responding" -ForegroundColor Green
} catch {
    Write-Host "WARNING: API may not be responding" -ForegroundColor Yellow
    Write-Host "Make sure port-forward is active:" -ForegroundColor Yellow
    Write-Host "  kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000" -ForegroundColor Yellow
}
Write-Host ""

# Menu
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Which tests would you like to run?" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Smoke Test Only (30 seconds)" -ForegroundColor White
Write-Host "  2. Quick Validation (Smoke + Load, ~6 minutes)" -ForegroundColor White
Write-Host "  3. Full Suite (All 5 tests, ~2 hours)" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Starting SMOKE TEST..." -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Cyan
        & $k6Exe run load_test_smoke.js
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "[PASS] Smoke test completed successfully" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "[FAIL] Smoke test failed" -ForegroundColor Red
        }
    }

    "2" {
        Write-Host ""
        Write-Host "Starting QUICK VALIDATION (Smoke + Load)..." -ForegroundColor Green

        Write-Host ""
        Write-Host "[1/2] SMOKE TEST (30 seconds)" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Cyan
        & $k6Exe run load_test_smoke.js
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "[FAIL] Smoke test failed. Stopping." -ForegroundColor Red
            exit 1
        }
        Write-Host ""
        Write-Host "[PASS] Smoke test completed successfully" -ForegroundColor Green

        Write-Host ""
        Write-Host "[2/2] LOAD TEST (5 minutes)" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Cyan
        & $k6Exe run load_test_load.js
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "[FAIL] Load test failed" -ForegroundColor Red
            exit 1
        }
        Write-Host ""
        Write-Host "[PASS] Load test completed successfully" -ForegroundColor Green
        Write-Host ""
        Write-Host "Quick validation complete!" -ForegroundColor Green
    }

    "3" {
        Write-Host ""
        Write-Host "Starting FULL TEST SUITE (All 5 tests, ~2 hours)..." -ForegroundColor Green

        # Smoke Test
        Write-Host ""
        Write-Host "[1/5] SMOKE TEST (30 seconds)" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Cyan
        & $k6Exe run load_test_smoke.js
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] Smoke test failed. Stopping." -ForegroundColor Red
            exit 1
        }
        Write-Host "[PASS] Smoke test completed" -ForegroundColor Green

        # Load Test
        Write-Host ""
        Write-Host "[2/5] LOAD TEST (5 minutes)" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Cyan
        & $k6Exe run load_test_load.js
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] Load test failed. Stopping." -ForegroundColor Red
            exit 1
        }
        Write-Host "[PASS] Load test completed" -ForegroundColor Green

        # Spike Test
        Write-Host ""
        Write-Host "[3/5] SPIKE TEST (5 minutes)" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Cyan
        & $k6Exe run load_test_spike.js
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] Spike test failed. Stopping." -ForegroundColor Red
            exit 1
        }
        Write-Host "[PASS] Spike test completed" -ForegroundColor Green

        # Stress Test
        Write-Host ""
        Write-Host "[4/5] STRESS TEST (30 minutes)" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "NOTE: This test will take ~30 minutes. Monitor progress above." -ForegroundColor Yellow
        & $k6Exe run load_test_stress.js
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] Stress test failed. Stopping." -ForegroundColor Red
            exit 1
        }
        Write-Host "[PASS] Stress test completed" -ForegroundColor Green

        # Endurance Test
        Write-Host ""
        Write-Host "[5/5] ENDURANCE TEST (60 minutes)" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "NOTE: This test will take ~60 minutes. Monitor progress above." -ForegroundColor Yellow
        & $k6Exe run load_test_endurance.js
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] Endurance test failed." -ForegroundColor Red
            exit 1
        }
        Write-Host "[PASS] Endurance test completed" -ForegroundColor Green

        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "ALL TESTS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Results are displayed above. You can:" -ForegroundColor Green
        Write-Host "  1. Review the metrics (latency, error rates, throughput)" -ForegroundColor Green
        Write-Host "  2. Compare against SLO thresholds" -ForegroundColor Green
        Write-Host "  3. Proceed to Phase 6 (Integration Testing)" -ForegroundColor Green
        Write-Host ""
    }

    default {
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}
