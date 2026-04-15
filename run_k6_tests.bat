@echo off
REM Ryzanstein Phase 5 Load Testing - k6 Execution Script
REM This script adds k6 to PATH and runs the load tests

setlocal enabledelayedexpansion

echo ============================================================
echo  Phase 5: Load Testing Execution
echo  Ryzanstein LLM API
echo ============================================================
echo.

REM Set k6 location
set K6_PATH=C:\Users\sgbil\Downloads\k6-v0.50.0-windows-amd64\k6-v0.50.0-windows-amd64
set K6_EXE=%K6_PATH%\k6.exe

REM Verify k6 exists
if not exist "%K6_EXE%" (
    echo ERROR: k6.exe not found at:
    echo %K6_EXE%
    echo.
    echo Please verify the k6 download location.
    echo Expected: C:\Users\sgbil\Downloads\k6-v0.50.0-windows-amd64\k6-v0.50.0-windows-amd64\k6.exe
    pause
    exit /b 1
)

echo [OK] k6 found at: %K6_EXE%
echo.

REM Add k6 to PATH for this session
set PATH=%K6_PATH%;%PATH%

REM Verify k6 works
echo [CHECK] Verifying k6 version...
"%K6_EXE%" version
if errorlevel 1 (
    echo ERROR: k6 command failed
    pause
    exit /b 1
)
echo.

REM Navigate to project directory
cd /d s:\Ryot
if errorlevel 1 (
    echo ERROR: Cannot navigate to s:\Ryot
    pause
    exit /b 1
)

echo [OK] Working directory: %cd%
echo.

REM Verify API is running
echo [CHECK] Verifying API health...
curl -s http://localhost:8000/health > nul 2>&1
if errorlevel 1 (
    echo WARNING: API may not be responding
    echo Make sure port-forward is active:
    echo   kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000
    echo.
)

REM Ask user which tests to run
echo ============================================================
echo  Which tests would you like to run?
echo ============================================================
echo.
echo  1. Smoke Test Only (30 seconds)
echo  2. Quick Validation (Smoke + Load, ~6 minutes)
echo  3. Full Suite (All 5 tests, ~2 hours)
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto smoke_only
if "%choice%"=="2" goto quick_validation
if "%choice%"=="3" goto full_suite
echo Invalid choice. Exiting.
pause
exit /b 1

:smoke_only
echo.
echo Starting SMOKE TEST...
echo ============================================================
"%K6_EXE%" run load_test_smoke.js
pause
exit /b 0

:quick_validation
echo.
echo Starting QUICK VALIDATION (Smoke + Load)...
echo ============================================================
echo.
echo [1/2] SMOKE TEST (30 seconds)
echo ============================================================
"%K6_EXE%" run load_test_smoke.js
if errorlevel 1 (
    echo Smoke test failed. Stopping.
    pause
    exit /b 1
)
echo.
echo [PASS] Smoke test completed successfully
echo.
pause
echo.
echo [2/2] LOAD TEST (5 minutes)
echo ============================================================
"%K6_EXE%" run load_test_load.js
if errorlevel 1 (
    echo Load test failed.
    pause
    exit /b 1
)
echo.
echo [PASS] Load test completed successfully
echo.
pause
exit /b 0

:full_suite
echo.
echo Starting FULL TEST SUITE (All 5 tests, ~2 hours)
echo ============================================================
echo.
echo [1/5] SMOKE TEST (30 seconds)
echo ============================================================
"%K6_EXE%" run load_test_smoke.js
if errorlevel 1 (
    echo Smoke test failed. Stopping.
    pause
    exit /b 1
)
echo [PASS] Smoke test completed
echo.
pause

echo [2/5] LOAD TEST (5 minutes)
echo ============================================================
"%K6_EXE%" run load_test_load.js
if errorlevel 1 (
    echo Load test failed. Stopping.
    pause
    exit /b 1
)
echo [PASS] Load test completed
echo.
pause

echo [3/5] SPIKE TEST (5 minutes)
echo ============================================================
"%K6_EXE%" run load_test_spike.js
if errorlevel 1 (
    echo Spike test failed. Stopping.
    pause
    exit /b 1
)
echo [PASS] Spike test completed
echo.
pause

echo [4/5] STRESS TEST (30 minutes)
echo ============================================================
echo NOTE: This test will take ~30 minutes. You can monitor progress in the output above.
echo ============================================================
"%K6_EXE%" run load_test_stress.js
if errorlevel 1 (
    echo Stress test failed. Stopping.
    pause
    exit /b 1
)
echo [PASS] Stress test completed
echo.
pause

echo [5/5] ENDURANCE TEST (60 minutes)
echo ============================================================
echo NOTE: This test will take ~60 minutes. You can monitor progress in the output above.
echo ============================================================
"%K6_EXE%" run load_test_endurance.js
if errorlevel 1 (
    echo Endurance test failed.
    pause
    exit /b 1
)
echo [PASS] Endurance test completed
echo.
echo ============================================================
echo ALL TESTS COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo Results are displayed above. You can:
echo   1. Review the metrics (latency, error rates, throughput)
echo   2. Compare against SLO thresholds
echo   3. Proceed to Phase 6 (Integration Testing)
echo.
pause
exit /b 0
