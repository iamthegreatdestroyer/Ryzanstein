@echo off
REM Phase 6: Quick Start - Opens 5 Monitoring Terminals
REM This batch file launches the PowerShell script to set up Phase 6

setlocal enabledelayedexpansion

echo ============================================================
echo  PHASE 6: INTEGRATED MONITORING SETUP
echo ============================================================
echo.

REM Get the script path
set SCRIPT=s:\Ryot\start_phase6_monitoring.ps1

REM Check if PowerShell script exists
if not exist "%SCRIPT%" (
    echo ERROR: Could not find %SCRIPT%
    pause
    exit /b 1
)

echo Starting Phase 6 monitoring setup...
echo.
echo This will:
echo   1. Open VSCode integrated terminals
echo   2. Start 5 port-forwards (API, Prometheus, Grafana, Jaeger, AlertMgr)
echo   3. Guide you through Phase 6 verification
echo.
pause

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "%SCRIPT%"

pause
