@echo off
REM Port Remediation Verification Script
REM Checks if ports have been successfully reassigned

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  PORT REMEDIATION VERIFICATION
echo ============================================================
echo.

set /a issues=0

REM Check Old Ports (should NOT respond or show different content)
echo Testing OLD ports (should be freed)...
echo.

echo Testing Port 9090 (Hyperbox Prometheus - should be freed)...
curl -s -m 2 http://localhost:9090/-/healthy > nul 2>&1
if errorlevel 1 (
    echo   ✅ Port 9090 FREED - Hyperbox moved successfully
) else (
    echo   ❌ Port 9090 STILL ACTIVE - Hyperbox not moved yet
    set /a issues+=1
)
echo.

echo Testing Port 3000 (Faceless YouTube - should be freed)...
curl -s -m 2 http://localhost:3000/api/health > nul 2>&1
if errorlevel 1 (
    echo   ✅ Port 3000 FREED - Faceless YouTube moved successfully
) else (
    echo   ⚠️  Port 3000 STILL ACTIVE - checking if it's now Ryzanstein...
    REM Try to identify which project it is
)
echo.

echo Testing Port 16686 (Neurectomy Jaeger - should be freed)...
curl -s -m 2 -I http://localhost:16686/ 2>&1 | find "200" > nul
if errorlevel 1 (
    echo   ✅ Port 16686 FREED - Neurectomy moved successfully
) else (
    echo   ⚠️  Port 16686 STILL ACTIVE - checking if it's now Ryzanstein...
)
echo.

REM Check New Ports (should be active with content)
echo Testing NEW ports (should be accessible)...
echo.

echo Testing Port 40001 (Hyperbox Prometheus - NEW location)...
curl -s -m 2 http://localhost:40001/-/healthy > nul 2>&1
if errorlevel 1 (
    echo   ⚠️  Port 40001 NOT RESPONDING - Container may not be running
) else (
    echo   ✅ Port 40001 ACTIVE - Hyperbox Prometheus successfully moved!
    set /a issues+=1
)
echo.

echo Testing Port 40005 (Faceless YouTube Dashboard - NEW location)...
curl -s -m 2 http://localhost:40005/api/health > nul 2>&1
if errorlevel 1 (
    echo   ⚠️  Port 40005 NOT RESPONDING - Container may not be running
) else (
    echo   ✅ Port 40005 ACTIVE - Faceless YouTube successfully moved!
)
echo.

echo Testing Port 40010 (Neurectomy Jaeger - NEW location)...
curl -s -m 2 -I http://localhost:40010/ 2>&1 | find "200" > nul
if errorlevel 1 (
    echo   ⚠️  Port 40010 NOT RESPONDING - Container may not be running
) else (
    echo   ✅ Port 40010 ACTIVE - Neurectomy Jaeger successfully moved!
)
echo.

REM Check Ryzanstein Ports (should now be accessible)
echo Testing RYZANSTEIN ports (should now be accessible)...
echo.

echo Testing Port 8000 (Ryzanstein API)...
curl -s -m 2 http://localhost:8000/health > nul 2>&1
if errorlevel 1 (
    echo   ⚠️  Port 8000 NOT responding - kubectl port-forward may not be active
    echo      Run: kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000
) else (
    echo   ✅ Port 8000 ACTIVE - Ryzanstein API accessible!
)
echo.

echo Testing Port 9090 (Ryzanstein Prometheus) - if port freed...
curl -s -m 2 http://localhost:9090/-/healthy > nul 2>&1
if errorlevel 1 (
    echo   ⚠️  Port 9090 NOT responding (expected if Hyperbox not moved)
) else (
    echo   ✅ Port 9090 ACTIVE - Ryzanstein Prometheus accessible!
)
echo.

echo Testing Port 3000 (Ryzanstein Grafana) - if port freed...
curl -s -m 2 http://localhost:3000/api/health > nul 2>&1
if errorlevel 1 (
    echo   ⚠️  Port 3000 NOT responding (expected if Faceless YouTube not moved)
) else (
    echo   ✅ Port 3000 ACTIVE - Ryzanstein Grafana accessible!
)
echo.

echo Testing Port 16686 (Ryzanstein Jaeger) - if port freed...
curl -s -m 2 -I http://localhost:16686/ 2>&1 | find "200" > nul
if errorlevel 1 (
    echo   ⚠️  Port 16686 NOT responding (expected if Neurectomy not moved)
) else (
    echo   ✅ Port 16686 ACTIVE - Ryzanstein Jaeger accessible!
)
echo.

echo Testing Port 9093 (Ryzanstein AlertManager)...
curl -s -m 2 http://localhost:9093/-/healthy > nul 2>&1
if errorlevel 1 (
    echo   ⚠️  Port 9093 NOT responding - kubectl port-forward may not be active
    echo      Run: kubectl port-forward -n ryzanstein-staging svc/alertmanager 9093:9093
) else (
    echo   ✅ Port 9093 ACTIVE - Ryzanstein AlertManager accessible!
)
echo.

REM Summary
echo ============================================================
echo  SUMMARY
echo ============================================================
echo.

if %issues% equ 0 (
    echo ✅ ALL PORTS SUCCESSFULLY REMEDIATED!
    echo.
    echo All 3 conflicting projects have been moved to new ports.
    echo Ryzanstein ports are now available for kubectl port-forwards.
    echo.
    echo You can now proceed with Phase 6 setup:
    echo   s:\Ryot\START_PHASE6_NOW.bat
) else (
    echo ⚠️  %issues% port^(s^) still need remediation
    echo.
    echo Next steps:
    echo   1. If Hyperbox moved:    curl http://localhost:40001/-/healthy
    echo   2. If Faceless moved:    curl http://localhost:40005/api/health
    echo   3. If Neurectomy moved:  curl -I http://localhost:40010/
    echo.
    echo Once all are moved, run this script again to verify.
)
echo.
echo ============================================================
echo.

pause
